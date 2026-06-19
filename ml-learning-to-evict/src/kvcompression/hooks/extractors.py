#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from concurrent.futures import Executor
from pathlib import Path
from typing import Optional

import safetensors.torch
import torch
from omegaconf import DictConfig
from torch import nn

from kvcompression.hooks.registrar import register_attention_forward_hook


class ExtractAttentionTensors:
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        sample_idx: int,
        output_path: Path,
        executor: Optional[Executor] = None,
        disable_kv_caching: bool = True,
    ):
        self.cfg = cfg
        self._model = model
        self._context_manager = register_attention_forward_hook(
            model=self._model,
            hook=self.store_attention_tensors,
            attention_wrapper=self.wrapped_attention_call,
        )
        self.executor = executor
        self.disable_kv_caching = disable_kv_caching

        self.sample_idx = sample_idx
        self.output_path = output_path
        self.output_path.mkdir(exist_ok=True, parents=True)

    def __enter__(self):
        if self.disable_kv_caching:
            self.original_cache_enabled = self._model.layers[0].attn.cache_enabled
            for layer in self._model.layers:
                layer.attn.cache_enabled = False
        return self._context_manager.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable_kv_caching:
            for layer in self._model.layers:
                layer.attn.cache_enabled = self.original_cache_enabled
        return self._context_manager.__exit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def wrapped_attention_call(
        q, k, v, mask, dropout_p, is_causal, original_attention_call, attn_module
    ):
        """Wrap the attention call in order to extract its inputs.

        This enables storing the KQV without having to recompute them from scratch.
        """
        attn_module.forward_q = q
        attn_module.forward_k = k
        attn_module.forward_v = v
        attn_module.forward_mask = mask
        attn_module.forward_dropout_p = dropout_p
        attn_module.forward_is_causal = is_causal
        return original_attention_call(
            q,
            k,
            v,
            mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    def _store_attention_tensors(
        self,
        layer_idx: int,
        attn_module: nn.Module,
        input_pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        num_heads = attn_module.num_heads
        num_kv_heads = attn_module.num_kv_heads
        heads_per_group = num_heads // num_kv_heads

        save_path = self.output_path / f"layer_{layer_idx:06d}"
        save_path.mkdir(exist_ok=True, parents=True)

        # Store input_pos once per layer
        safetensors.torch.save_file(
            {"tensor": input_pos}, save_path / "input_pos.safetensors"
        )

        if k.shape[0] == num_heads:
            # K/V are expanded to match Q - extract unique KV heads
            kv_indices = list(range(0, num_heads, heads_per_group))

            for kv_head_idx in range(num_kv_heads):
                kv_head_path = save_path / f"kv_head_{kv_head_idx:03d}"
                kv_head_path.mkdir(exist_ok=True, parents=True)

                start_q_idx = kv_indices[kv_head_idx]
                k_tensor = k[start_q_idx].contiguous()
                v_tensor = v[start_q_idx].contiguous()

                end_q_idx = start_q_idx + heads_per_group
                q_group = q[start_q_idx:end_q_idx].contiguous()

                attention_tensors = {"k": k_tensor, "v": v_tensor, "q_group": q_group}
                safetensors.torch.save_file(
                    attention_tensors, kv_head_path / "attention_tensors.safetensors"
                )
        else:
            # K/V are not expanded - they have num_kv_heads directly
            assert k.shape[0] == num_kv_heads, (
                f"Expected {num_kv_heads} KV heads, got {k.shape[0]}"
            )
            assert v.shape[0] == num_kv_heads, (
                f"Expected {num_kv_heads} KV heads, got {v.shape[0]}"
            )

            for kv_head_idx in range(num_kv_heads):
                kv_head_path = save_path / f"kv_head_{kv_head_idx:03d}"
                kv_head_path.mkdir(exist_ok=True, parents=True)

                k_tensor = k[kv_head_idx].clone()
                v_tensor = v[kv_head_idx].clone()

                start_q_idx = kv_head_idx * heads_per_group
                end_q_idx = start_q_idx + heads_per_group
                q_group = q[start_q_idx:end_q_idx].clone()

                attention_tensors = {"k": k_tensor, "v": v_tensor, "q_group": q_group}
                safetensors.torch.save_file(
                    attention_tensors, kv_head_path / "attention_tensors.safetensors"
                )

    def store_attention_tensors(
        self, module: nn.Module, args, kwargs, outputs, layer_idx: int
    ):
        x, y = args
        assert torch.allclose(x, y)
        assert module.forward_dropout_p == 0

        input_pos = kwargs["input_pos"]
        q = module.forward_q
        k = module.forward_k
        v = module.forward_v

        input_pos = input_pos.detach().to("cpu").squeeze(0)
        q = q.detach().to("cpu").squeeze(0)
        k = k.detach().to("cpu").squeeze(0)
        v = v.detach().to("cpu").squeeze(0)

        self._store_attention_tensors(
            layer_idx=layer_idx,
            attn_module=module,
            input_pos=input_pos,
            q=q,
            k=k,
            v=v,
        )
