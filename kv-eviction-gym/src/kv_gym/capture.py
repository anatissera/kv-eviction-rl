"""
Run one LLM prefill+decode on a GSM8K example and capture all per-head tensors.

One call to `capture()` yields Q, K, V, and future-attention weights for
every (layer, head) pair simultaneously. SharedKVVecEnv calls this once
per episode reset to feed all 56 environments.

Strategy: two forward passes, both with hooks on each layer's self_attn module.
  Pass 1 (prefill only):   capture Q, K, V for every head via q/k/v_proj hooks.
  Pass 2 (greedy decode):  accumulate per-token cross-attention from new tokens
                            back onto prompt positions to build future_attn.

We avoid output_attentions=True during generate() because HuggingFace's
attention output format differs between eager/sdpa/flash backends and
across transformer versions. Hooks are backend-agnostic.

Requires attn_implementation="eager" so the model exposes q/k/v_proj.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from kv_gym.vendor.prompts import format_gsm8k


@dataclass
class AllHeadCapture:
    Q:            Tensor  # [n_layers, n_heads, prompt_len, head_dim]
    K:            Tensor  # [n_layers, n_heads, prompt_len, head_dim]
    V:            Tensor  # [n_layers, n_heads, prompt_len, head_dim]
    future_attn:  Tensor  # [n_layers, n_heads, prompt_len]  normalized to sum=1
    prompt_len:   int
    gold_answer:  str


def capture(
    model,
    tokenizer,
    example: dict,
    device: torch.device,
    max_new_tokens: int = 64,
) -> AllHeadCapture:
    """Two-pass capture: Q/K/V from prefill, future_attn from generation.

    Args:
        model:          HuggingFace CausalLM with attn_implementation="eager".
        tokenizer:      Matching tokenizer.
        example:        GSM8K dict with "prompt_text" and "gold_answers".
        device:         Model device.
        max_new_tokens: Generation length for future_attn accumulation.

    Returns:
        AllHeadCapture with all tensors on CPU.
    """
    prompt_text, _ = format_gsm8k(example)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    n_layers = model.config.num_hidden_layers
    n_heads  = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    head_dim = model.config.hidden_size // n_heads

    # ------------------------------------------------------------------ #
    # Pass 1: prefill → capture Q, K, V for all heads                    #
    # ------------------------------------------------------------------ #

    captured_qkv: dict[int, dict[str, Tensor]] = {}

    def make_qkv_hook(layer_idx: int):
        def hook(module, args, kwargs, output):
            with torch.no_grad():
                # args[0] is hidden_states: [batch, seq, hidden]
                h = args[0] if args else kwargs.get("hidden_states")
                if h is None:
                    return
                q = module.q_proj(h)
                k = module.k_proj(h)
                v = module.v_proj(h)

                bsz, seq, _ = q.shape
                H   = module.num_heads
                KVH = module.num_key_value_heads
                D   = module.head_dim

                q = q.view(bsz, seq, H,   D).squeeze(0).permute(1, 0, 2)   # [H, T, D]
                k = k.view(bsz, seq, KVH, D).squeeze(0).permute(1, 0, 2)   # [KVH, T, D]
                v = v.view(bsz, seq, KVH, D).squeeze(0).permute(1, 0, 2)   # [KVH, T, D]

                if KVH != H:
                    repeats = H // KVH
                    k = k.repeat_interleave(repeats, dim=0)
                    v = v.repeat_interleave(repeats, dim=0)

                captured_qkv[layer_idx] = {
                    "Q": q.cpu(), "K": k.cpu(), "V": v.cpu()
                }
        return hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_hook(make_qkv_hook(i), with_kwargs=True)
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    Q_all = torch.stack([captured_qkv[i]["Q"] for i in range(n_layers)])  # [L, H, T, D]
    K_all = torch.stack([captured_qkv[i]["K"] for i in range(n_layers)])
    V_all = torch.stack([captured_qkv[i]["V"] for i in range(n_layers)])

    # ------------------------------------------------------------------ #
    # Pass 2: greedy decode → accumulate future_attn via attention hooks  #
    # ------------------------------------------------------------------ #
    # For each generated token, we capture the softmax attention weights
    # from that one new query position over all past key positions.
    # We sum those weights (over the prompt_len positions only) to get
    # a proxy for how important each prompt token is to the generation.

    future_attn = torch.zeros(n_layers, n_heads, prompt_len)

    def make_attn_hook(layer_idx: int):
        def hook(module, args, kwargs, output):
            with torch.no_grad():
                h = args[0] if args else kwargs.get("hidden_states")
                if h is None or h.shape[1] == prompt_len:
                    # Skip the prefill step (seq == prompt_len)
                    return
                # h: [1, 1, hidden] — single new token
                q_new = module.q_proj(h)                              # [1, 1, H*D]
                H   = module.num_heads
                KVH = module.num_key_value_heads
                D   = module.head_dim
                q_new = q_new.view(1, H, D)                           # [1, H, D]

                # Retrieve cached K up to this point and slice prompt portion
                # We use K_all (prompt keys) as a proxy — this avoids needing
                # access to the running KV cache during generation.
                K_prompt = K_all[layer_idx, :, :, :].to(device)      # [H, T, D]
                # q_new: [1, H, D] → [H, 1, D]
                q_new = q_new.permute(1, 0, 2)                        # [H, 1, D]
                scale = D ** -0.5
                logits = torch.bmm(q_new, K_prompt.transpose(1, 2)) * scale  # [H, 1, T]
                weights = torch.softmax(logits, dim=-1)               # [H, 1, T]
                future_attn[layer_idx] += weights[:, 0, :].cpu()     # [H, T]
        return hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_hook(make_attn_hook(i), with_kwargs=True)
        hooks.append(h)

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    for h in hooks:
        h.remove()

    # Normalize so each head's future_attn sums to 1
    row_sum = future_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    future_attn = future_attn / row_sum

    return AllHeadCapture(
        Q=Q_all,
        K=K_all,
        V=V_all,
        future_attn=future_attn,
        prompt_len=prompt_len,
        gold_answer=example["gold_answers"][0],
    )
