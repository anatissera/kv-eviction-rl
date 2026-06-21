"""
Attention-based reward shaping for KV-cache eviction.

Runs one clean (full-context) generate per episode to collect the attention
weights from generated tokens back to prompt positions.  The result is a
per-token "importance" distribution: importance[t] measures how much the
model attended to prompt position t while producing the answer.

This importance distribution serves as a dense reward-shaping signal that
supplements the sparse 0/1 correctness reward:

    terminal_reward = (1 - weight) * correctness + weight * alignment

where alignment = sum(importance[t] for t in kept_tokens) ∈ [0, 1].

Cost: one extra model.generate() call per episode reset.

Attention backend requirement
-----------------------------
output_attentions=True requires attn_implementation='eager' to return
non-empty matrices.  SDPA (default on CPU/MPS) and flash_attention_2 (CUDA)
return empty attention tuples; compute_token_importance will raise RuntimeError
rather than silently fall back.  Use use_attention_shaping=False to skip the
reference run and rely on pure correctness reward instead.
"""

import warnings
from typing import Optional

import torch
from torch import Tensor


def _importance_from_attentions(
    attentions,         # tuple[n_steps] of tuple[n_layers] of Tensor[1,H_q,1,seq]
    T: int,
    n_kv_heads: int,    # number of KV-heads (GQA groups)
) -> Optional[Tensor]:
    """Extract per-prompt-token importance from generate() attention output.

    GQA handling (SnapKV / Learning-to-Evict convention):
    Attention is shaped [1, H_q, 1, seq] with H_q query heads grouped into
    n_kv_heads KV groups of size H_q/n_kv_heads.  Within each group we take
    MAX before averaging over KV-heads: "keep a token if ANY query head in
    the group attends to it."  Plain mean would dilute tokens that are
    critical to just one head.
    """
    importance = torch.zeros(T, dtype=torch.float32)
    n_terms = 0

    for step_attns in attentions:
        for layer_attn in step_attns:
            if layer_attn is None:
                continue
            # layer_attn: [1, H_q, 1, seq_len]
            seq_len  = layer_attn.shape[-1]
            n_prompt = min(T, seq_len)
            n_q      = layer_attn.shape[1]

            per_q = layer_attn[0, :, 0, :n_prompt]  # [H_q, T_prompt]

            if n_q % n_kv_heads == 0:
                group_size = n_q // n_kv_heads
                # [H_kv, group_size, T_prompt] → max within group → [H_kv, T_prompt]
                per_kv = per_q.view(n_kv_heads, group_size, n_prompt).amax(dim=1)
                attn_to_prompt = per_kv.mean(dim=0).cpu()  # mean over KV-heads
            else:
                # Fallback if model layout differs from expected GQA structure
                attn_to_prompt = per_q.mean(dim=0).cpu()

            importance[:n_prompt] += attn_to_prompt
            n_terms += 1

    if n_terms == 0:
        return None

    importance /= n_terms
    total = importance.sum()
    if total > 1e-8:
        importance /= total
    return importance


def _importance_from_kv(K: Tensor, V: Tensor) -> Tensor:
    """KV-norm fallback: importance[t] ∝ max over layers/heads of ||K[t]|| + ||V[t]||.

    K: [n_layers, n_kv_heads, T, head_dim]
    V: [n_layers, n_kv_heads, T, head_dim]
    Returns [T] normalised importance.

    Max (not mean) follows the SnapKV convention: a token's importance is
    determined by whichever (layer, head) finds it most salient, not the
    average.  Mean would dilute tokens critical to specific heads.
    """
    score = (K.norm(dim=-1) + V.norm(dim=-1)).amax(dim=(0, 1))  # [T]
    total = score.sum()
    if total > 1e-8:
        score = score / total
    return score.cpu()


def compute_token_importance(
    model,
    input_ids:      Tensor,            # [1, prompt_len]
    K:              Optional[Tensor],  # [n_layers, n_kv_heads, T, head_dim], or None
    V:              Optional[Tensor],  # [n_layers, n_kv_heads, T, head_dim], or None
    max_new_tokens: int = 64,
    device:         torch.device | None = None,
) -> Tensor:
    """Return per-prompt-token importance as a [T] tensor summing to 1.

    Runs model.generate() with output_attentions=True and aggregates attention
    weights across steps and layers (GQA-aware: max within each KV group, then
    mean over KV-heads).  Requires attn_implementation='eager'.

    Raises RuntimeError if output_attentions returns empty tensors (SDPA / flash).

    Args:
        model:          The causal LM.
        input_ids:      Full prompt ids [1, T].
        K, V:           Unused — kept for call-site compatibility.
        max_new_tokens: Tokens to generate in the clean reference run.
        device:         Target device.
    """
    if device is None:
        device = next(model.parameters()).device

    T = input_ids.shape[1]
    # Infer n_kv_heads from K shape if available, else fall back to model config
    if K is not None:
        n_kv_heads = K.shape[1]
    else:
        n_kv_heads = getattr(model.config, "num_key_value_heads",
                             model.config.num_attention_heads)

    # ---- Attempt 1: generate with output_attentions ----
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        if hasattr(out, "attentions") and out.attentions:
            imp = _importance_from_attentions(out.attentions, T, n_kv_heads)
            if imp is not None:
                return imp

        impl = getattr(model.config, "_attn_implementation", "?")
        raise RuntimeError(
            f"[attention_shaping] output_attentions=True returned empty tensors "
            f"(attn_implementation='{impl}'). "
            "Switch to attn_implementation='eager' to enable true future-attention shaping, "
            "or set use_attention_shaping=False to disable it."
        )

    except (ValueError, NotImplementedError, RuntimeError) as exc:
        raise RuntimeError(
            f"[attention_shaping] output_attentions not supported: {exc}. "
            "Switch to attn_implementation='eager' or set use_attention_shaping=False."
        ) from exc


def attention_alignment(
    importance:   Tensor,  # [T], sums to 1
    kept_indices: Tensor,  # [k] long
) -> float:
    """Fraction of total attention mass captured by the kept token set.

    Range: [0, 1].  Perfect alignment = 1.0 (all important tokens kept).
    """
    return importance[kept_indices].sum().item()
