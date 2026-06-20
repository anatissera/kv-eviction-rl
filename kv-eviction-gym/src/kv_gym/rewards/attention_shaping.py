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

Attention backend compatibility
-------------------------------
output_attentions=True requires eager attention to return non-empty matrices.
SDPA (default on CPU/MPS) and flash_attention_2 (CUDA) both return empty
attention tuples.  When that happens, we automatically fall back to a
KV-norm-based importance proxy:

    importance[t] ∝ mean_over_layers_heads( ||K[t]|| + ||V[t]|| )

This proxy is the "heavy hitter" heuristic used by H2O / SnapKV and is a
reasonable stand-in when true attention weights are unavailable.  It never
returns None — shaping always provides a signal.
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

    Strategy (in order of preference):
      1. Full-context generate with output_attentions=True — true future attention.
         Requires attn_implementation='eager'.
      2. KV-norm proxy (||K[t]|| + ||V[t]||) — always available, same signal as
         the oracle baseline in eval.py.  Used when option 1 returns empty attentions
         (SDPA / flash_attention_2).

    Args:
        model:          The causal LM.
        input_ids:      Full prompt ids [1, T].
        K, V:           Pre-computed KV tensors from capture(); used for fallback.
                        Pass None to skip fallback and return uniform distribution.
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

        warnings.warn(
            "[attention_shaping] output_attentions returned empty tensors "
            f"(attn_implementation='{getattr(model.config, '_attn_implementation', '?')}')."
            " Using KV-norm proxy instead. "
            "Set attn_implementation='eager' for true future-attention shaping."
        )

    except (ValueError, NotImplementedError, RuntimeError) as exc:
        warnings.warn(
            f"[attention_shaping] output_attentions not supported ({type(exc).__name__}). "
            "Using KV-norm proxy instead."
        )

    # ---- Fallback: KV-norm proxy ----
    if K is not None and V is not None:
        return _importance_from_kv(K, V)

    # Last resort: uniform
    return torch.full((T,), 1.0 / T)


def attention_alignment(
    importance:   Tensor,  # [T], sums to 1
    kept_indices: Tensor,  # [k] long
) -> float:
    """Fraction of total attention mass captured by the kept token set.

    Range: [0, 1].  Perfect alignment = 1.0 (all important tokens kept).
    """
    return importance[kept_indices].sum().item()
