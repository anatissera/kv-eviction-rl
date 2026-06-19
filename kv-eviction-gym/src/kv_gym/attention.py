"""
Online batched attention recomputation after each eviction step.

At each step one token is evicted per (layer, head). We recompute the
attention scores exactly — no approximation — across all heads at once
using einsum. This is O(T² · D) per step, about 5M FLOPs for T=200, D=128:
microseconds on GPU, fast enough on CPU for development.

The output is the attention mass each surviving token receives from *all*
query positions (summed, not averaged). This is used as a per-token
importance signal in the observation.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def recompute_all_heads(
    Q:        Tensor,   # [n_layers, n_heads, prompt_len, head_dim]
    K:        Tensor,   # [n_layers, n_heads, prompt_len, head_dim]
    resident: Tensor,   # [n_layers, n_heads, prompt_len]  bool
    scale:    float | None = None,
) -> Tensor:            # [n_layers, n_heads, prompt_len]  — 0 for evicted
    """Batched exact attention recompute across all (layer, head) pairs.

    For each (l, h), computes softmax(Q[l,h] @ K[l,h,resident[l,h]].T / scale)
    and returns the column-sum (total attention mass per surviving key).
    Evicted positions receive 0.

    Args:
        Q:        Query tensors from the prefill (fixed throughout episode).
        K:        Key tensors from the prefill (fixed throughout episode).
        resident: Boolean mask — True means the token is still in the cache.
        scale:    1 / sqrt(head_dim). Computed from Q if not provided.

    Returns:
        Tensor [L, H, T] with attention mass per token (0 for evicted).
    """
    L, H, T, D = Q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    device = Q.device
    out = torch.zeros(L, H, T, device=device, dtype=Q.dtype)

    # We iterate over (layer, head) pairs rather than using a single batched
    # einsum because each head has a different resident mask (different set of
    # surviving tokens), making it hard to batch naively without padding.
    # For T=200 and 56 heads this loop is still fast (~1ms on CPU).
    for l in range(L):
        for h in range(H):
            res = resident[l, h]        # [T]  bool
            K_res = K[l, h][res]        # [k, D]  where k = res.sum()
            if K_res.shape[0] == 0:
                continue

            # Q: [T, D], K_res: [k, D]  → logits: [T, k]
            logits = Q[l, h] @ K_res.T * scale   # [T, k]
            weights = F.softmax(logits, dim=-1)   # [T, k]

            # Sum attention mass each key receives across all queries.
            # weights.sum(0): [k] — scatter back to full length
            attn_mass = torch.zeros(T, device=device, dtype=Q.dtype)
            attn_mass[res] = weights.sum(0)
            out[l, h] = attn_mass

    return out
