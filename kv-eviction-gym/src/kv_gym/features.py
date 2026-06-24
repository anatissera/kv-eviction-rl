"""
Per-token observation builder.

Feature set per token position t:
    K_head0[t] || K_head1[t]  (n_kv_heads * head_dim,)  all key vectors concatenated
    V_head0[t] || V_head1[t]  (n_kv_heads * head_dim,)  all value vectors concatenated

feature_dim = 2 * n_kv_heads * head_dim  (256 for Qwen2.5-1.5B: 2 heads × 64 dim × K+V)

Heads are concatenated, not averaged.  Averaging is lossy — heads that encode
different token aspects can have K vectors pointing in different directions, and
their mean can cancel or be geometrically meaningless.  Concatenation lets the
downstream MLP learn per-head weights independently.

Position is already encoded in K via RoPE rotation, so an explicit t/max_len
scalar is redundant.

The observation is zero-padded beyond the current cache size.  The action mask
prevents the policy from selecting those padded positions.
"""

import numpy as np
import torch
from torch import Tensor


def feature_dim(n_kv_heads: int, head_dim: int) -> int:
    """Total features per token: K and V halves, all KV-heads concatenated."""
    return 2 * n_kv_heads * head_dim


def build_obs(
    K:             Tensor,           # [n_envs, n_kv_heads, T, head_dim]
    V:             Tensor,           # [n_envs, n_kv_heads, T, head_dim]
    prompt_len:    int,
    max_len:       int,
    resident_mask: Tensor | None = None,  # [n_envs, T] bool — False = already evicted
) -> np.ndarray:                     # [n_envs, max_len, 2*n_kv_heads*head_dim]
    """Build the observation array for all environments at once.

    Heads are concatenated along the feature axis: [K_h0 | K_h1 | V_h0 | V_h1].
    Evicted positions (resident_mask==False) are zeroed.
    """
    n_envs, H, _, head_dim = K.shape
    fdim  = 2 * H * head_dim
    obs   = np.zeros((n_envs, max_len, fdim), dtype=np.float32)

    # [n_envs, H, T, D] → [n_envs, T, H*D]
    K_flat = K[:, :, :prompt_len, :].permute(0, 2, 1, 3).reshape(n_envs, prompt_len, H * head_dim)
    V_flat = V[:, :, :prompt_len, :].permute(0, 2, 1, 3).reshape(n_envs, prompt_len, H * head_dim)

    obs[:, :prompt_len, :H*head_dim]           = K_flat.cpu().numpy()
    obs[:, :prompt_len, H*head_dim:2*H*head_dim] = V_flat.cpu().numpy()

    if resident_mask is not None:
        evicted = ~resident_mask[:, :prompt_len].cpu().numpy()  # [n_envs, T]
        obs[:, :prompt_len][evicted] = 0.0

    return obs
