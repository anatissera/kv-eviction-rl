"""
Build the per-token observation tensor for one episode step.

feature_dim = 2 * head_dim + 5
For Qwen2-1.5B (head_dim=128): 261 dims per token position.

Per-token features (for position i):
    K[i]           (head_dim,)   key vector, normalized per-head per-episode
    V[i]           (head_dim,)   value vector, normalized per-head per-episode
    attn_score[i]  (1,)          attention mass ∈ [0, 1], 0 for evicted
    position[i]    (1,)          i / max_len  (relative position)
    is_resident[i] (1,)          1 if alive, 0 if evicted (mirrors action mask)

Global context (same value broadcast to all token rows):
    layer_frac     (1,)          layer_idx / n_layers
    head_frac      (1,)          head_idx  / n_heads

The fixed-size output [n_envs, max_len, feature_dim] is zero-padded for
positions beyond prompt_len. The action mask already prevents sampling from
those positions, so padding is safe.
"""

import numpy as np
import torch
from torch import Tensor


def build_obs(
    K:          Tensor,   # [n_envs, max_len, head_dim]
    V:          Tensor,   # [n_envs, max_len, head_dim]
    attn_score: Tensor,   # [n_envs, max_len]
    resident:   Tensor,   # [n_envs, max_len]  bool
    layer_frac: Tensor,   # [n_envs]
    head_frac:  Tensor,   # [n_envs]
    prompt_len: int,
    max_len:    int,
) -> np.ndarray:          # [n_envs, max_len, feature_dim]
    """Build the observation array for all environments at once."""
    n_envs, _, head_dim = K.shape
    feature_dim = 2 * head_dim + 5

    obs = np.zeros((n_envs, max_len, feature_dim), dtype=np.float32)

    # Normalize K and V per head so that scale differences don't dominate.
    # Divide by the per-head RMS norm (computed over the prompt_len tokens).
    K_norm = _rms_normalize(K[:, :prompt_len, :])   # [n_envs, prompt_len, head_dim]
    V_norm = _rms_normalize(V[:, :prompt_len, :])   # [n_envs, prompt_len, head_dim]

    K_np  = K_norm.cpu().numpy()
    V_np  = V_norm.cpu().numpy()
    attn  = attn_score[:, :prompt_len].cpu().numpy()  # [n_envs, prompt_len]
    res   = resident[:, :prompt_len].float().cpu().numpy()

    positions = np.arange(prompt_len, dtype=np.float32) / max(max_len - 1, 1)
    positions = positions[None, :, None]   # [1, prompt_len, 1]
    positions = np.broadcast_to(positions, (n_envs, prompt_len, 1))

    lf = layer_frac.cpu().numpy()[:, None, None]   # [n_envs, 1, 1]
    hf = head_frac.cpu().numpy()[:, None, None]

    # Fill feature slots
    obs[:, :prompt_len, :head_dim]              = K_np
    obs[:, :prompt_len, head_dim:2*head_dim]    = V_np
    obs[:, :prompt_len, 2*head_dim:2*head_dim+1] = attn[:, :, None]
    obs[:, :prompt_len, 2*head_dim+1:2*head_dim+2] = positions
    obs[:, :prompt_len, 2*head_dim+2:2*head_dim+3] = res[:, :, None]
    obs[:, :prompt_len, 2*head_dim+3:2*head_dim+4] = np.broadcast_to(lf, (n_envs, prompt_len, 1))
    obs[:, :prompt_len, 2*head_dim+4:2*head_dim+5] = np.broadcast_to(hf, (n_envs, prompt_len, 1))

    return obs


def _rms_normalize(x: Tensor) -> Tensor:
    """Normalize by RMS over the token dimension (per env, per feature)."""
    rms = x.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-6)
    return x / rms
