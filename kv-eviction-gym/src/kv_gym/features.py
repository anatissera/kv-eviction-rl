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


# ── Rich features (Phase 2, experiment E1) ──────────────────────────────────────
# The baseline per-token feature is just K||V, which the PerTokenMLP then LayerNorms
# — erasing magnitude, the exact signal `kv_norm` evicts on (see policy.py). Rich
# features append scale/position columns that BYPASS the LayerNorm so the policy can
# represent (and beat) the norm heuristic:
#   [0] kz  = within-observation standardized ||K|| (z-score over resident slots)
#   [1] vz  = within-observation standardized ||V||
#   [2] pos = slot / cache_size            (absolute position, 0=oldest)
#   [3] rec = (cache_size-1-slot)/max_len  (recency, 0=most recent)
# Standardized (not raw) norms give a scale-free "is this token's norm unusually
# small vs its neighbours" signal — directly what kv_norm's argmin needs — and are
# stable across the 5–50× per-layer norm variation.
N_EXTRA_RICH = 4


def extra_feature_dim(rich: bool) -> int:
    return N_EXTRA_RICH if rich else 0


def feature_dim(n_kv_heads: int, head_dim: int, rich: bool = False) -> int:
    """Total features per token: K and V halves (all KV-heads concatenated),
    plus the rich scale/position columns when `rich` is True."""
    return 2 * n_kv_heads * head_dim + extra_feature_dim(rich)


def build_extra_columns(
    K: Tensor,          # [B, H, S, D]  (B = n_envs in eval, or 1 per-layer in training)
    V: Tensor,          # [B, H, S, D]
    cache_size: int,
    max_len: int,
) -> np.ndarray:        # [B, S, N_EXTRA_RICH] float32
    """Compute the rich extra columns for the S resident slots. SINGLE source of
    truth used by BOTH training (`batched_env._obs_episode`) and eval
    (`build_obs`) so the two paths are byte-identical (no train/eval confound)."""
    kmean = K.float().norm(dim=-1).mean(dim=1)   # [B, S]  mean over heads of ||K||
    vmean = V.float().norm(dim=-1).mean(dim=1)   # [B, S]
    B, S = kmean.shape

    def _z(x: Tensor) -> Tensor:
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True)
        return (x - mu) / (sd + 1e-6)

    kz  = _z(kmean)                                        # [B, S]
    vz  = _z(vmean)
    slot = torch.arange(S, dtype=torch.float32).unsqueeze(0).expand(B, S)
    pos = slot / max(cache_size, 1)
    rec = (cache_size - 1 - slot).clamp(min=0) / max(max_len, 1)
    cols = torch.stack([kz, vz, pos, rec], dim=-1)         # [B, S, 4]
    return cols.cpu().numpy().astype(np.float32)


def build_obs(
    K:             Tensor,           # [n_envs, n_kv_heads, T, head_dim]
    V:             Tensor,           # [n_envs, n_kv_heads, T, head_dim]
    prompt_len:    int,
    max_len:       int,
    resident_mask: Tensor | None = None,  # [n_envs, T] bool — False = already evicted
    rich:          bool = False,
) -> np.ndarray:                     # [n_envs, max_len, 2*n_kv_heads*head_dim (+N_EXTRA_RICH)]
    """Build the observation array for all environments at once.

    Heads are concatenated along the feature axis: [K_h0 | K_h1 | V_h0 | V_h1].
    When `rich`, N_EXTRA_RICH scale/position columns are appended (see
    build_extra_columns). Evicted positions (resident_mask==False) are zeroed.
    """
    n_envs, H, _, head_dim = K.shape
    kv_dim = 2 * H * head_dim
    fdim   = kv_dim + extra_feature_dim(rich)
    obs    = np.zeros((n_envs, max_len, fdim), dtype=np.float32)

    # [n_envs, H, T, D] → [n_envs, T, H*D]
    K_flat = K[:, :, :prompt_len, :].permute(0, 2, 1, 3).reshape(n_envs, prompt_len, H * head_dim)
    V_flat = V[:, :, :prompt_len, :].permute(0, 2, 1, 3).reshape(n_envs, prompt_len, H * head_dim)

    obs[:, :prompt_len, :H*head_dim]           = K_flat.cpu().numpy()
    obs[:, :prompt_len, H*head_dim:2*H*head_dim] = V_flat.cpu().numpy()

    if rich:
        cols = build_extra_columns(K[:, :, :prompt_len, :], V[:, :, :prompt_len, :],
                                   prompt_len, max_len)          # [n_envs, prompt_len, 4]
        obs[:, :prompt_len, kv_dim:kv_dim + N_EXTRA_RICH] = cols

    if resident_mask is not None:
        evicted = ~resident_mask[:, :prompt_len].cpu().numpy()  # [n_envs, T]
        obs[:, :prompt_len][evicted] = 0.0

    return obs
