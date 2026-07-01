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
# — erasing magnitude AND position, the exact signals `kv_norm`/recency evict on
# (see policy.py). Rich features append scale/position columns that BYPASS the
# LayerNorm so the policy can represent (and beat) the norm heuristic.
#
# Column layout (2*H + 4 columns, H = n_kv_heads):
#   [0    : H  ]  kz_h    per-head standardized ||K||  (z-score over resident slots)
#   [H    : 2H ]  vz_h    per-head standardized ||V||
#   [2H       ]  kz_mean  standardized mean-over-heads ||K||  (exactly kv_norm's signal)
#   [2H+1     ]  vz_mean  standardized mean-over-heads ||V||
#   [2H+2     ]  rec      (cache_size-1-slot)/max_len   (recency, 0 = most recent)
#   [2H+3     ]  pos_orig original token position / POS_SCALE  (D2: prompt-vs-generated)
#
# WHY standardized (not raw) norms: scale-free "is this token's norm unusually small
# vs its neighbours" — directly what kv_norm's argmin needs — and stable across the
# 5–50× per-layer norm variation (the policy is LAYER-SHARED, so a raw high-norm
# layer would dominate). WHY per-head AND mean: heads encode different aspects; the
# mean is given explicitly so the policy can replicate kv_norm exactly. WHY pos_orig
# (original position, not slot index): slot index is post-compaction and loses the
# prompt/generated distinction; original position (0..T = prompt, >T = generated)
# recovers it. Normalized by a fixed POS_SCALE so train/eval need no prompt_len.
POS_SCALE = 1024.0   # > max T (232) + max_new_tokens (600); keeps pos_orig in [0,1)


def extra_feature_dim(rich: bool, n_kv_heads: int = 2) -> int:
    return (2 * n_kv_heads + 4) if rich else 0


def feature_dim(n_kv_heads: int, head_dim: int, rich: bool = False) -> int:
    """Total features per token: K and V halves (all KV-heads concatenated),
    plus the rich scale/position columns when `rich` is True."""
    return 2 * n_kv_heads * head_dim + extra_feature_dim(rich, n_kv_heads)


def build_extra_columns(
    K: Tensor,          # [B, H, S, D]  (B = n_envs in eval, or 1 per-layer in training)
    V: Tensor,          # [B, H, S, D]
    cache_size: int,
    max_len: int,
    orig_pos=None,      # [B, S] int array of each slot's ORIGINAL token position (or None)
) -> np.ndarray:        # [B, S, 2H+4] float32
    """Compute the rich extra columns for the S resident slots. SINGLE source of
    truth used by BOTH training (`batched_env._obs_episode`) and eval
    (`build_obs`) so the two paths are byte-identical (no train/eval confound).

    `orig_pos` is each slot's original sequence position (from the env's
    slot_to_pos in training, or the eval PositionTracker). If None, falls back to
    a slot-index proxy (kept only for callers that don't track positions)."""
    Knorm = K.float().norm(dim=-1)   # [B, H, S]  ||K|| per head
    Vnorm = V.float().norm(dim=-1)   # [B, H, S]
    B, H, S = Knorm.shape

    def _z(x: Tensor) -> Tensor:                 # standardize over the slot dim (-1)
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True)
        return (x - mu) / (sd + 1e-6)

    kz_h    = _z(Knorm).permute(0, 2, 1)         # [B, S, H]  per-head z
    vz_h    = _z(Vnorm).permute(0, 2, 1)         # [B, S, H]
    kz_mean = _z(Knorm.mean(dim=1)).unsqueeze(-1)  # [B, S, 1]  z of mean-over-heads
    vz_mean = _z(Vnorm.mean(dim=1)).unsqueeze(-1)

    slot = torch.arange(S, dtype=torch.float32).unsqueeze(0).expand(B, S)
    rec  = ((cache_size - 1 - slot).clamp(min=0) / max(max_len, 1)).unsqueeze(-1)
    if orig_pos is not None:
        pos = (torch.as_tensor(np.asarray(orig_pos), dtype=torch.float32).reshape(B, S)
               / POS_SCALE).unsqueeze(-1)
    else:
        pos = (slot / max(cache_size, 1)).unsqueeze(-1)

    cols = torch.cat([kz_h, vz_h, kz_mean, vz_mean, rec, pos], dim=-1)  # [B, S, 2H+4]
    return cols.cpu().numpy().astype(np.float32)


def build_obs(
    K:             Tensor,           # [n_envs, n_kv_heads, T, head_dim]
    V:             Tensor,           # [n_envs, n_kv_heads, T, head_dim]
    prompt_len:    int,
    max_len:       int,
    resident_mask: Tensor | None = None,  # [n_envs, T] bool — False = already evicted
    rich:          bool = False,
    orig_pos=None,                   # [n_envs, prompt_len] original positions per slot
) -> np.ndarray:                     # [n_envs, max_len, 2*n_kv_heads*head_dim (+2H+4)]
    """Build the observation array for all environments at once.

    Heads are concatenated along the feature axis: [K_h0 | K_h1 | V_h0 | V_h1].
    When `rich`, the 2H+4 scale/position columns are appended (see
    build_extra_columns). Evicted positions (resident_mask==False) are zeroed.
    """
    n_envs, H, _, head_dim = K.shape
    kv_dim   = 2 * H * head_dim
    n_extra  = extra_feature_dim(rich, H)
    fdim     = kv_dim + n_extra
    obs      = np.zeros((n_envs, max_len, fdim), dtype=np.float32)

    # [n_envs, H, T, D] → [n_envs, T, H*D]
    K_flat = K[:, :, :prompt_len, :].permute(0, 2, 1, 3).reshape(n_envs, prompt_len, H * head_dim)
    V_flat = V[:, :, :prompt_len, :].permute(0, 2, 1, 3).reshape(n_envs, prompt_len, H * head_dim)

    obs[:, :prompt_len, :H*head_dim]           = K_flat.cpu().numpy()
    obs[:, :prompt_len, H*head_dim:2*H*head_dim] = V_flat.cpu().numpy()

    if rich:
        cols = build_extra_columns(K[:, :, :prompt_len, :], V[:, :, :prompt_len, :],
                                   prompt_len, max_len, orig_pos=orig_pos)  # [n_envs, prompt_len, 2H+4]
        obs[:, :prompt_len, kv_dim:kv_dim + n_extra] = cols

    if resident_mask is not None:
        evicted = ~resident_mask[:, :prompt_len].cpu().numpy()  # [n_envs, T]
        obs[:, :prompt_len][evicted] = 0.0

    return obs
