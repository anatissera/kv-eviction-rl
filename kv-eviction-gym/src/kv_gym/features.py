"""
Per-token observation builder.

Feature set per token position t:
    K[t]  (head_dim,)  full key vector — RoPE already applied, encodes content + position
    V[t]  (head_dim,)  full value vector

feature_dim = 2 * head_dim  (128 for Qwen2.5-1.5B with head_dim=64)

Position is already encoded in K via RoPE rotation, so an explicit
t/max_len scalar is redundant. No is_resident (MaskablePPO handles that
via the action mask). No layer/head fracs (add if ablations show benefit).

The observation is zero-padded beyond prompt_len. The action mask already
prevents the policy from selecting those padded positions.
"""

import numpy as np
import torch
from torch import Tensor


def feature_dim(head_dim: int) -> int:
    return 2 * head_dim


def build_obs(
    K:          Tensor,   # [n_envs, max_len, head_dim]
    V:          Tensor,   # [n_envs, max_len, head_dim]
    prompt_len: int,
    max_len:    int,
) -> np.ndarray:          # [n_envs, max_len, 2*head_dim]
    """Build the observation array for all environments at once."""
    n_envs, _, head_dim = K.shape
    fdim = 2 * head_dim
    obs  = np.zeros((n_envs, max_len, fdim), dtype=np.float32)

    obs[:, :prompt_len, :head_dim]           = K[:, :prompt_len].cpu().numpy()
    obs[:, :prompt_len, head_dim:2*head_dim] = V[:, :prompt_len].cpu().numpy()

    return obs
