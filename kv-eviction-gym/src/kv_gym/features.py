"""
Per-token observation builder.

Feature set per token position t:
    K[t]       (head_dim,)  full key vector with RoPE — encodes content + position
    V[t]       (head_dim,)  full value vector
    t/max_len  (1,)         explicit relative position (easy positional prior)

feature_dim = 2 * head_dim + 1  (129 for Qwen2.5-1.5B with head_dim=64)

No is_resident (MaskablePPO gets that from the action mask).
No layer/head fracs (add back if ablations show benefit).

The observation is zero-padded beyond prompt_len. The action mask already
prevents the policy from selecting those padded positions.
"""

import numpy as np
import torch
from torch import Tensor


def feature_dim(head_dim: int) -> int:
    return 2 * head_dim + 1


def build_obs(
    K:          Tensor,   # [n_envs, max_len, head_dim]
    V:          Tensor,   # [n_envs, max_len, head_dim]
    prompt_len: int,
    max_len:    int,
) -> np.ndarray:          # [n_envs, max_len, 2*head_dim+1]
    """Build the observation array for all environments at once."""
    n_envs, _, head_dim = K.shape
    fdim = 2 * head_dim + 1
    obs  = np.zeros((n_envs, max_len, fdim), dtype=np.float32)

    K_np = K[:, :prompt_len].cpu().numpy()        # [n_envs, T, D]
    V_np = V[:, :prompt_len].cpu().numpy()

    positions = (
        np.arange(prompt_len, dtype=np.float32) / max(max_len - 1, 1)
    )  # [T]

    obs[:, :prompt_len, :head_dim]            = K_np
    obs[:, :prompt_len, head_dim:2*head_dim]  = V_np
    obs[:, :prompt_len, 2*head_dim]           = positions[None, :]

    return obs
