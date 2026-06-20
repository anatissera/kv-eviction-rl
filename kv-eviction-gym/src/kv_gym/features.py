"""
Per-token observation builder.

Feature set per token position t:
    K[t]  (head_dim,)  full key vector — RoPE already applied, encodes content + position
    V[t]  (head_dim,)  full value vector

feature_dim = 2 * head_dim  (128 for Qwen2.5-1.5B with head_dim=64)

Position is already encoded in K via RoPE rotation, so an explicit
t/max_len scalar is redundant. No layer/head fracs (add if ablations show benefit).

Evicted token positions are zeroed out via `resident_mask` so the observation
reflects the current cache state rather than the frozen prefill. This makes the
observation non-constant across eviction steps, giving the value function a useful
signal about how far into the episode we are.

The observation is zero-padded beyond prompt_len. The action mask already
prevents the policy from selecting those padded positions.
"""

import numpy as np
import torch
from torch import Tensor


def feature_dim(head_dim: int) -> int:
    return 2 * head_dim


def build_obs(
    K:             Tensor,           # [n_envs, T, head_dim]
    V:             Tensor,           # [n_envs, T, head_dim]
    prompt_len:    int,
    max_len:       int,
    resident_mask: Tensor | None = None,  # [n_envs, T] bool — False = already evicted
) -> np.ndarray:                     # [n_envs, max_len, 2*head_dim]
    """Build the observation array for all environments at once.

    Evicted positions (resident_mask==False) are zeroed so the policy and value
    function can observe which tokens have already been removed.
    """
    n_envs, _, head_dim = K.shape
    fdim = 2 * head_dim
    obs  = np.zeros((n_envs, max_len, fdim), dtype=np.float32)

    obs[:, :prompt_len, :head_dim]           = K[:, :prompt_len].cpu().numpy()
    obs[:, :prompt_len, head_dim:2*head_dim] = V[:, :prompt_len].cpu().numpy()

    if resident_mask is not None:
        # Zero out evicted positions: [n_envs, prompt_len] bool broadcast over fdim
        evicted = ~resident_mask[:, :prompt_len].cpu().numpy()  # [n_envs, T]
        obs[:, :prompt_len][evicted] = 0.0

    return obs
