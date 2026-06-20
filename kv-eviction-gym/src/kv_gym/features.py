"""
Per-token observation builder.

Feature set (3 scalars per token position):
    ||K[t]||   L2 norm of the key vector — proxy for token importance
    ||V[t]||   L2 norm of the value vector
    position   t / max_len  (relative position in prompt)

feature_dim = 3

No attention scores (avoids eager-only constraint, enabling flash_attn).
No raw K/V vectors (262× smaller observation than the original design).
No is_resident (MaskablePPO gets that from the action mask).
No layer/head fracs (add back later if ablations show benefit).

The observation is zero-padded beyond prompt_len. The action mask already
prevents the policy from selecting those padded positions.
"""

import numpy as np
import torch
from torch import Tensor

FEATURE_DIM = 3


def build_obs(
    K:          Tensor,   # [n_envs, max_len, head_dim]
    V:          Tensor,   # [n_envs, max_len, head_dim]
    prompt_len: int,
    max_len:    int,
) -> np.ndarray:          # [n_envs, max_len, 3]
    """Build the observation array for all environments at once."""
    n_envs = K.shape[0]
    obs = np.zeros((n_envs, max_len, FEATURE_DIM), dtype=np.float32)

    # Key and value L2 norms per token  [n_envs, prompt_len]
    k_norm = K[:, :prompt_len].norm(dim=-1).cpu().numpy()
    v_norm = V[:, :prompt_len].norm(dim=-1).cpu().numpy()

    # Relative position  [prompt_len]  (same for all envs)
    positions = (
        np.arange(prompt_len, dtype=np.float32) / max(max_len - 1, 1)
    )  # [T]

    obs[:, :prompt_len, 0] = k_norm
    obs[:, :prompt_len, 1] = v_norm
    obs[:, :prompt_len, 2] = positions[None, :]   # broadcast across envs

    return obs
