"""
PerTokenMLP — SB3 BaseFeaturesExtractor for token-level observation spaces.

Inspired by KVSamplingGroupedQueryAgent in ml-learning-to-evict.
Source: /Users/alexanderbodner/Documents/Udesa/5to/aprendizaje reforzado/tp final/ml-learning-to-evict

The observation is [max_len, feature_dim]. We apply the same small MLP
independently to each token (shared weights across positions), outputting one
scalar keep-score per token.

features_dim = max_len  (one keep-score per position)

SB3 then attaches its own Linear(max_len → max_len) actor head and a separate
Linear(max_len → 1) critic head. This means the actor head is a max_len×max_len
matrix (65,536 params for max_len=256) instead of the max_len*hidden×max_len
matrix (4.2M params) from the previous flat-features design. The per-token
scalar output is the right pre-logit representation — positions don't need to
know about each other before the final actor head mixes them.

Architecture per token:
    Linear(feature_dim → hidden) → LayerNorm → SiLU →
    Linear(hidden → hidden) → LayerNorm → SiLU →
    Linear(hidden → 1)

Output: [batch, max_len]  (one scalar per position)
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class PerTokenMLP(BaseFeaturesExtractor):
    """Shared per-token MLP feature extractor.

    Args:
        observation_space: Box([max_len, feature_dim])
        hidden:            Hidden dimension (default 64).
    """

    def __init__(self, observation_space: spaces.Box, hidden: int = 64):
        max_len, feature_dim = observation_space.shape
        super().__init__(observation_space, features_dim=max_len)

        self.max_len     = max_len
        self.feature_dim = feature_dim
        self.hidden      = hidden
        self.head_dim    = feature_dim // 2  # K and V halves

        # Normalize K and V separately before the MLP.
        # Without this, K-norms vary 5–50× across layers (attention sinks,
        # massive-activation tokens), causing high-norm layers to dominate gradients.
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.v_norm = nn.LayerNorm(self.head_dim)

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),   # one scalar keep-score per token
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: [batch, max_len, feature_dim]
        batch = observations.shape[0]
        x = observations.view(batch * self.max_len, self.feature_dim)

        # Normalize K and V sub-vectors independently before projection
        k = self.k_norm(x[:, :self.head_dim])
        v = self.v_norm(x[:, self.head_dim:])
        x = torch.cat([k, v], dim=-1)

        x = self.mlp(x)                        # [batch*max_len, 1]
        x = x.view(batch, self.max_len)        # [batch, max_len] — one score per token
        return x
