"""
PerTokenMLP — SB3 BaseFeaturesExtractor for token-level observation spaces.

Inspired by KVSamplingGroupedQueryAgent in ml-learning-to-evict.
Source: /Users/alexanderbodner/Documents/Udesa/5to/aprendizaje reforzado/tp final/ml-learning-to-evict

The observation is [max_len, feature_dim]. We apply the same small MLP
independently to each token (shared weights across positions), then flatten
to [max_len * hidden] for SB3's actor/critic heads to consume.

Same weights for all token positions = the policy learns token-agnostic
importance features, which generalises across different prompt lengths and
token orderings. This mirrors the design in the bandit version of the paper.

Architecture per token:
    Linear(feature_dim → hidden) → LayerNorm → SiLU →
    Linear(hidden → hidden) → LayerNorm → SiLU

Output: flatten → [max_len * hidden]
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class PerTokenMLP(BaseFeaturesExtractor):
    """Shared per-token MLP feature extractor.

    Args:
        observation_space: Box([max_len, feature_dim])
        hidden:            Hidden dimension (default 64). Increase to 128
                           if the policy underfits.
    """

    def __init__(self, observation_space: spaces.Box, hidden: int = 64):
        max_len, feature_dim = observation_space.shape
        features_dim = max_len * hidden
        super().__init__(observation_space, features_dim=features_dim)

        self.max_len     = max_len
        self.feature_dim = feature_dim
        self.hidden      = hidden

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: [batch, max_len, feature_dim]
        batch = observations.shape[0]
        x = observations.view(batch * self.max_len, self.feature_dim)
        x = self.mlp(x)                                   # [batch*max_len, hidden]
        x = x.view(batch, self.max_len * self.hidden)     # [batch, max_len*hidden]
        return x
