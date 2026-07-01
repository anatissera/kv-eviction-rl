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

    def __init__(self, observation_space: spaces.Box, hidden: int = 64, n_extra: int = 0):
        max_len, feature_dim = observation_space.shape
        super().__init__(observation_space, features_dim=max_len)

        self.max_len     = max_len
        self.feature_dim = feature_dim
        self.hidden      = hidden
        # n_extra = rich scale/position columns appended AFTER K||V (Phase 2 E1).
        # These deliberately BYPASS the K/V LayerNorm — they carry magnitude and
        # position, the very signals LayerNorm would erase and that `kv_norm` uses.
        self.n_extra     = n_extra
        self.kv_dim      = feature_dim - n_extra   # width of the K||V block
        self.kv_half     = self.kv_dim // 2        # size of K half = n_kv_heads * head_dim

        # Normalize K and V separately before the MLP.
        # K-norms vary 5–50× across layers (attention sinks, massive-activation
        # tokens), causing high-norm layers to dominate gradients.
        self.k_norm = nn.LayerNorm(self.kv_half)
        self.v_norm = nn.LayerNorm(self.kv_half)

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
        # Cast to float32: buffer stores fp16 to save RAM; network needs fp32.
        observations = observations.float()
        batch = observations.shape[0]
        x = observations.view(batch * self.max_len, self.feature_dim)

        # Padded positions (cache slots beyond current cache_size) are zero-filled.
        # Detect them on the K||V block (extra columns like pos_frac can be 0 for a
        # real slot 0, so they must not participate in the real/pad decision).
        # We multiply the output by this mask rather than skipping the forward pass,
        # because (a) indexing into a ragged batch is slower, and (b) the mask zeros
        # the gradient for padded positions, preventing LayerNorm from updating its
        # parameters based on semantically empty inputs.
        kv    = x[:, :self.kv_dim]
        extra = x[:, self.kv_dim:]             # [batch*max_len, n_extra], raw (not normed)
        is_real = (kv.abs().sum(dim=-1, keepdim=True) > 0).float()  # [batch*max_len, 1]

        # Normalize K and V sub-vectors independently before projection; the rich
        # extra columns bypass normalization (they carry magnitude/position).
        k = self.k_norm(kv[:, :self.kv_half])
        v = self.v_norm(kv[:, self.kv_half:])
        x = torch.cat([k, v, extra], dim=-1)

        x = self.mlp(x)                        # [batch*max_len, 1]
        x = x * is_real                        # zero padded positions, block gradient
        x = x.view(batch, self.max_len)        # [batch, max_len] — one score per token
        return x


class PerTokenAttention(BaseFeaturesExtractor):
    """Phase 2 · E4 — cross-token attention feature extractor.

    Drop-in replacement for PerTokenMLP (same I/O: [batch, max_len, feat] →
    [batch, max_len]) that adds what the per-token MLP structurally lacks: each
    slot's keep-score depends on the OTHER resident slots via self-attention. This
    enables REDUNDANCY reasoning ("evict A because B already carries its info"),
    which a per-token method cannot express — the hypothesized ceiling of kv_norm
    and PerTokenMLP (both per-token) is what this is meant to break (see PLAN.md D3).

    Permutation-equivariant over slots: no positional encoding is added here because
    position is already an explicit input feature (rich pos_orig/rec), so the encoder
    can use it without breaking equivariance. Padded slots are masked out of attention
    and zeroed at the output (same is_real convention as PerTokenMLP). Same K/V
    LayerNorm + rich-column bypass as PerTokenMLP.
    """

    def __init__(self, observation_space: spaces.Box, hidden: int = 64, n_extra: int = 0,
                 n_heads: int = 4, n_layers: int = 2):
        max_len, feature_dim = observation_space.shape
        super().__init__(observation_space, features_dim=max_len)
        self.max_len     = max_len
        self.feature_dim = feature_dim
        self.n_extra     = n_extra
        self.kv_dim      = feature_dim - n_extra
        self.kv_half     = self.kv_dim // 2

        self.k_norm = nn.LayerNorm(self.kv_half)
        self.v_norm = nn.LayerNorm(self.kv_half)
        self.input_proj = nn.Linear(feature_dim, hidden)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.out = nn.Linear(hidden, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.float()
        B = observations.shape[0]
        kv    = observations[..., :self.kv_dim]                # [B, max_len, kv_dim]
        extra = observations[..., self.kv_dim:]                # [B, max_len, n_extra]
        is_real = (kv.abs().sum(dim=-1) > 0)                   # [B, max_len] bool

        k = self.k_norm(kv[..., :self.kv_half])
        v = self.v_norm(kv[..., self.kv_half:])
        x = torch.cat([k, v, extra], dim=-1)                  # [B, max_len, feat]
        h = self.input_proj(x)                                # [B, max_len, hidden]

        # Attend only over resident slots. A fully-padded row can't occur during
        # eviction (cache_size > budget ⇒ ≥1 resident), but guard nan just in case.
        pad_mask = ~is_real                                    # True = ignore
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = torch.nan_to_num(h)
        s = self.out(h).squeeze(-1)                           # [B, max_len]
        s = s * is_real.float()                               # zero padded slots
        return s
