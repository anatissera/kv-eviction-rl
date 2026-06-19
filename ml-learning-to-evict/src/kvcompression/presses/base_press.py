#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import abc
from typing import Optional

import torch
from torch import nn


class BasePress(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for KV cache compression strategies (presses).

    A press determines how to rank KV cache entries for compression.
    Subclasses must implement the `sort` method to produce a ranking
    of tokens by importance.
    """

    @abc.abstractmethod
    def sort(
        self,
        hidden_states: Optional[torch.Tensor],
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Rank KV cache entries by importance for compression.

        Args:
            hidden_states: Hidden states from the model, shape [B, S, D].
            keys: Key tensors, shape [B, num_kv_heads, S, head_dim].
            values: Value tensors, shape [B, num_kv_heads, S, head_dim].
            queries: Query tensors, shape [B, num_q_heads, S, head_dim].
            attention_weights: Pre-computed attention weights, optional.

        Returns:
            Ranking tensor of shape [B, num_kv_heads, S] where values are
            token indices sorted by importance (index 0 = most important).
        """
        raise NotImplementedError()

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ):
        return self.sort(
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            queries=queries,
            attention_weights=attention_weights,
        )

    def requires_attention_weights(self) -> bool:
        """
        Override this method to indicate if the press requires attention weights.
        This helps the system know when to re-materialize attention weights.
        """
        return False

    def supports_gqa(self) -> bool:
        """Returns whether this press can handle GQA (multiple queries per KV head)."""
        return False
