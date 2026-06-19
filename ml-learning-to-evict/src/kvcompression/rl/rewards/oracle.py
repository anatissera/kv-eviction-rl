#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import torch

from kvcompression.attention_utils import (
    materialize_attention_weights,
)
from kvcompression.costs import (
    CostResult,
    batched_compute_cost,
)
from kvcompression.utils.utils import is_torch_compile_disabled


class Oracle:
    """
    Computes optimal KV cache compression costs for reward calculation.

    This oracle evaluates the quality of a KV ranking by computing the
    normalized future attention AUC cost.
    """

    @torch.compile(
        dynamic=True,
        disable=is_torch_compile_disabled(),
    )
    def compute_cost(
        self,
        kv_rankings: torch.Tensor,  # [1, S] (S=max_prompt_len_in_batch)
        kv_rankings_length: torch.Tensor,  # [B]
        all_keys: torch.Tensor,  # [B, L, H] (L=max_entire_sequence_length)
        all_queries: torch.Tensor,  # [B, L, H] or [B, num_queries, L, H]
        seq_lengths: torch.Tensor,  # [B]
    ) -> CostResult:
        """
        Compute compression costs for a given KV ranking.

        Args:
            kv_rankings: Token indices sorted by agent's predicted importance.
            kv_rankings_length: Valid length of rankings for each batch item.
            all_keys: Key tensors for the full sequence.
            all_queries: Query tensors, possibly with grouped query dimension.
            seq_lengths: Valid sequence lengths per batch item.

        Returns:
            CostResult containing cost metrics.
        """
        # GQA case
        if all_keys.ndim == 3 and all_queries.ndim == 4:
            all_keys = all_keys.unsqueeze(1)

        attention_weights = materialize_attention_weights(
            query=all_queries.to(torch.float32),
            key=all_keys.to(torch.float32),
            is_causal=True,
            key_seq_lengths=seq_lengths,
            query_seq_lengths=seq_lengths,
        )  # [B, L, L] or [B, num_queries, L, L]

        # Handle GQA by taking max over query heads
        if attention_weights.ndim == 4:
            attention_weights = attention_weights.amax(1)

        return batched_compute_cost(
            kv_rankings=kv_rankings,
            kv_rankings_length=kv_rankings_length.contiguous(),
            attention_weights=attention_weights,
            seq_lengths=seq_lengths,
        )
