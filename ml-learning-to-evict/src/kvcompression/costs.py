#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import math
from typing import NamedTuple, Optional

import torch
from torchtune import utils

from kvcompression.utils.utils import is_torch_compile_disabled

logger = utils.get_logger("INFO")


class CostResult(NamedTuple):
    """
    Container for cost computation results.

    Contains continuous metrics for training with differentiable rewards.
    """

    cost: torch.Tensor = None
    cost_per_cachesize: Optional[torch.Tensor] = None
    valid_lengths: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "CostResult":
        new_fields = {}
        for name, value in self._asdict().items():
            if isinstance(value, torch.Tensor):
                new_fields[name] = value.to(device)
            else:
                new_fields[name] = value
        return CostResult(**new_fields)


def reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute cumulative sum in reverse order along the specified dimension."""
    return x.flip(dim).cumsum(dim).flip(dim)


@torch.compile(
    dynamic=True,
    disable=is_torch_compile_disabled(),
)
def batched_compute_cost(
    kv_rankings: torch.Tensor,
    kv_rankings_length: torch.Tensor,
    attention_weights: torch.Tensor,
    seq_lengths: torch.Tensor,
) -> CostResult:
    """
    Compute cost for training - only computes what's needed for reward.

    This computes the continuous attention-based metrics needed for training rewards.
    Supports torch.compile for performance.

    Args:
        kv_rankings: Token indices sorted by predicted importance [B, S] or [B, H, S].
        kv_rankings_length: Valid length of rankings for each batch item [B] or [B, H].
        attention_weights: Attention weight matrices [B, T, L] or [B, num_q, T, L].
        seq_lengths: Valid sequence lengths per batch item [B].

    Returns:
        CostResult with cost, cost_per_cachesize, and valid_lengths populated.
    """
    if attention_weights.ndim == 2:
        attention_weights = attention_weights[None]
    if kv_rankings.ndim == 1:
        kv_rankings = kv_rankings[None]

    partitions = kv_rankings.shape[:-1]
    n_samples = math.prod(partitions)
    *attn_ind_partitions, T, L = attention_weights.shape

    # Handle GQA case
    if len(attn_ind_partitions) > 1:
        attn_ind_partitions = attn_ind_partitions[:-1]

    indn_samples = math.prod(attn_ind_partitions)

    kv_rankings = kv_rankings.view(
        n_samples // indn_samples, indn_samples, kv_rankings.shape[-1]
    )
    kv_rankings_length = kv_rankings_length.view(
        n_samples // indn_samples, indn_samples
    )

    j_indices = torch.arange(indn_samples, device=attention_weights.device).expand_as(
        kv_rankings_length
    )

    # Handle GQA by taking max over query heads
    if attention_weights.ndim == 4:
        attention_weights = attention_weights.amax(1)

    future_attention_continuous = reverse_cumsum(attention_weights, dim=-2)
    future_attention_continuous = future_attention_continuous.view(indn_samples, T, L)

    continuous_costs_dp = future_attention_continuous[
        j_indices[..., None], kv_rankings_length[..., None], kv_rankings
    ]

    # Normalize by sequence length for training stability
    continuous_costs_dp = continuous_costs_dp / (
        seq_lengths.view(1, indn_samples, 1).to(torch.float32) + 1e-8
    )
    along_ranking_masks = torch.arange(
        kv_rankings.shape[-1], device=kv_rankings.device
    ).expand_as(kv_rankings)
    padded_masks = along_ranking_masks >= kv_rankings_length.view(
        n_samples // indn_samples, indn_samples, 1
    ).expand_as(kv_rankings)
    continuous_costs_dp.masked_fill_(padded_masks, 0.0)
    continuous_costs_dp = continuous_costs_dp.view(n_samples, kv_rankings.shape[-1])

    # Compute AUC using weighted sum (equivalent to reverse_cumsum + sum)
    n = continuous_costs_dp.shape[-1]
    weights = torch.arange(
        1, n + 1, device=continuous_costs_dp.device, dtype=continuous_costs_dp.dtype
    )

    # Oracle's AUC (ideal ranking)
    ideal_continuous_importances, _ = torch.sort(
        continuous_costs_dp, dim=-1, descending=True
    )
    ideal_future_attention_auc = torch.sum(
        ideal_continuous_importances * weights, dim=-1
    )

    safe_denominator_attn = torch.clamp(ideal_future_attention_auc, min=1e-8)

    # Agent's AUC (normalized by oracle)
    future_attention_normalized = (
        continuous_costs_dp * weights
    ) / safe_denominator_attn[..., None]

    future_attention_auc_normalized = future_attention_normalized.sum(-1)

    return CostResult(
        cost=future_attention_auc_normalized.view(*partitions, -1),
        cost_per_cachesize=future_attention_normalized.view(*partitions, -1),
        valid_lengths=kv_rankings_length.view(*partitions, -1),
    )
