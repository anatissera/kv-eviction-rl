#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import torch
import pytest

from kvcompression.costs import batched_compute_cost, CostResult, reverse_cumsum


def test_batched_compute_cost_regression():
    """
    Regression test: verifies batched_compute_cost produces the same
    values as the old batched_compute_optimal_cutoff_train function.
    """
    torch.manual_seed(42)
    batch_size, sequence_length, time_length = 2, 4, 5

    kv_rankings = torch.stack(
        [torch.randperm(sequence_length) for _ in range(batch_size)]
    )
    attention_weights = torch.rand(batch_size, time_length, sequence_length)
    kv_rankings_length = torch.full((batch_size,), fill_value=sequence_length)
    seq_lengths = torch.full((batch_size,), fill_value=time_length)

    result = batched_compute_cost(
        kv_rankings=kv_rankings,
        kv_rankings_length=kv_rankings_length,
        attention_weights=attention_weights,
        seq_lengths=seq_lengths,
    )

    # Values computed from old repo's batched_compute_optimal_cutoff_train
    expected_cost = torch.tensor([[1.8310027122497559], [1.1014279127120972]])
    expected_cost_per_cachesize = torch.tensor(
        [
            [
                0.03966257721185684,
                0.20298855006694794,
                0.33546948432922363,
                1.2528821229934692,
            ],
            [
                0.13647699356079102,
                0.327397882938385,
                0.1460268199443817,
                0.49152615666389465,
            ],
        ]
    )
    expected_valid_lengths = torch.tensor([[4], [4]])

    assert isinstance(result, CostResult)
    assert result.cost is not None
    assert result.cost_per_cachesize is not None
    assert result.valid_lengths is not None

    assert torch.allclose(result.cost, expected_cost, atol=1e-6)
    assert torch.allclose(
        result.cost_per_cachesize, expected_cost_per_cachesize, atol=1e-6
    )
    assert torch.equal(result.valid_lengths, expected_valid_lengths)


@pytest.mark.parametrize(
    "batch_size,sequence_length,time_length",
    [
        (1, 2, 4),  # Single batch, small sequence
        (1, 4, 5),  # Single batch
        (3, 8, 10),  # Standard case
        (2, 16, 20),  # Larger case
    ],
)
def test_batched_compute_cost_output_structure(
    batch_size, sequence_length, time_length
):
    """Test that batched_compute_cost returns correct structure and shapes."""
    kv_rankings = torch.stack(
        [torch.randperm(sequence_length) for _ in range(batch_size)]
    )
    attention_weights = torch.rand(batch_size, time_length, sequence_length)
    kv_rankings_length = torch.full((batch_size,), fill_value=sequence_length)
    seq_lengths = torch.full((batch_size,), fill_value=time_length)

    result = batched_compute_cost(
        kv_rankings=kv_rankings,
        kv_rankings_length=kv_rankings_length,
        attention_weights=attention_weights,
        seq_lengths=seq_lengths,
    )

    assert isinstance(result, CostResult)
    assert result.cost.shape == (batch_size, 1)
    assert result.cost_per_cachesize.shape == (batch_size, sequence_length)
    assert result.valid_lengths.shape == (batch_size, 1)


def test_batched_compute_cost_oracle_ranking():
    """Test that oracle ranking (sorted by importance) achieves cost ~= 1.0."""
    torch.manual_seed(123)
    batch_size, sequence_length, time_length = 2, 6, 8

    attention_weights = torch.rand(batch_size, time_length, sequence_length)
    kv_rankings_length = torch.full((batch_size,), fill_value=sequence_length)
    seq_lengths = torch.full((batch_size,), fill_value=time_length)

    # Compute importance exactly as the function does:
    # 1. Reverse cumsum to get future attention
    future_attention = reverse_cumsum(attention_weights, dim=-2)  # [B, T, L]
    # 2. Select at the kv_rankings_length timestep (for each batch item)
    importance = future_attention[
        torch.arange(batch_size), kv_rankings_length
    ]  # [B, L]

    # Oracle ranking: sort by importance descending
    oracle_rankings = importance.argsort(dim=-1, descending=True)

    result = batched_compute_cost(
        kv_rankings=oracle_rankings,
        kv_rankings_length=kv_rankings_length,
        attention_weights=attention_weights,
        seq_lengths=seq_lengths,
    )

    # Oracle should achieve exactly 1.0
    assert torch.allclose(result.cost, torch.ones_like(result.cost), atol=1e-5)


@pytest.mark.parametrize(
    "batch_size,num_queries,time_length,sequence_length",
    [
        (1, 2, 4, 3),  # Small GQA
        (2, 4, 10, 8),  # Standard GQA
        (1, 7, 5, 4),  # Single batch, 7 queries (like Qwen)
    ],
)
def test_batched_compute_cost_gqa(
    batch_size, num_queries, time_length, sequence_length
):
    """Test GQA case with 4D attention weights."""
    kv_rankings = torch.stack(
        [torch.randperm(sequence_length) for _ in range(batch_size)]
    )
    attention_weights = torch.rand(
        batch_size, num_queries, time_length, sequence_length
    )
    kv_rankings_length = torch.full((batch_size,), fill_value=sequence_length)
    seq_lengths = torch.full((batch_size,), fill_value=time_length)

    result = batched_compute_cost(
        kv_rankings=kv_rankings,
        kv_rankings_length=kv_rankings_length,
        attention_weights=attention_weights,
        seq_lengths=seq_lengths,
    )

    assert isinstance(result, CostResult)
    assert result.cost.shape == (batch_size, 1)
    assert result.cost_per_cachesize.shape == (batch_size, sequence_length)
    assert result.valid_lengths.shape == (batch_size, 1)


def test_batched_compute_cost_with_padding():
    """Test with variable kv_rankings_length per batch item."""
    torch.manual_seed(789)
    batch_size, max_sequence_length, time_length = 3, 8, 10

    kv_rankings = torch.stack(
        [torch.randperm(max_sequence_length) for _ in range(batch_size)]
    )
    attention_weights = torch.rand(batch_size, time_length, max_sequence_length)
    # Variable lengths: 8, 6, 4
    kv_rankings_length = torch.tensor([8, 6, 4])
    seq_lengths = torch.full((batch_size,), fill_value=time_length)

    result = batched_compute_cost(
        kv_rankings=kv_rankings,
        kv_rankings_length=kv_rankings_length,
        attention_weights=attention_weights,
        seq_lengths=seq_lengths,
    )

    # Verify shapes
    assert result.cost.shape == (batch_size, 1)
    assert result.cost_per_cachesize.shape == (batch_size, max_sequence_length)
    assert result.valid_lengths.shape == (batch_size, 1)
    # Verify valid_lengths matches input
    assert torch.equal(result.valid_lengths.squeeze(-1), kv_rankings_length)


def test_cost_result_to_device():
    """Test that CostResult.to() method moves all tensors to specified device."""
    cost = torch.tensor([[1.0], [2.0]])
    cost_per_cachesize = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    valid_lengths = torch.tensor([[3], [3]])

    result = CostResult(
        cost=cost, cost_per_cachesize=cost_per_cachesize, valid_lengths=valid_lengths
    )

    # Move to same device (should work without error)
    result_cpu = result.to(torch.device("cpu"))

    assert result_cpu.cost.device == torch.device("cpu")
    assert result_cpu.cost_per_cachesize.device == torch.device("cpu")
    assert result_cpu.valid_lengths.device == torch.device("cpu")


def test_batched_compute_cost_cost_is_sum_of_per_cachesize():
    """Test that cost equals sum of cost_per_cachesize."""
    torch.manual_seed(456)
    batch_size, sequence_length, time_length = 4, 10, 15

    kv_rankings = torch.stack(
        [torch.randperm(sequence_length) for _ in range(batch_size)]
    )
    attention_weights = torch.rand(batch_size, time_length, sequence_length)
    kv_rankings_length = torch.full((batch_size,), fill_value=sequence_length)
    seq_lengths = torch.full((batch_size,), fill_value=time_length)

    result = batched_compute_cost(
        kv_rankings=kv_rankings,
        kv_rankings_length=kv_rankings_length,
        attention_weights=attention_weights,
        seq_lengths=seq_lengths,
    )

    # cost should be the sum of cost_per_cachesize
    assert torch.allclose(
        result.cost,
        result.cost_per_cachesize.sum(-1, keepdim=True),
        atol=1e-6,
    )
