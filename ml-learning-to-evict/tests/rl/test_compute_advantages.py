#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import pytest
import torch

from kvcompression.rl.trainers.compute_advantages import (
    compute_returns_and_advantage,
    compute_returns_and_advantage_vec,
)


def generate_test_data(batch_size, max_seq_len, length_distribution, device="cpu"):
    """Generates trajectories and episode lengths for testing."""
    if max_seq_len == 0:
        # Handle case where sequence length is zero
        trajectories = {
            "rewards": torch.empty((batch_size, 0), device=device, dtype=torch.float32),
            "log_probs": torch.empty(
                (batch_size, 0), device=device, dtype=torch.float32
            ),
            "entropies": torch.empty(
                (batch_size, 0), device=device, dtype=torch.float32
            ),
        }
        episode_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
        return trajectories, episode_lengths

    # Generate base random data
    rewards = torch.randn(batch_size, device=device)
    log_probs = torch.randn(batch_size, max_seq_len, device=device)
    entropies = torch.rand(batch_size, max_seq_len, device=device) * 0.1

    # Generate episode lengths based on distribution type
    if length_distribution == "zeros":
        episode_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
    elif length_distribution == "one_valid":
        episode_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
        if batch_size > 0:
            idx_ = torch.randint(0, batch_size, (1,)).item()
            len_ = torch.randint(1, max_seq_len + 1, (1,)).item()
            episode_lengths[idx_] = len_
    elif length_distribution == "max_len":
        episode_lengths = torch.full(
            (batch_size,), max_seq_len, device=device, dtype=torch.long
        )
    elif length_distribution == "min_len":
        episode_lengths = torch.ones(batch_size, device=device, dtype=torch.long)
    elif length_distribution == "mixed":
        # Ensure lengths are at least 1 if max_seq_len > 0
        min_len = 1 if max_seq_len > 0 else 0
        episode_lengths = torch.randint(
            min_len, max_seq_len + 1, (batch_size,), device=device, dtype=torch.long
        )
    elif length_distribution == "edge_n_equals_1":  # Specific case for N=1 check
        episode_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
        if batch_size >= 1:
            episode_lengths[0] = torch.randint(1, max_seq_len + 1, (1,)).item()
    elif length_distribution == "edge_n_equals_2":  # Specific case for N=2 check
        episode_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
        if batch_size >= 2:
            episode_lengths[0] = torch.randint(1, max_seq_len + 1, (1,)).item()
            episode_lengths[1] = torch.randint(1, max_seq_len + 1, (1,)).item()
    else:
        raise ValueError(f"Unknown length distribution: {length_distribution}")

    trajectories = {
        "rewards": rewards,
        "log_probs": log_probs,
        "entropies": entropies,
    }

    # Ensure lengths are not greater than max_seq_len (should be handled by randint)
    episode_lengths = torch.clamp(episode_lengths, max=max_seq_len)

    return trajectories, episode_lengths


# Define test cases: (batch_size, max_seq_len, length_distribution)
test_params = [
    # Edge Cases
    (5, 10, "zeros"),  # No valid episodes
    (1, 10, "mixed"),  # Batch size 1 (N=1 or N=0) -> None expected
    (5, 10, "edge_n_equals_1"),  # Exactly one valid episode (N=1) -> None expected
    (0, 10, "mixed"),  # Batch size 0
    (5, 0, "zeros"),  # Max sequence length 0
    (
        2,
        5,
        "edge_n_equals_2",
    ),  # Exactly two valid episodes (N=2) -> Calculation possible
    (3, 5, "edge_n_equals_2"),  # N=2, one episode ignored
    # Regular Cases
    (5, 10, "max_len"),  # All episodes full length
    (5, 1, "max_len"),  # All episodes length 1
    (5, 10, "min_len"),  # All episodes length 1 (if max_seq_len > 0)
    (8, 20, "mixed"),  # Mixed lengths, typical case
    (32, 50, "mixed"),  # Larger batch, longer sequences
    (2, 3, "mixed"),  # Small batch, small sequence
]


@pytest.mark.parametrize("batch_size, max_seq_len, length_distribution", test_params)
@pytest.mark.parametrize("normalize_advantages", [True, False])
def test_compute_returns_and_advantage(
    batch_size, max_seq_len, length_distribution, normalize_advantages: bool
):
    """
    Tests if the VectorizedRLOO implementation produces the same output as the
    OriginalRLOO implementation.
    """
    device = "cpu"  # Use CPU for consistency in testing
    trajectories, episode_lengths = generate_test_data(
        batch_size, max_seq_len, length_distribution, device=device
    )

    original_calculator = compute_returns_and_advantage
    vectorized_calculator = compute_returns_and_advantage_vec

    original_advantages = original_calculator(
        trajectories["rewards"],
        episode_lengths,
        normalize_advantages=normalize_advantages,
    )
    vectorized_advanatages = vectorized_calculator(
        trajectories["rewards"],
        episode_lengths,
        normalize_advantages=normalize_advantages,
    )

    if original_advantages is None:
        assert vectorized_advanatages is None, (
            "Mismatch for advantage computation:: Original is None, Vectorized is not."
        )
    elif vectorized_advanatages is None:
        assert original_advantages is None, (
            "Mismatch for advantage computation:: Vectorized is None, Original is not."
        )
    else:
        assert torch.allclose(
            original_advantages, vectorized_advanatages, atol=1e-6, rtol=1e-5
        ), "Value mismatch for advantage computation: using torch.allclose."
        assert torch.equal(original_advantages, vectorized_advanatages), (
            "Value mismatch for advantage computation: using torch.equal."
        )
