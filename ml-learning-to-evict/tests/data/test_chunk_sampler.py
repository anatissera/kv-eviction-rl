#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from typing import List

import pytest
from torch.utils.data import Dataset

from kvcompression.data.chunk_sampler import ChunkSampler


class DummyDataset(Dataset):
    def __init__(self, size: int = 100):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> int:
        return idx


def test_chunk_validation() -> None:
    """Test that invalid chunk values raise appropriate errors."""
    dataset = DummyDataset(10)

    # current_chunk must be < total_num_chunks
    with pytest.raises(ValueError):
        ChunkSampler(dataset, total_num_chunks=3, current_chunk=3)

    # Valid cases should work fine
    ChunkSampler(dataset, total_num_chunks=3, current_chunk=0)
    ChunkSampler(dataset, total_num_chunks=3, current_chunk=2)


def test_complete_coverage() -> None:
    """Test that all chunks together cover the entire dataset exactly once."""
    dataset_size = 100
    dataset = DummyDataset(dataset_size)

    # Test with different numbers of chunks
    for total_chunks in [1, 2, 3, 10, 33]:
        all_indices: List[int] = []

        # Collect indices from all chunks
        for chunk in range(total_chunks):
            sampler = ChunkSampler(dataset, total_chunks, chunk)
            chunk_indices = list(sampler)

            # Each chunk should have no duplicates
            assert len(chunk_indices) == len(set(chunk_indices))
            all_indices.extend(chunk_indices)

        # All indices should be covered exactly once
        assert sorted(all_indices) == list(range(dataset_size))
        assert len(set(all_indices)) == dataset_size


def test_uneven_chunk_distribution() -> None:
    """Test that chunk sizes are distributed correctly for non-divisible cases."""
    dataset = DummyDataset(10)

    # For 10 items divided into 3 chunks:
    # Ceiling division: (10 + 3 - 1) // 3 = 4
    # So we expect chunks of size 4, 4, and 2
    chunk0 = ChunkSampler(dataset, total_num_chunks=3, current_chunk=0)
    chunk1 = ChunkSampler(dataset, total_num_chunks=3, current_chunk=1)
    chunk2 = ChunkSampler(dataset, total_num_chunks=3, current_chunk=2)

    assert list(chunk0) == [0, 1, 2, 3]
    assert list(chunk1) == [4, 5, 6, 7]
    assert list(chunk2) == [8, 9]


def test_shuffle_determinism() -> None:
    """Test that shuffling with seeds works deterministically."""
    dataset = DummyDataset(50)
    sampler = ChunkSampler(dataset, total_num_chunks=2, current_chunk=0)

    # Without shuffle
    no_shuffle = list(sampler)

    # With shuffle and same seed, should be deterministic
    sampler1 = ChunkSampler(dataset, 2, 0, shuffle=True, seed=42)
    sampler2 = ChunkSampler(dataset, 2, 0, shuffle=True, seed=42)
    assert list(sampler1) == list(sampler2)

    # With shuffle, order should be different but content identical
    shuffled = list(sampler1)
    assert shuffled != no_shuffle  # Different order
    assert sorted(shuffled) == sorted(no_shuffle)  # Same indices
