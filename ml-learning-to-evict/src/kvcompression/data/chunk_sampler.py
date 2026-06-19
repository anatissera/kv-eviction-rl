#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from typing import Iterator, List, Optional

import numpy as np
from torch.utils.data import Dataset, Sampler


class ChunkSampler(Sampler[int]):
    """
    Samples elements from a specific chunk of the dataset using block assignment.

    This divides the dataset into contiguous blocks and assigns each chunk a block.
    The last chunk may have a different size to ensure all samples are covered.

    Args:
        data_source: Dataset to sample from
        total_num_chunks: Total number of chunks to divide the dataset into
        current_chunk: The chunk to sample from (0-indexed)
        shuffle: Whether to shuffle the indices within the chunk
        seed: Random seed for shuffle
    """

    def __init__(
        self,
        data_source: Dataset,
        total_num_chunks: int,
        current_chunk: int,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.data_source = data_source
        self.total_num_chunks = total_num_chunks
        self.current_chunk = current_chunk
        self.shuffle = shuffle
        self.seed = seed

        if current_chunk >= total_num_chunks:
            raise ValueError(
                f"current_chunk ({current_chunk}) must be less than total_num_chunks ({total_num_chunks})"
            )

        self.total_length = len(data_source)
        self.indices = self._get_indices()

    def _get_indices(self) -> List[int]:
        # Calculate ceiling division to ensure all elements are covered
        chunk_size = (
            self.total_length + self.total_num_chunks - 1
        ) // self.total_num_chunks
        start_idx = self.current_chunk * chunk_size
        end_idx = min(start_idx + chunk_size, self.total_length)

        indices = list(range(start_idx, end_idx))

        if self.shuffle and indices:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)

        return indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
