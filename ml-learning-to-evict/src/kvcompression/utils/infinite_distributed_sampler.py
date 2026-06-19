#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Infinite distributed sampler for RL training with epoch-level resume support.

This sampler provides an infinite stream of data by cycling through epochs
automatically, eliminating the need for iterator recreation on each step.
"""

import logging
from typing import Iterator, Optional

import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler

logger = logging.getLogger(__name__)


class InfiniteDistributedSampler(DistributedSampler):
    """
    Infinite distributed sampler that cycles through epochs automatically.

    Unlike StatefulDistributedSampler which recreates iterators for precise
    step-level resume, this sampler provides approximate epoch-level resume
    for better performance in RL training where exact reproducibility is
    less critical than training efficiency.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        """
        Initialize the infinite distributed sampler.

        Args:
            dataset: PyTorch dataset to sample from
            num_replicas: Number of processes participating in training
            rank: Rank of the current process
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling
            drop_last: Whether to drop last incomplete batch
        """
        if num_replicas is None and rank is None and not dist.is_initialized():
            num_replicas = 1
            rank = 0

        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        # Parent class already initializes self.epoch = 0
        logger.info(f"rank: {self.rank}: InfiniteDistributedSampler created...")

    def __iter__(self) -> Iterator[int]:
        """
        Create an infinite iterator that cycles through epochs automatically.

        Returns:
            Iterator that yields sample indices infinitely
        """
        while True:
            # self.epoch is maintained by parent class and set via set_epoch()
            yield from super().__iter__()
            # Move to next epoch for different shuffling
            self.set_epoch(self.epoch + 1)

    def state_dict(self) -> dict:
        """
        Get state dictionary for checkpointing.

        Returns:
            Dictionary containing current epoch for resume
        """
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load state from checkpoint for resume.

        Args:
            state_dict: Dictionary containing epoch to resume from
        """
        self.set_epoch(state_dict["epoch"])
        logger.info(
            f"rank: {self.rank}: Resumed InfiniteDistributedSampler from epoch {self.epoch}"
        )
