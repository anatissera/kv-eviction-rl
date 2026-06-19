#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import abc

import torch

from kvcompression.costs import CostResult


class RewardStrategy(abc.ABC):
    """Base class for reward computation strategies in RL training.

    Reward strategies take the oracle cost computation results and compute
    the final scalar reward for each episode.
    """

    @abc.abstractmethod
    def compute_reward(self, cost_result: CostResult) -> torch.Tensor:
        """Compute reward from oracle cost results.

        Args:
            cost_result: Oracle computation result containing all metrics

        Returns:
            torch.Tensor: Scalar reward for each sample in the batch, shape [batch_size]
        """
        ...
