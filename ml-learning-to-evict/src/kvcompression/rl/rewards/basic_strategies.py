#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import torch
from kvcompression.costs import CostResult
from .base_reward_strategy import RewardStrategy


class FutureAttentionAucNormalizedReward(RewardStrategy):
    """Reward based on negative normalized future attention AUC."""

    def compute_reward(self, cost_result: CostResult) -> torch.Tensor:
        return -cost_result.cost.squeeze(-1)
