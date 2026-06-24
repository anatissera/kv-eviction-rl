#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import torch

from kvcompression.presses.base_press import BasePress


class RandomPress(BasePress):
    """Prune random KV pairs"""

    def sort(
        self,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.rand(keys.shape[:-1], device=keys.device)
        return scores.argsort(-1, descending=False)
