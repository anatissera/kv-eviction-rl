#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from typing import Dict

import torch


class BaseEnvironment:
    """Base class for RL environments."""

    def reset(self, raise_stop_iteration: bool = False) -> Dict[str, torch.Tensor]:
        """Reset the environment for a new episode."""
        raise NotImplementedError

    def step(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Take a step in the environment."""
        raise NotImplementedError

    def get_valid_actions(self) -> Dict[str, torch.Tensor]:
        """Return information about valid actions for the current state.

        By default, all actions are valid. Override in subclasses.
        """
        return {}  # Empty dict means no constraints
