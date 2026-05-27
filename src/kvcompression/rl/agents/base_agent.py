#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import abc
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class BaseAgent(nn.Module, metaclass=abc.ABCMeta):
    """Base class for RL agents."""

    @abc.abstractmethod
    def forward(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute scores or logits for each possible action given an observation.

        Args:
            observation: Dictionary containing observation tensors (keys, values,
                queries, seq_lengths, etc.).

        Returns:
            Dictionary containing at least 'scores' tensor of shape [B, action_dim].
        """
        raise NotImplementedError()

    def sample_actions(
        self, logits: torch.Tensor, sample: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Sample actions from logits using a categorical distribution.

        Args:
            logits: Action logits of shape [B, action_dim].
            sample: If True, sample from distribution. If False, take argmax.

        Returns:
            Dictionary containing:
                - 'actions': Selected action indices [B]
                - 'log_probs': Log probabilities of selected actions [B]
                - 'entropy': Entropy of the action distribution [B]
        """
        probs = F.softmax(logits, dim=1)
        dist = Categorical(probs=probs)

        if sample:
            actions = dist.sample()
        else:
            actions = torch.argmax(logits, dim=1)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return {
            "actions": actions,
            "log_probs": log_probs,
            "entropy": entropy,
        }

    def get_action(
        self,
        observation: Dict[str, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Get action and log probability from raw observation.

        Args:
            observation: Dictionary containing observation tensors.
            action_mask: Boolean mask of valid actions [B, action_dim]. If None,
                created from observation's selection_history and seq_lengths.
            sample: If True, sample from distribution. If False, take argmax.

        Returns:
            Dictionary containing 'actions', 'log_probs', and 'entropy'.
        """
        if action_mask is None:
            valid_actions = {
                "selection_history": observation["selection_history"],
                "seq_lengths": observation["seq_lengths"],
            }
            action_mask = self.create_action_mask(observation, valid_actions)

        scores = self(observation)["scores"]
        scores[~action_mask] = -1e9
        return self.sample_actions(scores, sample)

    @torch.no_grad()
    def act(
        self, observation: Dict[str, torch.Tensor], sample: bool = False
    ) -> torch.Tensor:
        """Standalone action selection method for inference."""
        return self.get_action(observation=observation, sample=sample)["actions"]

    def create_action_mask(
        self,
        observation: Dict[str, torch.Tensor],
        valid_actions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Create action mask for sorting task.

        Masks out already-selected tokens and positions beyond sequence length.

        Args:
            observation: Dictionary containing observation tensors (unused here).
            valid_actions: Dictionary with 'selection_history' [B, S] boolean tensor
                indicating already-selected positions, and 'seq_lengths' [B].

        Returns:
            Boolean mask [B, S] where True indicates a valid action.
        """
        mask = ~valid_actions["selection_history"]
        seq_lengths = valid_actions["seq_lengths"]
        length_mask = (
            torch.arange(mask.shape[-1], device=mask.device)[None, :]
            < seq_lengths[:, None]
        )
        return mask & length_mask
