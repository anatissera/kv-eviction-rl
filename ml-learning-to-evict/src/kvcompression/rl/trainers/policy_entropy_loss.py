#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def compute_policy_entropy_loss(
    device: torch.device,
    entropy_coef: float,
    batch_size: int,
    action_dim: int,
    max_steps: int,
    seq_lengths: torch.Tensor,
    episode_lengths: torch.Tensor,
    actions: torch.Tensor,  # [batch_size, max_steps]
    policy_logits: torch.Tensor,  # [batch_size, action_dim]
    advantages: torch.Tensor,  # [A] - Flattened advantages for valid steps
):
    """
    Compute policy gradient loss with entropy regularization.

    Reconstructs action masks from episode history and computes the REINFORCE
    loss with entropy bonus for exploration.

    Args:
        device: Device for tensor operations.
        entropy_coef: Coefficient for entropy regularization term.
        batch_size: Number of episodes in the batch.
        action_dim: Size of the action space (sequence length).
        max_steps: Maximum episode length in the batch.
        seq_lengths: Valid sequence lengths per episode [B].
        episode_lengths: Actual episode lengths [B].
        actions: Actions taken at each step [B, max_steps].
        policy_logits: Policy network output scores [B, action_dim].
        advantages: Flattened advantages for valid steps [A].

    Returns:
        Tuple of (total_loss, entropy_loss, policy_loss, avg_entropy).
    """
    policy_logits = policy_logits
    advantages = advantages

    # Reconstruct per-step action masks and calculate log_probs/entropy
    initial_length_mask = torch.arange(action_dim, device=device).expand(
        batch_size, action_dim
    ) < seq_lengths.unsqueeze(1)
    # Expand to match shape (B, T, ActionDim)
    initial_length_mask_expanded = initial_length_mask.unsqueeze(1).expand(
        -1, max_steps, -1
    )

    actions_one_hot = F.one_hot(
        actions, num_classes=action_dim
    ).bool()  # [B, T, ActionDim]

    # Compute cumulative selections (True if selected at or before step t)
    cumulative_selected = torch.cumsum(actions_one_hot, dim=1)

    # Shift cumulative selections to get the mask *before* each step
    selections_before_t = torch.cat(
        (
            torch.zeros_like(cumulative_selected[:, :1, :]),
            cumulative_selected[:, :-1, :],
        ),
        dim=1,
    )  # [B, T, ActionDim]

    reconstructed_masks = (selections_before_t == 0) & initial_length_mask_expanded

    if policy_logits.shape == (batch_size, action_dim):
        scores_per_step_grad = policy_logits.unsqueeze(1).expand(-1, max_steps, -1)
    else:
        raise ValueError(f"Unexpected scores shape: {policy_logits.shape}")

    masked_logits_grad = scores_per_step_grad.clone()
    masked_logits_grad[~reconstructed_masks] = -1e9

    dist = Categorical(logits=masked_logits_grad)

    log_probs_all = dist.log_prob(actions)  # [B, MaxSteps]
    entropies_all = dist.entropy()  # [B, MaxSteps]

    # Create mask for valid steps based on episode lengths
    step_mask = torch.arange(max_steps, device=device).expand(
        batch_size, max_steps
    ) < episode_lengths.unsqueeze(1)

    # Select log_probs, entropies, and advantages for valid steps
    cat_log_probs = log_probs_all[step_mask]
    cat_entropies = entropies_all[step_mask]

    policy_loss = -(cat_log_probs * advantages).mean()
    entropy_loss = -cat_entropies.mean()
    total_loss = policy_loss + entropy_coef * entropy_loss

    return total_loss, entropy_loss, policy_loss, cat_entropies.mean()
