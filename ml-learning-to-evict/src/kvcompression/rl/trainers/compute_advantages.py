#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from typing import Optional

import torch


@torch.no_grad
def compute_returns_and_advantage(
    rewards: torch.Tensor, episode_lengths: torch.Tensor, normalize_advantages: bool
) -> Optional[torch.Tensor]:
    """Compute advantages using RLOO method for terminal rewards (Original Loop)."""

    valid_episodes = episode_lengths > 0
    if not valid_episodes.any():
        return None

    valid_indices = torch.where(valid_episodes)[0]

    N = len(valid_indices)
    if N <= 1:
        return None

    original_dtype = rewards.dtype
    valid_terminal_rewards = rewards[valid_indices].to(torch.float32)
    valid_lengths = episode_lengths[valid_indices]

    batch_sum = valid_terminal_rewards.sum()

    baselines = (batch_sum - valid_terminal_rewards) / (N - 1)

    advantages_per_episode = valid_terminal_rewards - baselines

    if normalize_advantages:
        # Normalize the per-episode advantages BEFORE expansion
        adv_mean = advantages_per_episode.mean()
        adv_std = advantages_per_episode.std()
        advantages_per_episode = (advantages_per_episode - adv_mean) / (adv_std + 1e-8)

    all_advantages = []

    for idx in range(N):
        normalized_advantage = advantages_per_episode[idx]

        length = valid_lengths[idx].item()

        episode_advantages = normalized_advantage.repeat(length)
        all_advantages.append(episode_advantages)

    if not all_advantages:
        return None

    cat_advantages = torch.cat(all_advantages)

    if cat_advantages.numel() > 0:
        return cat_advantages.to(original_dtype)
    else:
        return None


@torch.no_grad()
def compute_returns_and_advantage_vec(
    rewards: torch.Tensor, episode_lengths: torch.Tensor, normalize_advantages: bool
) -> Optional[torch.Tensor]:
    """
    Compute advantages using RLOO method for terminal rewards with vectorized data.
    This version is numerically stabilized for low-precision (bf16/fp16) inputs.
    """
    device = episode_lengths.device
    original_dtype = rewards.dtype
    rewards = rewards.to(device)

    valid_episodes_mask = episode_lengths > 0
    if not valid_episodes_mask.any():
        return None

    valid_indices = torch.where(valid_episodes_mask)[0]
    valid_lengths = episode_lengths[valid_indices]
    N = len(valid_indices)

    if N <= 1:
        return None

    # Promote to float32 for numerical stability
    valid_terminal_rewards_fp32 = rewards[valid_indices].to(torch.float32)

    batch_sum = valid_terminal_rewards_fp32.sum()

    # Leave-one-out baselines
    baselines = (batch_sum - valid_terminal_rewards_fp32) / (N - 1)

    advantages_per_episode = valid_terminal_rewards_fp32 - baselines
    if normalize_advantages:
        adv_mean = advantages_per_episode.mean()
        adv_std = advantages_per_episode.std()
        advantages_per_episode = (advantages_per_episode - adv_mean) / (adv_std + 1e-8)

    max_valid_len = valid_lengths.max().item()
    range_tensor = torch.arange(max_valid_len, device=device).expand(N, max_valid_len)
    step_mask = range_tensor < valid_lengths.unsqueeze(1)

    # Expand advantages and cast back to original dtype
    expanded_advantages = advantages_per_episode.unsqueeze(1).expand(N, max_valid_len)

    cat_advantages = expanded_advantages[step_mask]

    return cat_advantages.to(original_dtype) if cat_advantages.numel() > 0 else None
