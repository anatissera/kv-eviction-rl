#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import pytest
import torch
from torch.distributions import Categorical

from kvcompression.rl.trainers.policy_entropy_loss import compute_policy_entropy_loss

if torch.cuda.is_available():
    from kvcompression.rl.trainers.triton_policy_entropy_loss import (
        compute_policy_entropy_loss_triton,
    )

    FN_TO_TEST = {
        "vectorized": compute_policy_entropy_loss,
        "triton": compute_policy_entropy_loss_triton,
    }
else:
    FN_TO_TEST = {
        "vectorized": compute_policy_entropy_loss,
        "triton": None,
    }


def compute_policy_entropy_loss_simple(
    device: torch.device,
    entropy_coef: float,
    batch_size: int,
    max_steps: int,
    action_dim: int,
    seq_lengths: torch.Tensor,  # Shape: [B] - Initial number of valid actions per episode
    episode_lengths: torch.Tensor,  # Shape: [B] - Number of steps taken per episode
    actions: torch.Tensor,  # Shape: [B, T] - Actions taken at each step
    policy_logits: torch.Tensor,  # Shape: [B, ActionDim] - Initial scores (logits) from policy, requires_grad=True
    advantages: torch.Tensor,  # Shape: [A] - Flattened advantages for valid steps (A = sum(episode_lengths))
):
    """
    A step-by-step version of the loss computation focusing on clarity.

    Iterates through each episode and each step within that episode:
    1. Determines the mask of actions that were valid *at that specific moment*. It
       masks out previously selected actions in that episode.
    2. Calculates the log-probability of the action *actually taken* at that step,
       using the scores and the reconstructed mask.
    3. Calculates the entropy of the distribution defined by the masked scores.
    4. Collects these log-probs and entropies.
    5. Computes the final policy and entropy losses using the collected values
       and the provided advantages.
    """
    log_probs_list = []
    entropies_list = []

    for b_idx in range(batch_size):
        current_episode_length = episode_lengths[b_idx].item()
        num_initially_valid_actions = seq_lengths[b_idx].item()
        episode_actions_taken = actions[b_idx, :current_episode_length]
        episode_initial_scores = policy_logits[b_idx]

        # Keep track of actions selected so far within this episode
        actions_selected_in_episode_so_far = set()

        for t_idx in range(current_episode_length):
            # Step 1: Determine valid actions mask for this step
            # An action is valid if:
            #   a) It was initially valid (index < num_initially_valid_actions)
            #   b) It has NOT been selected in previous steps of this episode

            valid_action_mask = torch.zeros(action_dim, dtype=torch.bool, device=device)

            # Enable initially valid actions
            upper_bound = min(num_initially_valid_actions, action_dim)
            valid_action_mask[:upper_bound] = True

            # Disable actions selected in previous steps
            for previously_selected_action_idx in actions_selected_in_episode_so_far:
                if 0 <= previously_selected_action_idx < action_dim:
                    valid_action_mask[previously_selected_action_idx] = False

            # Step 2: Calculate log_prob and entropy using the masked logits
            step_logits = episode_initial_scores.clone()
            step_logits[~valid_action_mask] = -1e9

            try:
                step_distribution = Categorical(logits=step_logits)
            except ValueError as e:
                print(f"ERROR creating Categorical at episode={b_idx}, step={t_idx}")
                print(f"Logits: {step_logits}")
                print(f"Mask: {valid_action_mask}")
                print(f"Selected so far: {actions_selected_in_episode_so_far}")
                print(f"Initial valid count: {num_initially_valid_actions}")
                raise e

            action_taken_tensor = episode_actions_taken[t_idx]
            action_taken_idx = action_taken_tensor.item()

            if not (0 <= action_taken_idx < action_dim):
                raise IndexError(
                    f"Action index {action_taken_idx} out of bounds for action_dim {action_dim} "
                    f"at episode={b_idx}, step={t_idx}"
                )

            log_prob_for_step = step_distribution.log_prob(action_taken_tensor)
            entropy_for_step = step_distribution.entropy()

            log_probs_list.append(log_prob_for_step)
            entropies_list.append(entropy_for_step)

            # Step 3: Update state for the next step's mask calculation
            actions_selected_in_episode_so_far.add(action_taken_idx)

    # Handle case with no valid steps processed
    if not log_probs_list:
        if advantages.numel() != 0:
            raise ValueError(
                "Mismatch: No steps processed, but advantages were provided."
            )
        # Return zero losses, ensuring gradient requirements match input scores
        zero_loss = torch.tensor(
            0.0, device=device, requires_grad=policy_logits.requires_grad
        )
        zero_entropy = torch.tensor(0.0, device=device)
        return zero_loss, -zero_entropy, zero_loss.clone(), zero_entropy

    flat_log_probs = torch.stack(log_probs_list)
    flat_entropies = torch.stack(entropies_list)

    num_steps_processed = flat_log_probs.shape[0]
    if num_steps_processed != advantages.shape[0]:
        total_ep_steps_expected = episode_lengths.sum().item()
        raise ValueError(
            f"Step count mismatch: Processed {num_steps_processed} steps, "
            f"but expected {advantages.shape[0]} steps based on advantages tensor. "
            f"Sum of episode lengths was {total_ep_steps_expected}."
        )

    # Step 5: Calculate losses
    advantages_tensor = advantages.to(device).detach()

    policy_loss = -(flat_log_probs * advantages_tensor).mean()
    entropy_loss = -flat_entropies.mean()  # Negative entropy encourages exploration
    mean_entropy = flat_entropies.mean()
    total_loss = policy_loss + float(entropy_coef) * entropy_loss

    return total_loss, entropy_loss, policy_loss, mean_entropy


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("action_dim", [5, 10])
@pytest.mark.parametrize("max_steps", [3, 8])
@pytest.mark.parametrize("entropy_coef", [0.0, 0.01])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=["float32", "float64"],
)
@pytest.mark.parametrize(
    "optimize_fn",
    [
        "vectorized",
        "triton",
    ],
)
def test_loss_equivalence(
    batch_size, action_dim, max_steps, entropy_coef, dtype, optimize_fn
):
    """
    Tests if the simple loop-based loss calculation yields the same
    result as the original vectorized version.
    """
    if optimize_fn == "triton" and not torch.cuda.is_available():
        pytest.skip("Triton kernel requires CUDA")

    optimize_fn = FN_TO_TEST[optimize_fn]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    seq_lengths = torch.randint(1, action_dim + 1, (batch_size,), device=device)

    # Episode lengths clamped to seq_length to avoid simulating steps
    # after running out of initial actions.
    potential_episode_lengths = torch.randint(
        1, max_steps + 1, (batch_size,), device=device
    )
    episode_lengths = torch.min(potential_episode_lengths, seq_lengths)

    # Generate unique action sequences based on seq_lengths and episode_lengths
    actions = torch.full((batch_size, max_steps), 0, dtype=torch.long, device=device)
    for b in range(batch_size):
        ep_len = episode_lengths[b].item()
        seq_len = seq_lengths[b].item()

        if ep_len == 0 or seq_len == 0:
            continue

        num_actions_to_sample = ep_len

        possible_actions = torch.arange(seq_len, device=device)
        perm = torch.randperm(seq_len, device=device)
        sampled_actions = possible_actions[perm[:num_actions_to_sample]]

        actions[b, :num_actions_to_sample] = sampled_actions

    policy_logits = torch.randn(
        batch_size, action_dim, device=device, requires_grad=True, dtype=dtype
    )

    total_valid_steps = episode_lengths.sum().item()
    advantages = torch.randn(total_valid_steps, device=device, dtype=dtype)

    if policy_logits.grad is not None:
        policy_logits.grad = None

    (total_loss_orig, entropy_loss_orig, policy_loss_orig, mean_entropy_orig) = (
        optimize_fn(
            device=device,
            entropy_coef=entropy_coef,
            batch_size=batch_size,
            action_dim=action_dim,
            max_steps=max_steps,
            seq_lengths=seq_lengths,
            episode_lengths=episode_lengths,
            actions=actions,
            policy_logits=policy_logits,
            advantages=advantages,
        )
    )

    scores_batch_grad_simple = policy_logits.clone().detach().requires_grad_(True)

    if scores_batch_grad_simple.grad is not None:
        scores_batch_grad_simple.grad = None

    (
        total_loss_simple,
        entropy_loss_simple,
        policy_loss_simple,
        mean_entropy_simple,
    ) = compute_policy_entropy_loss_simple(
        device=device,
        entropy_coef=entropy_coef,
        batch_size=batch_size,
        action_dim=action_dim,
        max_steps=max_steps,
        seq_lengths=seq_lengths,
        episode_lengths=episode_lengths,
        actions=actions,
        policy_logits=scores_batch_grad_simple,
        advantages=advantages,
    )

    atol = 1e-5
    rtol = 1e-4

    assert not torch.isnan(total_loss_orig), "Original total loss is NaN"
    assert not torch.isnan(total_loss_simple), "Simple total loss is NaN"

    assert torch.allclose(total_loss_orig, total_loss_simple, atol=atol, rtol=rtol), (
        f"Total loss mismatch: Orig={total_loss_orig.item()}, Simple={total_loss_simple.item()}"
    )
    assert torch.allclose(
        entropy_loss_orig, entropy_loss_simple, atol=atol, rtol=rtol
    ), (
        f"Entropy loss mismatch: Orig={entropy_loss_orig.item()}, Simple={entropy_loss_simple.item()}"
    )
    assert torch.allclose(policy_loss_orig, policy_loss_simple, atol=atol, rtol=rtol), (
        f"Policy loss mismatch: Orig={policy_loss_orig.item()}, Simple={policy_loss_simple.item()}"
    )
    assert torch.allclose(
        mean_entropy_orig, mean_entropy_simple, atol=atol, rtol=rtol
    ), (
        f"Mean entropy mismatch: Orig={mean_entropy_orig.item()}, Simple={mean_entropy_simple.item()}"
    )

    total_loss_orig.backward()
    total_loss_simple.backward()

    assert policy_logits.grad is not None, (
        "Gradient missing for original function scores"
    )
    assert scores_batch_grad_simple.grad is not None, (
        "Gradient missing for simple function scores"
    )
    assert not torch.isnan(policy_logits.grad).any(), "Original gradient contains NaN"
    assert not torch.isnan(scores_batch_grad_simple.grad).any(), (
        "Simple gradient contains NaN"
    )
    assert torch.allclose(
        policy_logits.grad, scores_batch_grad_simple.grad, atol=atol, rtol=rtol
    ), (
        f"Gradient mismatch:\nOrig Grad:\n{policy_logits.grad}\nSimple Grad:\n{scores_batch_grad_simple.grad}"
    )

    print(
        f"\nTest Passed for: B={batch_size}, A={action_dim}, T={max_steps}, C={entropy_coef}"
    )
    print(
        f"  Losses (Orig/Simple): Total={total_loss_orig.item():.4f}/{total_loss_simple.item():.4f}, "
        f"Policy={policy_loss_orig.item():.4f}/{policy_loss_simple.item():.4f}, "
        f"Entropy={entropy_loss_orig.item():.4f}/{entropy_loss_simple.item():.4f}"
    )
