#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import logging

import torch
import torch.autograd as autograd
import triton
import triton.language as tl

pylogger = logging.getLogger(__name__)


@triton.jit
def _policy_entropy_loss_forward_kernel(
    # Input Tensors
    logits_ptr,  # Pointer to initial policy logits [B, ActionDim]
    actions_ptr,  # Pointer to actions taken [B, T_max] (int32)
    seq_lengths_ptr,  # Pointer to initial valid action counts [B] (int32)
    episode_lengths_ptr,  # Pointer to episode lengths [B] (int32)
    prefix_sum_lengths_ptr,  # Pointer to exclusive prefix sum of episode_lengths [B+1] (int32)
    # Output Tensors (per step, flattened across the batch)
    flat_log_probs_out_ptr,  # Pointer to output flat log_probs [A] (float64)
    flat_entropies_out_ptr,  # Pointer to output flat entropies [A] (float64)
    # Tensor Shapes/Strides
    batch_size: tl.int32,
    action_dim: tl.int32,
    max_steps: tl.int32,  # T_max dimension of actions tensor
    stride_logits_b: tl.int32,  # Stride for batch dim in logits
    stride_logits_a: tl.int32,  # Stride for action dim in logits
    stride_actions_b: tl.int32,  # Stride for batch dim in actions
    stride_actions_t: tl.int32,  # Stride for time dim in actions
    # Constants
    logit_dtype: tl.constexpr,  # Dtype of the input logits
    BLOCK_SIZE_ACTION: tl.constexpr,  # Triton block size, must be power of 2 >= action_dim
):
    """
    Triton kernel for the forward pass of the policy/entropy loss calculation.

    Computes log_probs of taken actions and entropy for each valid step (b, t)
    across the batch, handling the dynamically changing action mask.

    Writes results to flattened output tensors indexed by the cumulative sum
    of episode lengths. Uses numerically stable log-softmax computation.
    Intermediate log-softmax calculations are done in float32 (or float64 if input is float64)
    for stability, while outputs are stored in float64 for precision.
    """
    # Each program instance handles one episode (batch item)
    b_idx = tl.program_id(0)

    # Guard against out-of-bounds batch index
    if b_idx >= batch_size:
        return

    # Load episode-specific scalars
    episode_length: tl.int32 = tl.load(episode_lengths_ptr + b_idx)
    num_initially_valid: tl.int32 = tl.load(seq_lengths_ptr + b_idx)
    # Starting index for this episode in the flat output tensors
    flat_start_idx: tl.int32 = tl.load(prefix_sum_lengths_ptr + b_idx)

    logits_b_ptr = logits_ptr + b_idx * stride_logits_b
    actions_b_ptr = actions_ptr + b_idx * stride_actions_b

    a_indices = tl.arange(0, BLOCK_SIZE_ACTION)
    action_range_mask = a_indices < action_dim

    # Load logits, using -inf for masked values (indices >= action_dim)
    episode_logits = tl.load(
        logits_b_ptr + a_indices * stride_logits_a,
        mask=action_range_mask,
        other=-float("inf"),
    ).to(logit_dtype)

    # Determine compute dtype (use float32 for stability unless input is float64)
    if logit_dtype == tl.float64:
        compute_dtype = tl.float64
    else:
        compute_dtype = tl.float32

    # Initialize mask: action is valid if index < num_initially_valid and within action_dim
    valid_action_mask = (a_indices < num_initially_valid) & action_range_mask

    for t_idx in range(episode_length):
        # Set invalid action logits to -inf for this step
        step_logits = tl.where(valid_action_mask, episode_logits, -float("inf"))

        # Compute LogSoftmax (numerically stable, in compute_dtype)
        step_logits_compute = step_logits.to(compute_dtype)

        # Subtract max logit to prevent overflow in exp
        max_logit = tl.max(step_logits_compute, axis=0)
        stable_logits = tl.where(
            valid_action_mask, step_logits_compute - max_logit, -float("inf")
        )
        exp_logits = tl.exp(stable_logits)
        sum_exp_logits = tl.sum(exp_logits, axis=0)
        # log(sum(exp(logits))), adding max_logit back for correct scale
        log_sum_exp_val = (
            tl.log(sum_exp_logits + 1e-20) + max_logit
        )  # Result is compute_dtype

        # Log probabilities: logits - log_sum_exp
        step_log_probs = tl.where(
            valid_action_mask,
            step_logits_compute - log_sum_exp_val,
            -float("inf"),  # Log prob of invalid action is -inf
        )

        # Probabilities derived from log_probs for consistency
        step_probs = tl.exp(step_log_probs)
        step_probs = tl.where(valid_action_mask, step_probs, 0.0)

        # Entropy: H(p) = -sum(p_i * log(p_i)), with p_i=0 case handled
        entropy_terms = tl.where(step_probs > 0, step_probs * step_log_probs, 0.0)
        entropy_for_step = -tl.sum(entropy_terms, axis=0)  # Result is compute_dtype

        # Get action taken at this step
        action_taken_idx = tl.load(actions_b_ptr + t_idx * stride_actions_t)  # is int32

        # Get log_prob of the action taken using sum(where(mask, value, 0))
        action_taken_mask = (a_indices == action_taken_idx) & action_range_mask
        log_prob_for_step = tl.sum(
            tl.where(action_taken_mask, step_log_probs, 0.0), axis=0
        )  # Result is compute_dtype

        # Store results in flat output tensors, cast to float64 for precision
        current_flat_idx = flat_start_idx + t_idx
        tl.store(
            flat_log_probs_out_ptr + current_flat_idx, log_prob_for_step.to(tl.float64)
        )
        tl.store(
            flat_entropies_out_ptr + current_flat_idx, entropy_for_step.to(tl.float64)
        )

        # Update mask for the next step: mask out the action just taken
        valid_action_mask = valid_action_mask & (a_indices != action_taken_idx)


@triton.jit
def _policy_entropy_loss_backward_kernel(
    # Input Tensors (from forward pass context or original inputs)
    logits_ptr,  # Initial policy logits [B, ActionDim] (original dtype)
    actions_ptr,  # Actions taken [B, T_max] (int32)
    seq_lengths_ptr,  # Initial valid action counts [B] (int32)
    episode_lengths_ptr,  # Episode lengths [B] (int32)
    prefix_sum_lengths_ptr,  # Exclusive prefix sum of episode_lengths [B+1] (int32)
    advantages_ptr,  # Advantages tensor [A] (flattened, float64)
    # Gradient Input (from downstream loss)
    grad_output_ptr,  # Pointer to the single scalar gradient flowing back (float64)
    # Output Tensor (Gradient w.r.t. forward inputs)
    grad_logits_out_ptr,  # Output gradient w.r.t initial logits [B, ActionDim] (float64 buffer)
    # Tensor Shapes/Strides
    batch_size: tl.int32,
    action_dim: tl.int32,
    max_steps: tl.int32,  # T_max
    total_num_steps: tl.int32,  # Total steps A = sum(episode_lengths)
    stride_logits_b: tl.int32,
    stride_logits_a: tl.int32,
    stride_actions_b: tl.int32,
    stride_actions_t: tl.int32,
    stride_grad_logits_b: tl.int32,  # Strides for the output gradient tensor
    stride_grad_logits_a: tl.int32,
    # Constants
    logit_dtype: tl.constexpr,  # Dtype of the input logits
    entropy_coef: tl.float64,  # Entropy coefficient (use float64 for precision)
    BLOCK_SIZE_ACTION: tl.constexpr,
):
    """
    Triton kernel for the backward pass of the policy/entropy loss calculation.

    Computes the gradient of the total loss with respect to the initial policy logits.
    It recomputes the necessary intermediate values (probabilities, log-probabilities,
    entropy) from the forward pass for each step, using the same numerically stable
    methods. The gradient calculation combines the policy gradient term and the
    entropy gradient term. Uses float64 for intermediate calculations and accumulation
    to maintain precision. The final gradient is stored in a float64 buffer.
    """
    b_idx = tl.program_id(0)

    if b_idx >= batch_size:
        return

    episode_length: tl.int32 = tl.load(episode_lengths_ptr + b_idx)
    num_initially_valid: tl.int32 = tl.load(seq_lengths_ptr + b_idx)
    flat_start_idx: tl.int32 = tl.load(prefix_sum_lengths_ptr + b_idx)

    logits_b_ptr = logits_ptr + b_idx * stride_logits_b
    actions_b_ptr = actions_ptr + b_idx * stride_actions_b
    grad_logits_b_ptr = grad_logits_out_ptr + b_idx * stride_grad_logits_b

    # Load initial logits (with original dtype)
    a_indices = tl.arange(0, BLOCK_SIZE_ACTION)
    action_range_mask = a_indices < action_dim
    episode_logits = tl.load(
        logits_b_ptr + a_indices * stride_logits_a,
        mask=action_range_mask,
        other=-float("inf"),
    ).to(logit_dtype)

    # Gradient accumulator (float64)
    grad_logits_acc = tl.zeros((BLOCK_SIZE_ACTION,), dtype=tl.float64)

    grad_output_f64 = tl.load(grad_output_ptr).to(tl.float64)

    # Precompute averaging factor over total steps A (float64)
    inv_total_steps = 1.0 / total_num_steps if total_num_steps > 0 else 0.0
    inv_total_steps_f64 = inv_total_steps.to(tl.float64)

    valid_action_mask = (a_indices < num_initially_valid) & action_range_mask

    # Process each step taken in the episode
    for t_idx in range(episode_length):
        current_flat_idx = flat_start_idx + t_idx

        # 1. Recompute the distribution for the current step (in float64)
        step_logits = tl.where(valid_action_mask, episode_logits, -float("inf"))
        step_logits_f64 = step_logits.to(tl.float64)

        # Numerically stable softmax/log-softmax matching forward pass, but in float64
        max_logit = tl.max(step_logits_f64, axis=0)
        stable_logits = tl.where(
            valid_action_mask, step_logits_f64 - max_logit, -float("inf")
        )
        exp_logits = tl.exp(stable_logits)
        sum_exp = tl.sum(exp_logits, axis=0)
        log_sum_exp_val = tl.log(sum_exp + 1e-20) + max_logit  # float64

        # Recompute log_probs and probs in float64
        step_log_probs = tl.where(
            valid_action_mask,
            step_logits_f64 - log_sum_exp_val,
            -float("inf"),
        )
        probs = tl.exp(step_log_probs)
        probs = tl.where(valid_action_mask, probs, 0.0)

        # 2. Get action taken and corresponding advantage (float64)
        action_taken = tl.load(actions_b_ptr + t_idx * stride_actions_t)  # int32
        advantage = tl.load(advantages_ptr + current_flat_idx).to(tl.float64)

        # 3. Policy gradient component (float64)
        action_is_taken = (a_indices == action_taken) & valid_action_mask
        indicator = tl.where(action_is_taken, 1.0, 0.0).to(tl.float64)

        # Factor includes averaging and negative sign from loss definition
        policy_factor = -advantage * inv_total_steps_f64

        # Gradient contribution from policy loss: factor * (indicator - probs)
        policy_grad = tl.where(
            valid_action_mask,
            policy_factor * (indicator - probs),
            0.0,  # No gradient for initially invalid actions
        )

        # 4. Entropy gradient component (float64)
        # Recompute entropy in float64
        entropy_terms = tl.where(probs > 0, probs * step_log_probs, 0.0)
        entropy = -tl.sum(
            entropy_terms, axis=0
        )  # Scalar entropy for this step (float64)

        # Factor includes averaging and entropy coefficient
        entropy_factor = entropy_coef * inv_total_steps_f64

        # (log(p_i) + entropy) term; if p_i=0, gradient contribution is 0
        log_p_plus_entropy = tl.where(probs > 0, step_log_probs + entropy, 0.0)

        # Gradient contribution from entropy loss
        entropy_grad = tl.where(
            valid_action_mask,
            entropy_factor * probs * log_p_plus_entropy,
            0.0,  # No gradient for initially invalid actions
        )

        # 5. Combine gradients and apply outer gradient (float64)
        step_grad = (policy_grad + entropy_grad) * grad_output_f64
        grad_logits_acc += step_grad

        valid_action_mask = valid_action_mask & (a_indices != action_taken)

    tl.store(
        grad_logits_b_ptr + a_indices * stride_grad_logits_a,
        grad_logits_acc,  # float64
        mask=action_range_mask,
    )


class PolicyEntropyLossTriton(autograd.Function):
    """
    PyTorch autograd Function to compute the policy/entropy loss using Triton kernels.

    This class handles the setup, kernel launch, and gradient definition required
    to integrate the custom Triton implementation into PyTorch's automatic
    differentiation system. Respects input dtype.
    """

    @staticmethod
    def forward(
        ctx,
        policy_logits: torch.Tensor,  # Shape: [B, ActionDim], requires_grad=True
        actions: torch.Tensor,  # Shape: [B, T]
        seq_lengths: torch.Tensor,  # Shape: [B]
        episode_lengths: torch.Tensor,  # Shape: [B]
        advantages: torch.Tensor,  # Shape: [A] (A = sum(episode_lengths))
        entropy_coef: float,
    ):
        """
        Forward pass using the Triton kernel `_policy_entropy_loss_forward_kernel`.

        Args:
            ctx: Context object to save information for backward pass.
            policy_logits: Initial policy logits tensor.
            actions: Actions taken tensor.
            seq_lengths: Initial valid action counts tensor.
            episode_lengths: Actual episode lengths tensor.
            advantages: Flattened advantages tensor.
            entropy_coef: Entropy coefficient scalar.

        Returns:
            Tuple containing (total_loss, entropy_loss, policy_loss, mean_entropy),
            similar to the reference Python function.
        """
        # Input validation
        assert policy_logits.dim() == 2
        assert actions.dim() == 2
        assert seq_lengths.dim() == 1
        assert episode_lengths.dim() == 1
        assert advantages.dim() == 1
        assert (
            policy_logits.shape[0]
            == actions.shape[0]
            == seq_lengths.shape[0]
            == episode_lengths.shape[0]
        )
        assert policy_logits.is_cuda, (
            "Triton implementation requires tensors on CUDA device."
        )
        # Kernels expect int32 for indices/lengths
        if not actions.dtype == torch.int32:
            actions = actions.to(torch.int32)
        if not seq_lengths.dtype == torch.int32:
            seq_lengths = seq_lengths.to(torch.int32)
        if not episode_lengths.dtype == torch.int32:
            episode_lengths = episode_lengths.to(torch.int32)

        device = policy_logits.device
        batch_size, action_dim = policy_logits.shape
        max_steps = actions.shape[1]
        logit_dtype = policy_logits.dtype
        tl_logit_dtype = (
            tl.float16
            if logit_dtype == torch.float16
            else tl.bfloat16
            if logit_dtype == torch.bfloat16
            else tl.float32
            if logit_dtype == torch.float32
            else tl.float64
        )

        total_num_steps = int(episode_lengths.sum().item())

        if total_num_steps > 0 and total_num_steps != advantages.shape[0]:
            pylogger.warning(
                f"advantages shape {advantages.shape[0]} doesn't match "
                f"sum of episode_lengths {total_num_steps}. Using advantages shape "
                f"({advantages.shape[0]}) for loss calculation."
            )
            total_num_steps = advantages.shape[0]
        elif total_num_steps == 0:
            advantages = torch.empty(
                0, dtype=advantages.dtype, device=advantages.device
            )  # Ensure consistency if 0 steps

        # Handle empty batch / zero steps case
        if total_num_steps == 0:
            zero_loss = torch.tensor(
                0.0,
                device=device,
                dtype=logit_dtype,
                requires_grad=policy_logits.requires_grad,
            )
            zero_entropy_f64 = torch.tensor(0.0, device=device, dtype=torch.float64)
            ctx.save_for_backward(
                policy_logits, actions, seq_lengths, episode_lengths, advantages
            )
            ctx.entropy_coef = entropy_coef
            ctx.total_num_steps = 0
            # Need prefix_sum_lengths for backward type consistency
            ctx.prefix_sum_lengths = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device
            )
            ctx.logit_dtype = logit_dtype
            ctx.action_dim = action_dim
            return (
                zero_loss,
                -zero_entropy_f64,
                zero_loss.clone().detach(),
                zero_entropy_f64,
            )

        # Exclusive prefix sum of episode lengths for flat tensor indexing
        prefix_sum_lengths = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        )
        torch.cumsum(episode_lengths, dim=0, out=prefix_sum_lengths[1:])

        # Output tensors (float64 for precision)
        flat_log_probs = torch.empty(
            total_num_steps, dtype=torch.float64, device=device
        )
        flat_entropies = torch.empty(
            total_num_steps, dtype=torch.float64, device=device
        )

        # Triton launch: one program instance per batch item
        grid = (batch_size,)
        BLOCK_SIZE_ACTION = triton.next_power_of_2(action_dim)

        _policy_entropy_loss_forward_kernel[grid](
            policy_logits,
            actions,
            seq_lengths,
            episode_lengths,
            prefix_sum_lengths,
            flat_log_probs,
            flat_entropies,
            batch_size,
            action_dim,
            max_steps,
            policy_logits.stride(0),
            policy_logits.stride(1),
            actions.stride(0),
            actions.stride(1),
            logit_dtype=tl_logit_dtype,
            BLOCK_SIZE_ACTION=BLOCK_SIZE_ACTION,
        )

        # Calculate final losses using float64 kernel outputs
        advantages_f64 = advantages.to(device=device, dtype=torch.float64).detach()

        policy_loss = -(flat_log_probs * advantages_f64).mean()

        mean_entropy = flat_entropies.mean()
        entropy_loss = -mean_entropy  # Loss term encourages higher entropy

        total_loss_f64 = policy_loss + float(entropy_coef) * entropy_loss

        # Save tensors and constants needed for backward pass
        ctx.save_for_backward(
            policy_logits,
            actions,
            seq_lengths,
            episode_lengths,
            prefix_sum_lengths,
            advantages,
        )
        ctx.entropy_coef = entropy_coef
        ctx.total_num_steps = total_num_steps
        ctx.action_dim = action_dim
        ctx.logit_dtype = logit_dtype

        # Only total_loss propagates gradients; auxiliary outputs are detached
        return (
            total_loss_f64.to(logit_dtype),
            entropy_loss.to(logit_dtype).detach(),
            policy_loss.to(logit_dtype).detach(),
            mean_entropy.to(logit_dtype).detach(),
        )

    @staticmethod
    def backward(
        ctx, grad_total_loss, grad_entropy_loss, grad_policy_loss, grad_mean_entropy
    ):
        """
        Backward pass using the Triton kernel `_policy_entropy_loss_backward_kernel`.
        Returns gradient matching the input policy_logits dtype.
        """
        if ctx.total_num_steps == 0:
            policy_logits, _, _, _, _, _ = ctx.saved_tensors
            return torch.zeros_like(policy_logits), None, None, None, None, None

        (
            policy_logits,
            actions,
            seq_lengths,
            episode_lengths,
            prefix_sum_lengths,
            advantages,
        ) = ctx.saved_tensors
        entropy_coef = ctx.entropy_coef
        total_num_steps = ctx.total_num_steps
        action_dim = ctx.action_dim
        logit_dtype = ctx.logit_dtype
        # Map torch dtype to triton dtype
        tl_logit_dtype = (
            tl.float16
            if logit_dtype == torch.float16
            else tl.bfloat16
            if logit_dtype == torch.bfloat16
            else tl.float32
            if logit_dtype == torch.float32
            else tl.float64
        )

        if total_num_steps == 0:
            return (
                torch.zeros_like(policy_logits, dtype=logit_dtype),
                None,
                None,
                None,
                None,
                None,
            )

        device = policy_logits.device
        batch_size = policy_logits.shape[0]
        max_steps = actions.shape[1]

        # Ensure incoming gradient is float64
        if grad_total_loss is None:
            grad_total_loss_f64 = torch.tensor(1.0, dtype=torch.float64, device=device)
        else:
            grad_total_loss_f64 = grad_total_loss.to(device=device, dtype=torch.float64)

        # Gradient output buffer (float64 for accumulation in kernel)
        grad_logits_out_f64 = torch.zeros_like(policy_logits, dtype=torch.float64)
        advantages_f64 = advantages.to(device=device, dtype=torch.float64)

        grid = (batch_size,)
        BLOCK_SIZE_ACTION = triton.next_power_of_2(action_dim)

        _policy_entropy_loss_backward_kernel[grid](
            policy_logits,
            actions,
            seq_lengths,
            episode_lengths,
            prefix_sum_lengths,
            advantages_f64,
            grad_total_loss_f64,
            grad_logits_out_f64,
            batch_size,
            action_dim,
            max_steps,
            total_num_steps,
            policy_logits.stride(0),
            policy_logits.stride(1),
            actions.stride(0),
            actions.stride(1),
            grad_logits_out_f64.stride(0),
            grad_logits_out_f64.stride(1),
            logit_dtype=tl_logit_dtype,
            entropy_coef=float(entropy_coef),
            BLOCK_SIZE_ACTION=BLOCK_SIZE_ACTION,
        )

        # Convert float64 gradient back to the original dtype
        grad_logits_out_typed = grad_logits_out_f64.to(dtype=logit_dtype)

        # Only policy_logits requires a gradient
        return grad_logits_out_typed, None, None, None, None, None


def compute_policy_entropy_loss_triton(
    # Arguments kept consistent with the reference function for easy swapping
    device: torch.device,
    entropy_coef: float,
    batch_size: int,
    max_steps: int,
    action_dim: int,
    seq_lengths: torch.Tensor,
    episode_lengths: torch.Tensor,
    actions: torch.Tensor,
    policy_logits: torch.Tensor,
    advantages: torch.Tensor,
):
    """
    Computes the policy gradient and entropy loss using the optimized Triton implementation.

    This function serves as a user-friendly interface to the `PolicyEntropyLossTriton`
    autograd Function. It takes the same arguments as the reference implementation
    and respects the dtype of `policy_logits`.

    Args:
        device: The torch device (inferred from tensors).
        entropy_coef: Coefficient for the entropy bonus term.
        batch_size: Number of episodes (inferred).
        max_steps: Max sequence length T (inferred).
        action_dim: Action dimension (inferred).
        seq_lengths: Tensor of shape [B], initial valid action counts (int32/int64).
        episode_lengths: Tensor of shape [B], actual steps taken (int32/int64).
        actions: Tensor of shape [B, T], actions indices (int32/int64).
        policy_logits: Tensor of shape [B, ActionDim], initial policy logits (requires grad).
                       The dtype of this tensor determines the computation precision where possible
                       and the dtype of the returned total_loss and gradient.
        advantages: Tensor of shape [A], flattened advantages (A = sum(episode_lengths)).

    Returns:
        A tuple containing:
        - total_loss: The combined policy and entropy loss (with grad history, matches policy_logits.dtype).
        - entropy_loss: The negative average entropy (-mean(entropy)) (detached, float64).
        - policy_loss: The negative average policy gradient loss term (detached, float64).
        - mean_entropy: The average entropy across all steps taken (detached, float64).
    """
    inferred_device = policy_logits.device
    if not policy_logits.is_cuda:
        raise ValueError(
            "Triton implementation requires tensors to be on a CUDA device."
        )
    # Check other tensors are on the same device
    tensors_to_check = [actions, seq_lengths, episode_lengths, advantages]
    if not all(t.device == inferred_device for t in tensors_to_check):
        pylogger.warning(f"Moving input tensors to device {inferred_device}")
        actions = actions.to(inferred_device)
        seq_lengths = seq_lengths.to(inferred_device)
        episode_lengths = episode_lengths.to(inferred_device)
        advantages = advantages.to(inferred_device)
        if not all(t.device == inferred_device for t in tensors_to_check):
            raise ValueError(
                f"Failed to move all input tensors to device {inferred_device}."
            )

    if not policy_logits.requires_grad:
        pylogger.warning(
            "policy_logits does not require grad. Setting requires_grad=True."
        )
        policy_logits.requires_grad_(True)

    total_loss, entropy_loss, policy_loss, mean_entropy = PolicyEntropyLossTriton.apply(
        policy_logits,
        actions,
        seq_lengths,
        episode_lengths,
        advantages,
        entropy_coef,
    )

    return total_loss, entropy_loss, policy_loss, mean_entropy
