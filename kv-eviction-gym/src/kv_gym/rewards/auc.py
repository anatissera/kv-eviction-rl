"""
Phase 1 reward: future-attention AUC ratio.

Ported from ml-learning-to-evict/src/kvcompression/rl/rewards/future_attention_reward.py
Source: /Users/alexanderbodner/Documents/Udesa/5to/aprendizaje reforzado/tp final/ml-learning-to-evict

The reward is:
    AUC_policy = sum of future_attn[i] for i in resident tokens
    AUC_oracle = sum of future_attn[i] for the top-budget tokens (best possible)
    reward = AUC_policy / AUC_oracle  ∈ (0, 1]

AUC_oracle gives a reward of 1.0 only when the policy keeps exactly the
tokens that the oracle would have kept (highest future-attention tokens).
This signal requires no LLM call during RL steps — everything is computed
from tensors captured at episode reset.

Fully vectorized over all n_envs (all layer/head pairs) at once.
"""

import torch
from torch import Tensor


def future_attention_auc(
    future_attn: Tensor,  # [n_envs, prompt_len]  — from AllHeadCapture, normalized
    resident:    Tensor,  # [n_envs, prompt_len]  bool
    budget:      int,
) -> Tensor:              # [n_envs]  ∈ (0, 1]
    """Compute AUC_policy / AUC_oracle for every environment.

    Args:
        future_attn: Attention mass from generated tokens onto each prompt
                     token, captured at episode reset and normalized to sum=1.
        resident:    Boolean mask of surviving tokens at episode end.
        budget:      Number of tokens the policy is allowed to keep. Used to
                     compute the oracle (top-budget by future_attn).

    Returns:
        Per-environment reward in (0, 1]. Returns 0 if AUC_oracle == 0
        (degenerate case where no future attention exists).
    """
    # Policy AUC: total future attention mass on kept tokens
    auc_policy = (future_attn * resident.float()).sum(dim=-1)  # [n_envs]

    # Oracle AUC: total future attention mass on the best `budget` tokens
    topk_vals, _ = future_attn.topk(k=min(budget, future_attn.shape[-1]), dim=-1)
    auc_oracle = topk_vals.sum(dim=-1)  # [n_envs]

    # Ratio — clamp oracle to avoid division by zero
    reward = auc_policy / auc_oracle.clamp(min=1e-8)
    return reward
