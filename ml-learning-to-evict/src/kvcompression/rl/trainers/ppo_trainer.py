#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import logging
import warnings
from typing import Any, Dict, List, Optional, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Categorical
from torchtune import config

from kvcompression.metrics import MetricsManager
from kvcompression.rl.agents.base_agent import BaseAgent
from kvcompression.rl.environments.base_env import BaseEnvironment
from kvcompression.rl.trainers.rloo_gumbel_terminal_trainer import RLOOGumbelTrainer
from kvcompression.utils.utils import get_underlying_model

pylogger = logging.getLogger(__name__)


class KVValueNetwork(nn.Module):
    """
    Small critic network that estimates the expected episode return V(s).

    Takes the key and value tensors from one episode observation, mean-pools
    over the sequence dimension, and maps the resulting feature vector to a
    scalar. The architecture intentionally mirrors the policy agent's first
    projection so the two share a similar inductive bias.
    """

    def __init__(self, head_dim: int, hidden_size: int = 128):
        super().__init__()
        # Input: concatenation of mean-pooled keys and values = 2 * head_dim
        self.net = nn.Sequential(
            nn.Linear(2 * head_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs: observation dict with keys/values [B, S, D] and seq_lengths [B].

        Returns:
            Scalar value estimate per episode, shape [B].
        """
        keys = obs["keys"].to(torch.float32)       # [B, S, D]
        values = obs["values"].to(torch.float32)   # [B, S, D]
        seq_lengths = obs["seq_lengths"]            # [B]

        S = keys.shape[1]
        device = keys.device
        # Boolean mask [B, S] — True for valid positions
        mask = torch.arange(S, device=device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()  # [B, S, 1]

        # Mean-pool over valid positions
        keys_pool = (keys * mask_f).sum(1) / seq_lengths.float().unsqueeze(1).clamp(min=1)
        vals_pool = (values * mask_f).sum(1) / seq_lengths.float().unsqueeze(1).clamp(min=1)

        features = torch.cat([keys_pool, vals_pool], dim=-1)  # [B, 2*D]
        return self.net(features).squeeze(-1)  # [B]


def _compute_per_step_log_probs(
    device: torch.device,
    batch_size: int,
    action_dim: int,
    max_steps: int,
    seq_lengths: torch.Tensor,
    episode_lengths: torch.Tensor,
    actions: torch.Tensor,        # [B, max_steps]
    policy_logits: torch.Tensor,  # [B, action_dim]
) -> torch.Tensor:
    """
    Compute per-step log-probabilities for the taken actions.

    Reconstructs the available-action mask at each step (same logic as
    compute_policy_entropy_loss) and returns the log-prob for each valid
    (batch, step) pair as a flattened tensor [A].
    """
    initial_length_mask = (
        torch.arange(action_dim, device=device).expand(batch_size, action_dim)
        < seq_lengths.unsqueeze(1)
    )
    initial_length_mask_expanded = initial_length_mask.unsqueeze(1).expand(-1, max_steps, -1)

    actions_one_hot = F.one_hot(actions, num_classes=action_dim).bool()  # [B, T, A]
    cumulative_selected = torch.cumsum(actions_one_hot, dim=1)
    selections_before_t = torch.cat(
        (torch.zeros_like(cumulative_selected[:, :1, :]), cumulative_selected[:, :-1, :]),
        dim=1,
    )
    reconstructed_masks = (selections_before_t == 0) & initial_length_mask_expanded

    scores_per_step = policy_logits.unsqueeze(1).expand(-1, max_steps, -1)  # [B, T, A]
    masked_logits = scores_per_step.clone()
    masked_logits[~reconstructed_masks] = -1e9

    dist = Categorical(logits=masked_logits)
    log_probs_all = dist.log_prob(actions)  # [B, T]

    step_mask = (
        torch.arange(max_steps, device=device).expand(batch_size, max_steps)
        < episode_lengths.unsqueeze(1)
    )
    return log_probs_all[step_mask]  # [A]


class PPOTrainer(RLOOGumbelTrainer):
    """
    PPO trainer for KV cache eviction agents.

    Extends RLOOGumbelTrainer with:
    - A learned value network V(obs) for baseline subtraction.
    - PPO clipped surrogate objective (Schulman et al. 2017).
    - Multiple gradient update passes (ppo_epochs) over each collected batch.

    The environment produces a single terminal reward, so the return for every
    step within an episode equals that reward.  Advantages are computed as
    A = R - V(obs), giving a Monte-Carlo actor-critic update.

    The value network and the policy network are optimised with separate
    optimisers so their learning rates can be tuned independently.

    Usage in YAML (drop-in replacement for RLOOGumbelTrainer):

        trainer:
          _component_: kvcompression.rl.trainers.ppo_trainer.PPOTrainer
          clip_epsilon: 0.2
          ppo_epochs: 4
          value_coef: 0.5
          value_head_dim: 128    # head_dim of the agent (128 for Qwen2-1.5B)
          value_hidden_size: 128
          value_lr: 3.0e-4
          # ... all other RLOOGumbelTrainer args ...
    """

    def __init__(
        self,
        agent: BaseAgent,
        ema_agent: torch.nn.Module,
        env: BaseEnvironment,
        eval_env: BaseEnvironment,
        entropy_coef: float,
        max_grad_norm: float,
        normalize_advantages: bool,
        metrics_manager: MetricsManager,
        ema_burnin_steps: int,
        optimizer_config: Union[DictConfig, Dict[str, Any]],
        scheduler_config: Union[DictConfig, Dict[str, Any]],
        # PPO-specific
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 4,
        value_coef: float = 0.5,
        value_head_dim: int = 128,
        value_hidden_size: int = 128,
        value_lr: float = 3e-4,
        # Gumbel chunk size (inherited)
        gumbel_sampling_chunk_size: int = 16,
        device: Optional[Union[str, torch.device]] = None,
        wandb_config: Optional[Dict] = None,
        info_keys_to_log: Optional[List[str]] = None,
        info_keys_to_ignore: Optional[List[str]] = None,
    ):
        super().__init__(
            agent=agent,
            ema_agent=ema_agent,
            env=env,
            eval_env=eval_env,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            normalize_advantages=normalize_advantages,
            metrics_manager=metrics_manager,
            ema_burnin_steps=ema_burnin_steps,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            gumbel_sampling_chunk_size=gumbel_sampling_chunk_size,
            device=device,
            wandb_config=wandb_config,
            info_keys_to_log=info_keys_to_log,
            info_keys_to_ignore=info_keys_to_ignore,
        )

        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.value_coef = value_coef

        self.value_network = KVValueNetwork(
            head_dim=value_head_dim,
            hidden_size=value_hidden_size,
        ).to(self.device)

        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), lr=value_lr
        )

        pylogger.info(
            f"PPOTrainer | clip_epsilon={clip_epsilon} | ppo_epochs={ppo_epochs} "
            f"| value_coef={value_coef} | value_lr={value_lr}"
        )

    def _setup_metrics(self, metrics_manager: MetricsManager):
        super()._setup_metrics(metrics_manager)
        # Add PPO-specific metrics on top of RLOO metrics
        metrics_manager.add_metric_group(
            "PPO",
            [
                "ppo_policy_loss",
                "ppo_value_loss",
                "ppo_entropy_loss",
                "ppo_clip_fraction",
                "ppo_value_mean",
                "ppo_value_std",
                "ppo_approx_kl",
            ],
        )

    # ------------------------------------------------------------------
    # Trajectory collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_trajectories(self) -> Dict[str, Any]:
        """
        Collect trajectories using Gumbel-sort, then annotate with old
        log-probabilities and value estimates for the PPO update.
        """
        result = super().collect_trajectories()  # runs Gumbel sampling + reward

        obs = result["initial_observation"]
        actions = result["trajectory_data"]["actions"]
        episode_lengths = result["episode_lengths"]

        # Re-run agent (still no_grad) to get scores for log-prob computation
        scores = self.underlying_agent(obs)["scores"].to(torch.float32)
        if not torch.all(torch.isfinite(scores)):
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)

        batch_size, max_steps = actions.shape
        action_dim = self.env.action_dim
        seq_lengths = obs["seq_lengths"]

        old_log_probs = _compute_per_step_log_probs(
            device=self.device,
            batch_size=batch_size,
            action_dim=action_dim,
            max_steps=max_steps,
            seq_lengths=seq_lengths,
            episode_lengths=episode_lengths,
            actions=actions,
            policy_logits=scores,
        )

        values = self.value_network(obs)

        result["trajectory_data"]["old_log_probs"] = old_log_probs.detach()
        result["trajectory_data"]["values"] = values.detach()
        return result

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update_policy(
        self,
        initial_observation: Dict[str, torch.Tensor],
        trajectory_data: Dict[str, torch.Tensor],
        episode_lengths: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Run ppo_epochs gradient updates on the collected batch using the PPO
        clipped objective and an MSE value loss.

        Args:
            initial_observation: The initial env observation (K/V/Q tensors).
            trajectory_data: actions, rewards, old_log_probs, values.
            episode_lengths: Valid episode lengths [B].
            advantages: Per-step advantages [A] (may be None if batch too small).

        Returns:
            Dict of scalar loss metrics.
        """
        if advantages is None:
            warnings.warn("Skipping PPO update: advantages could not be computed.")
            return {
                "policy_loss": 0.0,
                "entropy_loss": 0.0,
                "total_loss": 0.0,
            }

        actions = trajectory_data["actions"]
        old_log_probs = trajectory_data["old_log_probs"]  # [A]
        old_values = trajectory_data["values"]             # [B]

        batch_size, max_steps = actions.shape
        action_dim = self.env.action_dim
        seq_lengths = initial_observation["seq_lengths"]
        rewards = trajectory_data["rewards"].to(torch.float32)  # [B]

        # Monte-Carlo returns: R_t = terminal reward (same for all steps)
        # Expand to per-episode (one return per episode)
        step_mask = (
            torch.arange(max_steps, device=self.device).expand(batch_size, max_steps)
            < episode_lengths.unsqueeze(1)
        )
        valid_ep_mask = episode_lengths > 0
        returns_per_episode = rewards[valid_ep_mask]               # [N]
        valid_lengths = episode_lengths[valid_ep_mask]             # [N]
        N = valid_ep_mask.sum().item()

        # Flatten returns to match the flattened advantages shape [A]
        # Each episode's steps get that episode's return as their target
        max_valid_len = valid_lengths.max().item()
        range_t = torch.arange(max_valid_len, device=self.device).expand(N, max_valid_len)
        flat_mask = range_t < valid_lengths.unsqueeze(1)
        returns_expanded = returns_per_episode.unsqueeze(1).expand(N, max_valid_len)
        flat_returns = returns_expanded[flat_mask]  # [A]

        # Accumulate loss metrics over PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_clip_frac = 0.0
        total_approx_kl = 0.0
        total_grad_norm = 0.0

        self.agent.train()
        self.value_network.train()

        for _ in range(self.ppo_epochs):
            # ---- policy forward pass ----
            new_scores = self.agent(initial_observation)["scores"].to(torch.float32)
            if not torch.all(torch.isfinite(new_scores)):
                new_scores = torch.nan_to_num(new_scores, nan=0.0, posinf=1e4, neginf=-1e4)

            new_log_probs = _compute_per_step_log_probs(
                device=self.device,
                batch_size=batch_size,
                action_dim=action_dim,
                max_steps=max_steps,
                seq_lengths=seq_lengths,
                episode_lengths=episode_lengths,
                actions=actions,
                policy_logits=new_scores,
            )  # [A]

            # ---- entropy ----
            # Compute entropy from the full per-step distribution (same reconstruction)
            initial_length_mask = (
                torch.arange(action_dim, device=self.device)
                .expand(batch_size, action_dim)
                < seq_lengths.unsqueeze(1)
            )
            initial_length_mask_expanded = initial_length_mask.unsqueeze(1).expand(
                -1, max_steps, -1
            )
            actions_one_hot = F.one_hot(actions, num_classes=action_dim).bool()
            cumulative_selected = torch.cumsum(actions_one_hot, dim=1)
            selections_before_t = torch.cat(
                (
                    torch.zeros_like(cumulative_selected[:, :1, :]),
                    cumulative_selected[:, :-1, :],
                ),
                dim=1,
            )
            reconstructed_masks = (selections_before_t == 0) & initial_length_mask_expanded
            scores_per_step = new_scores.unsqueeze(1).expand(-1, max_steps, -1)
            masked_logits = scores_per_step.clone()
            masked_logits[~reconstructed_masks] = -1e9
            dist = Categorical(logits=masked_logits)
            entropies_all = dist.entropy()  # [B, T]
            cat_entropies = entropies_all[step_mask]  # [A]

            # ---- PPO clipped surrogate ----
            log_ratio = new_log_probs - old_log_probs
            ratio = log_ratio.exp()
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            surr1 = ratio * advantages
            surr2 = clipped_ratio * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            clip_frac = ((ratio - 1).abs() > self.clip_epsilon).float().mean()
            approx_kl = ((ratio - 1) - log_ratio).mean()

            # ---- value loss ----
            new_values = self.value_network(initial_observation)   # [B]
            valid_new_values = new_values[valid_ep_mask]           # [N]
            valid_new_values_expanded = valid_new_values.unsqueeze(1).expand(
                N, max_valid_len
            )[flat_mask]  # [A]
            value_loss = F.mse_loss(valid_new_values_expanded, flat_returns)

            # ---- entropy loss ----
            entropy_loss = -cat_entropies.mean()

            # ---- total loss ----
            total_loss = (
                policy_loss
                + self.value_coef * value_loss
                + self.entropy_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.agent.parameters()) + list(self.value_network.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()
            self.value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_clip_frac += clip_frac.item()
            total_approx_kl += approx_kl.item()
            total_grad_norm += grad_norm.item()

        n = self.ppo_epochs
        ppo_metrics = {
            "ppo_policy_loss": total_policy_loss / n,
            "ppo_value_loss": total_value_loss / n,
            "ppo_entropy_loss": total_entropy_loss / n,
            "ppo_clip_fraction": total_clip_frac / n,
            "ppo_approx_kl": total_approx_kl / n,
            "ppo_value_mean": old_values.mean().item(),
            "ppo_value_std": old_values.std().item() if old_values.numel() > 1 else 0.0,
        }
        self.metrics_manager.add_batch(
            ppo_metrics,
            increment_step=False,
            distribute_average=True,
            distribute_local_nitems=1,
        )
        self.metrics_manager.add(
            "gradient_norm",
            total_grad_norm / n,
            distribute_average=True,
            distribute_local_nitems=1,
        )

        # Return the metrics in the format expected by the training loop.
        # Map PPO fields to names the base class logs: policy_loss, entropy_loss, total_loss.
        return {
            "policy_loss": total_policy_loss / n,
            "entropy_loss": total_entropy_loss / n,
            "total_loss": (total_policy_loss + self.value_coef * total_value_loss + self.entropy_coef * total_entropy_loss) / n,
            "gradient_norm": total_grad_norm / n,
        }

    # ------------------------------------------------------------------
    # Advantages: use value-based baseline instead of RLOO
    # ------------------------------------------------------------------

    def train_batch(self, step: int) -> Dict[str, float]:
        """
        Collect a batch, compute value-based advantages, run PPO update.

        Replaces RLOO advantage computation with V(obs)-subtracted returns so
        the value network is actually used.
        """
        self.agent.train()

        collection_result = self.collect_trajectories()

        initial_observation = collection_result["initial_observation"]
        trajectory_data = collection_result["trajectory_data"]
        episode_lengths = collection_result["episode_lengths"]

        rewards = trajectory_data["rewards"].to(torch.float32)
        old_values = trajectory_data["values"]  # [B]

        # Clip extreme rewards (same as RLOO trainer)
        clip_value = 200
        clamped_mask = rewards < -clip_value
        self.metrics_manager.add(
            "rewards_clamping_percentage",
            clamped_mask.float().mean().item() * 100,
            distribute_average=False,
            distribute_local_nitems=1,
        )
        torch.clamp_(rewards, min=-clip_value)

        # Compute per-episode advantages: A_i = R_i - V(obs_i)
        valid_mask = episode_lengths > 0
        if not valid_mask.any() or valid_mask.sum() <= 1:
            advantages = None
        else:
            per_episode_adv = rewards[valid_mask] - old_values[valid_mask]

            if self.normalize_advantages and per_episode_adv.numel() > 1:
                per_episode_adv = (per_episode_adv - per_episode_adv.mean()) / (
                    per_episode_adv.std() + 1e-8
                )

            # Clip advantages (same as RLOO)
            clip_adv = 2.0
            per_episode_adv = torch.clamp(per_episode_adv, -clip_adv, clip_adv)

            # Expand per-episode advantages to per-step for the loss function
            valid_lengths = episode_lengths[valid_mask]
            N = valid_mask.sum().item()
            max_valid_len = valid_lengths.max().item()
            range_t = torch.arange(max_valid_len, device=self.device).expand(N, max_valid_len)
            flat_mask = range_t < valid_lengths.unsqueeze(1)
            adv_expanded = per_episode_adv.unsqueeze(1).expand(N, max_valid_len)
            advantages = adv_expanded[flat_mask]  # [A]

        self.metrics_manager.add_batch(
            {
                "adv_before_norm_mean": rewards[valid_mask].mean().item() if valid_mask.any() else 0.0,
                "adv_before_norm_std": rewards[valid_mask].std().item() if valid_mask.sum() > 1 else 0.0,
            },
            distribute_average=False,
            distribute_local_nitems=1,
        )

        if advantages is not None:
            self.metrics_manager.add_batch(
                {
                    "adv_after_norm_mean": advantages.mean().item(),
                    "adv_after_norm_std": advantages.std().item() if advantages.numel() > 1 else 0.0,
                },
                distribute_average=False,
                distribute_local_nitems=1,
            )

        loss_metrics = self.update_policy(
            initial_observation,
            trajectory_data,
            episode_lengths,
            advantages,
        )

        self.scheduler.step()

        if step > self.ema_burnin_steps:
            self.ema_agent.update_parameters(get_underlying_model(self.agent))

        self.metrics_manager.add(
            "learning_rate",
            self.optimizer.param_groups[0]["lr"],
            distribute_average=True,
            distribute_local_nitems=1,
        )
        self.metrics_manager.add(
            "avg_advantage",
            advantages.mean().item() if advantages is not None else 0.0,
            distribute_average=False,
            distribute_local_nitems=1,
        )
        self.metrics_manager.add("current_step", step, distribute_average=False)
        self.metrics_manager.add(
            "scheduler_last_epoch", self.scheduler.last_epoch, distribute_average=False
        )
        self.metrics_manager.increment_step()

        return {
            "avg_reward": self.metrics_manager.get_latest("episode_reward", 0.0),
            **{key: self.metrics_manager.get_latest(key, 0.0) for key in self.info_keys_to_log},
            **loss_metrics,
        }
