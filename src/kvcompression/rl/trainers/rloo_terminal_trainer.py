#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import logging
import warnings
from typing import Any, Dict, List, Optional, Set, Union

import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Categorical
from torchtune import config
from tqdm import tqdm

from kvcompression.metrics import MetricsManager
from kvcompression.rl.agents.base_agent import BaseAgent
from kvcompression.rl.environments.base_env import BaseEnvironment
from kvcompression.rl.trainers.compute_advantages import (
    compute_returns_and_advantage_vec,
)
from kvcompression.utils.utils import (
    get_underlying_model,
)

pylogger = logging.getLogger(__name__)


class RLOOTerminalTrainer:
    """
    Generic Trainer using REINFORCE Leave-One-Out (RLOO) algorithm
    for environments with terminal rewards. Handles custom metrics
    from the environment's info dictionary.
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
        device: Optional[Union[str, torch.device]] = None,
        wandb_config: Optional[Dict] = None,
        info_keys_to_log: Optional[List[str]] = None,
        info_keys_to_ignore: Optional[List[str]] = None,
    ):
        """
        Initialize the RLOO trainer.

        Args:
            agent: The RL agent to train.
            ema_agent: Exponential moving average version of the agent for evaluation.
            env: Training environment.
            eval_env: Evaluation environment.
            entropy_coef: Coefficient for entropy regularization loss.
            max_grad_norm: Maximum gradient norm for clipping.
            normalize_advantages: Whether to normalize advantages.
            metrics_manager: Manager for logging metrics.
            ema_burnin_steps: Steps before starting EMA updates.
            optimizer_config: Configuration for the optimizer.
            scheduler_config: Configuration for the learning rate scheduler.
            device: Device to use for training.
            wandb_config: Optional W&B configuration.
            info_keys_to_log: Environment info keys to log.
            info_keys_to_ignore: Environment info keys to ignore.
        """
        self.normalize_advantages = normalize_advantages

        self.env = env
        self.eval_env = eval_env

        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Convert to DictConfig if needed for torchtune instantiate compatibility

        if not isinstance(optimizer_config, DictConfig):
            optimizer_config = OmegaConf.create(optimizer_config)
        if not isinstance(scheduler_config, DictConfig):
            scheduler_config = OmegaConf.create(scheduler_config)

        self.optimizer = config.instantiate(optimizer_config, params=agent.parameters())
        self.scheduler = config.instantiate(scheduler_config, optimizer=self.optimizer)

        self.agent = agent
        self.ema_agent = ema_agent
        self.ema_burnin_steps = ema_burnin_steps

        self.underlying_agent = get_underlying_model(agent)

        self.info_keys_to_log: Set[str] = (
            set(info_keys_to_log) if info_keys_to_log else set()
        )
        self.info_keys_to_ignore: Set[str] = (
            set(info_keys_to_ignore) if info_keys_to_ignore else set()
        )

        self.info_keys_to_log = self.info_keys_to_log - self.info_keys_to_ignore

        self.metrics_manager = metrics_manager
        self._setup_metrics(metrics_manager=self.metrics_manager)

        if self.device.type == "cuda":
            from kvcompression.rl.trainers.triton_policy_entropy_loss import (
                compute_policy_entropy_loss_triton,
            )

            self.loss_fn = compute_policy_entropy_loss_triton
        else:
            from kvcompression.rl.trainers.policy_entropy_loss import (
                compute_policy_entropy_loss,
            )

            self.loss_fn = compute_policy_entropy_loss

        agent_param_count = sum(p.numel() for p in self.agent.parameters())

        hparams = {
            "entropy_coef": entropy_coef,
            "max_grad_norm": max_grad_norm,
            "logged_info_keys": sorted(list(self.info_keys_to_log)),
            **(wandb_config or {}),
        }
        self.metrics_manager.log_hparams(hparams)

        self.metrics_manager.add(
            "agent_param_count",
            agent_param_count,
            distribute_average=False,
        )

    def _setup_metrics(self, metrics_manager: MetricsManager):
        """Define metric groups and dynamically add info keys."""
        metrics_manager.add_metric_group(
            "Losses", ["policy_loss", "entropy_loss", "total_loss"]
        )
        metrics_manager.add_metric_group("Model", ["agent_param_count"])
        training_metrics = [
            "episode_reward",
            "avg_advantage",
            "episode_length",
            "avg_entropy",
            "max_agent_score",
            "learning_rate",
            "gradient_norm",
            "current_step",
            "scheduler_last_epoch",
        ]
        training_metrics.extend(sorted(list(self.info_keys_to_log)))
        metrics_manager.add_metric_group("Training", training_metrics)

        evaluation_metrics = [
            "eval_reward",
            "eval_length",
            "eval_at_step",
        ]
        evaluation_metrics.extend(
            [f"eval_{key}" for key in sorted(list(self.info_keys_to_log))]
        )
        metrics_manager.add_metric_group("Evaluation", evaluation_metrics)

    @torch.no_grad()
    def collect_trajectories(self) -> Dict[str, Any]:
        """Collect trajectories by interacting with the environment."""
        self.agent.eval()

        initial_obs = self.env.reset()
        obs = initial_obs

        # Get max steps from observation and determine action space size
        # Use env attribute if available, otherwise infer from obs
        max_possible_steps = getattr(
            self.env, "max_steps", obs["seq_lengths"].max().item()
        )
        batch_size = obs["seq_lengths"].numel()

        trajectory_data = {
            "actions": torch.zeros(
                (batch_size, max_possible_steps),
                dtype=torch.long,
                device=obs["seq_lengths"].device,
            ),
            "rewards": torch.zeros(
                (batch_size,),
                dtype=torch.float32,
                device=obs["seq_lengths"].device,
            ),
        }

        info_data = {
            k: torch.zeros(
                (batch_size,),
                dtype=v.dtype,
                device=v.device,
            )
            for k, v in obs["info"].items()
        }

        device = obs["seq_lengths"].device
        active_episodes = torch.ones(batch_size, dtype=torch.bool, device=device)
        episode_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        current_timestep = torch.zeros(batch_size, dtype=torch.long, device=device)
        batch_idxs = torch.arange(batch_size, device=device)

        precomputed_scores = self.agent(obs)["scores"]

        # Sanitize precomputed scores to prevent NaN rewards
        if not torch.all(torch.isfinite(precomputed_scores)):
            precomputed_scores = torch.nan_to_num(
                precomputed_scores, nan=0.0, posinf=1e4, neginf=-1e4
            )

        max_scores = precomputed_scores.max(dim=-1).values

        # Masks out scores for tokens after the prompt
        valid_actions = self.env.get_valid_actions()

        action_mask = self.underlying_agent.create_action_mask(obs, valid_actions)

        actions_this_step = torch.zeros(batch_size, dtype=torch.long, device=device)

        for step in range(max_possible_steps):
            if not active_episodes.any():
                break

            active_indices = torch.where(active_episodes)[0]

            active_scores = precomputed_scores[active_indices]
            active_action_mask = action_mask[active_indices]
            active_scores[~active_action_mask] = -1e9

            probs = torch.softmax(active_scores.to(torch.float32), dim=-1)
            dist = Categorical(probs=probs)
            sampled_actions = dist.sample()

            actions_this_step.zero_()
            actions_this_step[active_indices] = sampled_actions

            next_obs = self.env.step(actions_this_step)

            active_timestep = current_timestep[active_indices]

            trajectory_data["actions"][active_indices, active_timestep] = (
                sampled_actions
            )

            current_timestep[active_indices] += 1

            # Handle finished episodes
            dones = next_obs["dones"]  # Shape: [B]
            newly_finished_mask = (
                active_episodes & dones
            )  # Mask of episodes that finished this step

            if newly_finished_mask.any():
                finished_indices = torch.where(newly_finished_mask)[0]
                episode_lengths[finished_indices] = current_timestep[finished_indices]
                active_episodes[finished_indices] = False

                env_rewards = next_obs["rewards"][finished_indices]

                if not torch.all(torch.isfinite(env_rewards)):
                    # Sanitize environment rewards as last resort
                    env_rewards = torch.nan_to_num(
                        env_rewards, nan=0.0, posinf=0.0, neginf=0.0
                    )

                trajectory_data["rewards"][finished_indices] = env_rewards

                for k, v in next_obs["info"].items():
                    info_data[k][finished_indices] = v[finished_indices].squeeze()

            obs = next_obs

            # Prevent selecting the same action again
            action_mask[batch_idxs, actions_this_step] = False

        # For episodes that never finished, set length to their current timestep
        unfinished = episode_lengths == 0
        episode_lengths[unfinished] = current_timestep[unfinished]

        valid_len_mask = episode_lengths > 0

        if valid_len_mask.any():
            num_valid = valid_len_mask.sum().item()

            self.metrics_manager.add(
                "episode_reward",
                trajectory_data["rewards"][valid_len_mask].mean().item(),
                distribute_average=False,
                distribute_local_nitems=num_valid,
            )
            self.metrics_manager.add(
                "episode_length",
                episode_lengths[valid_len_mask].float().mean().item(),
                distribute_average=False,
                distribute_local_nitems=num_valid,
            )
            self.metrics_manager.add(
                "max_agent_score",
                max_scores[valid_len_mask].float().mean().item(),
                distribute_average=False,
                distribute_local_nitems=num_valid,
            )

            for k, v in info_data.items():
                self.metrics_manager.add(
                    k,
                    v[valid_len_mask].float().mean().item(),
                    distribute_average=False,
                )
        else:
            self.metrics_manager.add("episode_reward", 0.0)
            self.metrics_manager.add("episode_length", 0.0)
            self.metrics_manager.add("max_agent_score", 0.0)

            for k, v in info_data.items():
                self.metrics_manager.add(k, 0.0)

        return {
            "initial_observation": initial_obs,
            "trajectory_data": trajectory_data,
            "episode_lengths": episode_lengths,
        }

    def update_policy(
        self,
        initial_observation: Dict[str, torch.Tensor],
        trajectory_data: Dict[str, torch.Tensor],
        episode_lengths: torch.Tensor,
        advantages: torch.Tensor,  # Pass advantages directly
    ) -> Dict[str, float]:
        """Update policy using processed trajectory data."""

        # If no advantages could be computed (e.g., batch size <= 1), skip update
        if advantages is None:
            warnings.warn(
                "Skipping policy update because advantages could not be computed (batch size <= 1 or no valid episodes)."
            )
            return {
                "policy_loss": 0.0,
                "entropy_loss": 0.0,
                "total_loss": 0.0,
            }

        actions = trajectory_data["actions"]
        batch_size, max_steps = actions.shape
        action_dim = self.env.action_dim
        seq_lengths = initial_observation["seq_lengths"]

        # Perform one forward pass with gradients enabled
        self.agent.train()
        scores_batch_grad = self.agent(initial_observation)["scores"].to(torch.float32)

        # Sanitize the logits to prevent CUDA assertions from NaN/inf values.
        if not torch.all(torch.isfinite(scores_batch_grad)):
            scores_batch_grad = torch.nan_to_num(
                scores_batch_grad, nan=0.0, posinf=1e4, neginf=-1e4
            )

        total_loss, entropy_loss, policy_loss, avg_entropy = self.loss_fn(
            device=self.device,
            entropy_coef=self.entropy_coef,
            batch_size=batch_size,
            action_dim=action_dim,
            max_steps=max_steps,
            seq_lengths=seq_lengths,
            episode_lengths=episode_lengths,
            actions=actions,
            policy_logits=scores_batch_grad,
            advantages=advantages,
        )

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        loss_metrics = {
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "gradient_norm": grad_norm.item(),
        }
        self.metrics_manager.add_batch(
            loss_metrics,
            increment_step=False,
            distribute_average=True,
            distribute_local_nitems=1,
        )
        self.metrics_manager.add(
            "avg_entropy",
            avg_entropy.item(),
            distribute_average=True,
            distribute_local_nitems=1,
        )

        return loss_metrics

    def train_batch(
        self,
        step: int,
    ) -> Dict[str, float]:
        """Perform one training batch: collect trajectories, compute advantage, update policy."""
        self.agent.train()

        collection_result = self.collect_trajectories()

        initial_observation = collection_result["initial_observation"]
        trajectory_data = collection_result["trajectory_data"]
        episode_lengths = collection_result["episode_lengths"]

        rewards = trajectory_data["rewards"].to(torch.float32)

        clip_value = 200
        clamped_mask = rewards < -clip_value
        clamping_percentage = clamped_mask.float().mean().item() * 100
        self.metrics_manager.add(
            "rewards_clamping_percentage",
            clamping_percentage,
            distribute_average=False,
            distribute_local_nitems=1,
        )
        torch.clamp_(rewards, min=-clip_value)

        # Log stats of advantages *before* normalization
        if rewards.numel() > 1:
            adv_before_norm_mean = rewards.mean().item()
            adv_before_norm_std = rewards.std().item()
            self.metrics_manager.add_batch(
                {
                    "adv_before_norm_mean": adv_before_norm_mean,
                    "adv_before_norm_std": adv_before_norm_std,
                },
                distribute_average=False,
                distribute_local_nitems=1,
            )

        advantages = compute_returns_and_advantage_vec(
            rewards,
            episode_lengths,
            normalize_advantages=self.normalize_advantages,
        )

        if advantages is not None:
            # Log stats of advantages *after* normalization
            self.metrics_manager.add_batch(
                {
                    "adv_after_norm_mean": advantages.mean().item(),
                    "adv_after_norm_std": advantages.std().item(),
                },
                distribute_average=False,
                distribute_local_nitems=1,
            )

            clip_value = 2.0

            clamped_mask = (advantages < -clip_value) | (advantages > clip_value)
            clamping_percentage = clamped_mask.float().mean().item() * 100
            self.metrics_manager.add(
                "adv_clamping_percentage",
                clamping_percentage,
                distribute_average=False,
                distribute_local_nitems=1,
            )

            torch.clamp_(advantages, min=-clip_value, max=clip_value)

        loss_metrics = self.update_policy(
            initial_observation,
            trajectory_data,
            episode_lengths,
            advantages,
        )

        self.scheduler.step()

        if step > self.ema_burnin_steps:
            self.ema_agent.update_parameters(self.underlying_agent)

        self.metrics_manager.add(
            "learning_rate",
            self.optimizer.param_groups[0]["lr"],
            distribute_average=True,
            distribute_local_nitems=1,
        )

        if advantages is not None:
            self.metrics_manager.add(
                "avg_advantage",
                advantages.mean().item(),
                distribute_average=False,
                distribute_local_nitems=1,
            )
        else:
            self.metrics_manager.add("avg_advantage", 0.0)

        self.metrics_manager.add("current_step", step, distribute_average=False)
        self.metrics_manager.add(
            "scheduler_last_epoch", self.scheduler.last_epoch, distribute_average=False
        )

        self.metrics_manager.increment_step()

        # Return latest logged metrics for convenience (e.g., for progress bars)
        latest_metrics = {
            "avg_reward": self.metrics_manager.get_latest("episode_reward", 0.0),
            **{
                key: self.metrics_manager.get_latest(key, 0.0)
                for key in self.info_keys_to_log
            },
            **loss_metrics,
        }

        return latest_metrics

    @torch.no_grad()
    def evaluate(
        self, num_episodes: Optional[int] = None, metric_prefix: str = "eval"
    ) -> Dict[str, float]:
        """Evaluate agent performance in batches, using vectorized info processing."""
        self.ema_agent.eval()

        total_rewards_sum = 0.0
        total_lengths_sum = 0
        info_metric_sums = {key: 0.0 for key in self.info_keys_to_log}
        total_episodes_processed = 0

        if hasattr(self.eval_env, "reset_iterator"):
            self.eval_env.reset_iterator()
        else:
            warnings.warn(
                "Evaluation environment does not have 'reset_iterator' method. Evaluation might fail after first run."
            )

        try:
            with tqdm(
                desc="Evaluation",
                unit="batch",
                total=num_episodes
                if num_episodes is not None
                else self.eval_env.length(),
            ) as pbar:
                while True:
                    if (
                        num_episodes is not None
                        and total_episodes_processed >= num_episodes
                    ):
                        break

                    # Batch processing (mirrors collect_trajectories structure)
                    obs = self.eval_env.reset(raise_stop_iteration=True)

                    current_batch_size = self.eval_env.current_batch_size

                    max_possible_steps = getattr(
                        self.eval_env, "max_steps", obs["seq_lengths"].max().item()
                    )

                    device = obs["seq_lengths"].device
                    active_episodes_batch = torch.ones(
                        current_batch_size, dtype=torch.bool, device=device
                    )
                    episode_lengths_batch = torch.zeros(
                        current_batch_size, dtype=torch.long, device=device
                    )
                    current_timestep_batch = torch.zeros(
                        current_batch_size, dtype=torch.long, device=device
                    )
                    # Need rewards per step temporarily to get final reward
                    rewards_batch_steps = torch.zeros(
                        (current_batch_size,),
                        dtype=torch.float32,
                        device=device,
                    )
                    info_data = {
                        k: torch.zeros(
                            (current_batch_size,),
                            dtype=v.dtype,
                            device=v.device,
                        )
                        for k, v in obs["info"].items()
                    }

                    actions_this_step = torch.zeros(
                        current_batch_size, dtype=torch.long, device=device
                    )

                    for step in range(max_possible_steps):
                        if not active_episodes_batch.any():
                            break
                        active_indices = torch.where(active_episodes_batch)[0]

                        active_obs = {
                            k: v[active_indices]
                            if isinstance(v, torch.Tensor)
                            and v.shape[0] == current_batch_size
                            else v
                            for k, v in obs.items()
                        }

                        actions_active = self.ema_agent.module.act(
                            active_obs, sample=False
                        )
                        actions_this_step.zero_()
                        actions_this_step[active_indices] = actions_active

                        next_obs = self.eval_env.step(actions_this_step)

                        current_timestep_batch[active_indices] += 1
                        dones = next_obs["dones"]
                        newly_finished_mask = active_episodes_batch & dones

                        if newly_finished_mask.any():
                            finished_indices = torch.where(newly_finished_mask)[0]
                            episode_lengths_batch[finished_indices] = (
                                current_timestep_batch[finished_indices]
                            )
                            active_episodes_batch[finished_indices] = False
                            rewards_batch_steps[finished_indices] = next_obs["rewards"][
                                finished_indices
                            ]
                            for k, v in next_obs["info"].items():
                                info_data[k][finished_indices] = v[
                                    finished_indices
                                ].squeeze()

                        obs = next_obs

                    # Set lengths for unfinished episodes in this batch
                    unfinished_mask_batch = (
                        episode_lengths_batch == 0
                    ) | active_episodes_batch
                    if unfinished_mask_batch.any():
                        episode_lengths_batch[unfinished_mask_batch] = (
                            max_possible_steps
                        )

                    valid_len_mask_batch = episode_lengths_batch > 0
                    valid_len_indices_batch = torch.where(valid_len_mask_batch)[0]
                    num_valid_in_batch = len(valid_len_indices_batch)

                    if num_valid_in_batch > 0:
                        for k, v in next_obs["info"].items():
                            info_metric_sums[k] += info_data[k].float().sum().item()

                        total_rewards_sum += (
                            rewards_batch_steps[valid_len_mask_batch].sum().item()
                        )
                        total_lengths_sum += (
                            episode_lengths_batch[valid_len_mask_batch].sum().item()
                        )
                        total_episodes_processed += num_valid_in_batch

                    pbar.update(1)

        except StopIteration:
            pass

        logged_metrics = [
            f"{metric_prefix}_reward",
            f"{metric_prefix}_length",
            f"{metric_prefix}_at_step",
        ]

        self.metrics_manager.add_batch(
            {
                f"{metric_prefix}_reward": total_rewards_sum,
                f"{metric_prefix}_length": total_lengths_sum,
            },
            increment_step=False,
            distribute_average=True,
            distribute_local_nitems=total_episodes_processed,
        )

        for key in self.info_keys_to_log:
            total_sum = info_metric_sums[key]
            logged_metrics.append(f"{metric_prefix}_{key}")

            self.metrics_manager.add(
                f"{metric_prefix}_{key}",
                total_sum / total_episodes_processed
                if total_episodes_processed > 0
                else 0.0,
                distribute_average=False,
            )

        if self.metrics_manager.is_rank_zero:
            self.metrics_manager.add(
                f"{metric_prefix}_at_step",
                self.metrics_manager.step_counter,
                distribute_average=False,
            )

        return logged_metrics

    def plot_training_curves(self, smoothing_window=50):
        fig, _ = self.metrics_manager.plot_metric_grid(
            metrics_to_plot=[
                "episode_reward",
                "episode_length",
            ],
            window_size=smoothing_window,
            title="Training curves",
            log_on_wandb=True,
        )
        return fig

    def plot_loss_curves(self, smoothing_window=50):
        fig, _ = self.metrics_manager.plot_metric_grid(
            metrics_to_plot=[
                "policy_loss",
                "entropy_loss",
                "total_loss",
            ],
            window_size=smoothing_window,
            title="Loss curves",
            log_on_wandb=True,
        )

        return fig

    def plot_eval_metrics(self):
        fig, _ = self.metrics_manager.plot_metric_grid(
            metrics_to_plot=[
                "eval_reward",
                "eval_length",
            ],
            log_on_wandb=True,
            title="Evaluation Curves",
            marker="o",
            linestyle="-",
        )
        return fig
