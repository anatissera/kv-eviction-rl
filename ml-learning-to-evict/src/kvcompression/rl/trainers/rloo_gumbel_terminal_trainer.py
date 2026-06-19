#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import logging
import warnings
from typing import Any, Dict

import torch
from tqdm import tqdm

from kvcompression.rl.trainers.rloo_terminal_trainer import RLOOTerminalTrainer

pylogger = logging.getLogger(__name__)


class RLOOGumbelTrainer(RLOOTerminalTrainer):
    """
    This "hybrid" version chunks the entire pipeline (Gumbel-Sort and reward
    computation) to provide full control over peak memory usage. It uses a
    chunk-aware environment method for reward calculation.
    """

    def __init__(self, *args, gumbel_sampling_chunk_size: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.gumbel_sampling_chunk_size = gumbel_sampling_chunk_size
        pylogger.info(
            f"RLOOGumbelTrainer initialized with chunk size: {self.gumbel_sampling_chunk_size}"
        )

    @torch.no_grad()
    def collect_trajectories(self) -> Dict[str, Any]:
        self.agent.eval()

        initial_obs = self.env.reset()

        seq_lengths = initial_obs["seq_lengths"]
        batch_size = seq_lengths.shape[0]
        max_possible_steps = getattr(self.env, "max_steps", seq_lengths.max().item())
        device = seq_lengths.device

        precomputed_scores = self.agent(initial_obs)["scores"].to(torch.float32)
        if not torch.all(torch.isfinite(precomputed_scores)):
            precomputed_scores = torch.nan_to_num(
                precomputed_scores, nan=0.0, posinf=1e4, neginf=-1e4
            )

        valid_scores_mask = torch.isfinite(precomputed_scores)
        if valid_scores_mask.any():
            valid_scores = precomputed_scores[valid_scores_mask]
            self.metrics_manager.add_batch(
                {
                    "agent_score_mean": valid_scores.mean().item(),
                    "agent_score_std": valid_scores.std().item(),
                    "agent_score_min": valid_scores.min().item(),
                    "agent_score_max": valid_scores.max().item(),
                },
                distribute_average=False,
                distribute_local_nitems=1,
            )

        info_keys_to_collect = list(self.info_keys_to_log)
        actions = torch.empty(
            (batch_size, max_possible_steps), dtype=torch.long, device=device
        )
        rewards = torch.empty(batch_size, dtype=torch.float32, device=device)
        all_info = {key: [] for key in set(info_keys_to_collect)}

        chunk_size = self.gumbel_sampling_chunk_size
        max_steps_arange = torch.arange(max_possible_steps, device=device)[None, :]

        for i in range(0, batch_size, chunk_size):
            start_idx, end_idx = i, min(i + chunk_size, batch_size)

            scores_chunk = precomputed_scores[start_idx:end_idx]

            # Cast to float32 for numerical stability
            scores_chunk_fp32 = scores_chunk.to(torch.float32)

            # Gumbel noise generation
            finfo = torch.finfo(scores_chunk_fp32.dtype)
            noise = torch.rand_like(scores_chunk_fp32).clamp_(
                min=finfo.eps, max=1.0 - finfo.eps
            )
            gumbel_noise = -torch.log(-torch.log(noise))

            perturbed_scores_fp32 = scores_chunk_fp32 + gumbel_noise

            padding_mask_chunk = max_steps_arange < seq_lengths[start_idx:end_idx, None]
            perturbed_scores_fp32[~padding_mask_chunk] = -1e9

            # Gumbel-Sort
            actions_chunk = torch.argsort(
                perturbed_scores_fp32, dim=-1, descending=True
            )

            # Reward calculation using the chunk-aware method
            batch_indices_chunk = torch.arange(start_idx, end_idx, device=device)

            terminal_results_chunk = self.env.compute_rewards_for_chunk(
                actions_chunk=actions_chunk,
                batch_indices_chunk=batch_indices_chunk,
            )

            actions[start_idx:end_idx] = actions_chunk
            rewards[start_idx:end_idx] = terminal_results_chunk.pop("rewards")
            for key, val in terminal_results_chunk.items():
                if key in all_info:
                    all_info[key].append(val)

        info_data = {key: torch.cat(val, dim=0) for key, val in all_info.items() if val}

        if not torch.all(torch.isfinite(rewards)):
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        episode_lengths = seq_lengths
        valid_len_mask = episode_lengths > 0

        if valid_len_mask.any():
            num_valid = valid_len_mask.sum().item()
            valid_rewards = rewards[valid_len_mask]

            self.metrics_manager.add_batch(
                {
                    "reward_mean": valid_rewards.mean().item(),
                    "reward_std": valid_rewards.std().item(),
                    "reward_min": valid_rewards.min().item(),
                    "reward_max": valid_rewards.max().item(),
                },
                distribute_average=False,
                distribute_local_nitems=num_valid,
            )

            self.metrics_manager.add(
                "episode_reward",
                valid_rewards.mean().item(),
                distribute_average=False,
                distribute_local_nitems=num_valid,
            )
            self.metrics_manager.add(
                "episode_length",
                episode_lengths[valid_len_mask].float().mean().item(),
                distribute_average=False,
                distribute_local_nitems=num_valid,
            )
            for k, v in info_data.items():
                if k in self.info_keys_to_log:
                    self.metrics_manager.add(
                        k,
                        v[valid_len_mask].float().mean().item(),
                        distribute_average=False,
                    )

        else:
            self.metrics_manager.add("episode_reward", 0.0)
            self.metrics_manager.add("episode_length", 0.0)
            for k in self.info_keys_to_log:
                self.metrics_manager.add(k, 0.0)

        trajectory_data = {"actions": actions, "rewards": rewards}
        return {
            "initial_observation": initial_obs,
            "trajectory_data": trajectory_data,
            "episode_lengths": episode_lengths,
        }

    @torch.no_grad()
    def evaluate(
        self, num_episodes: int | None = None, metric_prefix: str = "eval"
    ) -> list[str]:
        """
        Evaluate agent performance using a vectorized, one-shot trajectory generation.
        This avoids the slow, iterative stepping and matches the optimization style
        of the `collect_trajectories` method.
        """
        self.ema_agent.eval()

        # Initialize accumulators
        total_rewards_sum = 0.0
        total_lengths_sum = 0.0
        info_metric_sums = {key: 0.0 for key in self.info_keys_to_log}
        total_episodes_processed = 0

        if hasattr(self.eval_env, "reset_iterator"):
            self.eval_env.reset_iterator()
        else:
            warnings.warn(
                "Evaluation environment does not have 'reset_iterator' method. "
                "Evaluation might fail after first run."
            )

        try:
            pbar_total = (
                num_episodes if num_episodes is not None else self.eval_env.length()
            )
            with tqdm(desc="Evaluation", unit="batch", total=pbar_total) as pbar:
                while True:
                    if (
                        num_episodes is not None
                        and total_episodes_processed >= num_episodes
                    ):
                        break

                    obs = self.eval_env.reset(raise_stop_iteration=True)
                    current_batch_size = self.eval_env.current_batch_size
                    seq_lengths = obs["seq_lengths"]
                    device = seq_lengths.device
                    max_possible_steps = getattr(
                        self.eval_env, "max_steps", seq_lengths.max().item()
                    )

                    scores = self.ema_agent.module(obs)["scores"]

                    max_steps_arange = torch.arange(max_possible_steps, device=device)[
                        None, :
                    ]
                    padding_mask = max_steps_arange < seq_lengths[:, None]
                    scores[~padding_mask] = -torch.inf

                    # Get deterministic action sequences via argsort (no Gumbel noise)
                    actions = torch.argsort(scores, dim=-1, descending=True)

                    chunk_size = self.gumbel_sampling_chunk_size
                    batch_rewards = []
                    batch_info = {key: [] for key in self.info_keys_to_log}

                    for i in range(0, current_batch_size, chunk_size):
                        start_idx, end_idx = i, min(i + chunk_size, current_batch_size)
                        actions_chunk = actions[start_idx:end_idx]
                        batch_indices_chunk = torch.arange(
                            start_idx, end_idx, device=device
                        )

                        # This assumes eval_env also has this method
                        terminal_results_chunk = (
                            self.eval_env.compute_rewards_for_chunk(
                                actions_chunk=actions_chunk,
                                batch_indices_chunk=batch_indices_chunk,
                            )
                        )

                        batch_rewards.append(terminal_results_chunk.pop("rewards"))
                        for key, val in terminal_results_chunk.items():
                            if key in batch_info:
                                batch_info[key].append(val)

                    rewards_tensor = torch.cat(batch_rewards, dim=0)
                    info_data = {
                        key: torch.cat(val, dim=0)
                        for key, val in batch_info.items()
                        if val
                    }

                    total_rewards_sum += rewards_tensor.sum().item()
                    total_lengths_sum += seq_lengths.sum().item()
                    total_episodes_processed += current_batch_size
                    for key, val_tensor in info_data.items():
                        info_metric_sums[key] += val_tensor.sum().item()

                    pbar.update(current_batch_size)

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
