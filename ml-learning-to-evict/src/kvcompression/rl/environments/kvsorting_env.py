#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torchtune
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torchtune import config
from torchtune.utils import get_logger

from kvcompression.rl.rewards.base_reward_strategy import RewardStrategy
from kvcompression.rl.rewards.basic_strategies import FutureAttentionAucNormalizedReward
from kvcompression.rl.rewards.oracle import Oracle
from kvcompression.utils.utils import batch_to_device

pylogger = get_logger("INFO")


class KVSortingEnvironmentV2:
    """
    Environment for learning to sort KV sequences by iteratively selecting elements.
    """

    def __init__(
        self,
        oracle: Oracle,
        dataloader: Dataset,
        device: Union[str, torch.device],
        reward_strategy: Union[str, RewardStrategy],
        info_keys_to_log: Optional[List[str]] = None,
    ):
        # Handle reward strategy - support both string (legacy) and component (new)
        if isinstance(reward_strategy, str):
            # Legacy string-based reward strategy - convert to component
            self.reward_strategy = self._create_legacy_reward_strategy(reward_strategy)
            pylogger.warning(
                f"String-based reward strategy '{reward_strategy}' is deprecated. "
                "Please update your config to use reward strategy components."
            )
        else:
            # New component-based reward strategy
            if not hasattr(reward_strategy, "compute_reward"):
                # Instantiate from config if needed
                reward_strategy = config.instantiate(OmegaConf.create(reward_strategy))
            self.reward_strategy = reward_strategy

        pylogger.info(f"Reward strategy in use: {type(self.reward_strategy).__name__}")

        self.oracle = oracle
        self.device = torch.device(device)
        self._dtype = torch.get_default_dtype()

        self.info_keys_to_log = info_keys_to_log or []

        self._info_key_map = {}

        self._info_key_dtypes = {}

        self.dataloader = dataloader
        self.dataloader_iter = iter(self.dataloader)

        self.current_batch: Optional[Dict[str, torch.Tensor]] = None
        self.prompt_len: Optional[torch.Tensor] = None
        self.max_prompt_len_in_batch: Optional[int] = None
        self.action_dim: Optional[int] = None
        self.max_entire_sequence_length: Optional[int] = None
        self.current_batch_size: int = 0
        self.selection_mask: Optional[torch.Tensor] = None
        self.output_kv_indices: Optional[torch.Tensor] = None
        self.step_count: Optional[torch.Tensor] = None

    def _create_legacy_reward_strategy(self, strategy_name: str) -> RewardStrategy:
        """Create reward strategy component from legacy string name."""
        strategy_map = {
            "future_attention_auc_normalized": FutureAttentionAucNormalizedReward,
        }

        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown legacy reward strategy: {strategy_name}")

        return strategy_map[strategy_name]()

    def _initialize_info_dict(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Creates a placeholder info dictionary with default values for all configured keys."""
        info = {}
        for log_key in self.info_keys_to_log:
            dtype = self._info_key_dtypes.get(log_key, self._dtype)
            init_value = -1 if not dtype.is_floating_point else float("nan")

            info[log_key] = torch.full(
                (batch_size,),
                init_value,
                device=self.device,
                dtype=dtype,
            )
        return info

    def _extract_info_from_oracle_result(
        self, cost_result, num_samples
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts configured metrics from the oracle's cost_result object.
        """
        info = {}

        for log_key in self.info_keys_to_log:
            # Use the map to find the internal attribute name
            attr_name = self._info_key_map.get(log_key, log_key)

            if (
                hasattr(cost_result, attr_name)
                and (value := getattr(cost_result, attr_name)) is not None
            ):
                info[log_key] = value.view(num_samples)
            else:
                dtype = self._info_key_dtypes.get(log_key, self._dtype)
                init_value = -1 if not dtype.is_floating_point else float("nan")
                info[log_key] = torch.full(
                    (num_samples,),
                    init_value,
                    device=self.device,
                    dtype=dtype,
                )
        return info

    def _compute_reward(self, cost_result) -> torch.Tensor:
        """Compute reward using the configured reward strategy."""
        return self.reward_strategy.compute_reward(cost_result)

    def length(self) -> Optional[int]:
        return len(self.dataloader)

    def recreate_iterator(self):
        """
        Recreates the dataloader iterator.

        With InfiniteDistributedSampler, this is only needed for resume,
        not for regular training steps.
        """
        torchtune.utils.log_rank_zero(
            pylogger,
            "Recreating DataLoader iterator (for resume only).",
            level=logging.DEBUG,
        )
        try:
            self.dataloader_iter = iter(self.dataloader)
        except Exception as e:
            pylogger.error(
                f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Error creating DataLoader iterator: {e}"
            )
            raise e

    def _get_next_batch(self, raise_stop_iteration: bool):
        """Retrieves the next batch from the infinite dataloader."""
        try:
            # With InfiniteDistributedSampler, this should never raise StopIteration
            batch = next(self.dataloader_iter)
        except StopIteration as e:
            # This should not happen with InfiniteDistributedSampler
            if raise_stop_iteration:
                raise e

            # Fallback: recreate iterator and try again
            torchtune.utils.log_rank_zero(
                pylogger,
                "Unexpected StopIteration with InfiniteDistributedSampler. Recreating iterator.",
                level=logging.WARNING,
            )
            self.recreate_iterator()
            batch = next(self.dataloader_iter)
        except Exception as e:
            pylogger.error(f"Error getting batch from dataloader: {e}")
            raise e

        batch_to_device(batch, self.device, non_blocking=True)
        return batch

    def reset_iterator(self):
        """Resets the internal dataloader iterator to start from the beginning."""
        pylogger.info("Resetting evaluation dataloader iterator.")
        # local fix (not in Apple's release): upstream assigned self.iterator here,
        # a dead attribute, so the eval dataloader never actually reset between evals
        self.dataloader_iter = iter(self.dataloader)

    def reset(self, raise_stop_iteration: bool = False) -> Dict[str, torch.Tensor]:
        """Resets the environment with the next batch."""
        self.current_batch = self._get_next_batch(
            raise_stop_iteration=raise_stop_iteration
        )

        required_keys = ["prompt_len", "keys", "values", "queries"]
        for key in required_keys:
            if key not in self.current_batch or self.current_batch[key] is None:
                raise ValueError(
                    f"Batch missing required key '{key}' or value is None after collation."
                )

        self.prompt_len = self.current_batch["prompt_len"]
        self.current_batch_size = self.prompt_len.numel()

        # Determine max_len from actual tensor shapes *after collation/padding*
        self.max_prompt_len_in_batch = self.prompt_len.amax()
        self.action_dim = self.max_prompt_len_in_batch

        self.max_entire_sequence_length = self.current_batch["keys"].shape[-2]
        if self.max_prompt_len_in_batch >= self.max_entire_sequence_length:
            raise RuntimeError(
                "The is nothing in the future in this batch! The model and the oracle have access to the same information."
                "This can happen if the LLM did not generate any token for the given prompt, or if random prompt len is enabled and the sampling contains a bug."
            )

        # Initialize state tensors
        self.selection_mask = torch.zeros(
            (self.current_batch_size, self.max_prompt_len_in_batch),
            dtype=torch.bool,
            device=self.device,
        )
        self.output_kv_indices = torch.full(
            (self.current_batch_size, self.max_prompt_len_in_batch),
            -1,  # Use -1 as placeholder for unselected steps
            dtype=torch.long,
            device=self.device,
        )
        self.step_count = torch.zeros(
            self.current_batch_size, dtype=torch.long, device=self.device
        )

        fake_info = self._initialize_info_dict(self.current_batch_size)
        obs = self._get_observation()
        obs["info"] = fake_info
        return obs

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        """Constructs the observation dictionary (views only for speed)."""
        if (
            self.current_batch is None
            or self.selection_mask is None
            or self.step_count is None
            or self.prompt_len is None
        ):
            raise RuntimeError("Environment must be reset before getting observation.")

        # The oracle computes the cost inside the environment, using the complete keys and queries
        # to compute the attention scores -- that, thus, includes the future.
        # From outside the env, the future MUST not be observable, which we guarantee by
        # only exposing keys/values up to max_prompt_len_in_batch.
        if self.current_batch["queries"].ndim == 3:
            # Single head format: [B, S, D]
            obs_dict = {
                "keys": self.current_batch["keys"][
                    :, : self.max_prompt_len_in_batch, :
                ],
                "values": self.current_batch["values"][
                    :, : self.max_prompt_len_in_batch, :
                ],
                "queries": self.current_batch["queries"][
                    :, : self.max_prompt_len_in_batch, :
                ],
                "selection_history": self.selection_mask,
                "seq_lengths": self.prompt_len,
                "step_count": self.step_count,
            }
        else:
            # Grouped query format: [B, num_kv_heads, S, D] or [B, num_kv_heads, queries_per_kv, S, D]
            obs_dict = {
                "keys": self.current_batch["keys"][
                    ..., : self.max_prompt_len_in_batch, :
                ],
                "values": self.current_batch["values"][
                    ..., : self.max_prompt_len_in_batch, :
                ],
                "queries": self.current_batch["queries"][
                    ..., : self.max_prompt_len_in_batch, :
                ],
                "selection_history": self.selection_mask,
                "seq_lengths": self.prompt_len,
                "step_count": self.step_count,
            }
        return obs_dict

    def step(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Takes a step by selecting the next element index (optimized)."""
        if (
            self.current_batch is None
            or self.selection_mask is None
            or self.output_kv_indices is None
            or self.step_count is None
            or self.prompt_len is None
        ):
            raise RuntimeError("Environment must be reset before stepping.")

        if actions.shape[0] != self.current_batch_size:
            raise ValueError(
                f"Actions tensor size ({actions.shape[0]}) != batch size ({self.current_batch_size})"
            )

        not_done_mask = self.step_count < self.prompt_len
        active_indices = torch.where(not_done_mask)[0]
        num_active = active_indices.numel()

        if num_active > 0:
            active_actions = actions[active_indices]
            active_steps = self.step_count[active_indices]
            self.selection_mask[active_indices, active_actions] = True
            self.output_kv_indices[active_indices, active_steps] = active_actions

        self.step_count += 1

        dones = self.step_count >= self.prompt_len

        rewards = torch.zeros(
            self.current_batch_size, device=self.device, dtype=torch.float32
        )

        info = self._initialize_info_dict(self.current_batch_size)

        just_finished_mask = dones & not_done_mask

        if torch.any(just_finished_mask):
            done_indices = torch.where(just_finished_mask)[0]

            final_rankings = self.output_kv_indices[done_indices]
            rankings_len = self.prompt_len[done_indices]

            # Include future tokens for oracle evaluation (seq_len > prompt_len)
            keys_for_oracle = self.current_batch["keys"][done_indices]
            queries_for_oracle = self.current_batch["queries"][done_indices]
            lengths_for_oracle = self.current_batch["lengths"][done_indices]

            cost_result = self.oracle.compute_cost(
                kv_rankings=final_rankings,
                kv_rankings_length=rankings_len,
                all_keys=keys_for_oracle.to(torch.float32),
                all_queries=queries_for_oracle.to(torch.float32),
                seq_lengths=lengths_for_oracle,
            )

            rewards[done_indices] = self._compute_reward(cost_result)
            cost_info = self._extract_info_from_oracle_result(
                cost_result, num_samples=rankings_len.shape[0]
            )
            for key, value in cost_info.items():
                info[key][done_indices] = value.to(info[key].dtype)

        next_observation = self._get_observation()
        next_observation["rewards"] = rewards
        next_observation["dones"] = dones
        next_observation["info"] = info

        return next_observation

    def get_valid_actions(self) -> Dict[str, torch.Tensor]:
        """Returns views/references to info needed for action masking (FAST)."""
        if self.selection_mask is None or self.prompt_len is None:
            raise RuntimeError(
                "Environment must be reset before getting valid actions."
            )
        return {
            "selection_history": self.selection_mask,
            "seq_lengths": self.prompt_len,
        }

    def compute_rewards_for_chunk(
        self,
        actions_chunk: torch.Tensor,
        batch_indices_chunk: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes terminal rewards for a chunk of the batch.

        This method is NOT stateless. It uses the environment's internal `current_batch`
        and slices it based on the provided indices. This guarantees that the data
        fed to the oracle for each sample is identical to the iterative `step()` path,
        while still allowing for chunked processing to manage memory.

        Args:
            actions_chunk (torch.Tensor): A tensor of action permutations for the chunk,
                                          shape [chunk_size, max_prompt_len].
            batch_indices_chunk (torch.Tensor): The original indices of these samples
                                                in the full batch, shape [chunk_size].

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing 'rewards' and other info metrics.
        """
        if self.current_batch is None:
            raise RuntimeError("Environment must be reset before calling this method.")

        # Slice the environment's internal state for this chunk
        keys_chunk = self.current_batch["keys"][batch_indices_chunk].to(torch.float32)
        queries_chunk = self.current_batch["queries"][batch_indices_chunk].to(
            torch.float32
        )
        lengths_chunk = self.current_batch["lengths"][batch_indices_chunk]
        prompt_len_chunk = self.prompt_len[batch_indices_chunk]

        max_len = actions_chunk.shape[1]
        padding_mask = (
            torch.arange(max_len, device=actions_chunk.device)[None, :]
            >= prompt_len_chunk[:, None]
        )

        # Zero out padded action indices; the oracle ignores them based on prompt_len_chunk
        actions_chunk[padding_mask] = 0

        cost_result = self.oracle.compute_cost(
            kv_rankings=actions_chunk,
            kv_rankings_length=prompt_len_chunk,
            all_keys=keys_chunk,
            all_queries=queries_chunk,
            seq_lengths=lengths_chunk,
        )

        info = self._extract_info_from_oracle_result(
            cost_result, num_samples=prompt_len_chunk.shape[0]
        )
        rewards = self._compute_reward(cost_result)

        if not torch.all(torch.isfinite(rewards)):
            pylogger.warning(f"NaN or Inf detected in rewards! Values: {rewards}")
            # Sanitize the rewards as a safety measure
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        info["rewards"] = rewards
        return info
