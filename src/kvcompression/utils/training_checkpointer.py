#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Training checkpointer for resume functionality.

This module provides a reusable checkpointing system for
checkpoint management with local storage.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from kvcompression.utils.utils import get_underlying_model

if TYPE_CHECKING:
    from kvcompression.metrics import MetricsManager
    from kvcompression.rl.trainers.rloo_terminal_trainer import RLOOTerminalTrainer

logger = logging.getLogger(__name__)


@dataclass
class ResumeInfo:
    """Container for resume information that gets broadcast to all ranks."""

    resume_success: bool
    step: int = 0
    best_eval_metric_value: float = -float("inf")
    last_eval_step: int = 0
    wandb_run_id: Optional[str] = None
    metrics_history: Dict[str, Any] = field(default_factory=dict)
    step_counter: int = 0
    random_states: Dict[str, Any] = field(default_factory=dict)


class TrainingCheckpointer:
    """
    Reusable training checkpointer with local storage.

    Features:
    - Local storage with hierarchical fallback
    - Manager pattern (only rank 0 handles I/O)
    - Full state preservation
    - Graceful error handling
    """

    # Constants
    DIR_NAME: str = "checkpoints"
    FILE_MATCH: str = "*.pth"
    FILE_FORMAT: str = "%012d.pth"

    def __init__(
        self,
        storage_dir: Path,
        config: DictConfig,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize the training checkpointer.

        Args:
            storage_dir: Base storage directory for checkpoints
            config: Training configuration
            rank: Current process rank
            world_size: Total number of processes
        """
        self.storage_dir = Path(storage_dir)
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_rank_zero = rank == 0
        self.is_distributed = world_size > 1

        self.checkpoint_interval = OmegaConf.select(
            config, "resume.checkpoint_interval", default=50
        )
        self.max_checkpoints = OmegaConf.select(
            config, "resume.max_checkpoints_to_keep", default=3
        )
        self.resume_enabled = OmegaConf.select(config, "resume.enabled", default=True)

    def _get_local_checkpoint_dir(self, metrics_manager: MetricsManager) -> Path:
        """Get run-specific local checkpoint directory."""
        run_dir = metrics_manager.default_run_dir(storage_dir=self.storage_dir)
        return run_dir / self.DIR_NAME

    def _get_latest_checkpoint(self, directory: Path) -> Optional[Path]:
        """Find the latest checkpoint in a directory based on step number."""
        if not directory.exists():
            return None

        checkpoint_files = list(directory.glob(self.FILE_MATCH))
        if not checkpoint_files:
            return None

        valid_checkpoints = []
        for checkpoint_file in checkpoint_files:
            try:
                step_number = int(checkpoint_file.stem)
                valid_checkpoints.append((step_number, checkpoint_file))
            except ValueError:
                # Skip files that don't have valid step numbers as names
                continue

        if not valid_checkpoints:
            return None

        return max(valid_checkpoints, key=lambda x: x[0])[1]

    def check_for_resume(self) -> Optional[Path]:
        """
        Check for resume checkpoints in local storage.

        Returns:
            Path to checkpoint file if found, None otherwise.
        """
        if not self.is_rank_zero or not self.resume_enabled:
            return None

        # No automatic resume from local storage - users can specify checkpoint path
        # in config if they want to resume from a specific checkpoint
        logger.info("No automatic resume checkpoint found. Starting fresh training.")
        return None

    def restore_best_checkpoint_if_exists(
        self, metrics_manager: MetricsManager
    ) -> bool:
        """
        Check for and restore best checkpoint if it exists.

        Args:
            metrics_manager: Metrics manager for determining local path

        Returns:
            True if best checkpoint was restored, False otherwise
        """
        if not self.is_rank_zero:
            return False

        try:
            local_best_path = metrics_manager.best_ckpt_path(
                storage_dir=self.storage_dir
            )

            if not local_best_path.exists():
                return False

            logger.info(f"Best checkpoint found at: {local_best_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to check for best checkpoint: {e}")
            return False

    def load_checkpoint(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from file without applying state.

        This method can be called by all ranks to load their own copy of the checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary if successful, None otherwise
        """
        try:
            # Load checkpoint on the correct device for this rank
            map_location: Union[str, Dict[str, str]] = (
                {"cuda:%d" % 0: "cuda:%d" % self.rank}
                if torch.cuda.is_available()
                else "cpu"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=map_location, weights_only=False
            )
            return checkpoint
        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Error loading checkpoint from {checkpoint_path}: {e}"
            )
            return None

    def _save_to_file(self, path: Path, checkpoint: Dict[str, Any]) -> None:
        """Save checkpoint to file with proper directory creation."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def save_checkpoint(
        self,
        step: int,
        agent: torch.nn.Module,
        trainer: RLOOTerminalTrainer,
        metrics_manager: MetricsManager,
        best_eval_metric_value: float,
        last_eval_step: int,
        sampler_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save checkpoint to local storage.

        Args:
            step: Current training step
            agent: Training agent (model)
            trainer: Training trainer instance
            metrics_manager: Metrics manager instance
            best_eval_metric_value: Current best evaluation metric value
            last_eval_step: Step when last evaluation was performed
            sampler_state: Optional sampler state for resume support
        """
        if not self.is_rank_zero:
            return

        try:
            checkpoint_filename = self.FILE_FORMAT % step

            checkpoint = self._create_checkpoint_dict(
                step,
                agent,
                trainer,
                metrics_manager,
                best_eval_metric_value,
                last_eval_step,
                sampler_state,
            )

            # Save to local storage (run-specific directory)
            local_checkpoint_dir = self._get_local_checkpoint_dir(metrics_manager)
            local_checkpoint_path = local_checkpoint_dir / checkpoint_filename
            self._save_to_file(local_checkpoint_path, checkpoint)

            logger.info(
                f"Resume checkpoint saved to local storage: {local_checkpoint_path}"
            )

        except Exception as e:
            logger.error(f"Error saving resume checkpoint: {e}")

    def save_best_checkpoint(
        self,
        step: int,
        agent: torch.nn.Module,
        trainer: RLOOTerminalTrainer,
        metrics_manager: MetricsManager,
        best_eval_metric_value: float,
        metric_name: str,
        best_model_path: Path,
        sampler_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save best model checkpoint to specified path."""
        if not self.is_rank_zero:
            return

        try:
            checkpoint = self._create_checkpoint_dict(
                step,
                agent,
                trainer,
                metrics_manager,
                best_eval_metric_value,
                0,
                sampler_state,
            )

            checkpoint[f"best_{metric_name}"] = best_eval_metric_value

            # Save to specified local path
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, best_model_path)

            logger.info(f"Best model checkpoint saved to: {best_model_path}")

        except Exception as e:
            logger.error(f"Error saving best model checkpoint: {e}")

    def _create_checkpoint_dict(
        self,
        step: int,
        agent: torch.nn.Module,
        trainer: RLOOTerminalTrainer,
        metrics_manager: MetricsManager,
        best_eval_metric_value: float,
        last_eval_step: int,
        sampler_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create checkpoint dictionary with proper EMA state preservation."""
        underlying_model = get_underlying_model(agent)
        model_state_dict: Dict[str, Any] = (
            underlying_model._orig_mod.state_dict()
            if hasattr(underlying_model, "_orig_mod")
            and underlying_model._orig_mod is not None
            else underlying_model.state_dict()
        )

        # Save complete AveragedModel state including n_averaged counter
        ema_model_state_dict: Optional[Dict[str, Any]] = None
        if hasattr(trainer, "ema_agent") and trainer.ema_agent:
            ema_model_state_dict = trainer.ema_agent.state_dict()

        wandb_run_id: Optional[str] = None
        if (
            hasattr(trainer, "metrics_manager")
            and hasattr(trainer.metrics_manager, "wandb")
            and trainer.metrics_manager.wandb
            and hasattr(trainer.metrics_manager.wandb, "run")
            and trainer.metrics_manager.wandb.run
        ):
            wandb_run_id = trainer.metrics_manager.wandb.run.id

        # Create checkpoint
        checkpoint: Dict[str, Any] = {
            # Model & Training State
            "model_state_dict": model_state_dict,
            "ema_model_state_dict": ema_model_state_dict,
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict()
            if trainer.scheduler
            else None,
            # Training Progress
            "step": step,
            "best_eval_metric_value": best_eval_metric_value,
            "last_eval_step": last_eval_step,
            # Metrics for WandB continuity
            "metrics_history": metrics_manager.metrics,
            "step_counter": metrics_manager.step_counter,
            "wandb_run_id": wandb_run_id,
            # Random States
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state()
            if torch.cuda.is_available() and torch.cuda.current_device() >= 0
            else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            # Configuration
            "config": OmegaConf.to_container(self.config, resolve=True),
            # Sampler State for InfiniteDistributedSampler resume
            "sampler_state": sampler_state,
        }

        return checkpoint

    def cleanup_old_checkpoints(self, metrics_manager: MetricsManager) -> None:
        """
        Clean up old checkpoints to avoid storage bloat.

        Args:
            metrics_manager: Metrics manager instance for path computation
        """
        if not self.is_rank_zero:
            return

        try:
            local_checkpoint_dir = self._get_local_checkpoint_dir(metrics_manager)
            if not local_checkpoint_dir.exists():
                return

            if self.max_checkpoints <= 0:
                logger.warning(
                    f"max_checkpoints must be > 0, got {self.max_checkpoints}. Skipping cleanup."
                )
                return

            checkpoint_files = list(local_checkpoint_dir.glob(self.FILE_MATCH))
            if len(checkpoint_files) <= self.max_checkpoints:
                return

            # Sort by step number (extracted from filename)
            try:
                sorted_checkpoints: List[Path] = sorted(
                    checkpoint_files, key=lambda p: int(p.stem)
                )
            except ValueError:
                # If step extraction fails, sort by modification time
                sorted_checkpoints = sorted(
                    checkpoint_files, key=lambda p: p.stat().st_mtime
                )

            checkpoints_to_remove = sorted_checkpoints[: -self.max_checkpoints]

            for old_checkpoint in checkpoints_to_remove:
                try:
                    old_checkpoint.unlink()
                except Exception as e:
                    logger.warning(
                        f"Failed to remove old checkpoint {old_checkpoint}: {e}"
                    )

        except Exception as e:
            logger.warning(f"Checkpoint cleanup failed: {e}")

    def should_save_checkpoint(self, step: int) -> bool:
        """Determine if checkpoint should be saved at this step."""
        return (
            self.resume_enabled
            and self.is_rank_zero
            and step % self.checkpoint_interval == 0
        )

    def broadcast_checkpoint_path(
        self, checkpoint_path: Optional[Path]
    ) -> Optional[Path]:
        """
        Broadcast checkpoint path from rank 0 to all ranks.

        Args:
            checkpoint_path: Path to checkpoint file (only valid on rank 0)

        Returns:
            Checkpoint path available on all ranks
        """
        if not self.is_distributed:
            return checkpoint_path

        path_list = [checkpoint_path] if self.is_rank_zero else [None]
        dist.broadcast_object_list(path_list, src=0)

        return path_list[0]

    def apply_random_states(self, resume_info: ResumeInfo) -> None:
        """
        Apply random states to all ranks from resume info.

        Args:
            resume_info: Resume information containing random states
        """
        if not resume_info.resume_success or not resume_info.random_states:
            return

        random_states = resume_info.random_states

        try:
            if random_states.get("torch_rng_state"):
                torch.set_rng_state(random_states["torch_rng_state"])
            if random_states.get("cuda_rng_state") and torch.cuda.is_available():
                torch.cuda.set_rng_state(random_states["cuda_rng_state"])
            if random_states.get("numpy_rng_state"):
                np.random.set_state(random_states["numpy_rng_state"])
            if random_states.get("python_rng_state"):
                random.setstate(random_states["python_rng_state"])

            logger.info("Random states applied for reproducibility")
        except Exception as e:
            logger.warning(f"Failed to apply random states: {e}")

    def sync_after_checkpoint(self) -> None:
        """Synchronize all ranks after checkpoint operations."""
        if self.is_distributed:
            dist.barrier()
