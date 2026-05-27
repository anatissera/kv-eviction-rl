#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, training, utils
from torchtune.recipe_interfaces import FTRecipeInterface
from tqdm import tqdm

from kvcompression import PROJECT_ROOT
from kvcompression.data.pad_collate import pad_collate_fn
from kvcompression.metrics._metrics_manager import MetricsManager
from kvcompression.rl.trainers.rloo_terminal_trainer import RLOOTerminalTrainer
from kvcompression.utils.infinite_distributed_sampler import InfiniteDistributedSampler
from kvcompression.utils.s3_utils import (
    get_corresponding_remote_path,
    upload_directory,
)
from kvcompression.utils.training_checkpointer import ResumeInfo, TrainingCheckpointer
from kvcompression.utils.utils import get_underlying_model, is_torch_compile_disabled

torch.set_float32_matmul_precision("high")

logger = utils.get_logger("INFO")
log_rank_zero = utils.log_rank_zero


class RLOOTrainingRecipeDistributed(FTRecipeInterface):
    """
    TorchTune-style recipe for distributed training of RL agents using RLOOTerminalTrainer,
    adapted with TorchTune distributed best practices using DDP.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)

        torch.set_default_dtype(self._dtype)

        self._setup_distributed()

        self._seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("deterministic_cudnn", False)
        )
        msg = f"Rank {self.rank}: Setting random seed to {self._seed}"
        if cfg.get("deterministic_cudnn", False):
            msg += " with deterministic CUDNN"
        log_rank_zero(logger, msg)

        secrets_asset = OmegaConf.select(cfg, "secrets.asset", default=None)
        if secrets_asset:
            from kvcompression.utils.s3_utils import load_remote_json_secrets

            load_remote_json_secrets(secrets_asset, rank=self.rank)

        training_storage_dir = OmegaConf.select(
            cfg, "training.storage_dir", default=None
        )
        if training_storage_dir:
            self.checkpointer: Optional[TrainingCheckpointer] = TrainingCheckpointer(
                storage_dir=Path(training_storage_dir),
                config=cfg,
                rank=self.rank,
                world_size=self.world_size,
            )
        else:
            self.checkpointer = None

    def _setup_distributed(self):
        """Initializes the distributed process group using TorchTune helpers."""
        is_distributed = False
        if dist.is_available() and self._cfg.distributed:
            backend = training.get_distributed_backend(self._device.type)
            dist.init_process_group(
                backend=backend, timeout=datetime.timedelta(hours=1)
            )
            is_distributed = True
        elif self._cfg.distributed:
            log_rank_zero("Distributed training requested but not available.")

        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0

        if self._device.type == "cuda":
            self._device = torch.device(f"cuda:{self.rank}")
            torch.cuda.set_device(self._device)

        if is_distributed:
            log_rank_zero(
                logger,
                f"Distributed training initialized: Rank {self.rank}/{self.world_size} on device {self._device} using backend {backend}",
            )
        else:
            log_rank_zero(
                logger,
                f"Single-gpu training initialized on device {self._device}",
            )

    def _fast_forward_to_step(self, target_step: int) -> None:
        """
        Fast-forward to approximate resume position using epoch-level granularity.

        Args:
            target_step: Target training step to resume from
        """
        if not hasattr(self._train_loader, "sampler") or target_step <= 0:
            return

        # Calculate approximate epoch for resume
        steps_per_epoch = len(self._train_loader)
        if steps_per_epoch > 0:
            target_epoch = target_step // steps_per_epoch
            steps_in_epoch = target_step % steps_per_epoch

            # Set sampler to correct epoch
            self._train_loader.sampler.set_epoch(target_epoch)

            # Recreate iterator once for resume
            if hasattr(self._trainer.env, "recreate_iterator"):
                self._trainer.env.recreate_iterator()

            # Fast-forward within epoch by consuming batches
            for _ in range(steps_in_epoch):
                try:
                    next(self._trainer.env.dataloader_iter)
                except StopIteration:
                    # Epoch ended, iterator will automatically cycle to next
                    break

            log_rank_zero(
                logger,
                f"Fast-forwarded to step {target_step}: epoch {target_epoch}, "
                f"consumed {min(steps_in_epoch, len(self._train_loader))} batches in current epoch",
            )

    def _should_save_checkpoint(self, current_step: int) -> bool:
        """
        Determine if checkpoint should be saved at this step.

        Args:
            current_step: Current training step

        Returns:
            True if checkpoint should be saved, False otherwise
        """
        if not self.checkpointer:
            return False

        return self.checkpointer.should_save_checkpoint(current_step)

    def _safe_checkpoint_save(
        self, step: int, best_eval_metric_value: float, last_eval_step: int
    ) -> None:
        """
        Save checkpoint with error handling.

        Args:
            step: Current training step
            best_eval_metric_value: Current best evaluation metric value
            last_eval_step: Step when last evaluation was performed
        """
        if not self.checkpointer:
            return

        try:
            self.checkpointer.save_checkpoint(
                step,
                self._agent,
                self._trainer,
                self._trainer.metrics_manager,
                best_eval_metric_value,
                last_eval_step,
                sampler_state=self._train_loader.sampler.state_dict()
                if self._train_loader
                else None,
            )
            self.checkpointer.cleanup_old_checkpoints(self._trainer.metrics_manager)
        except Exception as e:
            log_rank_zero(
                logger,
                f"Checkpoint save failed at step {step}: {e}. Continuing training...",
                level=logging.WARNING,
            )

    def setup(self) -> None:
        """Instantiates all components required for training."""
        log_rank_zero(logger, "Setting up recipe components...")

        if not OmegaConf.select(self._cfg, "remote_dir", default=None):
            raise ValueError(
                "Config key 'remote_dir' is required for uploading trained agents. "
                "Set it to your S3 bucket path (e.g., 's3://ml-learning-to-evict')."
            )

        # Phase 1: Resume detection (before any model creation)
        resume_info, checkpoint_data = self._detect_and_load_resume()

        # Phase 2: Setup data and environments
        self._setup_data_and_environments()

        # Phase 3: Setup model and agent
        self._setup_model_and_agent(
            checkpoint_data if resume_info.resume_success else None
        )

        # Phase 4: Setup trainer
        self._setup_trainer(resume_info, checkpoint_data)

        # Phase 4.5: Restore best checkpoint if resuming
        if resume_info.resume_success and self.checkpointer:
            self.checkpointer.restore_best_checkpoint_if_exists(
                self._trainer.metrics_manager
            )

        # Phase 5: Apply remaining resume state
        if resume_info.resume_success:
            self._apply_remaining_resume_state(resume_info, checkpoint_data)

        # Initialize training state variables
        self._resume_step = resume_info.step if resume_info.resume_success else 0
        self._last_eval_step = (
            resume_info.last_eval_step if resume_info.resume_success else None
        )
        self._best_eval_metric_value = (
            resume_info.best_eval_metric_value
            if resume_info.resume_success
            else -float("inf")
        )

        # Fast-forward to resume position if needed
        if resume_info.resume_success and self._resume_step > 0:
            # Restore sampler state if available
            if (
                "sampler_state" in checkpoint_data
                and checkpoint_data["sampler_state"] is not None
                and self._train_loader is not None
            ):
                self._train_loader.sampler.load_state_dict(
                    checkpoint_data["sampler_state"]
                )
                log_rank_zero(logger, "Restored sampler state from checkpoint")

            self._fast_forward_to_step(self._resume_step)

        if not resume_info.resume_success:
            log_rank_zero(logger, "Starting fresh training")

        if dist.is_initialized():
            dist.barrier()
        log_rank_zero(logger, "Setup complete.")

    def _detect_and_load_resume(self) -> tuple[ResumeInfo, Optional[Dict[str, Any]]]:
        """Phase 1: Detect and load resume checkpoint before any model creation."""
        log_rank_zero(logger, "Checking for resume checkpoint...")

        if not self.checkpointer:
            log_rank_zero(logger, "No checkpointer available. Starting fresh training.")
            return ResumeInfo(resume_success=False), None

        # Only rank 0 checks for resume, then broadcast the decision
        checkpoint_path = self.checkpointer.check_for_resume()
        checkpoint_path = self.checkpointer.broadcast_checkpoint_path(checkpoint_path)

        if not checkpoint_path:
            log_rank_zero(
                logger, "No resume checkpoint found. Starting fresh training."
            )
            return ResumeInfo(resume_success=False), None

        # All ranks load their own copy from disk
        checkpoint_data = self.checkpointer.load_checkpoint(checkpoint_path)
        if not checkpoint_data:
            log_rank_zero(
                logger,
                f"Failed to load checkpoint from {checkpoint_path}. Starting fresh training.",
                level=logging.ERROR,
            )
            # Make this fatal as requested
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}")

        # Extract resume info
        resume_info = ResumeInfo(
            resume_success=True,
            step=checkpoint_data["step"],
            best_eval_metric_value=checkpoint_data.get(
                "best_eval_metric_value",
                checkpoint_data.get("best_eval_reward", -float("inf")),
            ),
            last_eval_step=checkpoint_data.get("last_eval_step", 0),
            wandb_run_id=checkpoint_data.get("wandb_run_id"),
            metrics_history=checkpoint_data.get("metrics_history", {}),
            step_counter=checkpoint_data.get("step_counter", checkpoint_data["step"]),
            random_states={
                "torch_rng_state": checkpoint_data.get("torch_rng_state"),
                "cuda_rng_state": checkpoint_data.get("cuda_rng_state"),
                "numpy_rng_state": checkpoint_data.get("numpy_rng_state"),
                "python_rng_state": checkpoint_data.get("python_rng_state"),
            },
        )

        log_rank_zero(
            logger,
            f"Found resume checkpoint at step {resume_info.step}",
        )

        # Apply random states early for deterministic behavior
        if self.checkpointer:
            self.checkpointer.apply_random_states(resume_info)

        self._seed = training.set_seed(
            seed=self._cfg.seed, debug_mode=self._cfg.get("deterministic_cudnn", False)
        )

        return resume_info, checkpoint_data

    def _setup_data_and_environments(self) -> None:
        """Phase 2: Setup datasets, dataloaders, oracle, and environments."""
        self._train_loader = None
        self._eval_loader = None
        self._train_dset_len = 0

        if hasattr(self._cfg, "data"):
            logger.info("Data section found, setting up datasets...")
            self._train_dset = config.instantiate(self._cfg.data.dataset.train)
            self._val_dset = config.instantiate(self._cfg.data.dataset.val)
            self._train_dset_len = len(self._train_dset)
            self._val_dset_len = len(self._val_dset)

            # Create InfiniteDistributedSampler for efficient RL training
            train_sampler = InfiniteDistributedSampler(
                dataset=self._train_dset,
                seed=self._seed,
            )
            val_sampler = DistributedSampler(
                self._val_dset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                seed=self._seed,
            )

            self._train_loader = DataLoader(
                dataset=self._train_dset,
                batch_size=self._cfg.loader.train.batch_size,
                sampler=train_sampler,
                collate_fn=pad_collate_fn,
                num_workers=self._cfg.loader.train.dataloader_num_workers,
                prefetch_factor=self._cfg.loader.train.prefetch_factor,
                shuffle=False,
                pin_memory=self._cfg.loader.train.dataloader_pin_memory
                and (self._device.type == "cuda"),
                drop_last=True,
                persistent_workers=(self._cfg.loader.train.dataloader_num_workers > 0),
            )
            self._eval_loader = DataLoader(
                dataset=self._val_dset,
                batch_size=self._cfg.loader.eval.batch_size,
                sampler=val_sampler,
                collate_fn=pad_collate_fn,
                num_workers=self._cfg.loader.eval.dataloader_num_workers,
                prefetch_factor=self._cfg.loader.eval.prefetch_factor,
                shuffle=False,
                pin_memory=self._cfg.loader.eval.dataloader_pin_memory
                and (self._device.type == "cuda"),
                drop_last=True,
                persistent_workers=(self._cfg.loader.eval.dataloader_num_workers > 0),
            )
            log_rank_zero(
                logger,
                "DataLoaders created with InfiniteDistributedSampler for efficient RL training.",
            )
        else:
            log_rank_zero(
                logger,
                "No 'data.dataset' section in config. Environment must handle data generation/loading.",
            )

        log_rank_zero(logger, "Instantiating Oracle/Verifier...")
        with torch.device(self._device):
            self._oracle = config.instantiate(self._cfg.oracle)

        log_rank_zero(logger, "Instantiating Training Environment...")
        env_kwargs = {
            "device": self._device,
        }
        if self._train_loader is not None:
            env_kwargs["dataloader"] = self._train_loader
        if self._oracle:
            env_kwargs["oracle"] = self._oracle
        self._env = config.instantiate(self._cfg.environment.train, **env_kwargs)

        log_rank_zero(logger, "Instantiating Evaluation Environment...")
        env_kwargs = {
            "device": self._device,
        }
        if self._eval_loader is not None:
            env_kwargs["dataloader"] = self._eval_loader
        if self._oracle:
            env_kwargs["oracle"] = self._oracle
        self._eval_env = config.instantiate(self._cfg.environment.eval, **env_kwargs)

    def _setup_model_and_agent(self, checkpoint_data: Optional[Dict[str, Any]]) -> None:
        """Phase 3: Setup model and agent with optional checkpoint restoration."""
        log_rank_zero(logger, "Instantiating Agent...")
        with training.set_default_dtype(self._dtype), torch.device(self._device):
            model_to_wrap: nn.Module = config.instantiate(
                self._cfg.agent, device=self._device
            )

            # Separate EMA model instance, initialized with the training model's weights
            ema_model_instance: nn.Module = config.instantiate(
                self._cfg.agent, device=self._device
            )
            ema_model_instance.load_state_dict(model_to_wrap.state_dict())

            self._ema_agent: nn.Module = AveragedModel(
                ema_model_instance,
                avg_fn=torch.optim.swa_utils.get_ema_avg_fn(self._cfg.ema.decay),
                use_buffers=True,
            )

        # Restore model state before DDP wrapping if resuming
        if checkpoint_data:
            log_rank_zero(logger, "Restoring model state before DDP wrapping...")
            if "model_state_dict" in checkpoint_data:
                model_to_wrap.load_state_dict(checkpoint_data["model_state_dict"])
                log_rank_zero(logger, "Model state restored successfully.")
            else:
                raise RuntimeError("No model_state_dict found in checkpoint")

        # Compile before DDP wrapping
        if self._cfg.compile_agent and not is_torch_compile_disabled():
            log_rank_zero(logger, "Compiling the agent...")
            model_to_wrap.compile(dynamic=True)
            log_rank_zero(logger, "Agent compiled successfully.")

        if dist.is_initialized():
            find_unused = self._cfg.get("ddp_find_unused_parameters", False)

            self._agent = DDP(
                model_to_wrap,
                device_ids=[self.rank] if self._device.type == "cuda" else None,
                output_device=self.rank if self._device.type == "cuda" else None,
                find_unused_parameters=find_unused,
            )

            log_rank_zero(
                logger,
                f"Agent wrapped with DDP (find_unused_parameters={find_unused}).",
            )
        else:
            self._agent = model_to_wrap

    def _setup_trainer(
        self, resume_info: ResumeInfo, checkpoint_data: Optional[Dict[str, Any]]
    ) -> None:
        """Phase 4: Setup trainer with metrics manager."""
        # Create MetricsManager with resume info (all ranks have same info now)
        metrics_manager: MetricsManager = MetricsManager(
            use_wandb=OmegaConf.select(self._cfg, "wandb.enabled", default=False),
            wandb_project=OmegaConf.select(self._cfg, "wandb.project", default=None),
            wandb_entity=OmegaConf.select(self._cfg, "wandb.entity", default=None),
            wandb_name=OmegaConf.select(self._cfg, "wandb.name", default=None),
            wandb_run_id=resume_info.wandb_run_id
            if resume_info.resume_success
            else None,
            initial_metrics_history=resume_info.metrics_history
            if resume_info.resume_success
            else None,
            initial_step_counter=resume_info.step_counter
            if resume_info.resume_success
            else 0,
        )

        # Set up profiler before trainer instantiation
        profiler_output_dir = PROJECT_ROOT / "profiler"
        self._profiler, profiler_cfg = training.setup_torch_profiler(
            **self._cfg.profiler, output_dir=profiler_output_dir
        )
        if profiler_cfg["enabled"] and self._is_rank_zero:
            log_rank_zero(
                logger, f"Torch Profiler is enabled with config: {profiler_cfg}"
            )
            Path(profiler_output_dir).mkdir(parents=True, exist_ok=True)

        log_rank_zero(logger, "Instantiating Trainer...")
        self._trainer: RLOOTerminalTrainer = config.instantiate(
            self._cfg.trainer,
            agent=self._agent,
            ema_agent=self._ema_agent,
            env=self._env,
            eval_env=self._eval_env,
            device=self._device,
            metrics_manager=metrics_manager,
        )

        metrics_manager.log_hparams(OmegaConf.to_container(self._cfg, resolve=True))

    def _apply_remaining_resume_state(
        self, resume_info: ResumeInfo, checkpoint_data: Dict[str, Any]
    ) -> None:
        """Phase 5: Apply remaining resume state after trainer creation."""
        log_rank_zero(logger, "Restoring optimizer and scheduler state...")

        # Restore EMA model state
        if (
            "ema_model_state_dict" in checkpoint_data
            and checkpoint_data["ema_model_state_dict"]
        ):
            self._ema_agent.load_state_dict(checkpoint_data["ema_model_state_dict"])
            log_rank_zero(logger, "EMA model state restored successfully.")
        else:
            log_rank_zero(
                logger, "No EMA model state found in checkpoint - EMA will start fresh."
            )

        # Restore optimizer state
        if "optimizer_state_dict" in checkpoint_data:
            self._trainer.optimizer.load_state_dict(
                checkpoint_data["optimizer_state_dict"]
            )
            log_rank_zero(logger, "Optimizer state restored successfully.")
        else:
            raise RuntimeError("No optimizer_state_dict found in checkpoint")

        # Restore scheduler state
        if (
            "scheduler_state_dict" in checkpoint_data
            and checkpoint_data["scheduler_state_dict"]
        ):
            if self._trainer.scheduler:
                self._trainer.scheduler.load_state_dict(
                    checkpoint_data["scheduler_state_dict"]
                )
                log_rank_zero(
                    logger,
                    f"Scheduler state restored. Last epoch: {self._trainer.scheduler.last_epoch}",
                )
            else:
                log_rank_zero(logger, "No scheduler available to restore state to.")

        log_rank_zero(
            logger,
            f"Resuming training from step {resume_info.step} "
            f"(last eval at step {resume_info.last_eval_step})",
        )

    def train(self) -> None:
        """Runs the distributed training loop."""
        log_rank_zero(logger, "Starting training...")

        with self._profiler as prof:
            train_cfg = self._cfg.training
            num_epochs_or_steps = train_cfg.num_epochs_or_steps
            eval_interval = train_cfg.eval_interval

            eval_episodes = OmegaConf.select(train_cfg, "eval_episodes", default=None)
            save_best_model = OmegaConf.select(
                train_cfg, "save_best_model", default=False
            )

            if save_best_model:
                best_model_path = self._trainer.metrics_manager.best_ckpt_path(
                    storage_dir=OmegaConf.select(train_cfg, "storage_dir", default=None)
                )

            save_best_on_metric = OmegaConf.select(
                train_cfg, "save_best_on_metric", default="eval_reward"
            )
            best_eval_metric_value = self._best_eval_metric_value

            log_cfg = self._cfg.get("logging", {})
            tqdm_train_metrics = log_cfg.get(
                "tqdm_train_metrics", ["total_loss", "episode_reward"]
            )
            tqdm_eval_metrics = log_cfg.get(
                "tqdm_eval_metrics", ["eval_reward", "eval_length"]
            )
            tqdm_window_size = log_cfg.get("tqdm_smoothing_window", 50)

            # Calculate start step and last eval step
            start_step: int = self._resume_step
            last_eval_step: Optional[int] = self._last_eval_step

            # Validate training bounds to prevent edge cases
            if start_step >= num_epochs_or_steps:
                log_rank_zero(
                    logger,
                    f"Resume step {start_step} >= total steps {num_epochs_or_steps}. Training complete.",
                )
                return

            # Progress bar only on rank 0
            progress_bar = tqdm(
                range(start_step, num_epochs_or_steps),
                desc="Training Steps",
                disable=not self._is_rank_zero,
                initial=start_step,
                total=num_epochs_or_steps,
            )

            for current_step in range(start_step, num_epochs_or_steps):
                # Determine if we should evaluate at this step
                is_final_step = current_step == num_epochs_or_steps - 1

                if last_eval_step is None:
                    # Fresh start - evaluate at regular intervals or final step
                    should_evaluate = (
                        current_step + 1
                    ) % eval_interval == 0 or is_final_step
                else:
                    # Resume case - evaluate based on steps since last evaluation or final step
                    steps_since_last_eval = current_step + 1 - last_eval_step
                    should_evaluate = (
                        steps_since_last_eval >= eval_interval or is_final_step
                    )

                # Training Step (all Ranks)
                _ = self._trainer.train_batch(step=current_step)

                # Periodic checkpoint saving
                if self._should_save_checkpoint(current_step + 1):
                    self._safe_checkpoint_save(
                        current_step + 1,
                        best_eval_metric_value,
                        last_eval_step,
                    )

                if self._is_rank_zero:
                    tqdm_metrics_str = self._trainer.metrics_manager.format_for_tqdm(
                        tqdm_train_metrics,
                        window_size=tqdm_window_size,
                    )
                    lr = (
                        self._trainer.optimizer.param_groups[0]["lr"]
                        if self._trainer.optimizer
                        else float("nan")
                    )
                    progress_bar.set_postfix_str(f"{tqdm_metrics_str} LR: {lr:.1e}")
                    progress_bar.update(1)

                if self._profiler:
                    prof.step()

                # Evaluation
                if should_evaluate:
                    if self._eval_loader is not None and hasattr(
                        self._eval_loader.sampler, "set_epoch"
                    ):
                        epoch = (
                            (current_step + 1) // len(self._eval_loader)
                            if len(self._eval_loader) > 0
                            else 0
                        )
                        self._eval_loader.sampler.set_epoch(epoch)

                    log_rank_zero(
                        logger,
                        f"\n--- Evaluating at step {current_step + 1} (Rank {self.rank}) ---",
                    )

                    # Evaluation Logic (all ranks participate)
                    self._agent.eval()
                    with torch.no_grad():
                        logged_eval_metrics = self._trainer.evaluate(
                            num_episodes=eval_episodes
                        )

                    self._agent.train()

                    last_eval_step = current_step + 1

                    log_rank_zero(
                        logger,
                        f"\n--- Evaluating at step {current_step + 1} (Rank {self.rank}) DONE ---",
                    )

                    if self._is_rank_zero and logged_eval_metrics is not None:
                        eval_metrics_str = (
                            self._trainer.metrics_manager.format_for_tqdm(
                                tqdm_eval_metrics
                            )
                        )
                        log_rank_zero(
                            logger,
                            f"Step {current_step + 1} Eval (Aggregated): {eval_metrics_str}",
                        )

                        # Best model checkpointing (rank 0 only)
                        if save_best_model and self.checkpointer:
                            current_eval_metric = (
                                self._trainer.metrics_manager.get_latest(
                                    save_best_on_metric
                                )
                            )

                            if current_eval_metric is None:
                                log_rank_zero(
                                    logger,
                                    f"Warning: Metric '{save_best_on_metric}' not found. Skipping best model check.",
                                )
                            elif current_eval_metric > best_eval_metric_value:
                                best_eval_metric_value = current_eval_metric
                                log_rank_zero(
                                    logger,
                                    f"  -> New best model! {save_best_on_metric}: {best_eval_metric_value:.4f}. Saving to {best_model_path}...",
                                )

                                # Use checkpointer for best model saving
                                self.checkpointer.save_best_checkpoint(
                                    step=current_step + 1,
                                    agent=self._agent,
                                    trainer=self._trainer,
                                    metrics_manager=self._trainer.metrics_manager,
                                    best_eval_metric_value=best_eval_metric_value,
                                    metric_name=save_best_on_metric,
                                    best_model_path=best_model_path,
                                    sampler_state=self._train_loader.sampler.state_dict()
                                    if self._train_loader
                                    else None,
                                )

                                self._trainer.metrics_manager.add_batch(
                                    {
                                        f"best_{save_best_on_metric}": best_eval_metric_value,
                                        "best_model_step": current_step + 1,
                                    },
                                    increment_step=False,
                                    commit=False,
                                    distribute_average=False,
                                )

                        # Save resume checkpoint after evaluation
                        self._safe_checkpoint_save(
                            current_step + 1,
                            best_eval_metric_value,
                            last_eval_step,
                        )

                    if dist.is_initialized():
                        dist.barrier()

            if self._is_rank_zero:
                progress_bar.close()
            log_rank_zero(logger, "\nTraining finished.")

        if self._profiler and self._is_rank_zero:
            profiler_output_dir = OmegaConf.select(
                self._cfg, "profiler.trace_dir", default="./profiler_logs"
            )
            log_rank_zero(logger, f"Profiler trace saved to {profiler_output_dir}")

    def plot_results(self) -> None:
        """Plots training and evaluation curves based on config (Rank 0 only)."""
        if not self._is_rank_zero:
            return

        if not OmegaConf.select(self._cfg, "plotting.enabled", default=False):
            logger.info("Plotting is disabled in the configuration.")
            return

        logger.info("\nPlotting training results...")
        plot_cfg = self._cfg.plotting
        smoothing_window = OmegaConf.select(plot_cfg, "smoothing_window", default=50)

        if OmegaConf.select(plot_cfg, "plot_training_curves", default=True):
            self._trainer.plot_training_curves(smoothing_window=smoothing_window)
        if OmegaConf.select(plot_cfg, "plot_loss_curves", default=True):
            self._trainer.plot_loss_curves(smoothing_window=smoothing_window)
        if OmegaConf.select(plot_cfg, "plot_eval_metrics", default=True):
            self._trainer.plot_eval_metrics()

        custom_plots = OmegaConf.select(plot_cfg, "custom_metric_grids", default=[])
        for plot_spec in custom_plots:
            metrics = plot_spec.metrics
            window = OmegaConf.select(plot_spec, "window", default=smoothing_window)
            title = OmegaConf.select(plot_spec, "title", default=None)
            if metrics:
                logger.info(f"Plotting custom grid: {metrics} (window={window})")
                self._trainer.metrics_manager.plot_metric_grid(
                    metrics_to_plot=metrics,
                    window_size=window,
                    title=title,
                    log_on_wandb=True,
                )

        logger.info("Plotting complete (or saved to WandB if enabled).")

    @torch.inference_mode()
    def evaluate_best_model(self) -> None:
        """Loads the best saved model and runs a final evaluation (Rank 0 only)."""
        if not OmegaConf.select(self._cfg, "final_evaluation.enabled", default=False):
            logger.info("Final evaluation of the best model is disabled.")
            return
        if not OmegaConf.select(self._cfg, "training.save_best_model", default=False):
            logger.warning(
                "Final evaluation enabled, but 'training.save_best_model' is false. Skipping."
            )
            return

        best_model_path = self._trainer.metrics_manager.best_ckpt_path(
            storage_dir=OmegaConf.select(
                self._cfg.training, "storage_dir", default=None
            )
        )
        if not Path(best_model_path).exists():
            logger.warning(
                f"Best model path '{best_model_path}' not found. Skipping final evaluation."
            )
            return

        logger.info("\n--- Final Evaluation ---")
        logger.info(f"Loading best model from: {best_model_path}")
        try:
            # Load checkpoint on the correct device for each rank
            map_location = (
                {"cuda:%d" % 0: "cuda:%d" % self.rank}
                if self._device.type == "cuda"
                else self._device
            )
            checkpoint = torch.load(
                best_model_path, map_location=map_location, weights_only=False
            )

            underlying_model = get_underlying_model(self._agent)
            model_to_load = (
                underlying_model._orig_mod
                if hasattr(underlying_model, "_orig_mod")
                and underlying_model._orig_mod is not None
                else underlying_model
            )
            model_to_load.load_state_dict(checkpoint["model_state_dict"])
            self._ema_agent.load_state_dict(checkpoint["ema_model_state_dict"])
            log_rank_zero(logger, "Loaded best EMA model state for final evaluation.")

            log_rank_zero(
                logger,
                f"Best model loaded onto all ranks (saved at step {checkpoint.get('step', 'N/A')}).",
            )

            if dist.is_initialized():
                dist.barrier()

            final_eval_cfg = self._cfg.final_evaluation
            final_eval_episodes = final_eval_cfg.eval_episodes
            logger.info(f"Evaluating best model (episodes={final_eval_episodes})...")

            logged_metrics_names = self._trainer.evaluate(
                num_episodes=final_eval_episodes, metric_prefix="final"
            )

            if dist.is_initialized():
                dist.barrier()

            if self._is_rank_zero:
                log_rank_zero(
                    logger,
                    f"--- Final Test Metrics rank: {self.rank} ---",
                )
                for key in logged_metrics_names:
                    value = self._trainer.metrics_manager.get_latest(key)
                    if isinstance(value, float):
                        log_rank_zero(logger, f"  {key}: {value:.4f}")
                    else:
                        log_rank_zero(logger, f"  {key}: {value}")

        except FileNotFoundError:
            logger.error(
                f"Best model file not found at {best_model_path}. Skipping final evaluation."
            )
        except Exception as e:
            logger.error(f"Error during final evaluation: {e}", exc_info=True)

    def upload(self):
        if dist.is_initialized():
            dist.barrier()

        # Fall back to env var if remote_dir not set in config
        remote_dir = self._cfg.remote_dir
        if remote_dir is None:
            bucket = os.environ.get("KVCOMPRESSION_S3_BUCKET")
            if bucket:
                remote_dir = f"s3://{bucket}"
            else:
                log_rank_zero(
                    logger,
                    "Skipping upload: neither remote_dir in config nor KVCOMPRESSION_S3_BUCKET env var is set.",
                    level=logging.WARNING,
                )
                return

        if self._is_rank_zero:
            training_storage_dir = OmegaConf.select(
                self._cfg, "training.storage_dir", default=None
            )
            local_directory = self._trainer.metrics_manager.default_run_dir(
                storage_dir=training_storage_dir
            )
            remote_run_directory = get_corresponding_remote_path(
                path=local_directory,
                local_root=self._cfg.home_dir,
                remote_root=remote_dir,
            )
            upload_result = upload_directory(
                local_directory=local_directory,
                remote_destination_path=remote_run_directory,
            )
            if not upload_result:
                log_rank_zero(
                    logger,
                    "Upload of the run directory to remote storage failed. Results may be lost!",
                    level=logging.ERROR,
                )
            else:
                log_rank_zero(
                    logger,
                    "Upload of the run directory successful: {} -> {}".format(
                        local_directory, remote_run_directory
                    ),
                    level=logging.INFO,
                )

    def cleanup(self):
        """Finalize metrics logging and clean up distributed group."""
        self._trainer.metrics_manager.finalize()

        if dist.is_initialized():
            dist.destroy_process_group()
            log_rank_zero(logger, "Distributed process group destroyed.")


def launch_training(cfg: DictConfig, cleanup: bool = True) -> MetricsManager:
    if torch.cuda.is_available() and cfg.get("high_precision_matmul", False):
        log_rank_zero(logger, "Setting float32 matmul precision to 'high'")
        torch.set_float32_matmul_precision("high")

    recipe = RLOOTrainingRecipeDistributed(cfg=cfg)

    recipe.setup()

    metrics_manager = recipe._trainer.metrics_manager

    if recipe._is_rank_zero:
        config.log_config(recipe_name="RLOOTrainingRecipeDistributed", cfg=cfg)
        training_storage_dir = OmegaConf.select(
            cfg, "training.storage_dir", default=None
        )
        logger.info(
            f"Checkpoints and logs will be saved to: {metrics_manager.default_run_dir(storage_dir=training_storage_dir)}"
        )

    recipe.train()
    recipe.plot_results()
    recipe.evaluate_best_model()
    recipe.upload()

    if cleanup:
        recipe.cleanup()
    else:
        if dist.is_initialized():
            dist.barrier()

    return metrics_manager


@config.parse
def main(cfg: DictConfig) -> None:
    """Main entry point for the Distributed RLOO Training Recipe."""
    metrics_manager = launch_training(cfg, cleanup=False)
    metrics_manager.finalize()


if __name__ == "__main__":
    # Launch using torchrun:
    # torchrun --nproc_per_node NUM_GPUS your_script_name.py --config your_config.yaml [other args]
    sys.exit(main())
