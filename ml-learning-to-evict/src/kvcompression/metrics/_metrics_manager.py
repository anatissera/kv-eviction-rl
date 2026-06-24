#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

from kvcompression import PROJECT_ROOT

pylogger = logging.getLogger(__name__)


class MetricsManager:
    """
    Centralized manager for tracking, computing, and visualizing metrics.
    Handles metric aggregation automatically in distributed settings.
    """

    def __init__(
        self,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_run_id: Optional[str] = None,
        initial_metrics_history: Optional[Dict[str, List[float]]] = None,
        initial_step_counter: int = 0,
    ):
        """
        Initialize the metrics manager.

        Args:
            use_wandb: Whether to enable Weights & Biases logging.
            wandb_project: W&B project name (required if use_wandb=True).
            wandb_entity: W&B entity/team name.
            wandb_name: Display name for the W&B run.
            wandb_config: Configuration dict to log to W&B.
            wandb_run_id: Existing run ID to resume (enables run resumption).
            initial_metrics_history: Pre-existing metrics to restore from checkpoint.
            initial_step_counter: Starting step count (for checkpoint resumption).
        """
        self.metrics: Dict[str, List[float]] = initial_metrics_history or {}
        self.metric_groups: Dict[str, List[str]] = {}
        self._use_wandb = use_wandb
        self.step_counter = initial_step_counter
        self.wandb = None
        self.wandb_name = wandb_name
        self.wandb_run_id = wandb_run_id

        self._run_path: Optional[Path] = None

        # Distributed setup
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_rank_zero = self.rank == 0

        if self._use_wandb and self.is_rank_zero:
            try:
                import wandb

                self.wandb = wandb
                if wandb.run is None and not wandb.login(verify=True):
                    pylogger.warning("W&B login check failed. Disabling W&B logging.")
                    self._use_wandb = False
                    self.wandb = None
                elif wandb_project is None:
                    pylogger.warning("To log into W&B, provide a project name.")
                    self._use_wandb = False
                    self.wandb = None

                elif wandb.run is None:
                    # Handle resume case
                    if self.wandb_run_id:
                        wandb.init(
                            id=self.wandb_run_id,
                            name=wandb_name,
                            project=wandb_project,
                            entity=wandb_entity,
                            config=wandb_config or {},
                            resume="must",
                        )
                        pylogger.info(
                            f"Resumed W&B run {self.wandb_run_id} for project: {wandb_project}"
                        )
                    else:
                        wandb.init(
                            name=wandb_name,
                            project=wandb_project,
                            entity=wandb_entity,
                            config=wandb_config or {},
                        )
                        pylogger.info(
                            f"Initialized W&B logging for project: {wandb_project}"
                        )
                else:
                    pylogger.warning("W&B already initialized.")

            except ImportError:
                pylogger.warning(
                    "wandb package not found. Disabling W&B logging. Install with `pip install wandb`"
                )
                self._use_wandb = False
            except Exception as e:
                pylogger.warning(
                    f"Error initializing wandb: {e}. Disabling W&B logging."
                )
                self._use_wandb = False
        elif self._use_wandb and not self.is_rank_zero:
            self._use_wandb = False

    def _gather_and_sum_metric_on_rank0(
        self, value: Union[float, torch.Tensor]
    ) -> Optional[float]:
        """Aggregates metric (avg) using gather_object. Returns result on rank 0."""
        if not self.is_distributed or self.world_size == 1:
            return float(value.item() if isinstance(value, torch.Tensor) else value)

        local_value = float(value.item() if isinstance(value, torch.Tensor) else value)

        gathered_values = [None] * self.world_size if self.is_rank_zero else None

        # Gather Python objects from all ranks to rank 0.
        # Works regardless of whether local_value was calculated on CPU/GPU
        # and works with both 'nccl' and 'gloo' backends.
        dist.gather_object(
            local_value,
            gathered_values if self.is_rank_zero else None,
            dst=0,
        )

        if self.is_rank_zero:
            valid_values = [v for v in gathered_values if v is not None]
            if not valid_values:
                pylogger.warning("No valid metric values gathered.")
                return None
            return sum(valid_values)

        else:
            return None

    def default_run_dir(self, storage_dir: Optional[Path] = None) -> Path:
        if self._run_path:
            return self._run_path

        if storage_dir is None:
            pylogger.warning(
                "No storage directory provided. Using default: {PROJECT_ROOT}/agents"
            )
            storage_dir = PROJECT_ROOT / "agents"
        storage_dir = Path(storage_dir)

        run_path_list: List[Optional[Path]] = [None]

        if self.is_rank_zero:
            if self.is_wandb_enabled():
                run_id: str = self.wandb.run.id
            else:
                now = datetime.datetime.now()
                run_id = now.strftime("%Y-%m-%d_%H-%M-%S")

            run_path = storage_dir / run_id
            run_path.mkdir(exist_ok=True, parents=True)
            run_path_list[0] = run_path

        if self.is_distributed:
            pylogger.info(
                f"Rank {dist.get_rank()}: Waiting for broadcast of runpath..."
            )
            dist.broadcast_object_list(run_path_list, src=0)

        final_run_path: Optional[Path] = run_path_list[0]
        self._run_path = final_run_path

        return self._run_path

    def default_figures_dir(self, storage_dir: Optional[Path] = None) -> Path:
        figures_dir = self.default_run_dir(storage_dir=storage_dir) / "figures"
        if self.is_rank_zero:
            figures_dir.mkdir(exist_ok=True, parents=True)
        return figures_dir

    def best_ckpt_path(self, storage_dir: Optional[Path] = None) -> Path:
        return self.default_run_dir(storage_dir=storage_dir) / "best_ckpt.pth"

    def is_wandb_enabled(self) -> bool:
        return self._use_wandb and self.is_rank_zero

    def increment_step(self, increment: int = 1) -> None:
        """Increment the step counter (only relevant on rank 0)."""
        if self.is_rank_zero:
            self.step_counter += increment

    def add_metric_group(self, group_name: str, metrics: List[str]):
        """Add a new group of related metrics (rank 0 only)."""
        if self.is_rank_zero:
            self.metric_groups[group_name] = metrics
            for metric in metrics:
                if metric not in self.metrics:
                    self.metrics[metric] = []

    def add(
        self,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
        distribute_average: bool = True,
        distribute_local_nitems: Optional[int] = None,
    ):
        """Add a single metric value, aggregating across ranks."""
        if distribute_average:
            if distribute_local_nitems is None:
                raise ValueError(
                    "Impossible to average over procs without local number of items"
                )

        if distribute_average:
            aggregated_value = self._gather_and_sum_metric_on_rank0(value)
            n_items = self._gather_and_sum_metric_on_rank0(distribute_local_nitems)
        else:
            aggregated_value = value
            n_items = distribute_local_nitems

        if self.is_rank_zero and aggregated_value is not None:
            if distribute_average:
                aggregated_value = (
                    aggregated_value / n_items if n_items else aggregated_value
                )

            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(aggregated_value)
            log_step = step if step is not None else self.step_counter
            if self.is_wandb_enabled():
                self.wandb.log({metric_name: aggregated_value}, step=log_step)

    def add_batch(
        self,
        metric_dict: Dict[str, Union[float, torch.Tensor]],
        step: Optional[int] = None,
        increment_step: bool = True,
        commit: Optional[bool] = None,
        distribute_average: bool = True,
        distribute_local_nitems: Optional[int] = None,
    ):
        """Add multiple metric values, aggregating each across ranks."""
        if distribute_average:
            if distribute_local_nitems is None:
                raise ValueError(
                    "Impossible to average over procs without local number of items"
                )

        if distribute_average:
            n_items = self._gather_and_sum_metric_on_rank0(distribute_local_nitems)
        else:
            n_items = distribute_local_nitems

        aggregated_metrics = {}

        for metric_name, value in metric_dict.items():
            if distribute_average:
                aggregated_value = self._gather_and_sum_metric_on_rank0(value)
            else:
                aggregated_value = value

            if self.is_rank_zero and aggregated_value is not None:
                if distribute_average:
                    aggregated_value = (
                        aggregated_value / n_items if n_items else aggregated_value
                    )
                aggregated_metrics[metric_name] = aggregated_value
                if metric_name not in self.metrics:
                    self.metrics[metric_name] = []
                self.metrics[metric_name].append(aggregated_value)

        if self.is_rank_zero:
            log_step = step if step is not None else self.step_counter
            if self.is_wandb_enabled() and aggregated_metrics:
                self.wandb.log(aggregated_metrics, step=log_step, commit=commit)

            if increment_step:
                self.step_counter += 1

    def get_latest(
        self, metric_name: str, default: Optional[float] = None
    ) -> Optional[float]:
        """Get the most recent aggregated value of a metric (rank 0 only)."""
        if not self.is_rank_zero:
            return default
        values = self.metrics.get(metric_name)
        return values[-1] if values else default

    def get_running_average(
        self, metric_name: str, window_size: int = 10, default: Optional[float] = None
    ) -> Optional[float]:
        """Get the running average of aggregated values (rank 0 only)."""
        if not self.is_rank_zero:
            return default
        values = self.metrics.get(metric_name)
        if not values or len(values) == 0:
            return default
        relevant_values = values[-window_size:]
        return sum(relevant_values) / len(relevant_values)

    def get_all(self, metric_name: str) -> List[float]:
        """Get all historical aggregated values for a metric (rank 0 only)."""
        if not self.is_rank_zero:
            return []
        return self.metrics.get(metric_name, [])

    def plot_metric(
        self,
        metric_name: str,
        title: Optional[str] = None,
        window_size: Optional[int] = None,
        ax=None,
        log_on_wandb: bool = True,
        marker: Optional[str] = None,
        linestyle: Optional[str] = "-",
    ):
        """Plot a single metric (rank 0 only)."""
        if not self.is_rank_zero:
            return None

        values = self.get_all(metric_name)
        if not values:
            pylogger.warning(f"Rank 0: No data for metric: {metric_name}")
            return None

        standalone_plot = ax is None
        if standalone_plot:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        # Determine x-axis: use step counter or specific 'eval_at_step' if available
        if metric_name.startswith("eval_") and "eval_at_step" in self.metrics:
            x_values = self.get_all("eval_at_step")
            if len(x_values) != len(values):
                pylogger.warning(
                    f"Length mismatch for {metric_name} ({len(values)}) and eval_at_step ({len(x_values)}). Using step index."
                )
                x_values = np.arange(len(values))  # Fallback
        else:
            x_values = np.arange(len(values))

        ax.plot(
            x_values,
            values,
            alpha=0.3 if window_size else 1.0,
            label=metric_name,
            marker=marker,
            linestyle=linestyle,
        )

        if window_size and len(values) >= window_size:
            ma_x_start_idx = window_size - 1
            if len(x_values) > ma_x_start_idx:
                ma = np.convolve(
                    values, np.ones(window_size) / window_size, mode="valid"
                )
                ax.plot(
                    x_values[ma_x_start_idx:],
                    ma,
                    label=f"Moving Avg (win={window_size})",
                )

        ax.set_title(title or f"{metric_name} over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(useOffset=False)

        plot_name = title or metric_name

        if self.is_rank_zero and standalone_plot:
            fig.savefig(self.default_figures_dir() / f"{plot_name}_plot.svg")

        if log_on_wandb and self.is_wandb_enabled() and fig:
            try:
                self.wandb.log({plot_name: self.wandb.Image(fig)})

            except Exception as e:
                pylogger.error(f"Failed to log plot to W&B: {e}")
            finally:
                if standalone_plot:
                    plt.close(fig)

        if standalone_plot and not log_on_wandb:
            plt.close(fig)

        return fig if standalone_plot else ax

    def plot_metrics(
        self,
        metric_names: List[str],
        title: Optional[str] = None,
        window_size: Optional[int] = None,
        ax=None,
        log_on_wandb: bool = True,
    ):
        """Plot multiple metrics on the same graph (rank 0 only)."""
        if not self.is_rank_zero:
            return None

        standalone_plot = ax is None
        if standalone_plot:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        for metric_name in metric_names:
            values = self.get_all(metric_name)
            if not values:
                pylogger.warning(f"Rank 0: No data for metric: {metric_name}")
                continue

            x_values = np.arange(len(values))

            ax.plot(
                x_values, values, alpha=0.3 if window_size else 1.0, label=metric_name
            )

            if window_size and len(values) >= window_size:
                ma_x_start_idx = window_size - 1
                if len(x_values) > ma_x_start_idx:
                    ma = np.convolve(
                        values, np.ones(window_size) / window_size, mode="valid"
                    )
                    ax.plot(
                        x_values[ma_x_start_idx:],
                        ma,
                        label=f"{metric_name} MA (win={window_size})",
                    )

        ax.set_title(title or "Metrics over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_name = title or "_".join(metric_names)

        if self.is_rank_zero and standalone_plot:
            fig.savefig(self.default_figures_dir() / f"{plot_name}_plot.svg")

        if log_on_wandb and self.is_wandb_enabled() and fig:
            try:
                self.wandb.log({plot_name: self.wandb.Image(fig)})
            except Exception as e:
                pylogger.error(f"Failed to log plot to W&B: {e}")
            finally:
                if standalone_plot:
                    plt.close(fig)

        if standalone_plot and not log_on_wandb:
            plt.close(fig)

        return fig if standalone_plot else ax

    def plot_metric_grid(
        self,
        metrics_to_plot: Optional[Sequence[str]] = None,
        group_name: Optional[str] = None,
        window_size: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        log_on_wandb: bool = True,
        marker: Optional[str] = None,
        linestyle: Optional[str] = "-",
    ):
        """Plot multiple metrics in a grid layout (rank 0 only)."""
        if not self.is_rank_zero:
            return None, None

        if group_name and group_name in self.metric_groups:
            metrics_list = self.metric_groups[group_name]
        elif metrics_to_plot is not None:
            metrics_list = list(metrics_to_plot)
        else:
            metrics_list = list(self.metrics.keys())

        all_metrics_list = metrics_list
        metrics_list = [m for m in metrics_list if self.get_all(m)]
        n = len(metrics_list)
        if n == 0:
            pylogger.warning(
                f"Rank 0: No metrics with data to plot in grid. Provided metrics: {' '.join(all_metrics_list)}"
            )
            return None, None

        cols = min(3, n)
        rows = (n + cols - 1) // cols

        if figsize is None:
            figsize = (7 * cols, 5 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        for i, metric_name in enumerate(metrics_list):
            self.plot_metric(
                metric_name,
                title=metric_name,
                window_size=window_size,
                ax=axes_flat[i],
                log_on_wandb=False,  # Log the whole grid later
                marker=marker,
                linestyle=linestyle,
            )

        # Hide unused subplots
        for i in range(n, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        if title is not None:
            fig.suptitle(title)

        plt.tight_layout()

        plot_name = title or "_".join(metrics_list)
        if self.is_rank_zero:
            fig.savefig(self.default_figures_dir() / f"{plot_name}_grid.svg")

        if log_on_wandb and self.is_wandb_enabled():
            try:
                self.wandb.log({plot_name: self.wandb.Image(fig)})
            except Exception as e:
                pylogger.error(f"Failed to log grid plot to W&B: {e}")
            finally:
                plt.close(fig)
        elif not log_on_wandb:
            plt.close(fig)

        return fig, axes

    def format_for_tqdm(
        self, metrics: List[str], window_size: Optional[int] = None
    ) -> str:
        """Format aggregated metrics for TQDM (rank 0 only)."""
        if not self.is_rank_zero:
            return ""

        parts = []
        for metric in metrics:
            if window_size:  # Prioritize running average if window is specified
                value = self.get_running_average(metric, window_size)
                if value is not None:
                    parts.append(f"{metric}(avg{window_size})={value:.3f}")
                else:  # Fallback to latest if not enough data for window
                    value = self.get_latest(metric)
                    if value is not None:
                        parts.append(f"{metric}={value:.3f}")
            else:
                value = self.get_latest(metric)
                if value is not None:
                    parts.append(f"{metric}={value:.3f}")
        return ", ".join(parts)

    def log_hparams(
        self, hparams: Dict[str, Any], metrics: Optional[Dict[str, float]] = None
    ):
        """Log hyperparameters and aggregated final metrics to wandb (rank 0 only)."""
        n_items = self._gather_and_sum_metric_on_rank0(1.0)

        aggregated_final_metrics_rank0 = {}
        if metrics:
            for k, v in metrics.items():
                aggregated_sum = self._gather_and_sum_metric_on_rank0(v)
                if self.is_rank_zero:
                    aggregated_final_metrics_rank0[k] = (
                        aggregated_sum / n_items if n_items else aggregated_sum
                    )

        if self.is_wandb_enabled():
            self.wandb.config.update(hparams, allow_val_change=True)
            if aggregated_final_metrics_rank0:
                pylogger.info(
                    f"Logging final metrics: {aggregated_final_metrics_rank0}"
                )
                self.wandb.log(aggregated_final_metrics_rank0)
            else:
                pylogger.info("No final metrics provided or calculated to log.")

    def finalize(self):
        """Finish the W&B run, if active (rank 0 only)."""
        if self.is_wandb_enabled():
            self.wandb.finish()
            pylogger.info("W&B run finished.")
