#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
from torchtune import utils

from kvcompression import PROJECT_ROOT
from kvcompression.presses.base_press import BasePress
from kvcompression.utils.s3_utils import (
    download_file,
    get_s3_filesystem,
    ls_remote_folder,
)
from kvcompression.utils.utils import instantiate_agent_from_ckpt

LOCAL_AGENTS_DIR = PROJECT_ROOT / "agents"
REMOTE_AGENTS_DIR = os.environ.get("KV_REMOTE_AGENTS_DIR")

logger = utils.get_logger("INFO")
log_rank_zero = utils.log_rank_zero


class SamplerAgentPress(BasePress):
    """
    Compression strategy using a trained RL agent to rank KV cache entries.

    This press loads a trained agent checkpoint and uses it to score and rank
    tokens for KV cache compression. Supports automatic checkpoint download
    from remote storage if not available locally.
    """

    def __init__(
        self,
        ckpt_path: Optional[Path] = None,
        agent_id: Optional[str] = None,
        use_best_ckpt: Optional[bool] = None,
        local_agents_dir: Path = LOCAL_AGENTS_DIR,
        remote_agents_dir: str = REMOTE_AGENTS_DIR,
        query_selection_mode: str = "first",
        n_sinks: int = 0,
        n_running_window: int = 0,
        ckpt_model_key: str = "ema_model_state_dict",
    ):
        """
        Initialize the sampler agent press.

        Args:
            ckpt_path: Direct path to checkpoint file. If provided, agent_id is ignored.
            agent_id: Agent identifier for locating checkpoint in agents directory.
            use_best_ckpt: Checkpoint selection mode:
                - True: Use only 'best_ckpt.pth'
                - False: Use only latest resume checkpoint
                - None: Prefer best_ckpt.pth, fallback to latest resume
            local_agents_dir: Base directory for local agent checkpoints.
            remote_agents_dir: Base path in remote storage for agent checkpoints.
            query_selection_mode: How to handle GQA query groups:
                - "first": Use first query from each KV head group
                - "average": Average queries (not implemented)
                - "no_agg": Pass all queries to agent
            n_sinks: Number of initial tokens to always keep (attention sinks).
            n_running_window: Number of recent tokens to always keep.
            ckpt_model_key: Key in checkpoint dict containing model state.
        """
        super().__init__()
        self.world_size, self.rank = utils.get_world_size_and_rank()

        if ckpt_path is None and agent_id is None:
            raise ValueError(
                "Impossible to identify agent to use without ckpt_path or agent_id"
            )

        local_agents_dir = Path(local_agents_dir)

        if ckpt_path:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                raise ValueError(f"Checkpoint path provided but not valid: {ckpt_path}")
        else:
            if remote_agents_dir is None:
                raise EnvironmentError(
                    "Environment variable 'KV_REMOTE_AGENTS_DIR' must be set when using agent_id "
                    "without a direct ckpt_path"
                )
            relative_ckpt_path = self._find_checkpoint_path(
                agent_id=agent_id,
                local_agents_dir=local_agents_dir,
                remote_agents_dir=remote_agents_dir,
                use_best_ckpt=use_best_ckpt,
            )
            ckpt_path = local_agents_dir / agent_id / relative_ckpt_path

        if self.rank == 0 and not ckpt_path.exists():
            ckpt_remote_path = f"{remote_agents_dir}/{agent_id}/{relative_ckpt_path}"
            log_rank_zero(
                logger=logger,
                msg=f"Agent checkpoint '{ckpt_path.name}' for '{agent_id}' not found locally. "
                f"Attempting download: '{ckpt_remote_path}' -> '{ckpt_path}'",
            )
            download_file(
                remote_source_path=ckpt_remote_path,
                local_destination=ckpt_path,
            )

        log_rank_zero(
            logger=logger,
            msg=f"Agent '{agent_id}' found. Attempting instantiation from '{ckpt_path}'.",
        )

        if dist.is_initialized():
            dist.barrier()

        self.agent = instantiate_agent_from_ckpt(
            checkpoint_path=ckpt_path, model_key=ckpt_model_key
        )
        if query_selection_mode not in {"first", "average", "no_agg"}:
            raise ValueError(f"Unsupported query_selection_mode={query_selection_mode}")
        self.query_selection_mode = query_selection_mode
        self.n_sinks = n_sinks
        self.n_running_window = n_running_window

    def _find_latest_remote_resume_ckpt(self, remote_files: List[str]) -> Optional[str]:
        """Finds the path of the latest resume checkpoint from a list of remote files."""
        resume_checkpoints = [
            f for f in remote_files if "/checkpoints/" in f and f.endswith(".pth")
        ]
        if not resume_checkpoints:
            return None

        # Sort numerically by the checkpoint step number (e.g., '1000' from '1000.pth')
        resume_checkpoints.sort(key=lambda p: int(Path(p).stem))
        return resume_checkpoints[-1]

    def _find_checkpoint_path(
        self,
        agent_id: str,
        local_agents_dir: Path,
        remote_agents_dir: str,
        use_best_ckpt: Optional[bool],
    ) -> Path:
        """
        Determines the target checkpoint path by inspecting the remote repository first.

        Args:
            agent_id: The identifier for the agent.
            local_agents_dir: The base local directory for agents.
            remote_agents_dir: The base remote directory for agents.
            use_best_ckpt: Controls checkpoint selection logic:
                - True: Only 'best_ckpt.pth' is considered.
                - False: Only the latest resume checkpoint is considered.
                - None: 'best_ckpt.pth' is preferred, with fallback to the latest resume checkpoint.

        Returns:
            The expected local Path to the chosen checkpoint file.
        """
        fs = get_s3_filesystem()
        remote_agent_path = f"{remote_agents_dir}/{agent_id}"

        try:
            remote_files = ls_remote_folder(remote_agent_path, fs)
        except Exception as e:
            raise IOError(
                f"Could not list remote directory '{remote_agent_path}'. "
                f"Please ensure the agent_id is correct and you have S3 access. Original error: {e}"
            ) from e

        remote_best_ckpt = f"{remote_agent_path}/best_ckpt.pth"
        has_best_ckpt = remote_best_ckpt in remote_files
        latest_resume_ckpt = self._find_latest_remote_resume_ckpt(remote_files)

        target_remote_path = None
        if use_best_ckpt is True:
            if has_best_ckpt:
                target_remote_path = remote_best_ckpt
            else:
                raise FileNotFoundError(
                    f"use_best_ckpt=True, but '{remote_best_ckpt}' not found for agent '{agent_id}'."
                )
        elif use_best_ckpt is False:
            if latest_resume_ckpt:
                target_remote_path = latest_resume_ckpt
            else:
                raise FileNotFoundError(
                    f"use_best_ckpt=False, but no resume checkpoints found in '{remote_agent_path}/checkpoints/'."
                )
        elif use_best_ckpt is None:  # Default behavior
            if has_best_ckpt:
                target_remote_path = remote_best_ckpt
            elif latest_resume_ckpt:
                log_rank_zero(
                    logger=logger,
                    msg=f"best_ckpt.pth not found for agent '{agent_id}'. "
                    f"Falling back to latest resume checkpoint: '{Path(latest_resume_ckpt).name}'.",
                )
                target_remote_path = latest_resume_ckpt
            else:
                raise FileNotFoundError(
                    f"No suitable checkpoint found for agent '{agent_id}'. "
                    f"Neither 'best_ckpt.pth' nor any resume checkpoints exist in '{remote_agent_path}'."
                )

        relative_path = Path(target_remote_path).relative_to(remote_agent_path)
        return relative_path

    def requires_attention_weights(self) -> bool:
        return self.agent.requires_attention_weights()

    def _apply_sink_and_window(
        self, kv_ranking: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """
        Overrides an agent's token ranking to enforce sink and running window tokens.

        Args:
            kv_ranking: [B, n_heads, seq_len] tensor where values are token indices,
                        sorted by the agent's preference (e.g., kv_ranking[..., 0] is
                        the ID of the most important token).
            seq_len: The sequence length.

        Returns:
            A new ranking tensor with priority tokens placed at the beginning.
        """
        if self.n_sinks == 0 and self.n_running_window == 0:
            return kv_ranking

        device = kv_ranking.device

        # 1. Define the set of high-priority token indices.
        n_sinks = min(self.n_sinks, seq_len)
        n_running_window = min(self.n_running_window, seq_len)
        sink_indices = torch.arange(n_sinks, device=device)
        window_start_pos = max(n_sinks, seq_len - n_running_window)
        window_indices = torch.arange(window_start_pos, seq_len, device=device)
        priority_indices = torch.cat([sink_indices, window_indices])

        # 2. Convert the agent's ranking to per-token scores via argsort.
        agent_ranks = kv_ranking.argsort(dim=-1)

        # Higher score = more important. Best rank (0) gets highest score (seq_len).
        scores = seq_len - agent_ranks

        # 3. Override scores for priority tokens to be higher than any agent score.
        scores[..., priority_indices] = seq_len + 1

        # 4. Generate the final ranking by sorting on composite scores.
        new_ranking = torch.argsort(scores, dim=-1, descending=True)

        return new_ranking

    def sort(
        self,
        hidden_states: Optional[torch.Tensor],
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Rank KV cache entries using the trained agent.

        The agent scores each token position, and tokens are ranked by their
        scores in descending order. Optionally enforces sink tokens and a
        running window of recent tokens to always be kept.

        Args:
            hidden_states: Hidden states from the model, shape [B, S, D].
            keys: Key tensors, shape [B, num_kv_heads, S, head_dim].
            values: Value tensors, shape [B, num_kv_heads, S, head_dim].
            queries: Query tensors, shape [B, num_q_heads, S, head_dim].
            attention_weights: Pre-computed attention weights (unused).

        Returns:
            Ranking tensor of shape [B, num_kv_heads, S] where values are
            token indices sorted by importance (index 0 = most important).
        """
        batch_size, n_heads, seq_len, head_dim = keys.shape
        num_q_heads = queries.shape[1]

        if n_heads != num_q_heads:
            if self.query_selection_mode == "first":
                queries_per_kv = num_q_heads // n_heads
                # Reshape to group queries by KV heads, then take first from each group
                queries_grouped = queries.view(
                    batch_size, n_heads, queries_per_kv, seq_len, head_dim
                )
                grouped_queries = queries_grouped[
                    :, :, 0, :, :
                ]  # [B, n_heads, seq_len, head_dim]

                # Since we are reducing the group, consider the num_q_heads to be 1
                num_q_heads = 1
            elif self.query_selection_mode == "no_agg":
                # Pass all queries grouped by KV heads - let the agent handle them
                queries_per_kv = num_q_heads // n_heads
                grouped_queries = queries.view(
                    batch_size, n_heads, queries_per_kv, seq_len, head_dim
                )
            else:
                raise NotImplementedError(
                    f"Not yet implemented: {self.query_selection_mode=}"
                )
        else:
            # 1:1 mapping, just add query group dimension
            grouped_queries = queries.unsqueeze(2)  # [B, n_heads, 1, seq_len, head_dim]

        scores = self.agent(
            {
                "hidden_states": hidden_states.view(
                    batch_size, seq_len, hidden_states.shape[-1]
                )
                if hidden_states is not None
                else None,
                "keys": keys.view(
                    -1, seq_len, head_dim
                ),  # [B*n_heads, seq_len, head_dim]
                "values": values.view(
                    -1, seq_len, head_dim
                ),  # [B*n_heads, seq_len, head_dim]
                "queries": grouped_queries.view(
                    -1, num_q_heads, seq_len, head_dim
                ),  # [B*n_heads, queries_per_kv, seq_len, head_dim]
                "seq_lengths": torch.tensor([seq_len], device=keys.device).expand(
                    batch_size
                ),
            }
        )["scores"]

        kv_ranking = scores.argsort(dim=-1, descending=True)

        if self.n_sinks > 0 or self.n_running_window > 0:
            kv_ranking = self._apply_sink_and_window(kv_ranking, seq_len)

        return kv_ranking.view(batch_size, n_heads, seq_len)

    def supports_gqa(self) -> bool:
        return self.agent.supports_gqa()
