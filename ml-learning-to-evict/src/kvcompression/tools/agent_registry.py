#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Agent Registry for Orchestrated Training

Core functionality for discovering and validating agents from sweep-based training runs.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from kvcompression.utils.s3_utils import get_s3_filesystem, ls_remote_folder

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for discovering and validating orchestrated agents."""

    def __init__(self, bucket: str):
        """
        Args:
            bucket: S3 bucket name (e.g., 'kvcompression' or 'ml-learning-to-evict')
        """
        self.base_path = f"{bucket}/agents"
        self.fs = get_s3_filesystem()

    def list_sweeps(self, group: str) -> List[str]:
        """List all available sweep names for a group."""
        group_path = f"{self.base_path}/{group}/"
        try:
            items = ls_remote_folder(group_path, self.fs)
            return sorted(set(item[len(group_path) :].split("/")[0] for item in items))
        except Exception as e:
            logger.error(f"Cannot access {group_path}: {e}")
            return []

    def discover_agent(
        self,
        group: str,
        sweep_name: str,
        layer: int,
        head: int,
        strict_unique: bool = True,
    ) -> str:
        """Find unique agent in (group, sweep_name, layer, head) directory."""
        remote_path = f"{self.base_path}/{group}/{sweep_name}/layer_{layer:06d}/kv_head_{head:03d}/"

        try:
            items = ls_remote_folder(remote_path, self.fs)
        except Exception:
            raise FileNotFoundError(f"No directory found at {remote_path}")

        # Filter for agent directories (exclude figures/)
        agent_dirs = [
            item.rstrip("/").split("/")[-1]
            for item in items
            if item.endswith("/") and not item.endswith("figures/")
        ]

        if not agent_dirs:
            raise FileNotFoundError(f"No agent found in {remote_path}")

        if len(agent_dirs) > 1:
            if strict_unique:
                raise ValueError(
                    f"Multiple agents found for layer {layer}, head {head}: {agent_dirs}. "
                    f"Only one agent per (layer, head) combination is allowed in strict mode."
                )
            # Use lexicographically latest (most recent timestamp)
            agent_dirs.sort()
            latest_agent = agent_dirs[-1]
            logger.warning(
                f"Multiple agents in {remote_path}: {agent_dirs}. Using latest: {latest_agent}"
            )
            return latest_agent

        return agent_dirs[0]

    def resolve_checkpoint_path(
        self,
        group: str,
        sweep_name: str,
        layer_folder: str,
        head_folder: str,
        agent_id: str,
    ) -> str:
        """Resolve which checkpoint to use (best_ckpt.pth or latest resume).

        Returns the relative checkpoint filename (e.g., 'best_ckpt.pth' or 'checkpoints/4000.pth').
        """
        agent_path = f"{self.base_path}/{group}/{sweep_name}/{layer_folder}/{head_folder}/{agent_id}"
        best_ckpt = f"{agent_path}/best_ckpt.pth"

        try:
            if self.fs.exists(best_ckpt):
                return "best_ckpt.pth"
        except Exception:
            pass

        # Fallback to latest resume checkpoint
        resume = self._find_latest_resume_checkpoint_remote(agent_path)
        if resume:
            return resume  # relative path like "checkpoints/4000.pth"

        raise FileNotFoundError(f"No checkpoint found for agent at {agent_path}")

    def _find_latest_resume_checkpoint_remote(self, agent_path: str) -> Optional[str]:
        """Find the latest resume checkpoint in the checkpoints/ directory.

        Args:
            agent_path: Full remote path to the agent directory.

        Returns:
            Relative checkpoint path (e.g., 'checkpoints/4000.pth') or None.
        """
        checkpoints_path = f"{agent_path}/checkpoints/"
        try:
            items = ls_remote_folder(checkpoints_path, self.fs)
        except Exception:
            return None

        resume_checkpoints = [f for f in items if f.endswith(".pth")]
        if not resume_checkpoints:
            return None

        # Sort numerically by the checkpoint step number (e.g., '1000' from '1000.pth')
        resume_checkpoints.sort(key=lambda p: int(Path(p).stem))
        latest = resume_checkpoints[-1]

        # Return relative path from agent directory
        filename = Path(latest).name
        return f"checkpoints/{filename}"

    def validate_agent_checkpoint(
        self,
        group: str,
        sweep_name: str,
        layer_folder: str,
        head_folder: str,
        agent_id: str,
    ) -> bool:
        """Validate that agent has a checkpoint file."""
        try:
            self.resolve_checkpoint_path(
                group, sweep_name, layer_folder, head_folder, agent_id
            )
            return True
        except FileNotFoundError:
            return False

    def discover_all_agents(
        self, group: str, sweep_name: str, strict_unique: bool = True
    ) -> Dict[Tuple[str, str], str]:
        """Discover all available agents in a sweep.

        Returns:
            Dict mapping (layer_folder, head_folder) tuples to agent_ids for all found agents
        """
        sweep_path = f"{self.base_path}/{group}/{sweep_name}/"

        try:
            items = ls_remote_folder(sweep_path, self.fs)
        except Exception as e:
            logger.error(f"Cannot access sweep path {sweep_path}: {e}")
            return {}

        # Process paths: slice from len(sweep_path) and split on '/'
        path_tuples = set()
        for item in items:
            if len(item) > len(sweep_path):
                relative_path = item[len(sweep_path) :]
                path_parts = relative_path.split("/")

                # We expect: layer_xxx/kv_head_xxx/agent_id (using original folder names)
                if len(path_parts) >= 3:
                    layer_folder = path_parts[0]
                    head_folder = path_parts[1]
                    agent_id = path_parts[2]

                    # Only process layer folders
                    if not layer_folder.startswith("layer_"):
                        continue

                    # Only process head folders (kv_head_ is the standard format)
                    if not head_folder.startswith("kv_head_"):
                        continue

                    # Skip if agent_id is empty or is a sub-directory (like 'figures')
                    if not agent_id or agent_id == "figures":
                        continue

                    path_tuples.add((layer_folder, head_folder, agent_id))

        path_tuples = sorted(path_tuples)

        # Organize agents: for each (layer, head), use the lexicographically latest agent_id
        agents = {}
        layer_head_agents = {}

        # Group by (layer_folder, head_folder) and keep folder names throughout
        for layer_folder, head_folder, agent_id in path_tuples:
            key = (layer_folder, head_folder)
            if key not in layer_head_agents:
                layer_head_agents[key] = []
            layer_head_agents[key].append(agent_id)

        # For each (layer_folder, head_folder), select the latest agent and validate
        for (layer_folder, head_folder), agent_ids in layer_head_agents.items():
            # Check for duplicates if strict_unique is enabled
            if strict_unique and len(agent_ids) > 1:
                layer_num = int(layer_folder.split("_")[-1])
                head_num = int(head_folder.split("_")[-1])
                raise ValueError(
                    f"Multiple agents found for layer {layer_num}, head {head_num}: {agent_ids}. "
                    f"Only one agent per (layer, head) combination is allowed in strict mode."
                )

            # Sort by agent_id and take the latest (most recent timestamp)
            agent_ids.sort()
            latest_agent = agent_ids[-1]

            # Validate that the agent has a checkpoint using original folder names
            if self.validate_agent_checkpoint(
                group, sweep_name, layer_folder, head_folder, latest_agent
            ):
                agents[(layer_folder, head_folder)] = latest_agent
            else:
                logger.warning(
                    f"Agent {latest_agent} for {layer_folder}/{head_folder} has no valid checkpoint"
                )

        return agents

    def print_discovery_summary(
        self,
        agents: Dict[Tuple[str, str], str],
        verbose: bool = False,
        expected_nheads: int = 4,
    ) -> None:
        """Print a nice summary of discovered agents."""
        if not agents:
            logger.info("No agents discovered.")
            return

        # Group by layer number (converted from folder name)
        layers = {}
        layer_agent_mapping = {}
        for (layer_folder, head_folder), agent_id in agents.items():
            layer_num = int(layer_folder.split("_")[-1])
            head_num = int(head_folder.split("_")[-1])

            if layer_num not in layers:
                layers[layer_num] = []
                layer_agent_mapping[layer_num] = {}
            layers[layer_num].append(head_num)
            layer_agent_mapping[layer_num][head_num] = agent_id

        logger.info(f"Discovered agents ({len(agents)} total):")
        for layer_num in sorted(layers.keys()):
            heads = sorted(layers[layer_num])
            head_str = ",".join(map(str, heads))

            # Check if all 4 heads are present (assuming GQA with 4 heads)
            expected_heads = list(range(expected_nheads))
            missing_heads = [h for h in expected_heads if h not in heads]

            if not missing_heads:
                status = "✓✓✓✓"
                detail = f"({len(heads)}/{expected_nheads} heads: {head_str})"
            else:
                status = "".join("✓" if h in heads else "✗" for h in expected_heads)
                missing_str = ",".join(map(str, missing_heads))
                detail = f"({len(heads)}/{expected_nheads} heads: {head_str} - missing heads {missing_str})"

            logger.info(f"Layer {layer_num}: {status} {detail}")

            # Show detailed mappings if verbose
            if verbose:
                for head_num in sorted(heads):
                    agent_id = layer_agent_mapping[layer_num][head_num]
                    logger.info(f"  kv_head_{head_num:03d} -> '{agent_id}'")
