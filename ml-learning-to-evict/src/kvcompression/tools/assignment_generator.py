#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Assignment Generator for Orchestrated Agents

Core functionality for generating composite configurations that automatically
discover and load agents from sweep-based training runs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from kvcompression.tools.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


def generate_composites(
    group: str,
    sweep_name: str,
    bucket: str,
    registry: AgentRegistry,
    layers: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
    expected_nagents: Optional[int] = None,
    strict_unique: bool = True,
) -> List[Dict[str, Any]]:
    """Generate composites list from sweep parameters.

    Each composite entry includes `remote_agents_dir` so the config is
    self-contained and bucket-independent, and `use_best_ckpt` based on
    which checkpoint is actually available.
    """
    remote_agents_dir = f"{bucket}/agents"

    # Auto-discovery mode: discover all available agents
    if layers is None or heads is None:
        discovered_agents = registry.discover_all_agents(
            group, sweep_name, strict_unique=strict_unique
        )

        # Check if sweep exists by verifying if we found any agents
        if len(discovered_agents) == 0:
            raise ValueError(
                f"No agents found for sweep '{sweep_name}' in group '{group}'. "
                f"Please check that the sweep exists and has trained agents."
            )

        registry.print_discovery_summary(discovered_agents, verbose=False)

        # Validate expected count if provided
        if expected_nagents is not None:
            if len(discovered_agents) != expected_nagents:
                raise ValueError(
                    f"Expected {expected_nagents} agents but found {len(discovered_agents)}"
                )

        # Generate composites from discovered agents
        composites = []
        for (layer_folder, head_folder), agent_id in discovered_agents.items():
            # Extract numeric values for the composite
            layer_num = int(layer_folder.split("_")[-1])
            head_num = int(head_folder.split("_")[-1])

            # Build the hierarchical agent_id using the actual folder names
            hierarchical_agent_id = (
                f"{group}/{sweep_name}/{layer_folder}/{head_folder}/{agent_id}"
            )

            # Resolve checkpoint to determine use_best_ckpt
            ckpt_path = registry.resolve_checkpoint_path(
                group, sweep_name, layer_folder, head_folder, agent_id
            )
            use_best_ckpt = ckpt_path == "best_ckpt.pth"

            composite = {
                "layer": layer_num,
                "head": head_num,
                "press": {
                    "_component_": "kvcompression.presses.sampler_agent_press.SamplerAgentPress",
                    "agent_id": hierarchical_agent_id,
                    "remote_agents_dir": remote_agents_dir,
                    "use_best_ckpt": use_best_ckpt,
                    "query_selection_mode": "no_agg",  # For GQA evaluation
                },
            }
            composites.append(composite)

    # Explicit mode: use provided layers and heads
    else:
        # Discover all agents first to get the actual folder names
        all_discovered_agents = registry.discover_all_agents(
            group, sweep_name, strict_unique=strict_unique
        )

        # Check if sweep exists by verifying if we found any agents
        if len(all_discovered_agents) == 0:
            raise ValueError(
                f"No agents found for sweep '{sweep_name}' in group '{group}'. "
                f"Please check that the sweep exists and has trained agents."
            )

        composites = []
        missing_agents = []

        for layer in layers:
            for head in heads:
                # Look for the agent in the discovered agents using the actual folder names
                found_agent = None
                for (
                    layer_folder,
                    head_folder,
                ), agent_id in all_discovered_agents.items():
                    layer_num = int(layer_folder.split("_")[-1])
                    head_num = int(head_folder.split("_")[-1])

                    if layer_num == layer and head_num == head:
                        found_agent = (layer_folder, head_folder, agent_id)
                        break

                if found_agent is None:
                    missing_agents.append((layer, head))
                    continue

                layer_folder, head_folder, agent_id = found_agent

                # Build the hierarchical agent_id using the actual folder names
                hierarchical_agent_id = (
                    f"{group}/{sweep_name}/{layer_folder}/{head_folder}/{agent_id}"
                )

                # Resolve checkpoint to determine use_best_ckpt
                ckpt_path = registry.resolve_checkpoint_path(
                    group, sweep_name, layer_folder, head_folder, agent_id
                )
                use_best_ckpt = ckpt_path == "best_ckpt.pth"

                composite = {
                    "layer": layer,
                    "head": head,
                    "press": {
                        "_component_": "kvcompression.presses.sampler_agent_press.SamplerAgentPress",
                        "agent_id": hierarchical_agent_id,
                        "remote_agents_dir": remote_agents_dir,
                        "use_best_ckpt": use_best_ckpt,
                        "query_selection_mode": "no_agg",  # For GQA evaluation
                    },
                }
                composites.append(composite)

        if missing_agents:
            logger.error(
                f"Missing agents for {len(missing_agents)} (layer, head) combinations:"
            )
            for layer, head in missing_agents:
                logger.error(f"  - Layer {layer}, Head {head}")
            raise ValueError(
                f"Cannot generate config: {len(missing_agents)} agents missing"
            )

    return sorted(composites, key=lambda x: (x["layer"], x["head"]))


def write_composites_file(
    group: str,
    sweep_name: str,
    bucket: str,
    registry: AgentRegistry,
    layers: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
    expected_nagents: Optional[int] = None,
    output_dir: Path = Path("."),
    strict_unique: bool = True,
) -> Path:
    """Write composites to a YAML file in config_composites/ directory."""
    composites = generate_composites(
        group=group,
        sweep_name=sweep_name,
        bucket=bucket,
        registry=registry,
        layers=layers,
        heads=heads,
        expected_nagents=expected_nagents,
        strict_unique=strict_unique,
    )

    # Create output file path in config_composites directory
    config_composites_dir = output_dir / "config_composites"
    config_composites_dir.mkdir(parents=True, exist_ok=True)
    output_file = config_composites_dir / f"{sweep_name}.yaml"

    # Wrap composites in the expected format
    output_data = {"assignments": composites}

    # Write composites to file
    with open(output_file, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    return output_file
