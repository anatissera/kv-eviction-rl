#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
CLI wrapper for agent registry (Typer version)
"""

import logging

import typer

from kvcompression.cli_logging import setup_cli_logging
from kvcompression.tools.agent_registry import AgentRegistry

setup_cli_logging()
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="agent_registry",
    help="Discover and validate agents from sweep-based training runs",
    add_completion=False,
)


@app.command()
def list_sweeps(
    group: str = typer.Argument(
        help="Agent group (e.g., 'grouped')",
        default="grouped",
    ),
    bucket: str = typer.Option(
        ...,
        "--bucket",
        "-b",
        help="S3 bucket name (e.g., 'kvcompression')",
    ),
):
    """List all available sweep names for a group.

    This command queries the remote storage to find all sweep directories
    for the specified agent group.

    Examples:
        # List all sweeps for the 'grouped' agent group
        python agent_registry.py list-sweeps grouped --bucket kvcompression

        # List sweeps for a custom group
        python agent_registry.py list-sweeps my_custom_group --bucket kvcompression
    """
    registry = AgentRegistry(bucket=bucket)

    typer.echo(
        f"🔍 Searching for sweeps in group '{group}' (in base path: '{registry.base_path}')..."
    )
    sweeps = registry.list_sweeps(group)

    if sweeps:
        typer.echo(f"✅ Found {len(sweeps)} sweep(s) in group '{group}':")
        for sweep in sweeps:
            typer.echo(f"  📁 {sweep}")
    else:
        typer.echo(f"❌ No sweeps found in group '{group}'")
        raise typer.Exit(1)


@app.command()
def discover(
    sweep_name: str = typer.Argument(help="Sweep name to discover agents for"),
    group: str = typer.Argument("grouped", help="Agent group (e.g., 'grouped')"),
    bucket: str = typer.Option(
        ...,
        "--bucket",
        "-b",
        help="S3 bucket name (e.g., 'kvcompression')",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed agent mappings"
    ),
):
    """Discover and display all available agents in a sweep.

    This command provides a quick overview of what agents are available
    in a sweep, showing the layer-by-layer coverage with visual indicators.

    Examples:
        # Discover all agents in a sweep
        python agent_registry.py discover my_sweep --bucket kvcompression

        # Discover agents with detailed mappings
        python agent_registry.py discover my_sweep --bucket kvcompression --verbose

        # Discover agents in a custom group
        python agent_registry.py discover my_sweep custom_group --bucket kvcompression
    """
    registry = AgentRegistry(bucket=bucket)

    typer.echo(f"🔍 Discovering agents in sweep '{sweep_name}' (group: '{group}')")

    discovered_agents = registry.discover_all_agents(group, sweep_name)

    if not discovered_agents:
        typer.echo("❌ No agents found in sweep", err=True)
        raise typer.Exit(1)

    # Print discovery summary
    registry.print_discovery_summary(discovered_agents, verbose=verbose)

    typer.echo(f"✅ Discovery complete: {len(discovered_agents)} agents found")


@app.command()
def validate_sweep(
    sweep_name: str = typer.Argument(help="Sweep name to validate"),
    expected_layers: str = typer.Option(
        None,
        "--expected-layers",
        "-l",
        help="Comma-separated expected layer indices (e.g., '10,15'). If provided without heads, warns user.",
    ),
    expected_heads: str = typer.Option(
        None,
        "--expected-heads",
        "-h",
        help="Comma-separated expected head indices (e.g., '0,1,2,3'). If provided without layers, warns user.",
    ),
    expected_nagents: int = typer.Option(
        None,
        "--expected-nagents",
        "-n",
        help="Expected total number of agents (always validated if provided)",
    ),
    group: str = typer.Argument("grouped", help="Agent group (e.g., 'grouped')"),
    bucket: str = typer.Option(
        ...,
        "--bucket",
        "-b",
        help="S3 bucket name (e.g., 'kvcompression')",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed validation output"
    ),
):
    """Validate agents in a sweep.

    Simplified logic:
    - If both layers and heads specified: validate exact combinations
    - If only layers or only heads specified: warn user and auto-discover
    - If expected_nagents provided: always validate total count
    - Otherwise: auto-discover all agents

    Examples:
        # Auto-discover all agents
        python agent_registry.py validate-sweep my_sweep --bucket kvcompression

        # Auto-discover with count validation
        python agent_registry.py validate-sweep my_sweep --bucket kvcompression --expected-nagents 16

        # Validate specific combinations
        python agent_registry.py validate-sweep my_sweep --bucket kvcompression -l "10,15" -h "0,1,2,3"
    """
    registry = AgentRegistry(bucket=bucket)

    # Parse inputs if provided
    layers = None
    heads = None

    if expected_layers:
        try:
            layers = [int(x.strip()) for x in expected_layers.split(",")]
            if any(layer < 0 for layer in layers):
                typer.echo("❌ Error: Layer indices must be non-negative", err=True)
                raise typer.Exit(1)
        except ValueError as e:
            typer.echo(f"❌ Error parsing layers: {e}", err=True)
            raise typer.Exit(1)

    if expected_heads:
        try:
            heads = [int(x.strip()) for x in expected_heads.split(",")]
            if any(head < 0 for head in heads):
                typer.echo("❌ Error: Head indices must be non-negative", err=True)
                raise typer.Exit(1)
        except ValueError as e:
            typer.echo(f"❌ Error parsing heads: {e}", err=True)
            raise typer.Exit(1)

    # Warn for incomplete specification
    if layers and not heads:
        typer.echo(
            "⚠️ Warning: Layers specified without heads - cannot validate specific combinations. Auto-discovering all agents.",
            err=True,
        )
        layers = None
    elif heads and not layers:
        typer.echo(
            "⚠️ Warning: Heads specified without layers - cannot validate specific combinations. Auto-discovering all agents.",
            err=True,
        )
        heads = None

    # Auto-discovery mode (default or forced by incomplete spec)
    if layers is None or heads is None:
        typer.echo(
            f"🔍 Auto-discovering agents in sweep '{sweep_name}' (group: '{group}')"
        )

        discovered_agents = registry.discover_all_agents(group, sweep_name)

        if not discovered_agents:
            typer.echo("❌ No agents found in sweep", err=True)
            raise typer.Exit(1)

        registry.print_discovery_summary(discovered_agents, verbose=verbose)

        # Always validate expected count if provided
        if expected_nagents is not None:
            if len(discovered_agents) != expected_nagents:
                typer.echo(
                    f"❌ Expected {expected_nagents} agents but found {len(discovered_agents)}",
                    err=True,
                )
                raise typer.Exit(1)
            else:
                typer.echo(f"✅ Found expected {expected_nagents} agents")

        typer.echo("🎉 Agent discovery completed successfully!")
        return

    # Explicit validation mode
    typer.echo(f"🔍 Validating sweep '{sweep_name}' in group '{group}'")
    typer.echo(f"📊 Expected layers: {layers}")
    typer.echo(f"📊 Expected heads: {heads}")
    typer.echo(f"🎯 Total combinations to check: {len(layers) * len(heads)}")

    # Check all layer-head combinations
    missing = []
    present = []

    for layer in layers:
        for head in heads:
            try:
                agent_id = registry.discover_agent(group, sweep_name, layer, head)
                # Validate checkpoint exists
                layer_folder = f"layer_{layer:06d}"
                head_folder = f"kv_head_{head:03d}"
                if registry.validate_agent_checkpoint(
                    group, sweep_name, layer_folder, head_folder, agent_id
                ):
                    present.append((layer, head))
                else:
                    missing.append((layer, head))
            except FileNotFoundError:
                missing.append((layer, head))

    # Print results
    total_expected = len(layers) * len(heads)
    total_present = len(present)
    completion_rate = total_present / total_expected if total_expected > 0 else 0

    typer.echo(
        f"\n✅ Present: {total_present}/{total_expected} ({completion_rate:.1%})"
    )

    if missing:
        typer.echo(f"❌ Missing {len(missing)} agents:")
        for layer, head in missing:
            typer.echo(f"  🚫 Layer {layer}, Head {head}")

    if verbose and present:
        typer.echo(f"\n✅ Present {total_present} agents:")
        for layer, head in present:
            typer.echo(f"  ✓ Layer {layer}, Head {head}")

    # Always validate expected count if provided
    if expected_nagents is not None:
        if total_present != expected_nagents:
            typer.echo(
                f"❌ Expected {expected_nagents} agents but found {total_present}",
                err=True,
            )
            raise typer.Exit(1)
        else:
            typer.echo(f"✅ Found expected {expected_nagents} agents")

    # Exit with error if incomplete
    if completion_rate < 1.0:
        typer.echo("❌ Sweep validation failed", err=True)
        raise typer.Exit(1)
    else:
        typer.echo("🎉 Sweep validation passed!")


if __name__ == "__main__":
    app()
