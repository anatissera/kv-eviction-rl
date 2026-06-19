#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
CLI wrapper for assignment generator (Typer version)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import typer
import yaml

from kvcompression.cli_logging import setup_cli_logging
from kvcompression.tools.agent_registry import AgentRegistry
from kvcompression.tools.assignment_generator import write_composites_file

setup_cli_logging()
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="assignment_generator",
    help="Generate composite configurations from sweep parameters",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def main():
    """Generate composite configurations from sweep parameters."""


def _parse_layer_head_params(
    layers_str: Optional[str], heads_str: Optional[str]
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Parse layer and head parameters from CLI strings."""
    layer_list = None
    head_list = None

    if layers_str is not None:
        try:
            layer_list = [int(x.strip()) for x in layers_str.split(",")]
        except ValueError as e:
            typer.echo(f"Error parsing layers: {e}", err=True)
            typer.echo("Layers must be comma-separated integers", err=True)
            raise typer.Exit(1)

        # Validate ranges
        if any(layer < 0 for layer in layer_list):
            typer.echo("Error: Layer indices must be non-negative", err=True)
            raise typer.Exit(1)

    if heads_str is not None:
        try:
            head_list = [int(x.strip()) for x in heads_str.split(",")]
        except ValueError as e:
            typer.echo(f"Error parsing heads: {e}", err=True)
            typer.echo("Heads must be comma-separated integers", err=True)
            raise typer.Exit(1)

        # Validate ranges
        if any(head < 0 for head in head_list):
            typer.echo("Error: Head indices must be non-negative", err=True)
            raise typer.Exit(1)

    return layer_list, head_list


@app.command()
def write_composites(
    sweep_names: List[str] = typer.Argument(
        help="One or more sweep names from training"
    ),
    bucket: str = typer.Option(
        ...,
        "--bucket",
        "-b",
        help="S3 bucket name (e.g., 'kvcompression')",
    ),
    layers: Optional[str] = typer.Option(
        None,
        "--layers",
        "-l",
        help="Comma-separated layer indices (e.g., '10,15'). Auto-discover if not provided.",
    ),
    heads: Optional[str] = typer.Option(
        None,
        "--heads",
        "-h",
        help="Comma-separated head indices (e.g., '0,1,2,3'). Auto-discover if not provided.",
    ),
    group: str = typer.Option(
        "grouped", "--group", "-g", help="Agent group (e.g., 'grouped')"
    ),
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        "-o",
        help="Output directory for composites file",
    ),
    expected_nagents: Optional[int] = typer.Option(
        None,
        "--expected-nagents",
        "-n",
        help="Expected total number of agents (for validation)",
    ),
    strict_unique: bool = typer.Option(
        True,
        "--strict-unique/--no-strict-unique",
        help="Error if multiple agents exist for same (layer, head). Default: True",
    ),
):
    """Write composites to config_composites/ for from_file workflow.

    Creates files named '{sweep_name}.yaml' in the config_composites/ directory
    that can be referenced by evaluation configs using from_file: true.

    Examples:
        # Auto-discover all agents in multiple sweeps
        python assignment_generator.py write-composites sweep1 sweep2 sweep3 --bucket kvcompression

        # Single sweep
        python assignment_generator.py write-composites my_sweep --bucket kvcompression

        # Explicit layers and heads for multiple sweeps
        python assignment_generator.py write-composites sweep1 sweep2 --bucket kvcompression --layers "10,15" --heads "0,1,2,3"

        # With expected count validation per sweep
        python assignment_generator.py write-composites sweep1 sweep2 --bucket kvcompression --expected-nagents 8
    """
    # Parse parameters
    layer_list, head_list = _parse_layer_head_params(layers, heads)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create registry once for all sweeps
    registry = AgentRegistry(bucket=bucket)

    # Process each sweep
    for sweep_name in sweep_names:
        # Log what we're doing
        if layer_list is not None and head_list is not None:
            num_composites = len(layer_list) * len(head_list)
            typer.echo(
                f"Generating composites for sweep '{sweep_name}': {len(layer_list)} layers × {len(head_list)} heads = {num_composites} composites"
            )
        else:
            typer.echo(
                f"Auto-discovering agents in sweep '{sweep_name}' (group: '{group}')"
            )

        # Generate composites file
        try:
            output_file = write_composites_file(
                group=group,
                sweep_name=sweep_name,
                bucket=bucket,
                registry=registry,
                layers=layer_list,
                heads=head_list,
                expected_nagents=expected_nagents,
                output_dir=output_dir,
                strict_unique=strict_unique,
            )

            # Success message
            typer.echo(f"✅ Generated composites file: {output_file}")

            # Read back to show count
            with open(output_file, "r") as f:
                data = yaml.safe_load(f)
            composites = data["assignments"]
            typer.echo(f"📊 Total composites: {len(composites)}")

        except ValueError as e:
            typer.echo(
                f"❌ Composites generation failed for '{sweep_name}': {e}", err=True
            )
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"❌ Unexpected error for '{sweep_name}': {e}", err=True)
            raise typer.Exit(1)

    # Usage instructions (show once at the end)
    typer.echo("\n📋 Usage instructions:")
    typer.echo("Reference these files in your evaluation config:")
    typer.echo("   compression_configs:")
    for sweep_name in sweep_names:
        typer.echo(f"     {sweep_name}:")
        typer.echo("       from_file: true")
        typer.echo(
            f"       file_name: {sweep_name}  # optional, defaults to config name"
        )


if __name__ == "__main__":
    app()
