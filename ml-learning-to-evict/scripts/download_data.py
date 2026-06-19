#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Unified data download script for KV compression project.

This script provides a centralized way to download GQA format data from S3
with support for selective layer/head downloads and split-based filtering.
Optimized to use AWS CLI's include/exclude patterns for maximum efficiency.

Usage:
    # Download default ruler dataset with validation split
    python scripts/download_data.py download --splits val.txt

    # Download with selective layers and heads (optimized with patterns)
    python scripts/download_data.py download --layers 10,15 --heads 0,1,2,3 --splits val.txt

    # Ultra-efficient single layer/head download
    python scripts/download_data.py download --layers 10 --heads 0 --splits val.txt

    # Download different dataset under gqa/
    python scripts/download_data.py download --dataset_name my_custom_dataset --model_path Qwen2.5-7B-Instruct/temperature_0.00

    # Download with different model
    python scripts/download_data.py download --dataset_name simonjegou_ruler --model_path Llama-3-8B-Instruct/temperature_0.00

    # Custom target directory
    python scripts/download_data.py download --target_dir /path/to/custom/data

    # Preview commands without executing (dry run)
    python scripts/download_data.py download --dataset_name test --layers 10 --heads 0,1 --dry_run

    # Full example with all parameters
    python scripts/download_data.py download \\
        --dataset_name experiment_v2 \\
        --model_path Qwen2.5-7B-Instruct/temperature_0.00 \\
        --layers 10,15,20 \\
        --heads 0,1,2,3 \\
        --splits val.txt,train.txt \\
        --target_dir custom_data \\
        --include_tokens false \\
        --dry_run

Optimizations:
    - Uses a single recursive `aws s3 cp` command.
    - Generates highly specific --include patterns to filter data on the S3 side,
      minimizing network traffic and API calls.
    - Avoids downloading unwanted samples when splits are specified.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Set

import fire


class DataDownloader:
    """Unified data downloader for KV compression datasets."""

    def __init__(self):
        bucket = os.environ.get("KVCOMPRESSION_S3_BUCKET")
        if not bucket:
            raise ValueError(
                "Environment variable 'KVCOMPRESSION_S3_BUCKET' must be set. "
                "Add it to your .env file, e.g.: KVCOMPRESSION_S3_BUCKET=ml-learning-to-evict"
            )
        self.base_s3_path = f"s3://{bucket}/data"

    def download(
        self,
        dataset_name: str = "simonjegou_ruler",
        model_path: str = "Qwen2.5-7B-Instruct/temperature_0.00",
        layers: Optional[str] = None,
        heads: Optional[str] = None,
        include_tokens: bool = True,
        sample_level_only: bool = False,
        target_dir: str = "/dev/shm/data",
        dry_run: bool = False,
        quiet: bool = True,
        **kwargs,
    ) -> None:
        """
        Main download entry point. Dispatches to the correct handler.

        Args:
            dataset_name: Dataset name within gqa/ (e.g., "simonjegou_ruler").
            model_path: Model path within dataset (e.g., "Qwen2.5-7B-Instruct/temperature_0.00").
            layers: Comma-separated layer indices.
            heads: Comma-separated head indices.
            include_tokens: Whether to download token files.
            sample_level_only: If True, download only sample-level files (no layer/head activations).
            target_dir: Local target directory.
            dry_run: If True, only print commands without executing them.
            quiet: If True, downloads in quiet mode.
        """
        if len(kwargs) > 0:
            print("Unknown options: ", kwargs)
            return

        self._download_gqa_data(
            dataset_name,
            model_path,
            layers,
            heads,
            include_tokens,
            sample_level_only,
            target_dir,
            dry_run,
            quiet=quiet,
        )

    def _download_gqa_data(
        self,
        dataset_name: str,
        model_path: str,
        layers: Optional[str],
        heads: Optional[str],
        include_tokens: bool,
        sample_level_only: bool,
        target_dir: str,
        dry_run: bool,
        quiet: bool,
    ) -> None:
        """
        Download GQA format data with selective filtering using optimized patterns.

        This method uses a single `aws s3 cp --recursive` command with precise
        --include patterns for maximum efficiency, significantly reducing S3 API calls
        and network round trips compared to individual file downloads.
        """
        print(f"Preparing to download GQA data for dataset: {dataset_name}")

        # Log what will be downloaded
        if sample_level_only:
            print("📄 Sample-level-only mode: ENABLED")
            print(
                "   Will download: all_text.txt, all_tokens.safetensors, generated_text.txt"
            )
            if include_tokens:
                print("   Will download: prompt_ntokens.safetensors")
            print("   Will NOT download: layer/head activation data")
        else:
            print("📄 Sample-level-only mode: DISABLED")
            print("   Will download: sample-level files + layer/head activation data")
            if layers is not None:
                print(f"   Layers: {layers}")
            else:
                print("   Layers: ALL")
            if heads is not None:
                print(f"   Heads: {heads}")
            else:
                print("   Heads: ALL")

        # Parse parameters
        layer_indices = self._parse_indices(layers) if layers is not None else None
        head_indices = self._parse_indices(heads) if heads is not None else None

        # Handle Fire's string boolean parsing
        if isinstance(include_tokens, str):
            include_tokens = include_tokens.lower() in ("true", "1", "yes", "on")

        # Setup paths
        remote_base = f"{self.base_s3_path}/gqa_safetensors/{dataset_name}/{model_path}"
        local_base = (
            Path(target_dir) / "gqa_safetensors" / dataset_name / Path(model_path)
        )

        # Always download all samples and split files
        print("Downloading all samples and split files (no filtering by splits).")

        # Build and execute the main download command
        self._download_with_patterns(
            remote_base,
            local_base,
            None,  # No sample filtering
            layer_indices,
            head_indices,
            include_tokens,
            sample_level_only,
            dry_run,
            quiet=quiet,
        )

        print(
            f"\n✅ GQA data download process initiated. Target directory: {local_base}"
        )

        # Perform sanity check to verify all expected files are present
        if not dry_run:
            self._verify_download_completeness(
                local_base,
                None,  # No sample filtering
                layer_indices,
                head_indices,
                include_tokens,
                sample_level_only,
                None,  # No split filtering
            )

    def _parse_indices(self, indices_str) -> Set[int]:
        """Parse comma-separated indices string into a set of integers."""
        if indices_str is None:
            return set()
        elif isinstance(indices_str, int):
            return {indices_str}
        elif isinstance(indices_str, str):
            return {int(idx.strip()) for idx in indices_str.split(",")}
        elif isinstance(indices_str, tuple):
            return {int(idx) for idx in indices_str}
        else:
            raise ValueError(
                f"Invalid indices format: expected a string, int, or tuple, got {type(indices_str)}"
            )

    def _get_sample_ids_from_splits(
        self, remote_base: str, local_base: Path, split_files: List[str], dry_run: bool
    ) -> Set[str]:
        """Download split files and extract sample IDs from them."""
        sample_ids = set()
        if dry_run:
            print("  (dry run - skipping reading of split files from S3)")
            return sample_ids

        for split_file in split_files:
            split_path = local_base / split_file
            try:
                # Download split file to target directory
                self._run_aws_command(
                    ["s3", "cp", f"{remote_base}/{split_file}", str(split_path)],
                    dry_run=False,
                )

                # Read the downloaded file
                with open(split_path, "r") as f:
                    for line in f:
                        sample_id = line.strip()
                        if sample_id:
                            sample_ids.add(sample_id)

            except subprocess.CalledProcessError:
                print(
                    f"Warning: Could not download split file {split_file}. It may not exist on remote."
                )
            except FileNotFoundError:
                print(f"Warning: Split file {split_file} not found after download.")

        return sample_ids

    def _download_with_patterns(
        self,
        remote_base: str,
        local_base: Path,
        sample_ids: Optional[Set[str]],
        layer_indices: Optional[Set[int]],
        head_indices: Optional[Set[int]],
        include_tokens: bool,
        sample_level_only: bool,
        dry_run: bool = False,
        quiet: bool = True,
    ) -> None:
        """Builds and executes a single, efficient aws s3 cp command using include/exclude patterns."""
        print("\nBuilding optimized download command...")

        include_patterns = self._build_include_patterns(
            sample_ids, layer_indices, head_indices, include_tokens, sample_level_only
        )

        if not include_patterns:
            print("No data to download based on the provided filters. Skipping.")
            return

        if quiet:
            cmd = ["s3", "cp", "--recursive", "--quiet", "--exclude", "*"]
        else:
            cmd = ["s3", "cp", "--recursive", "--exclude", "*"]

        for pattern in include_patterns:
            cmd.extend(["--include", pattern])
        cmd.extend([remote_base, str(local_base)])

        print(
            f"Downloading with a single command and {len(include_patterns)} include patterns..."
        )
        self._run_aws_command(cmd, dry_run=dry_run)

    def _build_include_patterns(
        self,
        sample_ids: Optional[Set[str]],
        layer_indices: Optional[Set[int]],
        head_indices: Optional[Set[int]],
        include_tokens: bool,
        sample_level_only: bool,
    ) -> List[str]:
        """
        Builds a list of --include patterns for the `s3 cp` command.

        This is the core of the optimization. It generates specific paths for each
        file/directory to download, allowing S3 to do all the filtering server-side.
        """
        patterns = []

        # Always include split files at the dataset root level
        patterns.extend(["train.txt", "val.txt", "test.txt", "all.txt"])

        # If sample_ids is an empty set (e.g., from a non-existent split file), return no patterns.
        if sample_ids is not None and not sample_ids:
            return patterns  # Return just the *.txt pattern

        # Determine the set of sample prefixes to iterate over.
        # If specific samples are given, we iterate through them.
        # Otherwise, we use a wildcard to match all samples.
        sample_prefixes = sorted(list(sample_ids)) if sample_ids else ["sample_*"]

        layer_dirs = (
            [f"layer_{i:06d}" for i in sorted(list(layer_indices))]
            if layer_indices
            else ["layer_*"]
        )
        head_dirs = (
            [f"kv_head_{i:03d}" for i in sorted(list(head_indices))]
            if head_indices
            else ["kv_head_*"]
        )

        for sample_prefix in sample_prefixes:
            # Add patterns for sample-level files
            patterns.append(f"{sample_prefix}/all_text.txt")
            patterns.append(f"{sample_prefix}/all_tokens.safetensors")
            patterns.append(f"{sample_prefix}/generated_text.txt")
            patterns.append(f"{sample_prefix}/ground_truth.safetensors")
            patterns.append(f"{sample_prefix}/context_len.safetensors")
            if include_tokens:
                patterns.append(f"{sample_prefix}/prompt_ntokens.safetensors")

            # Skip layer/head patterns if sample_level_only is True
            if sample_level_only:
                continue

            # Add patterns for layer and head combinations
            for layer_dir in layer_dirs:
                patterns.append(f"{sample_prefix}/{layer_dir}/input_pos.safetensors")
                for head_dir in head_dirs:
                    # This includes attention_tensors.safetensors
                    patterns.append(
                        f"{sample_prefix}/{layer_dir}/{head_dir}/attention_tensors.safetensors"
                    )

        return patterns

    def _run_aws_command(self, args: List[str], dry_run: bool = False) -> None:
        """Runs an aws CLI command, creating parent directories and handling errors."""
        full_command = ["aws"] + args
        print(f"Executing: {' '.join(full_command)}")

        if dry_run:
            print("  (dry run - command not executed)")
            return

        try:
            # Ensure target directory exists before copying
            if args[0] == "s3" and args[1] == "cp":
                target_path_str = args[-1]
                if not target_path_str.startswith("s3://"):
                    target_path = Path(target_path_str)
                    # For recursive copy, the target is a directory. For single file, it's the parent.
                    dir_to_create = (
                        target_path if "--recursive" in args else target_path.parent
                    )
                    dir_to_create.mkdir(parents=True, exist_ok=True)

            subprocess.run(full_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed with exit code {e.returncode}")
            print(f"   STDOUT: {e.stdout.strip()}")
            print(f"   STDERR: {e.stderr.strip()}")
            raise

    def _verify_download_completeness(
        self,
        local_base: Path,
        sample_ids: Optional[Set[str]],
        layer_indices: Optional[Set[int]],
        head_indices: Optional[Set[int]],
        include_tokens: bool,
        sample_level_only: bool,
        split_files: Optional[List[str]],
    ) -> None:
        """Verifies that all expected files were downloaded successfully."""
        print("\nVerifying download completeness...")
        missing_files = []

        if not local_base.exists():
            print(
                f"❌ Verification FAILED! Target directory does not exist: {local_base}"
            )
            return

        # Check for split files at the base level first
        if split_files:
            for split_file in split_files:
                if not (local_base / split_file).exists():
                    missing_files.append(f"Split file: {split_file}")

        # Determine which samples to check
        if sample_ids:
            sample_dirs = [local_base / sample_id for sample_id in sample_ids]
        else:
            sample_dirs = [
                d
                for d in local_base.iterdir()
                if d.is_dir() and d.name.startswith("sample_")
            ]
            if not sample_dirs:
                print(
                    "✅ Verification PASSED (no samples were expected or downloaded)."
                )
                return

        for sample_dir in sample_dirs:
            if not sample_dir.exists():
                missing_files.append(
                    f"Sample directory: {sample_dir.relative_to(local_base)}"
                )
                continue

            sample_name = sample_dir.name

            # Check sample-level files
            required_sample_files = [
                "all_text.txt",
                "all_tokens.safetensors",
                "generated_text.txt",
            ]
            if include_tokens:
                required_sample_files.append("prompt_ntokens.safetensors")

            for f in required_sample_files:
                if not (sample_dir / f).exists():
                    missing_files.append(f"{sample_name}/{f}")

            # Skip layer/head verification if sample_level_only is True
            if sample_level_only:
                continue

            # Determine layers to check for this sample
            actual_layer_dirs = [
                d
                for d in sample_dir.iterdir()
                if d.is_dir() and d.name.startswith("layer_")
            ]
            if layer_indices:
                expected_layer_dirs = [
                    sample_dir / f"layer_{i:06d}" for i in layer_indices
                ]
            else:
                expected_layer_dirs = (
                    actual_layer_dirs  # Check what was actually downloaded
                )

            if not expected_layer_dirs and (layer_indices is not None):
                missing_files.append(f"{sample_name}/ (no layer directories found)")

            for layer_dir in expected_layer_dirs:
                if not layer_dir.exists():
                    missing_files.append(f"{sample_name}/{layer_dir.name}")
                    continue

                if not (layer_dir / "input_pos.safetensors").exists():
                    missing_files.append(
                        f"{sample_name}/{layer_dir.name}/input_pos.safetensors"
                    )

                # Determine heads to check for this layer
                actual_head_dirs = [
                    d
                    for d in layer_dir.iterdir()
                    if d.is_dir() and d.name.startswith("kv_head_")
                ]
                if head_indices:
                    expected_head_dirs = [
                        layer_dir / f"kv_head_{i:03d}" for i in head_indices
                    ]
                else:
                    expected_head_dirs = actual_head_dirs

                if not expected_head_dirs and (head_indices is not None):
                    missing_files.append(
                        f"{sample_name}/{layer_dir.name}/ (no head directories found)"
                    )

                for head_dir in expected_head_dirs:
                    if not head_dir.exists():
                        missing_files.append(
                            f"{sample_name}/{layer_dir.name}/{head_dir.name}"
                        )
                        continue

                    if not (head_dir / "attention_tensors.safetensors").exists():
                        missing_files.append(
                            f"{sample_name}/{layer_dir.name}/{head_dir.name}/attention_tensors.safetensors"
                        )

        # Report results
        if missing_files:
            print(
                f"❌ Download verification FAILED! Missing {len(missing_files)} files/directories:"
            )
            for missing in missing_files[:20]:
                print(f"   - {missing}")
            if len(missing_files) > 20:
                print(f"   ... and {len(missing_files) - 20} more")
            print(
                "\nSuggestion: Re-run the download command. Some files may have failed to transfer."
            )
        else:
            print(
                "✅ Download verification PASSED! All expected files appear to be present."
            )


def main():
    """Main entry point using Fire CLI."""
    fire.Fire(DataDownloader)


if __name__ == "__main__":
    main()
