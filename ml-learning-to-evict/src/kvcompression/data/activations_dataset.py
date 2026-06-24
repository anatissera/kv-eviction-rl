#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from pathlib import Path
from typing import Union

import safetensors.torch
import torch
from torch.utils.data import Dataset
from torchtune import training, utils

from kvcompression import PROJECT_ROOT

logger = utils.get_logger("INFO")
log_rank_zero = utils.log_rank_zero


class QKVDataset(Dataset):
    """
    Unified dataset for GQA hierarchical data structure.

    Supports both single query and full query group modes.
    Always targets a specific layer and KV head combination.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        dtype,
        target_layer_idx: int,
        kv_head_idx: int,
        split_file: str,
        query_idx_within_group: int = None,
        keep_dim: bool = False,
        mmap: bool = True,
        randomize_prompt_len: bool = False,
        min_prompt_len: int = 16,
        **kwargs,
    ):
        """
        Initialize the QKV dataset.

        Args:
            dataset_path: Path to the dataset directory.
            dtype: Data type for loaded tensors.
            target_layer_idx: Target transformer layer index.
            kv_head_idx: Target KV head index within the layer.
            split_file: Filename containing the list of samples to use.
            query_idx_within_group: If set, select a specific query from the group.
                If None, return all queries in the group.
            keep_dim: Whether to keep the head dimension in output tensors.
            mmap: Whether to use memory-mapped loading for efficiency.
            randomize_prompt_len: Whether to randomly vary prompt length during training.
            min_prompt_len: Minimum prompt length when randomizing.
            **kwargs: Additional arguments (ignored).
        """
        self.dataset_path = Path(dataset_path)
        self.dtype = training.get_dtype(dtype=dtype)
        self.target_layer_idx = target_layer_idx
        self.kv_head_idx = kv_head_idx
        self.split_file = split_file
        self.query_idx_within_group = query_idx_within_group
        self.keep_dim = keep_dim
        self.mmap = mmap
        self.randomize_prompt_len = randomize_prompt_len
        self.min_prompt_len = min_prompt_len

        if not self.dataset_path.is_absolute():
            if PROJECT_ROOT:
                self.dataset_path = PROJECT_ROOT / dataset_path
            else:
                self.dataset_path = Path.cwd() / dataset_path

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        self.split_samples = self._load_split_samples()
        self.data_dirs = self._discover_data_dirs()

        if len(self.data_dirs) == 0:
            raise ValueError(
                f"No data found for layer {target_layer_idx}, head {kv_head_idx} "
                f"in {self.dataset_path}"
            )

        mode = (
            f"query_{query_idx_within_group}"
            if query_idx_within_group is not None
            else "full_group"
        )
        log_rank_zero(
            logger,
            f"Initialized QKVDataset for layer {target_layer_idx}, head {kv_head_idx} "
            f"({mode}, keep_dim={keep_dim}) with {len(self.data_dirs)} samples "
            f"from {self.dataset_path} (split: {split_file})",
        )

    def _load_split_samples(self):
        """Load sample names from split file and validate they exist."""
        split_path = self.dataset_path / self.split_file
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        samples = set(line.strip() for line in split_path.read_text().splitlines())

        for sample in samples:
            if not (self.dataset_path / sample).exists():
                raise FileNotFoundError(
                    f"Sample directory not found: {self.dataset_path / sample}"
                )

        return samples

    def _discover_data_dirs(self):
        """Find data directories for specific layer/head combination."""
        data_dirs = []
        layer_pattern = f"layer_{self.target_layer_idx:06d}"
        head_pattern = f"kv_head_{self.kv_head_idx:03d}"

        for sample_name in sorted(self.split_samples):
            sample_dir = self.dataset_path / sample_name
            target_dir = sample_dir / layer_pattern / head_pattern

            if target_dir.exists():
                required_files = ["attention_tensors.safetensors"]
                if all((target_dir / f).exists() for f in required_files):
                    data_dirs.append(target_dir)

        return data_dirs

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx):
        data_dir = self.data_dirs[idx]

        # Load aggregated attention tensors
        if self.mmap:
            attention_tensors = safetensors.torch.load_file(
                data_dir / "attention_tensors.safetensors", device="cpu"
            )
        else:
            attention_tensors = safetensors.torch.load_file(
                data_dir / "attention_tensors.safetensors"
            )

        queries = attention_tensors["q_group"]
        keys = attention_tensors["k"]
        values = attention_tensors["v"]

        # This is the total sequence length (prompt + generation)
        total_seq_len = keys.shape[0]

        if self.randomize_prompt_len:
            #  Ensure there is both a prompt to sort and a future to evaluate against.
            if total_seq_len <= self.min_prompt_len:
                # Sequence too short to randomize; use almost the entire sequence as prompt
                prompt_len = torch.tensor(max(1, total_seq_len - 1))
            else:
                # Upper bound is exclusive, ensuring at least one future token
                prompt_len = torch.randint(self.min_prompt_len, total_seq_len, size=())
        else:
            prompt_len_data = safetensors.torch.load_file(
                data_dir.parent.parent / "prompt_ntokens.safetensors"
            )
            prompt_len = prompt_len_data["tensor"]

        if self.query_idx_within_group is not None:
            queries = queries[self.query_idx_within_group]  # [seq_len, head_dim]

        if self.keep_dim:
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)
            if self.query_idx_within_group is not None:
                queries = queries.unsqueeze(0)

        sample = {
            "id": torch.tensor(idx),
            "queries": queries.to(self.dtype),
            "keys": keys.to(self.dtype),
            "values": values.to(self.dtype),
            "prompt_len": prompt_len.clone()
            if isinstance(prompt_len, torch.Tensor)
            else torch.tensor(prompt_len),
        }

        return sample
