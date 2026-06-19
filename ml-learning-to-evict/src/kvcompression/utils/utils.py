#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtune import config

from kvcompression.constants import DISABLE_TORCH_COMPILER_VARNAME

pylogger = logging.getLogger(__name__)


def get_underlying_model(model_instance):
    """Retrieves the underlying nn.Module, unwrapping DDP or DP if necessary."""
    if isinstance(model_instance, (DDP, DataParallel)):
        return model_instance.module
    else:
        return model_instance


def instantiate_agent_from_ckpt(checkpoint_path: Path, model_key: str) -> nn.Module:
    """Instantiate the agent corresponding to the checkpoint and loads its weights

    Args:
        checkpoint_path (Path): the path of the checkpoint to use, in .pt format.
        model_key (str): the key in the checkpoint dict to load the model state from
                        (e.g., "ema_model_state_dict" or "model_state_dict")

    Returns:
        nn.Module: the instantiated agent
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent_config = OmegaConf.create(checkpoint["config"]["agent"])
    agent: nn.Module = config.instantiate(agent_config, device="cpu")

    model_state_dict = checkpoint[model_key]

    # Handle wrapped model state dicts (like AveragedModel) by checking for "module." prefix
    if model_state_dict is not None and any(
        key.startswith("module.") for key in model_state_dict.keys()
    ):
        # Unwrap the state dict: remove "module." prefix and wrapper-specific keys
        model_state_dict = {
            key[7:]: value
            for key, value in model_state_dict.items()
            if key.startswith("module.")
        }

    agent.load_state_dict(model_state_dict, strict=True)
    return agent


max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def _select_seed_randomly(
    min_seed_value: int = min_seed_value, max_seed_value: int = max_seed_value
) -> int:
    return random.randint(min_seed_value, max_seed_value)  # noqa: S3


def seed_everything(seed: Optional[int] = None) -> int:
    """Set seed for pseudo-random number generators in: pytorch, numpy, python.random.

    Args:
        seed: the integer value seed for global random state.
            If ``None``, a random seed will be selected.

    """
    if seed is None:
        seed = _select_seed_randomly(min_seed_value, max_seed_value)
        pylogger.warn(f"No seed found, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        pylogger.warn(
            f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    pylogger.info(f"Seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def is_torch_compile_disabled() -> bool:
    """Check if torch.compile is disabled via environment variable."""
    return os.environ.get(DISABLE_TORCH_COMPILER_VARNAME, False)


def batch_to_device(batch: dict, device: torch.device, **kwargs) -> None:
    """
    Move all tensors in a batch dictionary to the specified device in-place.

    Args:
        batch: Dictionary containing tensors or nested dictionaries of tensors.
        device: Target device to move tensors to.
        **kwargs: Additional arguments passed to tensor.to().
    """
    for k, v in batch.items():
        if isinstance(v, dict):
            batch_to_device(v, device, **kwargs)
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device, **kwargs)
        else:
            raise ValueError(
                f"""To use batch_to_device, all elements in the batch must be a dict or Tensor.
Got key "{k}" with value of type {type(v)}"""
            )
