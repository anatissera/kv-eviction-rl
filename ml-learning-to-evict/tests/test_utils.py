#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torchtune.models.llama2 import llama2 as Llama2Builder


def fixed_init_tensor(
    shape: torch.Size,
    min_val: Union[float, int] = 0.0,
    max_val: Union[float, int] = 1.0,
    nonlinear: bool = False,
    dtype: torch.dtype = torch.float,
):
    """
    Utility for generating deterministic tensors of a given shape. In general stuff
    like torch.ones, torch.eye, etc can result in trivial outputs. This utility
    generates a range tensor [min_val, max_val) of a specified dtype, applies
    a sine function if nonlinear=True, then reshapes to the appropriate shape.
    """
    n_elements = math.prod(shape)
    if n_elements == 0:
        return torch.empty(shape, dtype=dtype)
    step_size = (max_val - min_val) / n_elements
    x = torch.arange(min_val, max_val, step_size, dtype=dtype)[:n_elements]
    if x.numel() < n_elements:
        x_extension = torch.full(
            (n_elements - x.numel(),), x[-1] if x.numel() > 0 else min_val, dtype=dtype
        )
        x = torch.cat((x, x_extension))

    x = x.reshape(shape)
    if nonlinear:
        return torch.sin(x)
    return x


@torch.no_grad
def fixed_init_model(
    model: nn.Module,
    min_val: Union[float, int] = 0.0,
    max_val: Union[float, int] = 1.0,
    nonlinear: bool = False,
    dtype: Optional[torch.dtype] = None,
):
    """
    This utility initializes all parameters of a model deterministically using the
    function fixed_init_tensor above. See that docstring for details of each parameter.
    """
    for _, param in model.named_parameters():
        param.copy_(
            fixed_init_tensor(
                param.shape,
                min_val=min_val,
                max_val=max_val,
                nonlinear=nonlinear,
                dtype=param.dtype if dtype is None else dtype,
            )
        )


def base_test_model(config: dict, device: str = "cpu") -> nn.Module:
    """
    Provides a base, deterministically initialized Llama2 model.
    KV cache is NOT setup here.

    Args:
        config: Model configuration dictionary.
        device: Device to place the model on ("cpu" or "cuda").

    Note: If you need the model on GPU, call model.setup_caches() first,
    then model.to(device) to ensure caches are on the correct device.
    """
    model = Llama2Builder(**config)
    fixed_init_model(model, min_val=-0.02, max_val=0.02)
    model.eval()
    return model.to(torch.device(device))
