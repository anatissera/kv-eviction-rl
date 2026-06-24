#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os

import pytest
import torch

from kvcompression.constants import DISABLE_TORCH_COMPILER_VARNAME
from kvcompression.utils.utils import seed_everything

os.environ[DISABLE_TORCH_COMPILER_VARNAME] = "1"

seed_everything(0)

CUDA_AVAILABLE = torch.cuda.is_available()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_cuda: mark test as requiring CUDA (GPU)"
    )


@pytest.fixture(autouse=True)
def skip_cuda_tests_if_no_gpu(request):
    """Auto-skip tests marked with requires_cuda if no GPU available."""
    if request.node.get_closest_marker("requires_cuda"):
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available - FlexAttention tests require GPU")


@pytest.fixture(scope="class")
def model_base_config_fixture():
    # Note: embed_dim / num_heads must be >= 16 for FlexAttention on CUDA
    return {
        "vocab_size": 1024,
        "embed_dim": 64,  # 64 / 4 = 16 head_dim (minimum for FlexAttention)
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "max_seq_len": 64,
    }
