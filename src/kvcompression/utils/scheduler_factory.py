#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Custom scheduler factory for complex scheduler configurations."""

import torch
from torch.optim import Optimizer


class SequentialWarmupCosineScheduler(torch.optim.lr_scheduler.SequentialLR):
    """SequentialLR with linear warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 50,
        total_steps: int = 1000,
        warmup_start_factor: float = 0.01,
        cosine_eta_min: float = 1e-6,
    ):
        cosine_steps = total_steps - warmup_steps

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=cosine_eta_min
        )

        super().__init__(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
