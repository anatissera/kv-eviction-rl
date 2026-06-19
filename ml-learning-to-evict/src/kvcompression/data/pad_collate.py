#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from collections import defaultdict

import torch
import torch.nn.functional as F

# Predefined padding sizes computation
# Describe start-end padding amount for each dimension, in reverse:
# the first pair of padding values, refer to the last dimension.
DEFAULT_PADDING_BEHAVIOR = {
    "queries": lambda max_len, x: (0, 0, 0, max_len - x.shape[-2]),
    "keys": lambda max_len, x: (0, 0, 0, max_len - x.shape[-2]),
    "values": lambda max_len, x: (0, 0, 0, max_len - x.shape[-2]),
    "attention_masks": lambda max_len, x: (
        0,
        max_len - x.shape[1],
        0,
        max_len - x.shape[0],
    ),
    "hidden_states": lambda max_len, x: (0, 0, 0, max_len - x.shape[0]),
}

DEFAULT_PAD_VALUE = {
    "queries": 0,
    "keys": 0,
    "values": 0,
    "attention_masks": False,
    "hidden_states": 0,
}


def pad_collate_fn(samples):
    """
    Fast collate function with torch.nn.functional.pad and custom padding values.

    Args:
        samples: List of dictionaries containing tensors to collate.

    Returns:
        Batched dictionary with padded tensors and validity masks.
    """
    lengths = torch.tensor([item["keys"].shape[-2] for item in samples])
    max_len = lengths.max().item()
    valid_sequence = torch.arange(max_len) < lengths[..., None]

    prompt_lengths = torch.tensor([item["prompt_len"] for item in samples])
    max_prompt_len = prompt_lengths.max().item()
    valid_prompt = torch.arange(max_prompt_len) < prompt_lengths[..., None]

    batch = {}
    batch["valid_sequence"] = valid_sequence
    batch["valid_prompt"] = valid_prompt
    batch["lengths"] = lengths

    padded_items = defaultdict(list)
    for sample in samples:
        for key, value in sample.items():
            if key in batch:
                continue
            if key in DEFAULT_PADDING_BEHAVIOR:
                padding = DEFAULT_PADDING_BEHAVIOR[key](max_len=max_len, x=value)
                value = F.pad(value, padding, value=DEFAULT_PAD_VALUE[key])
            padded_items[key].append(value)
    batch = batch | {k: torch.stack(v) for k, v in padded_items.items()}
    return batch
