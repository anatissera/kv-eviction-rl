#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import logging

import torch
from datasets import load_dataset
from torchtune import utils
from torchtune.data import Message
from torchtune.models.qwen2_5._tokenizer import Qwen2_5Tokenizer

log = utils.get_logger("DEBUG")

logger = logging.getLogger(__name__)


def _add_row_id(example, idx):
    """Adds a 'id' column to each example in the dataset."""
    example["id"] = idx
    return example


class KVPressPreprocessingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, data_dir: str, tokenizer, max_context_length):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.hf_dataset = load_dataset(dataset_name, data_dir=data_dir, split="test")
        self.hf_dataset = self.hf_dataset.map(_add_row_id, with_indices=True)

        self.tokenizer: Qwen2_5Tokenizer = tokenizer
        self.max_context_length = max_context_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]

        messages = [
            Message(role="user", content=row["context"] + row["question"]),
            Message(role="assistant", content=row["answer_prefix"], eot=False),
        ]
        sample = {}
        tokens, mask = self.tokenizer.tokenize_messages(messages, add_eos=False)
        sample["tokens"] = tokens
        sample["mask"] = mask
        sample["id"] = row["id"]
        return sample
