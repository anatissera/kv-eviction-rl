#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import concurrent
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import safetensors.torch
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torchtune import config, generation, training, utils
from torchtune.data import padded_collate
from tqdm import tqdm

from kvcompression.data.chunk_sampler import ChunkSampler
from kvcompression.data.kvpress_dataset import KVPressPreprocessingDataset
from kvcompression.hooks.extractors import ExtractAttentionTensors
from kvcompression.utils.s3_utils import upload_directory

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._compile = cfg.compile
        self.executor_max_workers = cfg.executor_max_workers
        training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        ckpt_dict = checkpointer.load_checkpoint()

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[training.MODEL_KEY],
            compile_model=self._compile,
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._sampler, self._dataloader = self._setup_data(cfg)

    def _setup_data(self, cfg) -> Tuple[ChunkSampler, DataLoader]:
        if "data_dir" in cfg:
            self._ds = KVPressPreprocessingDataset(
                dataset_name=cfg.dataset_name,
                data_dir=cfg.data_dir,
                tokenizer=self._tokenizer,
                max_context_length=self._model.max_seq_len,
            )
        else:
            self._ds = config.instantiate(
                cfg.dataset,
                tokenizer=self._tokenizer,
            )

        sampler = ChunkSampler(
            data_source=self._ds,
            total_num_chunks=cfg.total_num_chunks,
            current_chunk=cfg.current_chunk,
            shuffle=False,
        )
        dataloader = DataLoader(
            dataset=self._ds,
            batch_size=1,
            sampler=sampler,
            collate_fn=partial(
                padded_collate,
                pad_direction="left",
                keys_to_pad=["tokens", "mask"],
                padding_idx=self._tokenizer.pad_id,
            ),
            drop_last=False,
        )

        return sampler, dataloader

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
        compile_model: bool,
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if compile_model:
            training.compile_model(model)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")

        return model

    @torch.inference_mode()
    def generate(self, cfg: DictConfig):
        # Ensure the cache is setup on the right device, with only as many tokens as we need
        if cfg.enable_kv_cache:
            with self._device:
                self._model.setup_caches(
                    batch_size=1,
                    dtype=self._dtype,
                    decoder_max_seq_len=self._model.max_seq_len,
                )

        generation_output_path = Path(cfg.output_dir)

        if "dataset_name" in cfg:
            generation_output_path = generation_output_path / cfg.dataset_name.replace(
                "/", "_"
            )

        generation_output_path = (
            generation_output_path
            / cfg.model_name
            / f"temperature_{cfg.temperature:0.2f}"
        )

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.executor_max_workers
        )
        for batch in tqdm(self._dataloader, total=len(self._sampler)):
            sample_idx = batch.pop("id").item()

            utils.batch_to_device(batch, self._device)

            prompt_tokens = batch["tokens"]

            sample_outputpath = generation_output_path / f"sample_{sample_idx:07d}"
            sample_outputpath.mkdir(exist_ok=True, parents=True)

            generated_tokens, _ = generation.generate(
                model=self._model,
                prompt=prompt_tokens,
                max_generated_tokens=cfg.max_new_tokens,
                pad_id=self._tokenizer.pad_id,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                stop_tokens=self._tokenizer.stop_tokens,
                custom_generate_next_token=None,
            )
            self._model.reset_caches()

            with ExtractAttentionTensors(
                cfg=cfg,
                model=self._model,
                sample_idx=sample_idx,
                output_path=sample_outputpath,
                executor=executor,
                disable_kv_caching=True,
            ):
                masks = torch.tril(
                    torch.ones(
                        generated_tokens.shape[1],
                        generated_tokens.shape[1],
                        dtype=torch.bool,
                        device=prompt_tokens.device,
                    )
                ).unsqueeze(0)
                input_pos = torch.arange(
                    0, generated_tokens.shape[1], device=generated_tokens.device
                ).unsqueeze(0)

                self._model(generated_tokens, input_pos=input_pos, mask=masks)

            executor.submit(
                self.save_generation,
                generated_tokens.cpu(),
                prompt_tokens.numel(),
                sample_idx,
                local_sample_dir_path=Path(cfg.home_dir) / sample_outputpath,
                remote_sample_dir_path=f"{cfg.remote_dir}/{sample_outputpath}",
            )

        executor.shutdown(wait=True)

    def save_generation(
        self,
        generated_tokens,
        prompt_len,
        sample_idx,
        local_sample_dir_path,
        remote_sample_dir_path,
    ):
        (local_sample_dir_path / "all_text.txt").write_text(
            self._tokenizer.decode(generated_tokens[0].tolist())
        )
        (local_sample_dir_path / "generated_text.txt").write_text(
            self._tokenizer.decode(generated_tokens[0].tolist()[prompt_len:])
        )
        safetensors.torch.save_file(
            {"tensor": generated_tokens[0]},
            local_sample_dir_path / "all_tokens.safetensors",
        )
        safetensors.torch.save_file(
            {"tensor": torch.tensor(prompt_len)},
            local_sample_dir_path / "prompt_ntokens.safetensors",
        )

        # Add ground truth storage for OASST2 datasets
        if hasattr(self._ds, "get_ground_truth_info"):
            gt_info = self._ds.get_ground_truth_info(sample_idx)
            # Convert to tensor if needed for consistent storage format
            gt_tokens = gt_info["ground_truth_tokens"]
            if not isinstance(gt_tokens, torch.Tensor):
                gt_tokens = torch.tensor(gt_tokens, dtype=torch.long)

            safetensors.torch.save_file(
                {"tensor": gt_tokens},
                local_sample_dir_path / "ground_truth.safetensors",
            )
            safetensors.torch.save_file(
                {"tensor": torch.tensor(gt_info["context_len"])},
                local_sample_dir_path / "context_len.safetensors",
            )

        upload_successful = upload_directory(
            local_sample_dir_path, remote_sample_dir_path
        )
        if upload_successful:
            shutil.rmtree(local_sample_dir_path)
        else:
            logger.warning(
                f"Upload failed or was skipped for '{local_sample_dir_path}'. "
                "Local directory will NOT be deleted."
            )


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)

    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
