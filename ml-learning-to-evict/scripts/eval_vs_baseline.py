#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
#
"""
Minimal KV-cache-compression evaluation: trained KVP agents vs a heuristic baseline.

This is a convenience harness (not part of the original repo) for the Qwen2-1.5B
pipeline-validation experiment. The repository ships no standalone CLI evaluator and
no StreamingLLM/H2O baselines; the only in-repo heuristic press is RandomPress, and the
intended qualitative demo lives in notebooks/inference_demo.ipynb. This script wires the
*verified* generation path (see tests/test_generate_with_compression.py) into a small
quantitative comparison.

For each evaluated sample it greedily decodes a continuation three ways at the same KV
budget and reports how closely each compressed run reproduces the uncompressed reference:

  1. reference : DummyCompressionStrategy (no eviction; equals plain greedy decoding)
  2. random    : global RandomPress heuristic baseline
  3. learned   : per-(layer,head) trained SamplerAgentPress composite  [if available]

The learned path needs a trained agent for *every* (layer, head) of the model. If the
local agents/ directory is incomplete it is skipped with a message (train the full
28 x 2 = 56 sweep, or use the inference notebook, for a full learned evaluation).

Prompts are read from the activations produced by unroll_and_store (the sample_*/ dirs),
so this requires only the data-generation output, not a re-download of RULER.

Usage:
  uv run python scripts/eval_vs_baseline.py \
      --preprocess-config configs/preprocess_qwen1b.yaml \
      --sweep-name qwen1b_validation \
      --num-samples 1 --cache-size 256 --max-new-tokens 64 --device cuda
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import safetensors.torch
import torch
from omegaconf import OmegaConf
from torchtune import config, training, utils

from kvcompression import PROJECT_ROOT
from kvcompression.hooks.compressor import KVCompressor, LayerHeadCompressionConfig
from kvcompression.kv_cache.compression_strategy_protocol import (
    DummyCompressionStrategy,
    PressCompressionStrategy,
)
from kvcompression.presses.random_press import RandomPress
from kvcompression.presses.sampler_agent_press import SamplerAgentPress
from kvcompression.utils.generation import generate_with_compression

logger = utils.get_logger("INFO")


def build_model(cfg, device, dtype):
    """Load the torchtune LLM exactly like the data-generation recipe does."""
    checkpointer = config.instantiate(cfg.checkpointer)
    ckpt_dict = checkpointer.load_checkpoint()
    with training.set_default_dtype(dtype), device:
        model = config.instantiate(cfg.model)
    model.load_state_dict(ckpt_dict[training.MODEL_KEY])
    model.eval()
    return model


def find_eval_samples(base: Path, num_samples: int):
    """Return (prompt_tokens, prompt_len) for up to num_samples generated examples."""
    if not base.exists():
        raise FileNotFoundError(
            f"No generated data at {base}. Run data generation first."
        )

    # Prefer the validation split if present, else any samples.
    val_file = base / "val.txt"
    if val_file.exists():
        sample_names = [s.strip() for s in val_file.read_text().splitlines() if s.strip()]
    else:
        sample_names = sorted(p.name for p in base.glob("sample_*"))

    out = []
    for name in sample_names:
        sdir = base / name
        tok_f = sdir / "all_tokens.safetensors"
        plen_f = sdir / "prompt_ntokens.safetensors"
        if not (tok_f.exists() and plen_f.exists()):
            continue
        all_tokens = safetensors.torch.load_file(tok_f)["tensor"]
        prompt_len = int(safetensors.torch.load_file(plen_f)["tensor"].item())
        prompt = all_tokens[:prompt_len].view(1, -1).to(torch.long)
        out.append((prompt, prompt_len))
        if len(out) >= num_samples:
            break

    if not out:
        raise FileNotFoundError(f"No usable sample_*/ dirs with tokens under {base}.")
    return out


def discover_learned_composite(
    sweep_name: str, num_layers: int, num_kv_heads: int, device
) -> Optional[List[LayerHeadCompressionConfig]]:
    """Build a per-(layer,head) learned composite from local checkpoints.

    Returns None if the sweep does not cover every (layer, head) of the model.
    """
    grouped = PROJECT_ROOT / "agents" / "grouped" / sweep_name
    if not grouped.exists():
        logger.info(f"No local agents at {grouped}; skipping learned evaluation.")
        return None

    configs: List[LayerHeadCompressionConfig] = []
    for layer in range(num_layers):
        for head in range(num_kv_heads):
            ckpt = grouped / f"layer_{layer:06d}" / f"kv_head_{head:03d}" / "best_ckpt.pth"
            if not ckpt.exists():
                logger.info(
                    f"Learned composite incomplete (missing {ckpt}); train all "
                    f"{num_layers * num_kv_heads} agents for a full learned eval. Skipping."
                )
                return None
            press = SamplerAgentPress(ckpt_path=ckpt, query_selection_mode="no_agg")
            press.agent.to(device)
            configs.append(
                LayerHeadCompressionConfig(
                    layer=layer,
                    head=head,
                    strategy=PressCompressionStrategy(press),
                    strategy_name=f"SamplerAgent(L{layer},H{head})",
                )
            )
    logger.info(f"Loaded learned composite with {len(configs)} agents.")
    return configs


@torch.inference_mode()
def generate_once(model, prompt, strategies, cache_size, max_new_tokens, stop_tokens):
    """Greedy decode under a list of compression strategies.

    Every run goes through KVCompressor so cache setup/teardown is identical and clean
    across methods (the context manager allocates a CompressibleKVCache on enter and
    tears it down on exit). Use [DummyCompressionStrategy()] for the no-eviction reference.

    Mirrors tests/test_generate_with_compression.py: KVCompressor is entered first, then
    setup_caches is called inside the context (so the compressible cache is installed).
    """
    compressor = KVCompressor(model=model, compression_strategies=strategies)
    with compressor:
        with torch.device(prompt.device):
            model.setup_caches(batch_size=1, dtype=next(model.parameters()).dtype)
        return generate_with_compression(
            model=model,
            prompt=prompt,
            kv_compressor=compressor,
            target_cache_size=cache_size,
            max_generated_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
        )


def token_match_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of generated positions that agree (over the shorter continuation)."""
    a, b = a.view(-1), b.view(-1)
    n = min(a.numel(), b.numel())
    if n == 0:
        return 1.0
    return (a[:n] == b[:n]).float().mean().item()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preprocess-config", default="configs/preprocess_qwen1b.yaml")
    p.add_argument("--sweep-name", default="qwen1b_validation")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--cache-size", type=int, default=256, help="KV tokens kept per head after prefill")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/eval/eval_results.json")
    args = p.parse_args()

    device = utils.get_device(device=args.device)
    cfg = OmegaConf.load(args.preprocess_config)
    dtype = training.get_dtype(dtype=cfg.dtype, device=device)

    # output_dir resolves to "<DATA_ROOT>/gqa_safetensors"; build the temperature dir under it.
    output_dir = Path(str(cfg.output_dir))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    base = output_dir / "simonjegou_ruler" / cfg.model_name / "temperature_0.00"

    logger.info(f"Loading model {cfg.model_name} on {device} ({dtype})...")
    model = build_model(cfg, device, dtype)
    num_layers = len(model.layers)
    num_kv_heads = model.layers[0].attn.num_kv_heads
    stop_tokens = None  # decode a fixed-length continuation for a clean comparison

    samples = find_eval_samples(base, args.num_samples)
    samples = [(prompt.to(device), plen) for prompt, plen in samples]
    logger.info(
        f"Evaluating {len(samples)} sample(s): cache_size={args.cache_size}, "
        f"max_new_tokens={args.max_new_tokens}, model={num_layers}L x {num_kv_heads}KV-heads."
    )

    learned = discover_learned_composite(args.sweep_name, num_layers, num_kv_heads, device)

    per_sample = []
    for i, (prompt, plen) in enumerate(samples):
        budget = max(1, min(args.cache_size, plen - 1))
        row = {"sample": i, "prompt_len": plen, "cache_size": budget}

        ref = generate_once(
            model, prompt, [DummyCompressionStrategy()], plen, args.max_new_tokens, stop_tokens
        )
        ref_cont = ref[:, plen:]

        rnd = generate_once(
            model, prompt, [PressCompressionStrategy(RandomPress())], budget,
            args.max_new_tokens, stop_tokens,
        )
        row["random_match"] = token_match_ratio(rnd[:, plen:], ref_cont)

        if learned is not None:
            try:
                lrn = generate_once(
                    model, prompt, learned, budget, args.max_new_tokens, stop_tokens
                )
                row["learned_match"] = token_match_ratio(lrn[:, plen:], ref_cont)
            except Exception as e:  # learned composite is the untested path; never hard-fail eval
                logger.warning(f"Learned composite eval failed on sample {i}: {e}")
                row["learned_match"] = None

        logger.info(f"  sample {i}: {row}")
        per_sample.append(row)

    def _mean(key):
        vals = [r[key] for r in per_sample if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "model": cfg.model_name,
        "num_samples": len(per_sample),
        "cache_size": args.cache_size,
        "max_new_tokens": args.max_new_tokens,
        "metric": "token_match_ratio_vs_uncompressed_reference (higher = closer to full cache)",
        "random_baseline_mean": _mean("random_match"),
        "learned_mean": _mean("learned_match"),
        "per_sample": per_sample,
    }

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    logger.info("=" * 60)
    logger.info(f"RandomPress baseline match vs reference: {summary['random_baseline_mean']}")
    if summary["learned_mean"] is not None:
        logger.info(f"Learned KVP agents  match vs reference: {summary['learned_mean']}")
    else:
        logger.info("Learned KVP eval skipped (incomplete local sweep). See script docstring.")
    logger.info(f"Wrote {out_path}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
