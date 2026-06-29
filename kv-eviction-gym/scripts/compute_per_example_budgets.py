#!/usr/bin/env python3
"""Compute per-example KV-budget calibration from the FreeGrowthCache.

The free-growth cache stores, for each hard training example, the tokens
generated during free growth (the prefix decoded greedily until the KV
cache fills to budget+1 tokens). These are the examples where EOS was NOT
reached within budget, i.e., the ones actually used for training.

For each such example i we set:
    budget[i] = T[i] + len(cached[i]) - K

where:
    T[i]          = prompt token length (from the tokenizer)
    len(cached[i])= number of cached free-growth tokens
    K             = desired number of RL eviction steps per episode

Why this works:
    n_needed = budget+1 - T = len(cached) - K + 1
    Since n_needed <= len(cached) and no EOS in the first n_needed tokens
    (the cache only stores pre-EOS tokens), the cache lookup is a full hit
    and the episode is NOT skipped.

    The RL agent then operates for K steps: evict 1 → decode 1 per step.
    EOS may appear at or after step K, completing the episode.

Outputs a JSON file: { "42": 380, "137": 420, ... }
    Keys are original example indices (0..n_examples-1).
    Values are the calibrated budget for that example.

Usage (on the VM):
    python scripts/compute_per_example_budgets.py \
        --cache-dir ~/.kv_eviction_cache \
        --model-name qwen-1.5b \
        --n-examples 1000 \
        --eviction-steps 100 \
        --output per_example_budgets.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kv_gym.free_growth_cache import FreeGrowthCache
from kv_gym.vendor.gsm8k import load_gsm8k
from kv_gym.vendor.prompts import format_gsm8k
from kv_gym.vendor.loader import KNOWN_MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir",     default="~/.kv_eviction_cache")
    parser.add_argument("--model-name",    default="qwen-1.5b",
                        choices=list(KNOWN_MODELS.keys()))
    parser.add_argument("--n-examples",    type=int, default=1000,
                        help="Number of training examples (must match what training used).")
    parser.add_argument("--probe-n",       type=int, default=32,
                        help="Number of probe examples appended after training examples.")
    parser.add_argument("--seed",          type=int, default=0)
    parser.add_argument("--eviction-steps", type=int, default=100, dest="K",
                        help="K: number of RL eviction steps per episode. "
                             "Larger K = more learning signal per episode, "
                             "smaller K = more completions (correctness reward).")
    parser.add_argument("--min-budget",    type=int, default=300,
                        help="Discard examples with calibrated budget below this.")
    parser.add_argument("--max-budget",    type=int, default=600,
                        help="Discard examples with calibrated budget above this.")
    parser.add_argument("--output",        default="per_example_budgets.json")
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args()

    model_id = KNOWN_MODELS[args.model_name]
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading {args.n_examples + args.probe_n} GSM8K examples (seed={args.seed})...")
    all_examples = load_gsm8k(n=args.n_examples + args.probe_n, seed=args.seed, split="train")
    examples = all_examples[:args.n_examples]
    print(f"  training examples: {len(examples)}")

    cache = FreeGrowthCache(args.cache_dir, args.model_name)
    n_cached, total_tokens = cache.stats()
    print(f"FreeGrowthCache: {n_cached} entries, {total_tokens:,} total tokens")

    budgets: dict[str, int] = {}
    stats = {"n_cached": 0, "n_no_cache": 0, "n_too_short": 0,
             "n_budget_too_low": 0, "n_budget_too_high": 0, "n_ok": 0}

    prompt_lens: list[int] = []
    cached_lens: list[int] = []
    computed_budgets: list[int] = []

    for i, example in enumerate(examples):
        cached = cache.get(i)
        if cached is None:
            stats["n_no_cache"] += 1
            continue
        stats["n_cached"] += 1

        cached_len = len(cached)
        if cached_len <= args.K:
            # Not enough cached tokens to guarantee K eviction steps
            stats["n_too_short"] += 1
            if args.verbose:
                print(f"  [{i:04d}] skip: cached_len={cached_len} <= K={args.K}")
            continue

        # Tokenize prompt to get T
        prompt_txt, _ = format_gsm8k(example)
        T = tokenizer(prompt_txt, return_tensors="pt")["input_ids"].shape[1]

        budget = T + cached_len - args.K
        if budget < args.min_budget:
            stats["n_budget_too_low"] += 1
            if args.verbose:
                print(f"  [{i:04d}] skip: budget={budget} < min={args.min_budget} "
                      f"(T={T}, cached={cached_len})")
            continue
        if budget > args.max_budget:
            stats["n_budget_too_high"] += 1
            if args.verbose:
                print(f"  [{i:04d}] skip: budget={budget} > max={args.max_budget} "
                      f"(T={T}, cached={cached_len})")
            continue

        budgets[str(i)] = budget
        stats["n_ok"] += 1
        prompt_lens.append(T)
        cached_lens.append(cached_len)
        computed_budgets.append(budget)
        if args.verbose:
            print(f"  [{i:04d}] budget={budget}  T={T}  cached_len={cached_len}")

    print(f"\nSummary:")
    print(f"  total training examples : {args.n_examples}")
    print(f"  with cache entry        : {stats['n_cached']}")
    print(f"  no cache entry          : {stats['n_no_cache']}")
    print(f"  cached_len <= K={args.K}        : {stats['n_too_short']}")
    print(f"  budget < min={args.min_budget}         : {stats['n_budget_too_low']}")
    print(f"  budget > max={args.max_budget}         : {stats['n_budget_too_high']}")
    print(f"  calibrated (usable)     : {stats['n_ok']}")

    if computed_budgets:
        arr = np.array(computed_budgets)
        pl  = np.array(prompt_lens)
        cl  = np.array(cached_lens)
        print(f"\nBudget distribution:")
        print(f"  min={arr.min()}  p25={np.percentile(arr,25):.0f}  "
              f"median={np.median(arr):.0f}  p75={np.percentile(arr,75):.0f}  max={arr.max()}")
        print(f"Prompt length distribution:")
        print(f"  min={pl.min()}  median={np.median(pl):.0f}  max={pl.max()}")
        print(f"Cached length distribution:")
        print(f"  min={cl.min()}  median={np.median(cl):.0f}  max={cl.max()}")

    out_path = Path(args.output)
    out_path.write_text(json.dumps(budgets, indent=2))
    print(f"\nWrote {len(budgets)} calibrated budgets → {out_path}")


if __name__ == "__main__":
    main()
