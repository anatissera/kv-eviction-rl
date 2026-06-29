#!/usr/bin/env python3
"""Compute per-example base budgets from the FreeGrowthCache.

For each training example that has cached free-growth tokens, compute:
    base_budget[i] = T[i] + len(cached[i])

where:
    T[i]           = prompt token length
    len(cached[i]) = number of cached free-growth tokens (no EOS in cache)

At runtime, the eviction curriculum sets the effective budget:
    ep.budget[i] = base_budget[i] - eviction_k

where eviction_k starts small (easy) and grows (harder) over training.

With eviction_k applied:
    n_needed = ep.budget + 1 - T = cached_len - eviction_k + 1
    Since n_needed <= cached_len, the cache lookup is a full hit → no EOS
    → episode NOT skipped.
    The RL agent must then generate eviction_k+ tokens before EOS appears.

Only examples with cached_len >= min_cached_len are kept (ensures full hit
even at the hardest eviction_k in the curriculum).

Outputs per_example_budgets.json: { "42": 540, "137": 480, ... }
    Keys are original example indices (0..n_examples-1).
    Values are base_budget = T + cached_len (NO eviction_k subtracted).

Usage (on the VM):
    python scripts/compute_per_example_budgets.py \\
        --cache-dir ~/.kv_eviction_cache \\
        --model-name qwen-1.5b \\
        --n-examples 1000 \\
        --min-cached-len 150 \\
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
    parser.add_argument("--cache-dir",      default="~/.kv_eviction_cache")
    parser.add_argument("--model-name",     default="qwen-1.5b",
                        choices=list(KNOWN_MODELS.keys()))
    parser.add_argument("--n-examples",     type=int, default=1000,
                        help="Number of training examples (must match training run).")
    parser.add_argument("--probe-n",        type=int, default=32)
    parser.add_argument("--seed",           type=int, default=0)
    parser.add_argument("--min-cached-len", type=int, default=150,
                        help="Minimum cached_len to include. Must be > eviction_k_end "
                             "so the cache hit is guaranteed even at the hardest K.")
    parser.add_argument("--min-base-budget", type=int, default=350,
                        help="Minimum base_budget (T + cached_len) to include.")
    parser.add_argument("--max-base-budget", type=int, default=750,
                        help="Maximum base_budget. Clips examples with very long prompts.")
    parser.add_argument("--output",         default="per_example_budgets.json")
    parser.add_argument("--verbose",        action="store_true")
    args = parser.parse_args()

    model_id = KNOWN_MODELS[args.model_name]
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading {args.n_examples + args.probe_n} GSM8K examples (seed={args.seed})...")
    all_examples = load_gsm8k(n=args.n_examples + args.probe_n,
                               seed=args.seed, split="train")
    examples = all_examples[:args.n_examples]
    print(f"  training examples: {len(examples)}")

    cache = FreeGrowthCache(args.cache_dir, args.model_name)
    n_cached, total_tokens = cache.stats()
    print(f"FreeGrowthCache: {n_cached} entries, {total_tokens:,} total tokens")

    budgets: dict[str, int] = {}
    stats = {"no_cache": 0, "too_short": 0, "base_too_low": 0,
             "base_too_high": 0, "ok": 0}

    prompt_lens:   list[int] = []
    cached_lens:   list[int] = []
    base_budgets:  list[int] = []

    for i, example in enumerate(examples):
        cached = cache.get(i)
        if cached is None:
            stats["no_cache"] += 1
            continue

        cached_len = len(cached)
        if cached_len < args.min_cached_len:
            stats["too_short"] += 1
            if args.verbose:
                print(f"  [{i:04d}] skip: cached_len={cached_len} < min={args.min_cached_len}")
            continue

        prompt_txt, _ = format_gsm8k(example)
        T = tokenizer(prompt_txt, return_tensors="pt")["input_ids"].shape[1]

        base_budget = T + cached_len
        if base_budget < args.min_base_budget:
            stats["base_too_low"] += 1
            if args.verbose:
                print(f"  [{i:04d}] skip: base_budget={base_budget} < min={args.min_base_budget}")
            continue
        if base_budget > args.max_base_budget:
            stats["base_too_high"] += 1
            if args.verbose:
                print(f"  [{i:04d}] skip: base_budget={base_budget} > max={args.max_base_budget}")
            continue

        budgets[str(i)] = base_budget
        stats["ok"] += 1
        prompt_lens.append(T)
        cached_lens.append(cached_len)
        base_budgets.append(base_budget)
        if args.verbose:
            print(f"  [{i:04d}] base_budget={base_budget}  T={T}  cached_len={cached_len}")

    print(f"\nSummary:")
    print(f"  total training examples     : {args.n_examples}")
    print(f"  with cache entry            : {n_cached}")
    print(f"  no cache entry              : {stats['no_cache']}")
    print(f"  cached_len < {args.min_cached_len}            : {stats['too_short']}")
    print(f"  base_budget < {args.min_base_budget}           : {stats['base_too_low']}")
    print(f"  base_budget > {args.max_base_budget}           : {stats['base_too_high']}")
    print(f"  calibrated (usable)         : {stats['ok']}")

    if base_budgets:
        arr = np.array(base_budgets)
        pl  = np.array(prompt_lens)
        cl  = np.array(cached_lens)
        print(f"\nBase budget distribution (T + cached_len):")
        print(f"  min={arr.min()}  p25={np.percentile(arr,25):.0f}  "
              f"median={np.median(arr):.0f}  p75={np.percentile(arr,75):.0f}  max={arr.max()}")
        print(f"Prompt length distribution:")
        print(f"  min={pl.min()}  median={np.median(pl):.0f}  max={pl.max()}")
        print(f"Cached token count distribution:")
        print(f"  min={cl.min()}  median={np.median(cl):.0f}  max={cl.max()}")
        print(f"\nAt eviction_k_start=20:  effective budgets in "
              f"[{arr.min()-20}, {arr.max()-20}]  (easy)")
        print(f"At eviction_k_end=150:   effective budgets in "
              f"[{arr.min()-150}, {arr.max()-150}]  (hard)")

    out_path = Path(args.output)
    out_path.write_text(json.dumps(budgets, indent=2))
    print(f"\nWrote {len(budgets)} base budgets → {out_path}")
    print("NOTE: values are T + cached_len (no K subtracted).")
    print("      eviction_k is applied at runtime by the curriculum callback.")


if __name__ == "__main__":
    main()
