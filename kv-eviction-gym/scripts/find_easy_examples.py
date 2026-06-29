#!/usr/bin/env python3
"""Find easy examples for RL training via full-cache greedy inference.

"Easy" = the model finishes (EOS) within a short budget under ideal conditions
(no KV eviction, greedy decoding). These examples are currently SKIPPED by the
training loop because the free-growth skip logic discards any episode where EOS
appears during free-growth.

Strategy:
    Run greedy inference (do_sample=False) with full KV cache on all training
    examples. Record gen_len[i] = number of tokens until EOS. Filter to examples
    where gen_len is in [min_gen_len, max_gen_len].

Budget math:
    base_budget[i] = T[i] + gen_len[i]

    At runtime: ep.budget = base_budget[i] - eviction_k
      → n_fg = ep.budget - T = gen_len - eviction_k  (free-growth tokens)
      → EOS at position gen_len → does NOT appear in first n_fg tokens (n_fg < gen_len)
      → episode NOT skipped
      → RL phase: agent makes eviction_k steps, EOS appears at step ~eviction_k
      → correctness reward fires

    K-curriculum (eviction_k_start → eviction_k_end):
      - K=eviction_k_start (easy): few evictions needed, EOS close
      - K=eviction_k_end (hard): more compression required

    Constraint: eviction_k_end < min_gen_len (so n_fg > 0 even at hardest stage).

Usage (on the VM, ~30 min for 1000 examples on L4):
    python scripts/find_easy_examples.py \\
        --model-name qwen-1.5b \\
        --n-examples 1000 \\
        --max-gen-len 400 \\
        --min-gen-len 150 \\
        --max-new-tokens 600 \\
        --output per_example_budgets_easy.json

Output format (same as compute_per_example_budgets.py):
    {"42": 490, "137": 310, ...}
    Keys: original example indices. Values: base_budget = T + gen_len.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kv_gym.vendor.gsm8k import load_gsm8k
from kv_gym.vendor.prompts import format_gsm8k_chat
from kv_gym.vendor.loader import KNOWN_MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name",   default="qwen-1.5b",
                        choices=list(KNOWN_MODELS.keys()))
    parser.add_argument("--n-examples",   type=int, default=1000)
    parser.add_argument("--probe-n",      type=int, default=32)
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=600,
                        help="Generation cap. Only examples that finish before this count.")
    parser.add_argument("--min-gen-len",  type=int, default=150,
                        help="Minimum gen_len. Must be > eviction_k_end (e.g. 100).")
    parser.add_argument("--max-gen-len",  type=int, default=400,
                        help="Maximum gen_len. EOS must appear within this many tokens.")
    parser.add_argument("--batch-size",   type=int, default=1,
                        help="Inference batch size. 1 is safest for variable-length outputs.")
    parser.add_argument("--output",       default="per_example_budgets_easy.json")
    parser.add_argument("--raw-output",   default="gen_lengths_raw.json",
                        help="Save all gen_lens (before filtering) for analysis.")
    args = parser.parse_args()

    model_id = KNOWN_MODELS[args.model_name]
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).cuda()
    model.eval()
    eos_id = tokenizer.eos_token_id

    print(f"Loading {args.n_examples + args.probe_n} GSM8K examples (seed={args.seed})...")
    all_examples = load_gsm8k(n=args.n_examples + args.probe_n, seed=args.seed, split="train")
    examples = all_examples[:args.n_examples]
    print(f"  training examples: {len(examples)}")

    raw: dict[str, dict] = {}
    budgets: dict[str, int] = {}

    for i, ex in enumerate(examples):
        prompt, _ = format_gsm8k_chat(tokenizer, ex)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        T = ids.shape[1]

        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=eos_id,
            )

        gen = out[0, T:]
        gen_len = len(gen)
        finished = eos_id in gen

        raw[str(i)] = {"T": T, "gen_len": gen_len, "finished": finished}

        if (i + 1) % 50 == 0:
            easy_so_far = sum(
                1 for v in raw.values()
                if v["finished"] and args.min_gen_len <= v["gen_len"] <= args.max_gen_len
            )
            print(f"  {i+1}/{len(examples)}  easy so far: {easy_so_far}")

    # Save raw gen_lens for analysis
    Path(args.raw_output).write_text(json.dumps(raw, indent=2))
    print(f"\nSaved raw gen_lens → {args.raw_output}")

    # Filter and compute base budgets
    gen_lens_all = [v["gen_len"] for v in raw.values()]
    finished_all = [v["finished"] for v in raw.values()]

    print(f"\nAll examples:")
    print(f"  finished (EOS within {args.max_new_tokens} tokens): "
          f"{sum(finished_all)}/{len(finished_all)} "
          f"({sum(finished_all)/len(finished_all):.1%})")
    g = np.array(gen_lens_all)
    print(f"  gen_len: min={g.min()} p25={np.percentile(g,25):.0f} "
          f"p50={np.percentile(g,50):.0f} p75={np.percentile(g,75):.0f} max={g.max()}")

    for idx_str, v in raw.items():
        if not v["finished"]:
            continue
        gen_len = v["gen_len"]
        if not (args.min_gen_len <= gen_len <= args.max_gen_len):
            continue
        T = v["T"]
        base_budget = T + gen_len
        budgets[idx_str] = base_budget

    print(f"\nFiltered (min_gen_len={args.min_gen_len}, max_gen_len={args.max_gen_len}):")
    print(f"  calibrated examples: {len(budgets)}")

    if budgets:
        bvals = np.array(list(budgets.values()))
        glens = np.array([raw[k]["gen_len"] for k in budgets])
        tlens = np.array([raw[k]["T"] for k in budgets])
        print(f"  base_budget (T+gen_len): min={bvals.min()} "
              f"p50={np.median(bvals):.0f} max={bvals.max()}")
        print(f"  gen_len:  min={glens.min()} p50={np.median(glens):.0f} max={glens.max()}")
        print(f"  T:        min={tlens.min()} p50={np.median(tlens):.0f} max={tlens.max()}")
        print(f"\n  At eviction_k_start=20:  RL phase ~20 steps, EOS near end")
        print(f"  At eviction_k_end=100:   RL phase ~100 steps, need more compression")
        print(f"  Suggested max_len: {bvals.max() + 5}  (max base_budget + 5)")
        print(f"  Suggested max_new_tokens: {glens.max() + tlens.max() + 50}")

    Path(args.output).write_text(json.dumps(budgets, indent=2))
    print(f"\nWrote {len(budgets)} base budgets → {args.output}")
    print("NOTE: values are T + gen_len (no K subtracted). K applied at runtime.")


if __name__ == "__main__":
    main()
