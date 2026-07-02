"""E5 Golden-BC step 1 — generate future-attention traces for TRAIN examples.

For each of the first --n TRAIN examples (same seeded shuffle as train.py, so
these are exactly the examples the policy trains on), runs one full-cache
greedy decode with eager attention and saves the per-layer SUFFIX-MAX future
attention matrix (see scripts/oracle_eval.py):

    SUF[l][t, j] = max over decode steps t' >= t (and heads) of
                   attention(query t' → key j)

The golden label for an eviction at generated-step t in layer l is then
argmin over valid slots s of SUF[l][t, orig_pos(s)] — the exact quantity the
oracle used to beat kv_norm by +7pp (FINDINGS §11).

Output: <out-dir>/ex{idx:05d}.npz with l00..lNN float16 arrays + T, n_steps.
Resume-safe: skips existing files. ~15-25MB/example compressed.

Usage:
  python scripts/trace_gen.py --config configs/e1_rich.yaml --n 400 \
      --out-dir traces
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_eval import full_trace_with_future_attention  # noqa: E402

from kv_gym.capture import capture                         # noqa: E402
from kv_gym.vendor.loader import load_model_and_tokenizer  # noqa: E402
from kv_gym.vendor.gsm8k import load_gsm8k                 # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=400,
                   help="How many TRAIN examples (prefix of the train split) to trace")
    p.add_argument("--out-dir", default="traces")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm, tokenizer, device = load_model_and_tokenizer(
        name=cfg.get("model_name", "qwen-1.5b"),
        device=device,
        attn_implementation="eager",   # required for output_attentions
    )

    seed    = cfg.get("seed", 0)
    max_new = cfg.get("max_new_tokens", 600)
    max_len = cfg.get("max_len", 288)
    budget  = cfg.get("budget_min", 256)

    # TRAIN examples = prefix of the same seeded shuffle used by train.py.
    examples = load_gsm8k(n=args.n, seed=seed, split="train")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"trace gen: {len(examples)} train examples → {out_dir} (eager)")

    n_done = n_skip = 0
    for idx, ex in enumerate(examples):
        path = out_dir / f"ex{idx:05d}.npz"
        if path.exists():
            continue
        try:
            cap = capture(llm, tokenizer, ex, device)
            T = cap.prompt_len
            if T > max_len or T >= budget:
                np.savez_compressed(path, skip=np.int8(1), T=np.int32(T))
                n_skip += 1
                print(f"[{idx}] skip T={T}")
                continue
            _, suf, n_steps = full_trace_with_future_attention(
                llm, tokenizer, cap.input_ids, device, max_new)
            arrays = {f"l{l:02d}": suf[l] for l in range(len(suf))}
            np.savez_compressed(path, skip=np.int8(0), T=np.int32(T),
                                n_steps=np.int32(n_steps), **arrays)
            n_done += 1
            print(f"[{idx}] T={T} steps={n_steps} → {path.name}")
        except Exception as e:  # noqa: BLE001 — log, mark, continue
            print(f"[{idx}] ERROR {e}")
            # do NOT write the file → retried on next run
    print(f"done: {n_done} traced, {n_skip} skipped")


if __name__ == "__main__":
    main()
