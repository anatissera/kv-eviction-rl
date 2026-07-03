"""E7b pool screening: which filtered TRAIN examples are FULL-CACHE solvable?

The E7 arena (long-gen, budget 256) has its bar at zero (probe: kv_norm =
random = 0/16) — signal exists only when an episode is solvable at all. On
sdpa, full-cache solves ~68-88% of long-gen examples; training groups drawn
from unsolvable ones produce 50 consecutive zero-signal rollouts
(repeats_per_problem=50). Screening the pool to full-solvable examples makes
EVERY group informative.

Writes <out>/pool_screen.json: [{"idx": i, "full_correct": 0/1, "T": prompt_len,
"gen_len": n}] for the first --n examples of the FILTERED (min_answer_words)
seeded shuffle — the same universe e7_repeat.yaml trains on. Resume-safe.

Usage:
  python scripts/screen_pool.py --config configs/e7_repeat.yaml --n 96 \
      --out-dir runs/pool_screen
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kv_gym.capture import capture
from kv_gym.eval_core import score_full_cache
from kv_gym.vendor.loader import load_model_and_tokenizer
from kv_gym.vendor.gsm8k import load_gsm8k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=96)
    p.add_argument("--out-dir", default="runs/pool_screen")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm, tokenizer, device = load_model_and_tokenizer(
        name=cfg.get("model_name", "qwen-1.5b"),
        device=device,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    examples = load_gsm8k(n=args.n, seed=cfg.get("seed", 0), split="train",
                          min_answer_words=cfg.get("min_answer_words", 0))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pool_screen.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try: done.add(json.loads(line)["idx"])
            except Exception: pass
        print(f"resume: {len(done)} screened")

    max_new = cfg.get("max_new_tokens", 600)
    for idx, ex in enumerate(examples):
        if idx in done:
            continue
        try:
            cap = capture(llm, tokenizer, ex, device)
            corr, _, _ = score_full_cache(llm, tokenizer, cap.input_ids,
                                          cap.gold_answer, device, max_new)
            row = {"idx": idx, "full_correct": int(corr), "T": cap.prompt_len}
            print(f"[{idx}] full_correct={int(corr)} T={cap.prompt_len}")
        except Exception as e:  # noqa: BLE001
            row = {"idx": idx, "error": str(e)[:120]}
            print(f"[{idx}] ERROR {e}")
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")

    rows = [json.loads(l) for l in open(path)]
    good = [r["idx"] for r in rows if r.get("full_correct") == 1]
    print(f"\nscreen done: {len(good)}/{len(rows)} full-solvable")
    print("first 32 solvable indices:", good[:32])


if __name__ == "__main__":
    main()
