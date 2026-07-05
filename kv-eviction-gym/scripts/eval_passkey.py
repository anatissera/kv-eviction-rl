"""DATASET-CAUSALITY test: the same eviction arms on a RULER-style passkey task.

Hypothesis (FINDINGS 24): the whole GSM8K null is dataset-driven. GSM8K prompts
are short and DENSE (nothing redundant) and the decisive content is GENERATED
reasoning whose future utility is unpredictable. Apple's KVP wins on RULER:
long prompts that are mostly FILLER plus a needle, where future utility is
predictable from content. This script builds that regime at our scale:

  prompt  = filler sentences + "The secret code is NNNN." buried at a random
            depth + the question. T ~ 230-270 tokens.
  budget  = 176 < T  -> eviction pressure from step 1; every decode step one
            victim per layer, victims are mostly PROMPT tokens (prefill-eviction
            flavor). Generation is short (the answer).

Arms (paired per example, one eager model instance, same as oracle_eval):
  full, random, kv_norm, attn_cur (H2O-style current attention),
  oracle_fut (golden eviction from the full trace).

Predicted signature IF the dataset hypothesis is right (RULER-like):
  full ~ 1.0;  kv_norm ~ random (norms don't know where the needle is);
  oracle_fut >> kv_norm (future attention trivially keeps the needle).
That reproduces the qualitative Apple result and cleanly separates
"our method/env is broken" from "GSM8K has no learnable eviction signal".

Resume-safe JSONL, same format as oracle_eval.

Usage:
  python scripts/eval_passkey.py --n 96 --budget 176 --out-dir runs/passkey_eval
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_eval import full_trace_with_future_attention, make_oracle_evict_fn  # noqa: E402

from kv_gym.capture import capture                                  # noqa: E402
from kv_gym.eval_core import (                                       # noqa: E402
    run_online_episode, make_kv_norm_evict_fn, make_attention_evict_fn,
    make_random_evict_fn,
)
from kv_gym.vendor.loader import load_model_and_tokenizer            # noqa: E402
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract   # noqa: E402

FILLER = [
    "The weather in the northern valley stays mild for most of the spring season.",
    "Local markets open early and sell fresh produce from the nearby farms.",
    "The old library was renovated last year and now hosts weekly reading clubs.",
    "Trains between the two cities run every hour during the working week.",
    "The museum's new wing displays paintings from the early modern period.",
    "Cyclists prefer the river path because it avoids the steep northern hills.",
    "The bakery on the corner is known for its sourdough and rye loaves.",
    "Evening classes at the community center cover pottery and woodworking.",
    "The harbor gets crowded when the fishing boats return before sunset.",
    "A small observatory on the ridge opens to visitors on clear nights.",
    "The annual festival brings musicians from all over the region.",
    "Most offices in the district close early on the last Friday of the month.",
]


def make_passkey_examples(n: int, seed: int, n_filler: int = 14):
    """n examples in the project-wide schema (prompt_text, gold_answers)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        code = rng.randint(10000, 99999)
        sents = [rng.choice(FILLER) for _ in range(n_filler)]
        depth = rng.randint(1, n_filler - 1)   # never first/last for uniformity
        sents.insert(depth, f"Remember this: the secret code is {code}.")
        # Long forced generation BEFORE the answer: each decode step evicts one
        # slot per layer, so eviction count ~ generation length. Counting to 40
        # (~200 tokens) forces ~200 evictions/layer; the needle must SURVIVE the
        # whole generation to be retrievable at the end (v1 with a short answer
        # gave everything=1.0: only ~15 evictions, needle always survived).
        prompt = (" ".join(sents)
                  + " Task: first count from 1 to 40, writing every number. "
                    "After the counting, write exactly one final sentence: "
                    "'The secret code is X.' where X is the secret code "
                    "mentioned above.")
        out.append({"prompt_text": prompt, "gold_answers": [str(code)], "raw_chat": True,
                    "depth_frac": depth / n_filler})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=96)
    p.add_argument("--budget", type=int, default=176)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/passkey_eval")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm, tokenizer, device = load_model_and_tokenizer(
        name="qwen-1.5b", device=device, attn_implementation="eager")
    L = llm.config.num_hidden_layers
    n_sinks, n_recent = 4, 32

    examples = make_passkey_examples(args.n, args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "passkey_results.jsonl"
    done = set()
    if results_path.exists():
        for line in open(results_path):
            try: done.add(json.loads(line)["idx"])
            except Exception: pass
        print(f"resume: {len(done)} done")

    kv_fn   = make_kv_norm_evict_fn(L, n_sinks, n_recent)
    attn_fn = make_attention_evict_fn(L, n_sinks, n_recent)
    rng     = np.random.default_rng(args.seed)

    for idx, ex in enumerate(examples):
        if idx in done:
            continue
        try:
            cap = capture(llm, tokenizer, ex, device)
            T = cap.prompt_len
            gold = ex["gold_answers"][0]

            full_text, suf, n_steps = full_trace_with_future_attention(
                llm, tokenizer, cap.input_ids, device, args.max_new_tokens)
            c_full = float(flexible_extract(full_text, [gold]))
            orc_fn = make_oracle_evict_fn(suf, L, T, n_sinks, n_recent)
            rnd_fn = make_random_evict_fn(
                L, np.random.default_rng(rng.integers(1 << 30)), n_sinks, n_recent)

            def run(fn):
                text, _, _, trunc, _, _ = run_online_episode(
                    llm, tokenizer, cap.input_ids, args.budget,
                    args.max_new_tokens, device, fn)
                return float(flexible_extract(text, [gold]))

            row = {"idx": idx, "T": T, "depth": ex["depth_frac"],
                   "steps_full": n_steps, "full": c_full,
                   "random": run(rnd_fn), "kv_norm": run(kv_fn),
                   "attn_cur": run(attn_fn), "oracle_fut": run(orc_fn)}
            del suf
            print(f"[{idx}] T={T} full={row['full']:.0f} rnd={row['random']:.0f} "
                  f"kv={row['kv_norm']:.0f} attn={row['attn_cur']:.0f} "
                  f"orc={row['oracle_fut']:.0f}")
        except Exception as e:  # noqa: BLE001
            row = {"idx": idx, "error": str(e)[:200]}
            print(f"[{idx}] ERROR {e}")
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    rows = [json.loads(l) for l in open(results_path)]
    ok = [r for r in rows if "error" not in r]
    if ok:
        n = len(ok)
        print(f"\n==== PASSKEY ARENA ({n} examples, budget={args.budget}) ====")
        for k in ("full", "oracle_fut", "attn_cur", "kv_norm", "random"):
            print(f"  {k:10s} {sum(r[k] for r in ok)/n:.4f}")
        d = sum(r["oracle_fut"] - r["kv_norm"] for r in ok) / n
        print(f"  PAIRED oracle_fut - kv_norm = {d:+.4f}")


if __name__ == "__main__":
    main()
