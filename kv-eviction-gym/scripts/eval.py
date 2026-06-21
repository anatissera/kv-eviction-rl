"""
Evaluate a trained eviction policy against baselines on GSM8K test set.

All strategies run ONLINE: at each decode step, if cache_size > budget, one token
is evicted per layer before the next token is generated.  This matches how the
policy was trained and gives every method the same hard memory constraint.

Strategies compared:
  full       — no eviction (upper bound; cache grows freely)
  learned    — trained MaskablePPO policy with per-layer independent eviction
  streaming  — attention-sink: always evict the oldest non-sink token (slot n_sinks)
  attn_layer — per-layer attention oracle: evict the slot whose original prompt position
               has the lowest future-attention importance (requires eager attention)
  kv_norm    — per-layer: evict the slot with the lowest current ||K||+||V|| norm
  random     — evict a uniformly random slot (per layer, seeded)

Correlation analysis (when attn_layer is available):
  For each eviction decision made by the learned policy, compute the attention-importance
  rank percentile of the chosen slot among remaining slots:
    0.0 = always evicts the least-attended token (mimics oracle)
    0.5 = random
    1.0 = always evicts the most-attended token (anti-correlated)

The online decode loop, eviction strategies and helpers live in kv_gym.eval_core so the
training-time monitoring probe (kv_gym.probe) measures exactly the same thing.

Usage:
    python scripts/eval.py --model runs/my_run/best_model \\
                           --config configs/train.yaml \\
                           --budget 180 --n 100 \\
                           --output runs/my_run/eval_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sb3_contrib import MaskablePPO

from kv_gym.capture import capture
from kv_gym.vendor.loader import load_model_and_tokenizer
from kv_gym.vendor.gsm8k import load_gsm8k
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract
from kv_gym.eval_core import (
    capture_per_layer_attention,
    make_attention_evict_fn,
    make_kv_norm_evict_fn,
    make_learned_evict_fn,
    make_random_evict_fn,
    make_streaming_evict_fn,
    run_online_episode,
    score_full_cache,
    wilson_ci,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, help="Path to saved MaskablePPO checkpoint")
    p.add_argument("--config",  default="configs/quickstart.yaml")
    p.add_argument("--budget",  type=int, default=180,
                   help="Hard cache budget (tokens). Must be >= typical prompt length.")
    p.add_argument("--n",       type=int, default=100, help="Number of eval examples")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--n-sinks", type=int, default=4,
                   help="Attention sink count for StreamingLLM baseline")
    p.add_argument("--output",  default=None, help="Path to save JSON results (optional)")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    defaults_path = Path(path).parent / "model_defaults.json"
    if defaults_path.exists():
        with open(defaults_path) as f:
            all_defaults = json.load(f)
        model_key = cfg.get("model_name", "")
        for k, v in all_defaults.get(model_key, {}).items():
            if not k.startswith("_") and k not in cfg:
                cfg[k] = v
    return cfg


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    device_cfg = cfg.get("device", "auto")
    device = None if device_cfg == "auto" else torch.device(device_cfg)
    llm, tokenizer, device = load_model_and_tokenizer(
        name=cfg.get("model_name", "qwen-1.5b"),
        device=device,
        attn_implementation=cfg.get("attn_implementation", None),
    )

    budget         = args.budget
    max_new_tokens = cfg.get("max_new_tokens", 524)
    max_len        = cfg.get("max_len", 512)
    examples       = load_gsm8k(n=args.n, seed=args.seed, split="test")

    L = llm.config.num_hidden_layers

    policy = MaskablePPO.load(args.model, device=device)

    # Probe whether attention oracle is available (requires eager attention)
    print("Probing attention oracle (requires attn_implementation='eager')...")
    _probe_cap = capture(llm, tokenizer, examples[0], device)
    _probe_imp = capture_per_layer_attention(llm, _probe_cap.input_ids, device,
                                             max_new_tokens=4)
    attn_available = _probe_imp is not None
    impl = getattr(llm.config, "_attn_implementation", "unknown")
    print(f"  Attention oracle: {'AVAILABLE' if attn_available else f'UNAVAILABLE ({impl})'}")
    if not attn_available:
        print("  → attn_layer baseline and correlation analysis will be skipped.")
    print()

    scores: dict[str, list[float]] = {
        "full": [], "learned": [], "streaming": [],
        "attn_layer": [], "kv_norm": [], "random": [],
    }
    all_correlation: list[float] = []
    per_example_rows: list[dict] = []

    for ex_idx, ex in enumerate(examples):
        cap = capture(llm, tokenizer, ex, device)
        T   = cap.prompt_len

        if T > max_len:
            print(f"  [skip] ex {ex_idx}: T={T} > max_len={max_len}")
            continue
        if T >= budget:
            print(f"  [skip] ex {ex_idx}: T={T} >= budget={budget} (nothing to evict)")
            continue

        per_layer_imp = (
            capture_per_layer_attention(llm, cap.input_ids, device, max_new_tokens)
            if attn_available else None
        )

        gold = cap.gold_answer
        ids  = cap.input_ids

        # ---- full cache (upper bound, no eviction) ----
        scores["full"].append(score_full_cache(
            llm, tokenizer, ids, gold, device, max_new_tokens,
        ))

        def _score(evict_fn, ref_imp=None):
            text, corr, _ = run_online_episode(
                llm, tokenizer, ids, budget, max_new_tokens, device,
                evict_fn, per_layer_imp=ref_imp,
            )
            return flexible_extract(text, [gold]), corr

        ex_rng = np.random.default_rng(args.seed + ex_idx)

        # ---- learned ----
        s, corr = _score(
            make_learned_evict_fn(policy, L, max_len),
            ref_imp=per_layer_imp,
        )
        scores["learned"].append(s)
        all_correlation.extend(corr)

        # ---- streaming ----
        scores["streaming"].append(_score(make_streaming_evict_fn(args.n_sinks, L))[0])

        # ---- per-layer attention oracle ----
        if per_layer_imp is not None:
            scores["attn_layer"].append(_score(make_attention_evict_fn(per_layer_imp, L))[0])

        # ---- kv norm (per-layer, online) ----
        scores["kv_norm"].append(_score(make_kv_norm_evict_fn(L))[0])

        # ---- random ----
        scores["random"].append(_score(make_random_evict_fn(L, ex_rng))[0])

        row = {
            "example": ex_idx, "prompt_len": T,
            "full":      scores["full"][-1],
            "learned":   scores["learned"][-1],
            "streaming": scores["streaming"][-1],
            "kv_norm":   scores["kv_norm"][-1],
            "random":    scores["random"][-1],
        }
        if per_layer_imp is not None:
            row["attn_layer"] = scores["attn_layer"][-1]
        per_example_rows.append(row)

        attn_str = f"attn={row['attn_layer']:.0f}  " if "attn_layer" in row else ""
        print(f"  ex {ex_idx:3d}  T={T:3d}  "
              f"full={row['full']:.0f}  learned={row['learned']:.0f}  "
              f"streaming={row['streaming']:.0f}  {attn_str}"
              f"kv_norm={row['kv_norm']:.0f}  random={row['random']:.0f}")

    # ── Summary ──
    n         = len(scores["full"])
    full_mean = float(np.mean(scores["full"])) if n else 0.0

    print(f"\n{'='*62}")
    print(f"Results: {n} examples  budget={budget}  "
          f"n_sinks={args.n_sinks}  max_new_tokens={max_new_tokens}")
    print(f"{'='*62}")
    print(f"{'strategy':<14} {'mean':>6}  {'95% CI':>15}  {'vs full':>8}")
    print("-" * 50)

    summary = {"budget": budget, "n_examples": n, "strategies": {}}
    for name, vals in [
        ("full",       scores["full"]),
        ("learned",    scores["learned"]),
        ("streaming",  scores["streaming"]),
        ("attn_layer", scores["attn_layer"]),
        ("kv_norm",    scores["kv_norm"]),
        ("random",     scores["random"]),
    ]:
        if not vals:
            continue
        mean   = float(np.mean(vals))
        lo, hi = wilson_ci(sum(vals), len(vals))
        ratio  = mean / max(full_mean, 1e-8)
        print(f"{name:<14} {mean:>6.3f}  [{lo:.3f}, {hi:.3f}]  {ratio:>8.3f}")
        summary["strategies"][name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi,
                                        "vs_full": ratio}

    if all_correlation:
        mean_pct = float(np.mean(all_correlation))
        print(f"\nCorrelation: PPO vs attention oracle")
        print(f"  Mean attention rank percentile of evicted tokens: {mean_pct:.3f}")
        print(f"  (0.0 = always evicts least-attended; 0.5 = random)")
        summary["correlation"] = {
            "mean_attention_rank_percentile": mean_pct,
            "n_decisions": len(all_correlation),
        }
    else:
        print("\nCorrelation unavailable (attention oracle not supported).")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary["per_example"] = per_example_rows
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
