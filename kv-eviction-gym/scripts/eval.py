"""
Evaluate a trained eviction policy against baselines.

Four strategies compared on a held-out set of GSM8K examples:

  full     — no eviction (upper bound)
  learned  — the trained MaskablePPO policy with per-layer independent eviction
  oracle   — keep top-budget tokens by ||K|| + ||V|| norm (same set per layer)
  random   — keep a random subset of budget tokens (same set per layer)

Offline eval approach
---------------------
This script runs an OFFLINE evaluation: the policy sees the prefill K/V and
makes T−budget sequential eviction decisions to compress the prompt down to
`budget` tokens, then generates with the compressed cache.

The policy was TRAINED online (one eviction per decode step) so the eval
distribution differs from training.  This gives a lower bound on true
performance.  A proper online eval would replicate full episodes.

`budget` must be smaller than the prompt length T (otherwise nothing is
evicted).  Pass `--budget` explicitly; a typical value for GSM8K is 64–100.

Usage:
    python scripts/eval.py --model checkpoints/run --config configs/train.yaml --budget 80
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml

from sb3_contrib import MaskablePPO

from kv_gym.capture import capture
from kv_gym.features import build_obs, feature_dim
from kv_gym.rewards.per_layer_generate import generate_with_per_layer_eviction
from kv_gym.vendor.loader import load_model_and_tokenizer
from kv_gym.vendor.gsm8k import load_gsm8k
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, help="Path to saved MaskablePPO checkpoint")
    p.add_argument("--config", default="configs/quickstart.yaml")
    p.add_argument("--budget", type=int, default=None,
                   help="Tokens to keep per layer (must be < prompt length T). "
                        "Required for meaningful eval; defaults to 64 if not set.")
    p.add_argument("--n",      type=int, default=50, help="Number of eval examples")
    p.add_argument("--seed",   type=int, default=42)
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


def score_full_cache(model, tokenizer, input_ids, gold, device, max_new_tokens):
    """Generate with the full unevicted cache (upper bound)."""
    T = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    text = tokenizer.decode(out[0, T:], skip_special_tokens=True)
    return flexible_extract(text, [gold])


def score_with_resident(model, tokenizer, input_ids, resident, gold, device, max_new_tokens):
    """Per-layer cache-slicing eviction → score correctness.

    resident: [L, T] bool — True = keep.  Each layer uses its OWN mask.
    """
    text = generate_with_per_layer_eviction(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        resident_mask=resident,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    return flexible_extract(text, [gold])


def oracle_resident(K, V, budget, T, L):
    """Top-budget tokens by combined ||K||+||V|| norm, same set in all layers."""
    score = (K.norm(dim=-1) + V.norm(dim=-1)).amax(dim=(0, 1))  # [T], SnapKV max
    _, topk = score.topk(min(budget, T))
    resident = torch.zeros(L, T, dtype=torch.bool)
    resident[:, topk] = True
    return resident


def random_resident(budget, T, L, rng):
    """Random subset of budget tokens, same set in all layers."""
    kept = rng.choice(T, size=min(budget, T), replace=False)
    resident = torch.zeros(L, T, dtype=torch.bool)
    resident[:, kept] = True
    return resident


def wilson_ci(successes, n, z=1.96):
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - margin), float(centre + margin)


def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    device_cfg = cfg.get("device", "auto")
    device = None if device_cfg == "auto" else torch.device(device_cfg)
    llm, tokenizer, device = load_model_and_tokenizer(
        name=cfg.get("model_name", "qwen-1.5b"),
        device=device,
        attn_implementation=cfg.get("attn_implementation", None),
    )

    # Offline eval: budget = number of prompt tokens to KEEP after compression.
    # Must be < T (prompt length); default 64 works for most GSM8K examples.
    budget         = args.budget if args.budget is not None else 64
    max_new_tokens = cfg.get("max_new_tokens", 524)
    max_len        = cfg.get("max_len", 256)
    # Eval always uses the held-out test split, independent of training data
    examples       = load_gsm8k(n=args.n, seed=args.seed, split="test")

    L      = llm.config.num_hidden_layers
    H      = getattr(llm.config, "num_key_value_heads", llm.config.num_attention_heads)
    D      = llm.config.hidden_size // llm.config.num_attention_heads
    n_envs = L   # one env per layer (matching training setup)
    fdim   = feature_dim(H, D)

    policy = MaskablePPO.load(args.model, device=device)

    full_scores, learned_scores, oracle_scores, random_scores = [], [], [], []

    for ex_idx, ex in enumerate(examples):
        cap = capture(llm, tokenizer, ex, device)
        T   = cap.prompt_len

        if T > max_len:
            print(f"  [skip] example {ex_idx}: T={T} > max_len={max_len}")
            continue
        if T <= budget:
            print(f"  [skip] example {ex_idx}: T={T} <= budget={budget}, nothing to evict")
            continue

        # K/V all heads, shape [L, H, T, D] — heads concatenated to match training obs
        K_layer = cap.K
        V_layer = cap.V

        # ---------- full cache (upper bound) ----------
        full_scores.append(score_full_cache(
            llm, tokenizer, cap.input_ids, cap.gold_answer, device, max_new_tokens,
        ))

        # ---------- learned — per-layer independent eviction ----------
        resident = torch.ones(L, T, dtype=torch.bool)
        for _ in range(T - budget):
            obs   = build_obs(K_layer, V_layer, T, max_len, resident_mask=resident)
            masks = np.zeros((n_envs, max_len), dtype=bool)
            masks[:, :T] = resident.numpy()
            actions, _ = policy.predict(obs, action_masks=masks, deterministic=True)
            for l, tok in enumerate(int(a) for a in actions):
                if tok < T and resident[l, tok]:
                    resident[l, tok] = False

        learned_scores.append(score_with_resident(
            llm, tokenizer, cap.input_ids, resident,
            cap.gold_answer, device, max_new_tokens,
        ))

        # ---------- oracle — global norm ranking, same mask all layers ----------
        oracle_scores.append(score_with_resident(
            llm, tokenizer, cap.input_ids,
            oracle_resident(cap.K, cap.V, budget, T, L),
            cap.gold_answer, device, max_new_tokens,
        ))

        # ---------- random (seeded per example for reproducibility) ----------
        ex_rng = np.random.default_rng(args.seed + ex_idx)
        random_scores.append(score_with_resident(
            llm, tokenizer, cap.input_ids,
            random_resident(budget, T, L, ex_rng),
            cap.gold_answer, device, max_new_tokens,
        ))

    n = len(full_scores)
    full_mean = float(np.mean(full_scores)) if n else 0.0

    print(f"\nResults over {n} examples  (budget={budget}, max_new_tokens={max_new_tokens}):")
    print(f"{'strategy':<10} {'mean':>6}  {'95% CI':>15}  {'vs full':>8}")
    print("-" * 48)
    for name, vals in [
        ("full",    full_scores),
        ("learned", learned_scores),
        ("oracle",  oracle_scores),
        ("random",  random_scores),
    ]:
        if not vals:
            continue
        mean  = float(np.mean(vals))
        lo, hi = wilson_ci(sum(vals), len(vals))
        ratio = mean / max(full_mean, 1e-8)
        print(f"{name:<10} {mean:>6.3f}  [{lo:.3f}, {hi:.3f}]  {ratio:>8.3f}")

    if learned_scores and oracle_scores:
        print(f"\nLearned / Oracle: {np.mean(learned_scores) / max(np.mean(oracle_scores), 1e-8):.3f}")
    if learned_scores and random_scores:
        print(f"Learned / Random: {np.mean(learned_scores) / max(np.mean(random_scores), 1e-8):.3f}")


if __name__ == "__main__":
    main()
