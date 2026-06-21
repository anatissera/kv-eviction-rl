"""
Evaluate a trained eviction policy against baselines on GSM8K test set.

Six strategies compared on a held-out set of GSM8K examples:

  full       — no eviction (upper bound)
  learned    — trained MaskablePPO policy with per-layer independent eviction
  streaming  — attention-sink: keep first n_sinks tokens + most recent (budget-n_sinks)
  attn_layer — per-layer attention oracle: keep top-budget by future attention per layer
               (requires attn_implementation='eager'; skipped otherwise)
  kv_norm    — keep top-budget tokens by ||K||+||V|| norm (global, same set per layer)
  random     — keep a random subset of budget tokens (same set per layer)

Also reports correlation between PPO eviction decisions and the per-layer attention
oracle (when available): a mean "attention rank percentile" near 0.0 means the policy
preferentially evicts low-attention tokens (like the oracle); near 0.5 is random.

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
    python scripts/eval.py --model checkpoints/run --config configs/train.yaml --budget 80 --n 100 --n-sinks 4
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
    p.add_argument("--model",   required=True, help="Path to saved MaskablePPO checkpoint")
    p.add_argument("--config",  default="configs/quickstart.yaml")
    p.add_argument("--budget",  type=int, default=None,
                   help="Tokens to keep per layer (must be < prompt length T). "
                        "Required for meaningful eval; defaults to 64 if not set.")
    p.add_argument("--n",       type=int, default=100, help="Number of eval examples")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--n-sinks", type=int, default=4,
                   help="Number of initial tokens to treat as attention sinks (StreamingLLM)")
    p.add_argument("--output",  default=None,
                   help="Path to save JSON results (optional)")
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


def streaming_resident(budget: int, T: int, L: int, n_sinks: int = 4) -> torch.Tensor:
    """StreamingLLM: keep first n_sinks tokens (attention sinks) + most recent tokens.

    The remaining (budget - n_sinks) slots go to the most recent tokens, since
    recency strongly predicts attention in autoregressive generation.
    Same mask applied to all layers.
    """
    n_sinks  = min(n_sinks, budget, T)
    n_recent = min(budget - n_sinks, T - n_sinks)
    resident = torch.zeros(L, T, dtype=torch.bool)
    resident[:, :n_sinks] = True
    if n_recent > 0:
        resident[:, T - n_recent:T] = True
    return resident


def kv_norm_resident(K: torch.Tensor, V: torch.Tensor, budget: int, T: int, L: int) -> torch.Tensor:
    """Top-budget tokens by combined ||K||+||V|| norm (SnapKV max over layers/heads).

    Global ranking: same set of tokens kept in every layer.
    """
    score = (K.norm(dim=-1) + V.norm(dim=-1)).amax(dim=(0, 1))  # [T]
    _, topk = score.topk(min(budget, T))
    resident = torch.zeros(L, T, dtype=torch.bool)
    resident[:, topk] = True
    return resident


def random_resident(budget: int, T: int, L: int, rng) -> torch.Tensor:
    """Random subset of budget tokens, same set in all layers."""
    kept = rng.choice(T, size=min(budget, T), replace=False)
    resident = torch.zeros(L, T, dtype=torch.bool)
    resident[:, kept] = True
    return resident


def capture_per_layer_attention(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 64,
) -> torch.Tensor | None:
    """Run a reference generate with output_attentions=True and return [L, T] importance.

    For each layer independently, sums attention from generated tokens back to
    each prompt position (GQA-aware: max within KV group, then mean over KV groups).
    Returns None if output_attentions is unsupported (SDPA / flash attention).
    """
    T          = input_ids.shape[1]
    n_layers   = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)

    importance = torch.zeros(n_layers, T, dtype=torch.float32)
    counts     = torch.zeros(n_layers, dtype=torch.float32)

    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        if not (hasattr(out, "attentions") and out.attentions):
            return None

        for step_attns in out.attentions:
            for l, layer_attn in enumerate(step_attns):
                if layer_attn is None or l >= n_layers:
                    continue
                # layer_attn: [1, H_q, 1, seq_len]
                seq_len  = layer_attn.shape[-1]
                n_prompt = min(T, seq_len)
                n_q      = layer_attn.shape[1]
                per_q    = layer_attn[0, :, 0, :n_prompt]  # [H_q, T_prompt]

                if n_q % n_kv_heads == 0:
                    group_size = n_q // n_kv_heads
                    # max within KV group (SnapKV convention), then mean over KV groups
                    per_kv = per_q.view(n_kv_heads, group_size, n_prompt).amax(dim=1)
                    attn_to_prompt = per_kv.mean(dim=0).cpu()
                else:
                    attn_to_prompt = per_q.mean(dim=0).cpu()

                importance[l, :n_prompt] += attn_to_prompt
                counts[l] += 1

    except (RuntimeError, ValueError, NotImplementedError):
        return None

    counts = counts.clamp(min=1).unsqueeze(1)
    importance /= counts
    totals = importance.sum(dim=1, keepdim=True).clamp(min=1e-8)
    importance /= totals   # normalize each layer to sum to 1
    return importance       # [L, T]


def attention_resident(per_layer_imp: torch.Tensor, budget: int, T: int, L: int) -> torch.Tensor:
    """Keep top-budget tokens per layer independently by future attention importance."""
    resident = torch.zeros(L, T, dtype=torch.bool)
    k = min(budget, T)
    for l in range(L):
        _, topk = per_layer_imp[l, :T].topk(k)
        resident[l, topk] = True
    return resident


def attention_percentile_of_eviction(
    per_layer_imp: torch.Tensor,  # [L, T], normalized per layer
    layer: int,
    evicted_tok: int,
    resident: torch.Tensor,       # [L, T] BEFORE eviction of evicted_tok
) -> float:
    """Fraction of remaining tokens with LOWER importance than the evicted token.

    0.0 = policy evicted the least important token (matches attention oracle).
    0.5 = random.
    1.0 = policy evicted the most important token (anti-correlated).
    """
    remaining  = resident[layer].nonzero(as_tuple=True)[0]
    n_rem      = len(remaining)
    if n_rem <= 1:
        return 0.5
    imp_evicted = per_layer_imp[layer, evicted_tok].item()
    imp_others  = per_layer_imp[layer, remaining].cpu()
    n_lower     = (imp_others < imp_evicted).sum().item()
    return n_lower / (n_rem - 1)


def wilson_ci(successes: float, n: int, z: float = 1.96):
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p      = successes / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - margin), float(centre + margin)


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

    budget         = args.budget if args.budget is not None else 64
    max_new_tokens = cfg.get("max_new_tokens", 524)
    max_len        = cfg.get("max_len", 256)
    examples       = load_gsm8k(n=args.n, seed=args.seed, split="test")

    L      = llm.config.num_hidden_layers
    H      = getattr(llm.config, "num_key_value_heads", llm.config.num_attention_heads)
    D      = llm.config.hidden_size // llm.config.num_attention_heads
    n_envs = L

    policy = MaskablePPO.load(args.model, device=device)

    # Check whether attention oracle is available before the main loop
    print("Probing attention oracle availability (requires attn_implementation='eager')...")
    _probe_cap = capture(llm, tokenizer, examples[0], device)
    _probe_imp = capture_per_layer_attention(llm, _probe_cap.input_ids, device, max_new_tokens=4)
    attn_available = _probe_imp is not None
    if attn_available:
        print("  Attention oracle: AVAILABLE")
    else:
        impl = getattr(llm.config, "_attn_implementation", "unknown")
        print(f"  Attention oracle: UNAVAILABLE (attn_implementation='{impl}'). "
              "Skipping attn_layer baseline and correlation analysis.")

    results_per_example = []
    full_scores, learned_scores, streaming_scores = [], [], []
    attn_scores, kv_norm_scores, random_scores    = [], [], []
    correlation_samples: list[float] = []   # per eviction-decision, attention rank percentile

    for ex_idx, ex in enumerate(examples):
        cap = capture(llm, tokenizer, ex, device)
        T   = cap.prompt_len

        if T > max_len:
            print(f"  [skip] example {ex_idx}: T={T} > max_len={max_len}")
            continue
        if T <= budget:
            print(f"  [skip] example {ex_idx}: T={T} <= budget={budget}, nothing to evict")
            continue

        K_layer = cap.K  # [L, H, T, D]
        V_layer = cap.V

        # Attention importance for this example (per-layer) — may be None
        per_layer_imp = (
            capture_per_layer_attention(llm, cap.input_ids, device, max_new_tokens)
            if attn_available else None
        )

        # ---------- full cache (upper bound) ----------
        full_scores.append(score_full_cache(
            llm, tokenizer, cap.input_ids, cap.gold_answer, device, max_new_tokens,
        ))

        # ---------- learned — per-layer independent eviction with correlation tracking ----------
        resident = torch.ones(L, T, dtype=torch.bool)
        for _ in range(T - budget):
            obs   = build_obs(K_layer, V_layer, T, max_len, resident_mask=resident)
            masks = np.zeros((n_envs, max_len), dtype=bool)
            masks[:, :T] = resident.numpy()
            actions, _ = policy.predict(obs, action_masks=masks, deterministic=True)
            for l, tok in enumerate(int(a) for a in actions):
                if tok < T and resident[l, tok]:
                    if per_layer_imp is not None:
                        pct = attention_percentile_of_eviction(per_layer_imp, l, tok, resident)
                        correlation_samples.append(pct)
                    resident[l, tok] = False

        learned_scores.append(score_with_resident(
            llm, tokenizer, cap.input_ids, resident,
            cap.gold_answer, device, max_new_tokens,
        ))

        # ---------- StreamingLLM: sinks + recency ----------
        streaming_scores.append(score_with_resident(
            llm, tokenizer, cap.input_ids,
            streaming_resident(budget, T, L, args.n_sinks),
            cap.gold_answer, device, max_new_tokens,
        ))

        # ---------- per-layer attention oracle ----------
        if per_layer_imp is not None:
            attn_scores.append(score_with_resident(
                llm, tokenizer, cap.input_ids,
                attention_resident(per_layer_imp, budget, T, L),
                cap.gold_answer, device, max_new_tokens,
            ))

        # ---------- KV norm oracle — global, same mask all layers ----------
        kv_norm_scores.append(score_with_resident(
            llm, tokenizer, cap.input_ids,
            kv_norm_resident(cap.K, cap.V, budget, T, L),
            cap.gold_answer, device, max_new_tokens,
        ))

        # ---------- random ----------
        ex_rng = np.random.default_rng(args.seed + ex_idx)
        random_scores.append(score_with_resident(
            llm, tokenizer, cap.input_ids,
            random_resident(budget, T, L, ex_rng),
            cap.gold_answer, device, max_new_tokens,
        ))

        row = {
            "example": ex_idx,
            "prompt_len": T,
            "full":      full_scores[-1],
            "learned":   learned_scores[-1],
            "streaming": streaming_scores[-1],
            "kv_norm":   kv_norm_scores[-1],
            "random":    random_scores[-1],
        }
        if per_layer_imp is not None:
            row["attn_layer"] = attn_scores[-1]
        results_per_example.append(row)
        print(f"  ex {ex_idx:3d}  T={T:3d}  "
              f"full={row['full']:.0f}  learned={row['learned']:.0f}  "
              f"streaming={row['streaming']:.0f}  "
              + (f"attn={row['attn_layer']:.0f}  " if "attn_layer" in row else "")
              + f"kv_norm={row['kv_norm']:.0f}  random={row['random']:.0f}")

    # ---- Summary ----
    n = len(full_scores)
    full_mean = float(np.mean(full_scores)) if n else 0.0

    print(f"\n{'='*60}")
    print(f"Results: {n} examples  budget={budget}  n_sinks={args.n_sinks}  max_new_tokens={max_new_tokens}")
    print(f"{'='*60}")
    print(f"{'strategy':<14} {'mean':>6}  {'95% CI':>15}  {'vs full':>8}")
    print("-" * 50)

    all_rows = [
        ("full",       full_scores),
        ("learned",    learned_scores),
        ("streaming",  streaming_scores),
        ("attn_layer", attn_scores),
        ("kv_norm",    kv_norm_scores),
        ("random",     random_scores),
    ]
    summary = {"budget": budget, "n_examples": n, "strategies": {}}
    for name, vals in all_rows:
        if not vals:
            continue
        mean   = float(np.mean(vals))
        lo, hi = wilson_ci(sum(vals), len(vals))
        ratio  = mean / max(full_mean, 1e-8)
        print(f"{name:<14} {mean:>6.3f}  [{lo:.3f}, {hi:.3f}]  {ratio:>8.3f}")
        summary["strategies"][name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "vs_full": ratio}

    if correlation_samples:
        mean_pct = float(np.mean(correlation_samples))
        print(f"\nCorrelation with attention oracle:")
        print(f"  Mean attention rank percentile of evicted tokens: {mean_pct:.3f}")
        print(f"  (0.0 = always evicts least-attended token = oracle; 0.5 = random)")
        summary["correlation"] = {
            "mean_attention_rank_percentile": mean_pct,
            "n_decisions": len(correlation_samples),
        }
    else:
        print("\nCorrelation analysis unavailable (attention oracle not supported).")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary["per_example"] = results_per_example
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
