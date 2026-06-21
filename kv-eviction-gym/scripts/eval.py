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
from torch import Tensor

from sb3_contrib import MaskablePPO

from kv_gym.capture import capture
from kv_gym.features import build_obs
from kv_gym.vendor.loader import load_model_and_tokenizer
from kv_gym.vendor.gsm8k import load_gsm8k
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract


# ── Cache utilities ───────────────────────────────────────────────────────────

def _get_cache_size(past_kv) -> int:
    if hasattr(past_kv, "layers"):
        return past_kv.layers[0].keys.shape[2]
    return past_kv.key_cache[0].shape[2]


def _extract_all_kv(past_kv, L: int) -> tuple[Tensor, Tensor]:
    """Return K, V stacked over layers: [L, H, seq, D] on CPU float32."""
    K_list, V_list = [], []
    for l in range(L):
        if hasattr(past_kv, "layers"):
            k, v = past_kv.layers[l].keys, past_kv.layers[l].values
        else:
            k, v = past_kv.key_cache[l], past_kv.value_cache[l]
        K_list.append(k.squeeze(0).float().cpu())
        V_list.append(v.squeeze(0).float().cpu())
    return torch.stack(K_list), torch.stack(V_list)


def _evict_per_layer(past_kv, evict_slots: list[int]) -> None:
    """Remove one slot per layer from the live DynamicCache."""
    if hasattr(past_kv, "layers"):
        for l, s in enumerate(evict_slots):
            seq = past_kv.layers[l].keys.shape[2]
            keep = torch.tensor([i for i in range(seq) if i != s],
                                device=past_kv.layers[l].keys.device)
            past_kv.layers[l].keys   = past_kv.layers[l].keys.index_select(2, keep)
            past_kv.layers[l].values = past_kv.layers[l].values.index_select(2, keep)
    else:
        for l, s in enumerate(evict_slots):
            seq = past_kv.key_cache[l].shape[2]
            keep = torch.tensor([i for i in range(seq) if i != s],
                                device=past_kv.key_cache[l].device)
            past_kv.key_cache[l]   = past_kv.key_cache[l].index_select(2, keep)
            past_kv.value_cache[l] = past_kv.value_cache[l].index_select(2, keep)


# ── Position tracker ──────────────────────────────────────────────────────────

class PositionTracker:
    """Tracks original sequence position of each cache slot, per layer.

    Necessary for the attention oracle: after evictions, cache slots no longer
    correspond 1-to-1 to original positions, so we can't look up importance
    directly by slot index.
    """

    def __init__(self, L: int, T: int):
        self._pos: list[list[int]] = [list(range(T)) for _ in range(L)]

    def evict(self, layer: int, slot: int) -> None:
        self._pos[layer].pop(slot)

    def append_all(self, original_pos: int) -> None:
        """New generated token appended to every layer at the given position."""
        for p in self._pos:
            p.append(original_pos)

    def original_pos(self, layer: int, slot: int) -> int:
        return self._pos[layer][slot]

    def size(self, layer: int = 0) -> int:
        return len(self._pos[layer])


# ── Strategy factories ────────────────────────────────────────────────────────

def make_learned_evict_fn(policy, L: int, max_len: int):
    """PPO policy: observe current K/V and predict one eviction slot per layer."""
    def evict(K: Tensor, V: Tensor, tracker: PositionTracker) -> list[int]:
        cache_size = K.shape[2]
        obs   = build_obs(K, V, cache_size, max_len)          # [L, max_len, feat]
        masks = np.zeros((L, max_len), dtype=bool)
        masks[:, :cache_size] = True
        actions, _ = policy.predict(obs, action_masks=masks, deterministic=True)
        return [int(a) for a in actions]
    return evict


def make_streaming_evict_fn(n_sinks: int, L: int):
    """StreamingLLM: evict the oldest non-sink slot (always slot n_sinks)."""
    def evict(K: Tensor, V: Tensor, tracker: PositionTracker) -> list[int]:
        return [n_sinks] * L
    return evict


def make_attention_evict_fn(per_layer_imp: Tensor, L: int):
    """Attention oracle: evict the slot whose original position has least future attention.

    per_layer_imp: [L, T_prompt] normalised importance from a reference generate.
    Generated tokens that appear after the prefill get importance 0 (not in the
    prefill importance tensor) — the oracle naturally prefers to evict them over
    high-importance prompt tokens.
    """
    T_imp = per_layer_imp.shape[1]

    def evict(K: Tensor, V: Tensor, tracker: PositionTracker) -> list[int]:
        slots = []
        for l in range(L):
            n = tracker.size(l)
            imps = [
                per_layer_imp[l, tracker.original_pos(l, s)].item()
                if tracker.original_pos(l, s) < T_imp else 0.0
                for s in range(n)
            ]
            slots.append(int(np.argmin(imps)))
        return slots

    return evict


def make_kv_norm_evict_fn(L: int):
    """KV-norm: per layer, evict the slot with the lowest current ||K||+||V|| norm."""
    def evict(K: Tensor, V: Tensor, tracker: PositionTracker) -> list[int]:
        # K, V: [L, H, seq, D] — mean over heads
        norms = (K.norm(dim=-1) + V.norm(dim=-1)).mean(dim=1)  # [L, seq]
        return [int(norms[l].argmin().item()) for l in range(L)]
    return evict


def make_random_evict_fn(L: int, rng):
    """Random: uniformly random slot per layer."""
    def evict(K: Tensor, V: Tensor, tracker: PositionTracker) -> list[int]:
        return [int(rng.integers(0, K.shape[2])) for _ in range(L)]
    return evict


# ── Online episode runner ─────────────────────────────────────────────────────

@torch.no_grad()
def run_online_episode(
    model,
    tokenizer,
    input_ids:      Tensor,
    budget:         int,
    max_new_tokens: int,
    device:         torch.device,
    evict_fn,
    per_layer_imp:  Tensor | None = None,  # [L, T_prompt] for correlation tracking
) -> tuple[str, list[float]]:
    """Run one full online episode.

    At each decode step the cache grows by one token.  When cache_size > budget
    the evict_fn removes one token per layer before decoding continues.

    Returns:
        generated text and a list of attention-rank percentiles for each eviction
        decision (empty if per_layer_imp is None).
    """
    L   = model.config.num_hidden_layers
    T   = input_ids.shape[1]
    eos = tokenizer.eos_token_id

    out      = model(input_ids=input_ids.to(device), use_cache=True)
    past_kv  = out.past_key_values
    next_tok = int(out.logits[0, -1].argmax())
    true_pos = T
    generated: list[int] = []
    correlation: list[float] = []
    tracker = PositionTracker(L, T)

    for _ in range(max_new_tokens):
        if eos is not None and next_tok == eos:
            break
        generated.append(next_tok)

        cache_size = _get_cache_size(past_kv)
        if cache_size > budget:
            K, V = _extract_all_kv(past_kv, L)
            slots = evict_fn(K, V, tracker)

            # Correlation: importance rank percentile of chosen slot vs attention oracle
            if per_layer_imp is not None:
                T_imp = per_layer_imp.shape[1]
                for l, slot in enumerate(slots):
                    n = tracker.size(l)
                    imp_chosen = (
                        per_layer_imp[l, tracker.original_pos(l, slot)].item()
                        if tracker.original_pos(l, slot) < T_imp else 0.0
                    )
                    all_imps = [
                        per_layer_imp[l, tracker.original_pos(l, s)].item()
                        if tracker.original_pos(l, s) < T_imp else 0.0
                        for s in range(n)
                    ]
                    n_lower = sum(1 for imp in all_imps if imp < imp_chosen)
                    if n - 1 > 0:
                        correlation.append(n_lower / (n - 1))

            _evict_per_layer(past_kv, slots)
            for l, s in enumerate(slots):
                tracker.evict(l, s)

        pos      = torch.tensor([[true_pos]], device=device)
        step_out = model(
            input_ids=torch.tensor([[next_tok]], device=device),
            past_key_values=past_kv,
            position_ids=pos,
            cache_position=pos.squeeze(0),
            use_cache=True,
        )
        next_tok = int(step_out.logits[0, -1].argmax())
        past_kv  = step_out.past_key_values
        tracker.append_all(true_pos)
        true_pos += 1

    return tokenizer.decode(generated, skip_special_tokens=True), correlation


# ── Misc helpers ──────────────────────────────────────────────────────────────

def score_full_cache(model, tokenizer, input_ids, gold, device, max_new_tokens):
    T = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    text = tokenizer.decode(out[0, T:], skip_special_tokens=True)
    return flexible_extract(text, [gold])


def capture_per_layer_attention(model, input_ids, device, max_new_tokens=64):
    """Run reference generate with output_attentions=True → [L, T] per-layer importance.

    Returns None if output_attentions is unsupported (SDPA / flash_attention_2).
    """
    T          = input_ids.shape[1]
    n_layers   = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, "num_key_value_heads",
                         model.config.num_attention_heads)
    importance = torch.zeros(n_layers, T)
    counts     = torch.zeros(n_layers)

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
                seq_len  = layer_attn.shape[-1]
                n_prompt = min(T, seq_len)
                n_q      = layer_attn.shape[1]
                per_q    = layer_attn[0, :, 0, :n_prompt]
                if n_q % n_kv_heads == 0:
                    g = n_q // n_kv_heads
                    per_kv = per_q.view(n_kv_heads, g, n_prompt).amax(dim=1)
                    attn   = per_kv.mean(dim=0).cpu()
                else:
                    attn = per_q.mean(dim=0).cpu()
                importance[l, :n_prompt] += attn
                counts[l] += 1

    except (RuntimeError, ValueError, NotImplementedError):
        return None

    counts = counts.clamp(min=1).unsqueeze(1)
    importance /= counts
    importance /= importance.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return importance  # [L, T]


def wilson_ci(successes: float, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p      = successes / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - margin), float(centre + margin)


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
            text, corr = run_online_episode(
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
