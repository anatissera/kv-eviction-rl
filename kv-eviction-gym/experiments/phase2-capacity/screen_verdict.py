#!/usr/bin/env python3
"""E0 capacity-screen verdict.

Loads the 5 screen variants' probe curves (probe_on_train, so learned/kv_norm are
on the SAME overfit examples) and reports, per variant, the best paired gap
(learned − kv_norm) reached at ANY probe (best-case overfit capacity). Then maps
the pattern to which deficiency was the real bottleneck (see PLAN.md §3).

Usage: screen_verdict.py <dir-with clean CSVs named e0_*_probe_curve.csv>
"""
import csv, sys, os

VARIANTS = ["e0_baseline", "e0_rich", "e0_rich_s4", "e0_rich_warm", "e0_attn"]
LABEL = {
    "e0_baseline": "baseline (no rich)",
    "e0_rich":     "rich features",
    "e0_rich_s4":  "rich + S4",
    "e0_rich_warm":"rich + warm-start",
    "e0_attn":     "rich + attention",
}

def load(path):
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        return []
    out = []
    for r in rows:
        try:
            out.append((float(r["correct_learned"]), float(r["correct_kv_norm"]),
                        float(r.get("timestep", 0))))
        except (KeyError, ValueError):
            pass
    return out

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    print("#" * 64)
    print("# E0 CAPACITY SCREEN — best overfit gap (learned − kv_norm) per variant")
    print("#" * 64 + "\n")
    best = {}
    for v in VARIANTS:
        rows = load(os.path.join(d, f"{v}_probe_curve.csv"))
        if not rows:
            print(f"  {LABEL[v]:22s}  NO DATA")
            best[v] = None
            continue
        gaps = [(lr - kv, lr, kv, ts) for (lr, kv, ts) in rows]
        bg, blr, bkv, bts = max(gaps, key=lambda x: x[0])
        best[v] = bg
        flag = "✓ BEATS" if bg > 0.02 else ("≈ ties" if bg > -0.02 else "✗ below")
        print(f"  {LABEL[v]:22s}  best learned−kv_norm = {bg:+.3f}  "
              f"(learned={blr:.3f} kv_norm={bkv:.3f} @ {bts/1e3:.0f}k)  {flag}")
    print()
    def beats(v): return best.get(v) is not None and best[v] > 0.02
    def ties(v):  return best.get(v) is not None and best[v] > -0.02
    print("VERDICT:")
    if beats("e0_rich") and not beats("e0_baseline"):
        print("  → REPRESENTATION was the bottleneck (D1/D2). Rich features unlock beating")
        print("    kv_norm; baseline cannot. SCALE e0_rich (+ e0_rich_s4 if it adds).")
    elif beats("e0_rich_warm") and not beats("e0_rich"):
        print("  → EXPLORATION was the bottleneck (D4). Warm-start reaches >kv_norm where")
        print("    from-scratch rich does not. SCALE e0_rich_warm.")
    elif beats("e0_attn") and not (beats("e0_rich") or beats("e0_rich_warm")):
        print("  → PER-TOKEN ARCHITECTURE was the ceiling (D3). Only cross-token attention")
        print("    beats kv_norm. SCALE e0_attn.")
    elif any(beats(v) for v in VARIANTS):
        winners = [LABEL[v] for v in VARIANTS if beats(v)]
        print(f"  → Multiple/other winners beat kv_norm: {winners}. Scale the best.")
    elif any(ties(v) for v in VARIANTS if v != "e0_baseline"):
        print("  → Nothing BEATS kv_norm even on the overfit set, but rich variants TIE it.")
        print("    kv_norm may be near-optimal per-token; the reward/objective is the ceiling.")
        print("    Rethink objective (e.g. reward = accuracy-retention directly) before scaling.")
    else:
        print("  → Nothing reaches kv_norm even overfitting. Check the pipeline (features/")
        print("    reward wiring) before spending more compute.")

if __name__ == "__main__":
    main()
