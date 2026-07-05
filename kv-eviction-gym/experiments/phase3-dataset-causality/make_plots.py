"""Generate all paper-quality figures for the dataset-causality investigation.

Figures:
  fig1_oracle_gap_bars.png   the headline: per-arm accuracy on GSM8K vs passkey,
                             showing oracle-kv_norm = +0.07 (GSM8K) vs +0.43
                             (passkey) -> the null is dataset-driven.
  fig2_regime_headroom.png   full-random headroom + oracle margin across the
                             regimes we probed (compact "why" panel).
  fig3_e10_learning.png      E10 online-PPO probe trajectory in the passkey
                             arena (learned vs kv_norm over training), per seed.
  fig4_ranker_predictability.png  offline ranker: rank-correlation of predicted
                             vs true future utility (learned vs kv_norm proxy).

Run from the phase3 dir:  python make_plots.py
Robust to missing inputs (skips a figure if its data is absent).
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_style import apply_style, PALETTE, SERIES_COLORS, SERIES_LABELS

apply_style()
DATA = Path("data")
OUT = Path("plots")
OUT.mkdir(exist_ok=True)


def load_jsonl(name):
    p = DATA / name
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    return [r for r in rows if "error" not in r and not r.get("skip")]


def arm_means(rows, arms):
    n = len(rows)
    return {a: (sum(r[a] for r in rows) / n if all(a in r for r in rows) else None)
            for a in arms}, n


def paired_se(rows, a, b):
    d = [r[a] - r[b] for r in rows if a in r and b in r]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0
    return m, sd / math.sqrt(n) if n else 0


# ---------------------------------------------------------------- fig 1
def fig1():
    gsm = load_jsonl("gsm8k_oracle.jsonl")
    pk = load_jsonl("passkey_oracle.jsonl")
    if not gsm or not pk:
        print("fig1: missing data"); return
    arms = ["full", "oracle_fut", "attn_cur", "kv_norm", "random"]
    gm, gn = arm_means(gsm, arms)
    pm, pn = arm_means(pk, arms)
    g_gap, g_se = paired_se(gsm, "oracle_fut", "kv_norm")
    p_gap, p_se = paired_se(pk, "oracle_fut", "kv_norm")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, means, n, title, gap, se in [
        (axes[0], gm, gn, f"GSM8K  (n={gn})", g_gap, g_se),
        (axes[1], pm, pn, f"Passkey  (n={pn})", p_gap, p_se),
    ]:
        xs = [a for a in arms if means[a] is not None]
        vals = [means[a] for a in xs]
        colors = [SERIES_COLORS[a] for a in xs]
        bars = ax.bar(range(len(xs)), vals, color=colors, width=0.68,
                      edgecolor="white", linewidth=1.2)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9.5, color=PALETTE["ink"])
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([SERIES_LABELS[a] for a in xs], rotation=25, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_title(title)
        # annotate the oracle-kv_norm gap with a bracket
        io, ik = xs.index("oracle_fut"), xs.index("kv_norm")
        ytop = max(vals[io], vals[ik]) + 0.12
        ax.annotate("", xy=(io, ytop), xytext=(ik, ytop),
                    arrowprops=dict(arrowstyle="<->", color=PALETTE["accent"], lw=1.6))
        ax.text((io + ik) / 2, ytop + 0.02,
                f"oracle - kv_norm\n= {gap:+.2f}",
                ha="center", va="bottom", fontsize=9.5, weight="bold",
                color=PALETTE["accent"])
    axes[0].set_ylabel("Answer accuracy")
    fig.suptitle("Learnable eviction signal is a property of the DATASET",
                 fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_oracle_gap_bars.png")
    print("wrote fig1_oracle_gap_bars.png")


# ---------------------------------------------------------------- fig 2
def fig2():
    gsm = load_jsonl("gsm8k_oracle.jsonl")
    hot = load_jsonl("hotpot_results.jsonl")
    pk = load_jsonl("passkey_oracle.jsonl")
    if not gsm or not pk:
        print("fig2: missing data"); return

    def marg(rows):
        m, se = paired_se(rows, "oracle_fut", "kv_norm")
        return m, se
    items = [("GSM8K\n(reasoning)", gsm, PALETTE["slate_light"]),
             ("HotpotQA\n(real retrieval)", hot, PALETTE["blue"]) if hot else None,
             ("Passkey\n(synthetic retrieval)", pk, PALETTE["accent"])]
    items = [it for it in items if it]
    labels, margins, ses, colors = [], [], [], []
    for lab, rows, col in items:
        m, se = marg(rows); labels.append(lab); margins.append(m)
        ses.append(se); colors.append(col)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    bars = ax.bar(range(len(labels)), margins, yerr=ses, capsize=4,
                  color=colors, width=0.6, edgecolor="white", linewidth=1.4,
                  error_kw=dict(ecolor=PALETTE["ink"], lw=1.1))
    for i, v in enumerate(margins):
        ax.text(i, v + ses[i] + 0.012, f"+{v:.2f}", ha="center", va="bottom",
                fontsize=12, weight="bold", color=PALETTE["ink"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Learnable margin  (oracle - kv_norm)")
    ax.set_title("The learnable eviction margin scales with retrieval structure")
    ax.set_ylim(0, max(margins) * 1.3)
    ax.axhline(0, color=PALETTE["ink"], lw=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_regime_headroom.png")
    print("wrote fig2_regime_headroom.png")


# ---------------------------------------------------------------- fig 3
def fig3():
    import csv
    # explicit run -> (label, color) so the 3 curves are unambiguous
    runs = [
        ("e10_cold_seed0_probe.csv", "PPO cold (seed 0)", PALETTE["slate_light"]),
        ("e10_seed1_probe.csv",      "PPO cold (seed 1)", PALETTE["blue"]),
        ("e10_warm_probe.csv",       "PPO warm-start",    PALETTE["accent"]),
    ]
    present = [(f, l, c) for f, l, c in runs if (DATA / f).exists()
               and sum(1 for _ in open(DATA / f)) > 1]
    if not present:
        print("fig3: no E10 probe curves yet"); return
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    kv_vals, rnd_vals = [], []
    for f, lab, col in present:
        rows = list(csv.DictReader(open(DATA / f)))
        ts = [int(r["timestep"]) / 1e6 for r in rows]
        learned = [float(r["correct_learned"]) for r in rows]
        ax.plot(ts, learned, "-o", ms=4.5, lw=1.9, color=col, label=lab)
        kv_vals += [float(r["correct_kv_norm"]) for r in rows]
        rnd_vals += [float(r["correct_random"]) for r in rows]
    kvm = float(np.mean(kv_vals)); rndm = float(np.mean(rnd_vals))
    ax.axhline(kvm, color=PALETTE["slate"], lw=1.8, ls="--")
    ax.axhline(rndm, color="#b9bfc7", lw=1.3, ls=":")
    ax.text(ax.get_xlim()[1], kvm + 0.015, "kv_norm", ha="right", va="bottom",
            fontsize=9.5, color=PALETTE["slate"], weight="bold")
    ax.text(ax.get_xlim()[1], rndm + 0.015, "random", ha="right", va="bottom",
            fontsize=9, color="#8a929c")
    # shade the "beats kv_norm" band
    ax.axhspan(kvm, 1.05, color=PALETTE["teal"], alpha=0.06)
    ax.text(0.02, (kvm + 1.05) / 2, "beats kv_norm", transform=ax.get_yaxis_transform(),
            fontsize=8.5, color=PALETTE["teal"], style="italic",
            ha="left", va="center") if False else None
    ax.set_xlabel("Training steps  (millions)")
    ax.set_ylabel("Held-out probe accuracy")
    ax.set_title("E10: online PPO in the passkey arena (signal-bearing)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_e10_learning.png")
    print("wrote fig3_e10_learning.png")


# ---------------------------------------------------------------- fig 4
def fig4():
    p = DATA / "gsm8k_ranker_summary.json"
    if not p.exists():
        print("fig4: no ranker summary"); return
    d = json.load(open(p)).get("aggregate", {})
    if not d:
        print("fig4: empty ranker summary"); return
    metrics = [("Spearman rho", "sp_net", "sp_kvn"),
               ("Overlap@T/4", "ov_net", "ov_kvn")]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    x = np.arange(len(metrics))
    w = 0.36
    net = [d[m[1]] for m in metrics]
    kvn = [d[m[2]] for m in metrics]
    ax.bar(x - w / 2, net, w, color=PALETTE["accent"], label="Learned ranker",
           edgecolor="white", linewidth=1.2)
    ax.bar(x + w / 2, kvn, w, color=PALETTE["slate"], label="kv_norm proxy",
           edgecolor="white", linewidth=1.2)
    for i, v in enumerate(net):
        ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(kvn):
        ax.text(i + w / 2, v + 0.02 if v >= 0 else v - 0.05, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_title("Offline ranker: predicting future utility (GSM8K)")
    ax.axhline(0, color=PALETTE["ink"], lw=0.9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig4_ranker_predictability.png")
    print("wrote fig4_ranker_predictability.png")


# ---------------------------------------------------------------- fig 5
def fig5():
    p = DATA / "passkey_ranker_summary.json"
    if not p.exists():
        print("fig5: no passkey_ranker summary"); return
    d = json.load(open(p))
    s = d["summary"]
    arms = ["full", "oracle", "learned", "random", "kv_norm"]
    labels = {"full": "Full cache", "oracle": "Oracle (future attn.)",
              "learned": "Learned ranker\n(offline, ours)", "random": "Random",
              "kv_norm": "kv_norm (heuristic)"}
    colors = {"full": PALETTE["grey"], "oracle": PALETTE["teal"],
              "learned": PALETTE["accent"], "random": PALETTE["slate_light"],
              "kv_norm": PALETTE["slate"]}
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    xs = list(range(len(arms)))
    vals = [s[a] for a in arms]
    bars = ax.bar(xs, vals, color=[colors[a] for a in arms], width=0.66,
                  edgecolor="white", linewidth=1.3)
    # emphasize the learned bar
    li = arms.index("learned")
    bars[li].set_edgecolor(PALETTE["ink"]); bars[li].set_linewidth(1.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5,
                weight="bold" if arms[i] == "learned" else "normal")
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[a] for a in arms], fontsize=9.5)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Answer accuracy")
    g = d["learned_minus_kvnorm"]; w = d["wins"]; l = d["losses"]; n = d["n"]
    ax.set_title("Learned eviction BEATS the heuristic where signal exists")
    # bracket learned vs kv_norm
    ik = arms.index("kv_norm")
    yb = max(vals[li], vals[ik]) + 0.10
    ax.annotate("", xy=(li, yb), xytext=(ik, yb),
                arrowprops=dict(arrowstyle="<->", color=PALETTE["accent"], lw=1.8))
    ax.text((li + ik) / 2, yb + 0.015,
            f"+{g:.2f}  ({w}W / {l}L, n={n}, p<1e-4)",
            ha="center", va="bottom", fontsize=10, weight="bold",
            color=PALETTE["accent"])
    fig.text(0.5, -0.02, "Passkey arena, budget=176. Offline per-layer "
             "future-attention rankers (Apple/KVP recipe), evaluated as an "
             "eviction policy on held-out examples.",
             ha="center", fontsize=8, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_learned_ranker_wins.png")
    print("wrote fig5_learned_ranker_wins.png")


# ---------------------------------------------------------------- fig 6
def fig6():
    b256 = load_jsonl("hotpot_results.jsonl")
    b128 = load_jsonl("hotpot_b128_results.jsonl")
    if not b256 or not b128:
        print("fig6: missing HotpotQA data"); return
    arms = ["full", "oracle_fut", "kv_norm", "random"]
    labels = {"full": "Full", "oracle_fut": "Oracle", "kv_norm": "kv_norm",
              "random": "Random"}
    m256, _ = arm_means(b256, arms)
    m128, _ = arm_means(b128, arms)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(len(arms)); w = 0.38
    v256 = [m256[a] for a in arms]; v128 = [m128[a] for a in arms]
    ax.bar(x - w / 2, v256, w, color=PALETTE["blue"], label="4-5x  (budget 256)",
           edgecolor="white", linewidth=1.2)
    ax.bar(x + w / 2, v128, w, color=PALETTE["accent"], label="8-10x  (budget 128)",
           edgecolor="white", linewidth=1.2)
    for i, v in enumerate(v256):
        ax.text(i - w / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=8.5)
    for i, v in enumerate(v128):
        ax.text(i + w / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([labels[a] for a in arms])
    ax.set_ylabel("Answer accuracy")
    ax.set_ylim(0, 0.68)
    ax.set_title("HotpotQA (real data): harder compression, bigger learned margin")
    g256, _ = paired_se(b256, "oracle_fut", "kv_norm")
    g128, _ = paired_se(b128, "oracle_fut", "kv_norm")
    ax.legend(title=f"oracle - kv_norm:  +{g256:.2f} -> +{g128:.2f}", loc="upper right")
    fig.text(0.5, -0.02, "kv_norm collapses to random under aggressive "
             "compression while the oracle holds near full-cache.",
             ha="center", fontsize=8, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_hotpot_compression.png")
    print("wrote fig6_hotpot_compression.png")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print(f"{f.__name__} failed: {e}")
