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
    pk = load_jsonl("passkey_oracle.jsonl")
    if not gsm or not pk:
        print("fig2: missing data"); return
    # headroom = full - random ; margin = oracle - kv_norm
    def stats(rows):
        gm, _ = arm_means(rows, ["full", "random", "oracle_fut", "kv_norm"])
        head = (gm["full"] - gm["random"]) if gm["random"] is not None else None
        marg, _ = paired_se(rows, "oracle_fut", "kv_norm")
        return head, marg
    gh, gmarg = stats(gsm)
    ph, pmarg = stats(pk)
    labels = ["GSM8K", "Passkey"]
    margins = [gmarg, pmarg]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bars = ax.bar(labels, margins, color=[PALETTE["slate_light"], PALETTE["accent"]],
                  width=0.55, edgecolor="white", linewidth=1.4)
    for i, v in enumerate(margins):
        ax.text(i, v + 0.008, f"+{v:.2f}", ha="center", va="bottom",
                fontsize=12, weight="bold", color=PALETTE["ink"])
    ax.set_ylabel("Learnable margin  (oracle - kv_norm)")
    ax.set_title("How much a learned policy CAN win")
    ax.set_ylim(0, max(margins) * 1.25)
    ax.axhline(0, color=PALETTE["ink"], lw=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_regime_headroom.png")
    print("wrote fig2_regime_headroom.png")


# ---------------------------------------------------------------- fig 3
def fig3():
    curves = sorted(DATA.glob("e10_*_probe.csv"))
    if not curves:
        print("fig3: no E10 probe curves yet"); return
    import csv
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    seed_colors = [PALETTE["accent"], PALETTE["blue"]]
    kv_ref = None
    for i, cf in enumerate(curves):
        rows = list(csv.DictReader(open(cf)))
        ts = [int(r["timestep"]) / 1e6 for r in rows]
        learned = [float(r["correct_learned"]) for r in rows]
        kv = [float(r["correct_kv_norm"]) for r in rows]
        rnd = [float(r["correct_random"]) for r in rows]
        seed = cf.stem.replace("e10_", "").replace("_probe", "")
        ax.plot(ts, learned, "-o", ms=4, lw=1.8, color=seed_colors[i % 2],
                label=f"Learned ({seed})")
        kv_ref = (kv, rnd, ts)
    if kv_ref:
        kv, rnd, ts = kv_ref
        ax.axhline(np.mean(kv), color=PALETTE["slate"], lw=1.6, ls="--",
                   label="kv_norm (heuristic)")
        ax.axhline(np.mean(rnd), color=PALETTE["slate_light"], lw=1.2, ls=":",
                   label="random")
    ax.set_xlabel("Training steps  (millions)")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("E10: online PPO in the passkey arena")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=2)
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


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print(f"{f.__name__} failed: {e}")
