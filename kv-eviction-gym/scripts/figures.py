"""Generates every figure of the Results chapter.

Each cell produces one report figure and saves it to docs/imgs/.
Raw data lives in kv-eviction-gym/ (ab_results/ and
experiments/phase3-dataset-causality/data/). All figures share a cohesive cool
palette (teal -> slate) with one warm accent for the key comparison.

Run:  python scripts/figures.py
"""
# %% [markdown]
# # Results-chapter figures
# Learned KV-cache eviction. All figures in the report's style.

# %%
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_style import apply_style, PALETTE as C

apply_style(serif=True)

REPO = Path(__file__).resolve().parent.parent          # kv-eviction-gym/
AB = REPO / "ab_results"
D3 = REPO / "experiments" / "phase3-dataset-causality" / "data"
OUT = REPO.parent / "docs" / "imgs"
OUT.mkdir(parents=True, exist_ok=True)


def read_probe(path):
    rows = list(csv.DictReader(open(path)))
    return rows


def load_jsonl(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if "error" not in r and not r.get("skip")]


def paired_gap(rows, a="oracle_fut", b="kv_norm"):
    d = [r[a] - r[b] for r in rows if a in r and b in r]
    n = len(d); m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0
    return m, sd / n ** 0.5, n


# %% [markdown]
# ## Figure 1: E0 capacity screen (16 examples, overfit)
# Best paired gap per variant. Shows rich features / attention CAN beat
# kv_norm in capacity, the norm-blind baseline cannot, and S4 does not help.

# %%
def fig1_screen():
    variants = [
        ("e0_baseline_probe_curve.csv", "Baseline\n(norm-blind)", C["slate_light"]),
        ("e0_rich_probe_curve.csv", "Rich\nfeatures", C["blue"]),
        ("e0_attn_probe_curve.csv", "Cross-token\nattention", C["teal"]),
        ("e0_rich_s4_probe_curve.csv", "Rich +\ndense reward", C["slate"]),
    ]
    labels, gaps, colors = [], [], []
    for f, lab, col in variants:
        p = AB / f
        if not p.exists():
            continue
        rows = read_probe(p)
        best = max(float(r["correct_learned"]) - float(r["correct_kv_norm"]) for r in rows)
        labels.append(lab); gaps.append(best); colors.append(col)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(range(len(labels)), gaps, color=colors, width=0.66,
                  edgecolor="white", linewidth=1.3)
    for i, v in enumerate(gaps):
        ax.text(i, v + 0.004, f"+{v:.2f}", ha="center", va="bottom",
                fontsize=13, weight="bold")
    ax.axhline(0, color=C["ink"], lw=1.0)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylabel("Best advantage over kv_norm\n(paired contrast)")
    ax.set_title("Capacity per variant (16 examples)")
    ax.set_ylim(0, max(gaps) * 1.25)
    fig.text(0.5, -0.03, "Capacity: 16 examples, evaluated on the same examples "
             "it trained on (overfit). Peak over the run.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_screen.png")
    print("fig1_screen.png:", dict(zip([l.replace(chr(10), ' ') for l in labels],
                                       [round(g, 3) for g in gaps])))
    plt.close(fig)


fig1_screen()

# %% [markdown]
# ## Figure 2: scaled runs (paired-gap trajectory)
# s_rich, s_warm, s_attn at 2M steps. All three oscillate at or below 0 =
# parity. warm starts at kv_norm; attn stays below.

# %%
def fig2_scaled():
    runs = [
        ("s_rich_probe_curve.csv", "Rich features", C["blue"]),
        ("s_warm_probe_curve.csv", "Warm-start", C["accent"]),
        ("s_attn_probe_curve.csv", "Attention", C["slate"]),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for f, lab, col in runs:
        p = AB / f
        if not p.exists():
            continue
        rows = read_probe(p)
        # sort by timestep: the CSVs may contain concatenated preemption
        # attempts (timesteps that reset). We show the raw probes faint plus a
        # moving average (trend) so parity is readable without the zigzag of
        # resumed runs.
        pts = sorted((int(r["timestep"]) / 1e6,
                      float(r["correct_learned"]) - float(r["correct_kv_norm"]))
                     for r in rows)
        ts = np.array([t for t, _ in pts]); gap = np.array([g for _, g in pts])
        ax.plot(ts, gap, "o", ms=3.0, color=col, alpha=0.28)
        w = 5 if len(gap) >= 5 else max(1, len(gap))
        trend = np.convolve(gap, np.ones(w) / w, mode="valid")
        ts_t = ts[w - 1:]
        ax.plot(ts_t, trend, "-", lw=2.2, color=col, label=lab)
    ax.axhline(0, color=C["slate"], lw=1.6, ls="--")
    ax.text(ax.get_xlim()[1], 0.006, "kv_norm level", ha="right", va="bottom",
            fontsize=12, color=C["slate"], weight="bold")
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Advantage over kv_norm\n(paired contrast)")
    ax.set_title("Training trajectory at 2M steps")
    ax.legend(loc="lower right")
    fig.text(0.5, -0.04, "Points: each individual probe evaluation. Lines: "
             "trend (moving average over 5 probes) per arm.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_scaled_traj.png")
    print("fig2_scaled_traj.png written")
    plt.close(fig)


fig2_scaled()

# %% [markdown]
# ## Figure 3: wide eval (n=128, single stack)
# Accuracy per arm on 128 fresh examples. The per-token variants land within
# 2-3 examples of kv_norm (parity); attention is worse. Insight: full-random ~ 0.04.

# %%
def fig3_wide():
    # numbers from the wide eval (scripts/wide_eval.py, n=128, one stack) - FINDINGS 10
    arms = ["full", "kv_norm", "random", "s_rich", "s_warm", "s_attn"]
    vals = [0.742, 0.711, 0.703, 0.695, 0.688, 0.648]
    labels = ["Full\n(ceiling)", "kv_norm", "Random", "Rich", "Warm", "Attention"]
    colors = [C["grey"], C["slate"], C["slate_light"], C["blue"], C["accent"], C["teal"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(range(len(arms)), vals, color=colors, width=0.66,
                  edgecolor="white", linewidth=1.3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=12.5)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Accuracy (n=128)")
    ax.set_ylim(0.55, 0.78)
    ax.set_title("Accuracy per policy (wide evaluation, 128 examples)")
    # annotate the full-random headroom
    ax.annotate("", xy=(0, 0.742), xytext=(0, 0.703),
                arrowprops=dict(arrowstyle="<->", color=C["accent"], lw=1.6))
    ax.text(0.35, 0.722, "full - random\n= 0.04\n(little headroom)", fontsize=11.5,
            color=C["accent"], va="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_wide_eval.png")
    print("fig3_wide_eval.png written")
    plt.close(fig)


fig3_wide()

# %% [markdown]
# ## Figure 4: the future-attention oracle (GSM8K)
# Only FUTURE information beats kv_norm (+7pp); present attention (attn_cur) does not.

# %%
def fig4_oracle():
    rows = load_jsonl(D3 / "gsm8k_oracle.jsonl")
    n = len(rows)
    arms = ["full", "oracle_fut", "attn_cur", "kv_norm"]
    labels = ["Full", "Oracle\n(future attn.)", "Present attn.\n(H2O)", "kv_norm"]
    colors = [C["grey"], C["teal"], C["blue"], C["slate"]]
    vals = [sum(r[a] for r in rows) / n for a in arms]
    g, se, _ = paired_gap(rows, "oracle_fut", "kv_norm")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar(range(len(arms)), vals, color=colors, width=0.64,
                  edgecolor="white", linewidth=1.3)
    io = arms.index("oracle_fut")
    bars[io].set_edgecolor(C["ink"]); bars[io].set_linewidth(1.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=12.5)
    ik = arms.index("kv_norm")
    yb = max(vals[io], vals[ik]) + 0.06
    ax.annotate("", xy=(io, yb), xytext=(ik, yb),
                arrowprops=dict(arrowstyle="<->", color=C["accent"], lw=1.7))
    ax.text((io + ik) / 2, yb + 0.008, f"+{g:.2f} ({g/se:.1f}$\\sigma$)",
            ha="center", va="bottom", fontsize=12.5, weight="bold", color=C["accent"])
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(f"Accuracy (n={n})")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("Eviction policies: oracle vs heuristics (GSM8K)")
    fig.text(0.5, -0.02, "GSM8K, eager backend. Present attention does not beat "
             "kv_norm; only the future-attention oracle does.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_oracle.png")
    print(f"fig4_oracle.png: oracle-kv_norm = +{g:.3f} ({g/se:.1f} sigma), n={n}")
    plt.close(fig)


fig4_oracle()

# %% [markdown]
# ## Figure 5: headroom depends on the regime
# Low-difficulty regime (budget>=prompt) ~0.04 vs long generation ~0.45.

# %%
def fig5_regime():
    # Low-difficulty regime: from the GSM8K oracle (budget>=prompt, measured here).
    # Long-gen regime: from eval_e6c (128 filtered examples, budget 256, sdpa;
    #   full=0.680, random=0.227 -> headroom 0.453). FINDINGS 15. The local
    #   long-gen jsonl is under eager (flattened regime, unusable for headroom).
    benign = load_jsonl(D3 / "gsm8k_oracle.jsonl")
    nb = len(benign)
    full_b = sum(r["full"] for r in benign) / nb
    rnd_b = sum(r.get("random", r["kv_norm"]) for r in benign) / nb
    labels = ["Low difficulty\n(budget $\\geq$ prompt)", "Long generation\n(cache pressure)"]
    heads = [full_b - rnd_b, 0.680 - 0.227]
    colors = [C["slate_light"], C["accent"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    bars = ax.bar(range(len(labels)), heads, color=colors, width=0.55,
                  edgecolor="white", linewidth=1.4)
    for i, v in enumerate(heads):
        ax.text(i, v + 0.008, f"{v:.2f}", ha="center", va="bottom",
                fontsize=14, weight="bold")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylabel("full - random margin\n(how much eviction hurts)")
    ax.set_title("Learnable margin (full - random) per regime")
    ax.set_ylim(0, max(heads) * 1.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_regime.png")
    print("fig5_regime.png:", dict(zip([l.replace(chr(10), ' ') for l in labels],
                                       [round(h, 3) for h in heads])))
    plt.close(fig)


fig5_regime()

# %% [markdown]
# ## Figure 8: dataset spectrum (GSM8K -> real HotpotQA -> passkey)
# The oracle margin is a continuum, ordered by how distinguishable the
# important content is: dense reasoning (~0) -> real retrieval -> pure needle.

# %%
def _gap_se(path, a="oracle_fut", b="kv_norm"):
    rows = load_jsonl(path)
    d = [r[a] - r[b] for r in rows if a in r and b in r]
    n = len(d); m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0
    return m, sd / n ** 0.5, n


def fig8_spectrum():
    specs = [
        ("gsm8k_oracle.jsonl", "GSM8K\n(dense reasoning)", C["slate_light"]),
        ("hotpot_results.jsonl", "HotpotQA\n(real retrieval)", C["blue"]),
        ("passkey_oracle.jsonl", "Passkey\n(synthetic retrieval)", C["accent"]),
    ]
    labels, vals, ses, colors = [], [], [], []
    for f, lab, col in specs:
        g, se, n = _gap_se(D3 / f)
        labels.append(lab); vals.append(g); ses.append(se); colors.append(col)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(range(3), vals, yerr=ses, capsize=5, color=colors, width=0.6,
           edgecolor="white", linewidth=1.4, error_kw=dict(ecolor=C["ink"], lw=1.1))
    for i, (v, se) in enumerate(zip(vals, ses)):
        ax.text(i, v + se + 0.015, f"+{v:.2f}", ha="center", va="bottom",
                fontsize=14.5, weight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylabel("Oracle margin\n(oracle - kv_norm)")
    ax.set_title("Learnable margin per dataset")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.axhline(0, color=C["ink"], lw=0.9)
    fig.text(0.5, -0.03, "Same measurement on all three. The margin grows with "
             "how distinguishable the content to preserve is (dense $\\to$ pure needle).",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig8_dataset_spectrum.png")
    print("fig8_dataset_spectrum.png:", [(l.split(chr(10))[0], round(v, 3))
                                         for l, v in zip(labels, vals)])
    plt.close(fig)


fig8_spectrum()

# %% [markdown]
# ## Figure 9: the margin grows with compression (real HotpotQA + passkey)
# oracle-kv_norm vs cache budget: the more aggressive the compression (lower
# budget), the larger the learnable margin, on real and synthetic data.

# %%
def fig9_compression():
    # (budget, gap) per dataset, measured from the jsonl at each budget.
    hot = [
        (256, _gap_se(D3 / "hotpot_results.jsonl")[:2]),
        (128, _gap_se(D3 / "hotpot_b128_results.jsonl")[:2]),
    ]
    pk = [
        (176, _gap_se(D3 / "passkey_oracle.jsonl")[:2]),
        (128, _gap_se(D3 / "passkey_b128_results.jsonl")[:2]),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    # label offsets per (dataset, budget) so they don't sit on the lines
    offs = {("HotpotQA (real)", 256): (-6, 12), ("HotpotQA (real)", 128): (-2, -20),
            ("Passkey (synthetic)", 176): (-40, -4),
            ("Passkey (synthetic)", 128): (-6, 12)}
    for data, lab, col in [(hot, "HotpotQA (real)", C["blue"]),
                           (pk, "Passkey (synthetic)", C["accent"])]:
        bs = [b for b, _ in data]
        gs = [g for _, (g, se) in data]
        es = [se for _, (g, se) in data]
        ax.errorbar(bs, gs, yerr=es, marker="o", ms=8, lw=2.4, capsize=5,
                    color=col, label=lab, mec="white", mew=1.3)
        for b, g in zip(bs, gs):
            dx, dy = offs.get((lab, b), (0, 11))
            ax.annotate(f"+{g:.2f}", (b, g), textcoords="offset points",
                        xytext=(dx, dy), ha="center", fontsize=12.5,
                        weight="bold", color=col)
    ax.invert_xaxis()  # lower budget (more compression) to the right
    ax.set_xticks([256, 176, 128])
    ax.set_xlabel("Cache budget (tokens)  $\\rightarrow$ more compression")
    ax.set_ylabel("Oracle margin\n(oracle - kv_norm)")
    ax.set_title("Learnable margin vs compression")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    fig.text(0.5, -0.03, "The more aggressive the compression, the more kv_norm "
             "loses to the oracle. Holds on real and synthetic data.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig9_margin_vs_compression.png")
    print("fig9_margin_vs_compression.png written")
    plt.close(fig)


fig9_compression()

# %% [markdown]
# ## Figure 10: our online method on passkey (klC curve)
# The learned probe oscillates but its centre shifts above kv_norm in the
# second half of training; it touches the optimal policy (1.0).

# %%
def fig10_capstone():
    p = D3 / "e11_klC_probe.csv"
    rows = read_probe(p)
    pts = sorted((int(r["timestep"]) / 1e6, float(r["correct_learned"]))
                 for r in rows)
    ts = np.array([t for t, _ in pts]); acc = np.array([a for _, a in pts])
    kv = np.mean([float(r["correct_kv_norm"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    # shade the second half
    mid = ts[len(ts) // 2]
    ax.axvspan(mid, ts[-1], color=C["teal"], alpha=0.06)
    # raw probes + trend
    ax.plot(ts, acc, "o", ms=3.5, color=C["blue"], alpha=0.35)
    w = 5 if len(acc) >= 5 else max(1, len(acc))
    trend = np.convolve(acc, np.ones(w) / w, mode="valid")
    ax.plot(ts[w - 1:], trend, "-", lw=2.4, color=C["blue"],
            label="Our method (online)")
    ax.axhline(kv, color=C["slate"], lw=1.8, ls="--")
    ax.text(ts[-1], kv - 0.03, "kv_norm level", ha="right", va="top",
            fontsize=12, color=C["slate"], weight="bold")
    ax.axhline(1.0, color=C["teal"], lw=1.2, ls=":", alpha=0.7)
    ax.text(ts[0], 1.005, "optimal policy", ha="left", va="bottom",
            fontsize=11.5, color=C["teal"])
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Probe accuracy\n(learned policy)")
    ax.set_title("Our online method on passkey")
    ax.set_ylim(-0.05, 1.12)
    ax.legend(loc="lower right")
    fig.text(0.5, -0.04, "Points: raw probes. Line: trend (moving average). "
             "The band marks the second half, where the centre rises above kv_norm.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig10_capstone.png")
    h = len(acc) // 2
    print("fig10_capstone.png: 2nd half mean=%.3f vs kv_norm=%.3f (n=%d probes)"
          % (acc[h:].mean(), kv, len(acc)))
    plt.close(fig)


fig10_capstone()
print("\nAll figures written to", OUT)
