"""Figures for the stability sweep (E12) in the report's style.

Generates SEPARATE figures (one per question) into docs/imgs/, with the same
font (Latin Modern), palette and title conventions as figures.py: the title
says WHAT is measured, not the conclusion.

    python scripts/figures_e12.py

Excludes the 4 kvp-ab runs whose probe was built on 1-3 of 16 examples
(stale prompts.py without raw_chat: see docs/runs/13). Their anchors and gaps
are meaningless.
"""
import csv
import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_style, PALETTE as C

apply_style(serif=True)

REPO = Path(__file__).resolve().parent.parent          # kv-eviction-gym/
D4 = REPO / "experiments" / "phase4-stability" / "data"
P3 = REPO / "experiments" / "phase3-dataset-causality" / "data"
OUT = REPO.parent / "docs" / "imgs"
OUT.mkdir(parents=True, exist_ok=True)

# Probes built on 1-3 of 16 examples due to a stale prompts.py on kvp-ab.
INVALID = {"s_e12_seed2", "s_e12_seed3", "s_e12_seed4", "s_e12_cont_klC_seed1"}


def read(path):
    """Robust CSV: tolerates NUL bytes from interrupted scp downloads."""
    raw = Path(path).read_bytes().replace(b"\x00", b"")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))))
    return [r for r in rows
            if r.get("correct_learned") not in (None, "")
            and r.get("correct_kv_norm") not in (None, "")]


def series(run, origin=None):
    """(timesteps in M, learned, mean kv_norm). origin: CSV to prepend."""
    rows = read(D4 / f"{run}_probe.csv")
    if origin:
        rows = read(origin) + rows
    rows.sort(key=lambda r: int(float(r["timestep"])))
    ts = np.array([int(float(r["timestep"])) for r in rows]) / 1e6
    learned = np.array([float(r["correct_learned"]) for r in rows])
    kv = float(np.mean([float(r["correct_kv_norm"]) for r in rows]))
    return ts, learned, kv


def done(run):
    return (D4 / f"{run}.done").exists()


def smooth(y, w=5):
    if len(y) < w:
        return None, None
    trend = np.convolve(y, np.ones(w) / w, mode="valid")
    return np.arange(w - 1, len(y)), trend


# ---------------------------------------------------------------------------
# Figure 11: final paired gap per arm (finished, valid runs only)
# ---------------------------------------------------------------------------
def fig11_summary():
    # (run, readable label, which knob changes vs the base config)
    # Readable labels. Final order is computed from the gap (worst at the
    # bottom), so adding a new arm here is enough for it to enter the figure.
    arms = [
        # killed at 33%: entropy collapsed and everything truncated. Flagged
        # because the other bars average their full 3M runs.
        ("s_e12_entcoef",    "Less exploration (coef. 0.003)\nkilled at 33%"),
        ("s_e12_lrdecay_s0", "LR decay (seed 0)"),
        ("s_e12_epochs4",    "4 epochs per rollout"),
        ("s_e12_lrdecay_s1", "LR decay (seed 1)"),
        ("s_e12_klw15",      "Stronger dense reward"),
        ("s_e12_cont_klC",   "Base config extended to 10M steps"),
        ("s_e12_seed5",      "Base config (seed 5)"),
    ]
    rows = []
    for run, lab in arms:
        if not (D4 / f"{run}_probe.csv").exists() or not done(run):
            continue
        origin = P3 / "e11_klC_probe.csv" if run == "s_e12_cont_klC" else None
        _, learned, kv = series(run, origin)
        rows.append((learned.mean() - kv, lab))
    rows.sort()                                  # worst first => bottom of axis
    gaps = [g for g, _ in rows]
    labels = [l for _, l in rows]
    colors = [C["accent"] if g > 0 else C["slate_light"] for g in gaps]

    # horizontal bars: the labels are long and collide when vertical
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.barh(range(len(labels)), gaps, color=colors, height=0.6,
            edgecolor="white", linewidth=1.3)
    span = max(gaps) - min(gaps)
    for i, g in enumerate(gaps):
        ha = "left" if g > 0 else "right"
        off = span * 0.015 * (1 if g > 0 else -1)
        ax.text(g + off, i, f"{g:+.3f}", ha=ha, va="center",
                fontsize=12, weight="bold")
    ax.axvline(0, color=C["slate"], lw=1.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.set_xlabel("Mean advantage over kv_norm (paired contrast)")
    ax.set_title("Advantage over the heuristic per training variant")
    ax.set_xlim(min(gaps) - span * 0.18, max(gaps) + span * 0.18)
    ax.grid(axis="y", visible=False)
    # annotate the zero without covering bars: above the axis, outside the data
    ax.annotate("kv_norm level", xy=(0, len(labels) - 0.35),
                xytext=(span * 0.03, len(labels) - 0.35),
                fontsize=11.5, color=C["slate"], weight="bold", va="center")
    fig.text(0.5, -0.04, "Each bar averages every probe of its run against "
             "that same run's kv_norm. Finished runs only.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig11_e12_summary.png")
    print("fig11_e12_summary.png:", [(l.split(chr(10))[0], round(g, 3))
                                     for l, g in zip(labels, gaps)])
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 12: trajectories of the arms that attack PPO instability
# ---------------------------------------------------------------------------
def fig12_stability():
    arms = [
        ("s_e12_targetkl",  "More conservative\nclipping and target KL", C["accent"]),
        ("s_e12_entcoef",   "Less exploration\n(coef. 0.003, collapses)", C["teal"]),
        ("s_e12_entcoef6",  "Less exploration\n(coef. 0.006)", C["teal_light"]),
        ("s_e12_epochs15",  "15 epochs\nper rollout", C["blue"]),
    ]
    # an arm with 1-2 probes draws no trend: it would enter the legend as an
    # invisible line. It only shows once it has enough points.
    MIN_PROBES = 3
    present = []
    for r, l, c in arms:
        p = D4 / f"{r}_probe.csv"
        if p.exists() and len(read(p)) >= MIN_PROBES:
            present.append((r, l, c))
    if not present:
        print("fig12: no data from the stability arms yet")
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for run, lab, col in present:
        ts, learned, kv = series(run)
        # paired gap against the run's own anchor: comparable across arms even
        # though each VM yields a different kv_norm (same 16 examples, other GPU).
        gap = learned - kv
        ax.plot(ts, gap, "o", ms=3.2, color=col, alpha=0.30)
        idx, trend = smooth(gap)
        n_lab = f"{lab}  (n={len(gap)})"
        if trend is not None:
            ax.plot(ts[idx], trend, "-", lw=2.3, color=col, label=n_lab)
        else:
            ax.plot(ts, gap, "-", lw=2.0, color=col, label=n_lab)

    ax.axhline(0, color=C["slate"], lw=1.7, ls="--")
    ax.text(ax.get_xlim()[1], 0.012, "kv_norm level", ha="right", va="bottom",
            fontsize=11.5, color=C["slate"], weight="bold")
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Advantage over kv_norm\n(paired contrast)")
    ax.set_title("Trajectory of the variants that aim to stabilize training")
    ax.legend(loc="lower right", fontsize=10.5)
    fig.text(0.5, -0.04, "Points: individual evaluations. Lines: trend "
             "(moving average). Runs in progress, each against its own kv_norm.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig12_e12_stability.png")
    print("fig12_e12_stability.png:", [r for r, _, _ in present])
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 13: the isolated mechanism (optimization epochs per rollout)
# ---------------------------------------------------------------------------
def fig13_epochs():
    arms = [
        ("s_e12_epochs4",  "4 epochs",  C["slate_light"]),
        (None,             "10 epochs", C["accent"]),   # klC: the base config
        ("s_e12_epochs15", "15 epochs", C["blue"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels, fracs, colors = [], [], []
    for run, lab, col in arms:
        if run is None:                       # klC lives in phase3
            rows = read(P3 / "e11_klC_probe.csv")
            learned = np.array([float(r["correct_learned"]) for r in rows])
            kv = float(np.mean([float(r["correct_kv_norm"]) for r in rows]))
        else:
            if not (D4 / f"{run}_probe.csv").exists():
                continue
            _, learned, kv = series(run)
        if len(learned) < 5:                  # no useful data yet
            continue
        frac = float(np.mean(learned > kv))
        labels.append(f"{lab}\n(n={len(learned)})"); fracs.append(frac); colors.append(col)

    ax.bar(range(len(labels)), fracs, color=colors, width=0.55,
           edgecolor="white", linewidth=1.3)
    ax.axhline(0.5, color=C["slate"], lw=1.5, ls="--")
    for i, f in enumerate(fracs):
        # label inside the bar if it would touch the 50% line
        near_line = abs(f - 0.5) < 0.06
        y, va, col = ((f - 0.02, "top", "white") if near_line
                      else (f + 0.012, "bottom", C["ink"]))
        ax.text(i, y, f"{f:.0%}", ha="center", va=va,
                fontsize=13, weight="bold", color=col)
    ax.text(-0.42, 0.515, "half of the evaluations",
            ha="left", va="bottom", fontsize=11, color=C["slate"])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Evaluations above kv_norm\n(fraction of total)")
    ax.set_title("Optimization epochs per rollout")
    ax.set_ylim(0, max(fracs + [0.6]) * 1.28)
    n_shown = len(labels)
    note = ("Everything else identical across the runs."
            if n_shown >= 3 else
            "Everything else identical across the runs. "
            "The 15-epoch variant does not have enough evaluations yet.")
    fig.text(0.5, -0.04, note, ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig13_e12_epochs.png")
    print("fig13_e12_epochs.png:", list(zip([l.split(chr(10))[0] for l in labels],
                                            [round(f, 3) for f in fracs])))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 14: probe accuracy over training (one panel per arm)
# ---------------------------------------------------------------------------
def fig14_accuracy():
    """Raw probe accuracy vs steps, one panel per arm.

    Panels rather than an overlay because each run has its OWN kv_norm
    (0.375 to 0.688 depending on VM and seed: the same 16 examples yield
    different absolute accuracies on L4 vs T4). Overlaying raw accuracies
    against a single reference would suggest comparisons that are not valid;
    each panel carries its own kv_norm line.
    """
    # (run, label, origin to prepend so the axis starts at 0)
    arms = [
        ("s_e12_cont_klC",   "Base config extended to 10M steps",
         P3 / "e11_klC_probe.csv"),
        ("s_e12_seed5",      "Base config (seed 5)", None),
        ("s_e12_targetkl",   "More conservative clipping and target KL", None),
        ("s_e12_entcoef",    "Less exploration (coef. 0.003)", None),
        ("s_e12_entcoef6",   "Less exploration (coef. 0.006)", None),
        ("s_e12_epochs15",   "15 epochs per rollout", None),
        ("s_e12_epochs4",    "4 epochs per rollout", None),
        ("s_e12_lrdecay_s0", "LR decay (seed 0)", None),
        ("s_e12_lrdecay_s1", "LR decay (seed 1)", None),
        ("s_e12_klw15",      "Stronger dense reward", None),
    ]
    MIN_PROBES = 3
    present = [(r, l, o) for r, l, o in arms
               if (D4 / f"{r}_probe.csv").exists()
               and len(read(D4 / f"{r}_probe.csv")) >= MIN_PROBES]
    if not present:
        print("fig14: no data yet")
        return

    ncols = 3
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 3.5 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for i, (run, lab, origin) in enumerate(present):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        ts, learned, kv = series(run, origin)
        ax.plot(ts, learned, "o", ms=3.0, color=C["blue"], alpha=0.32)
        idx, trend = smooth(learned)
        if trend is not None:
            ax.plot(ts[idx], trend, "-", lw=2.2, color=C["blue"])
        ax.axhline(kv, color=C["slate"], ls="--", lw=1.5)
        ax.axhline(1.0, color=C["teal"], ls=":", lw=1.1, alpha=0.65)
        status = "" if done(run) else "  (in progress)"
        ax.set_title(f"{lab}{status}\nkv_norm = {kv:.2f}  (n={len(learned)})",
                     fontsize=12)
        ax.set_ylim(-0.05, 1.10)
        ax.set_xlim(0, None)
        ax.set_xlabel("Training steps (millions)", fontsize=11)
        if i % ncols == 0:
            ax.set_ylabel("Probe accuracy", fontsize=11)

    fig.suptitle("Probe accuracy over training",
                 fontsize=17, weight="bold", y=1.005)
    fig.text(0.5, -0.015,
             "Points: individual evaluations (16 examples each). Lines: "
             "trend (moving average). Dashed: that same run's kv_norm. "
             "Dotted: perfect policy.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig14_e12_accuracy.png")
    print("fig14_e12_accuracy.png:", [r for r, _, _ in present])
    plt.close(fig)


if __name__ == "__main__":
    fig11_summary()
    fig12_stability()
    fig13_epochs()
    fig14_accuracy()
    print("\nE12 figures written to", OUT)
