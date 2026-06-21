"""
Plot learning curves from a training run.

Reads runs/<run-name>/learning_curve.csv and saves:
    runs/<run-name>/learning_curve.png

Usage:
    python scripts/plot_curves.py --run runs/my_run
    python scripts/plot_curves.py --run runs/my_run --window 20
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run",    required=True, help="Path to run directory (e.g. runs/20240101_120000)")
    p.add_argument("--window", type=int, default=10, help="Rolling average window (rollouts)")
    return p.parse_args()


def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    padded = np.pad(arr, (w - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _read_csv_cols(csv_path: Path, cols: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, list] = {c: [] for c in cols}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for c in cols:
                try:
                    out[c].append(float(row.get(c, "nan")))
                except (ValueError, TypeError):
                    out[c].append(float("nan"))
    return {c: np.array(v) for c, v in out.items()}


def plot_probe(run_dir: Path) -> None:
    """Plot the fixed-probe monitoring curve (probe_curve.csv) if present."""
    csv_path = run_dir / "probe_curve.csv"
    if not csv_path.exists():
        return

    d = _read_csv_cols(csv_path, [
        "timestep", "correct_full", "correct_random", "correct_kv_norm",
        "correct_learned", "retention",
        "evict_attn_percentile", "evict_mean_pos_frac", "evict_sink_frac",
    ])
    ts = d["timestep"]
    if ts.size == 0:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ── Panel 1: correctness / retention vs fixed anchors ──
    ax1.plot(ts, d["retention"], color="crimson", linewidth=2, marker="o", ms=3,
             label="retention (learned/full on solvable)")
    ax1.plot(ts, d["correct_learned"], color="steelblue", linewidth=1.8, marker=".",
             label="correct: learned")
    for col, c, ls, lab in [
        ("correct_full",    "green",     "--", "full (ceiling)"),
        ("correct_kv_norm", "goldenrod", ":",  "kv_norm"),
        ("correct_random",  "gray",      ":",  "random (floor)"),
    ]:
        v = d[col]
        anchor = np.nanmean(v) if v.size else float("nan")
        ax1.axhline(anchor, color=c, linestyle=ls, linewidth=1.3, alpha=0.8, label=lab)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("correctness / retention")
    ax1.set_title("Fixed held-out probe — correctness")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    # ── Panel 2: eviction behavior (continuous, leads correctness) ──
    ax2.plot(ts, d["evict_attn_percentile"], color="purple", linewidth=2, marker="o", ms=3,
             label="evict attn percentile (0=oracle, .5=random)")
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax2.plot(ts, d["evict_mean_pos_frac"], color="teal", linewidth=1.5, marker=".",
             label="evict mean pos frac (0=old, 1=recent)")
    ax2.plot(ts, d["evict_sink_frac"], color="darkorange", linewidth=1.5, marker=".",
             label="evict sink frac")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("behavior")
    ax2.set_xlabel("timestep")
    ax2.set_title("Fixed held-out probe — eviction behavior")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Probe curves — {run_dir.name}", fontsize=12)
    plt.tight_layout()
    out_path = run_dir / "probe_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    args    = parse_args()
    run_dir = Path(args.run)
    csv_path = run_dir / "learning_curve.csv"

    # Probe curve is independent of the learning curve — plot it whenever present.
    plot_probe(run_dir)

    if not csv_path.exists():
        print(f"No learning_curve.csv found in {run_dir}. Run training first.")
        return

    timesteps, rew_means, len_means, corr_rates, align_means = [], [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timesteps.append(int(row["timestep"]))
                rew_means.append(float(row["ep_rew_mean"]))
                len_means.append(float(row["ep_len_mean"]))
                corr_rates.append(float(row.get("correctness_rate", "nan")))
                align_means.append(float(row.get("alignment_mean", "nan")))
            except (ValueError, KeyError):
                continue

    if not timesteps:
        print("CSV is empty — no data to plot yet.")
        return

    ts   = np.array(timesteps)
    rew  = np.array(rew_means)
    eln  = np.array(len_means)
    corr = np.array(corr_rates)
    aln  = np.array(align_means)

    has_corr = not np.all(np.isnan(corr))
    n_rows = 3 if has_corr else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)
    ax1, ax2 = axes[0], axes[-1]
    ax_corr  = axes[1] if has_corr else None

    # ── Episode reward ──
    ax1.plot(ts, rew, color="steelblue", alpha=0.35, linewidth=0.8, label="raw")
    smooth_rew = rolling_mean(rew, args.window)
    ax1.plot(ts, smooth_rew, color="steelblue", linewidth=2,
             label=f"rolling mean (w={args.window})")
    best_t   = ts[np.argmax(smooth_rew)]
    best_val = smooth_rew.max()
    ax1.axvline(best_t, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax1.scatter([best_t], [best_val], color="red", zorder=5,
                label=f"best: {best_val:.4f} @ {best_t:,}")
    ax1.set_ylabel("Mean episode reward")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Correctness rate ──
    if ax_corr is not None:
        valid = ~np.isnan(corr)
        ax_corr.plot(ts[valid], corr[valid], color="green", alpha=0.35, linewidth=0.8)
        smooth_corr = rolling_mean(corr[valid], args.window)
        ax_corr.plot(ts[valid], smooth_corr, color="green", linewidth=2,
                     label=f"correctness rate (w={args.window})")
        if not np.all(np.isnan(aln)):
            valid_a = ~np.isnan(aln)
            ax_corr.plot(ts[valid_a], rolling_mean(aln[valid_a], args.window),
                         color="goldenrod", linewidth=1.5, linestyle="--",
                         label="alignment mean")
        ax_corr.set_ylim(-0.05, 1.05)
        ax_corr.set_ylabel("Rate / score")
        ax_corr.legend(fontsize=9)
        ax_corr.grid(alpha=0.3)

    # ── Episode length ──
    ax2.plot(ts, eln, color="darkorange", alpha=0.35, linewidth=0.8)
    ax2.plot(ts, rolling_mean(eln, args.window), color="darkorange", linewidth=2)
    ax2.set_ylabel("Mean episode length (steps)")
    ax2.set_xlabel("Timestep")
    ax2.grid(alpha=0.3)

    run_name = run_dir.name
    fig.suptitle(f"Learning curves — {run_name}", fontsize=12)
    plt.tight_layout()

    out_path = run_dir / "learning_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    print(f"\nSummary ({len(ts)} rollouts):")
    print(f"  Final ep_rew_mean    : {rew[-1]:.4f}")
    print(f"  Best  ep_rew_mean    : {smooth_rew.max():.4f}  @ timestep {best_t:,}")
    if has_corr:
        valid_corr = corr[~np.isnan(corr)]
        print(f"  Final correctness    : {valid_corr[-1]:.1%}")
        print(f"  Best  correctness    : {valid_corr.max():.1%}")
    print(f"  Final ep_len_mean    : {eln[-1]:.1f}")


if __name__ == "__main__":
    main()
