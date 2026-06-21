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


def main():
    args    = parse_args()
    run_dir = Path(args.run)
    csv_path = run_dir / "learning_curve.csv"

    if not csv_path.exists():
        print(f"No learning_curve.csv found in {run_dir}. Run training first.")
        return

    timesteps, rew_means, len_means = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timesteps.append(int(row["timestep"]))
                rew_means.append(float(row["ep_rew_mean"]))
                len_means.append(float(row["ep_len_mean"]))
            except (ValueError, KeyError):
                continue

    if not timesteps:
        print("CSV is empty — no data to plot yet.")
        return

    ts  = np.array(timesteps)
    rew = np.array(rew_means)
    eln = np.array(len_means)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Episode reward
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

    # Episode length
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

    # Text summary
    print(f"\nSummary ({len(ts)} rollouts):")
    print(f"  Final ep_rew_mean : {rew[-1]:.4f}")
    print(f"  Best  ep_rew_mean : {smooth_rew.max():.4f}  @ timestep {best_t:,}")
    print(f"  Final ep_len_mean : {eln[-1]:.1f}")


if __name__ == "__main__":
    main()
