#!/usr/bin/env python3
"""Overlay probe retention curves of several runs for the S4 experiment.

Usage:
    python experiments/s4-entropy-shaping/compare_probe.py \
        --runs runs/none_v12:baseline runs/s4_kl_v1:S4-exact runs/s4_selfent_v1:S4-proxy \
        --out experiments/s4-entropy-shaping/comparison.png

Each --runs entry is `path:label`. Reads <path>/probe_curve.csv and plots
`retention` vs `timestep`. Also prints a small summary table (final + best
retention, and final kl_step_mean from learning_curve.csv if present).
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_probe(run_dir: Path):
    ts, ret = [], []
    p = run_dir / "probe_curve.csv"
    if not p.exists():
        return ts, ret
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                r = float(row["retention"])
            except (KeyError, ValueError):
                continue
            if r == r:  # not NaN
                ts.append(int(row["timestep"]))
                ret.append(r)
    return ts, ret


def _final_kl(run_dir: Path):
    p = run_dir / "learning_curve.csv"
    if not p.exists():
        return None
    last = None
    with open(p) as f:
        for row in csv.DictReader(f):
            v = row.get("kl_step_mean")
            if v not in (None, "", "nan"):
                try:
                    last = float(v)
                except ValueError:
                    pass
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="path:label entries")
    ap.add_argument("--out", default="experiments/s4-entropy-shaping/comparison.png")
    args = ap.parse_args()

    plt.figure(figsize=(9, 5.5))
    print(f"{'run':<22}{'final_ret':>10}{'best_ret':>10}{'final_kl':>12}")
    print("-" * 54)
    for entry in args.runs:
        path, _, label = entry.partition(":")
        label = label or Path(path).name
        run_dir = Path(path)
        ts, ret = _read_probe(run_dir)
        if not ts:
            print(f"{label:<22}{'(no probe_curve.csv)':>32}")
            continue
        plt.plot(ts, ret, marker="o", ms=3, lw=1.5, label=label)
        kl = _final_kl(run_dir)
        kl_s = f"{kl:.4f}" if kl is not None else "n/a"
        print(f"{label:<22}{ret[-1]:>10.3f}{max(ret):>10.3f}{kl_s:>12}")

    plt.xlabel("timestep")
    plt.ylabel("probe retention  (correct_learned / correct_full)")
    plt.title("S4 entropy shaping — probe retention vs baseline")
    plt.grid(alpha=0.3)
    plt.legend()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
