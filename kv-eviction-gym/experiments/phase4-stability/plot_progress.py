"""Progress plots for the E12 stability sweep (phase 4).

One panel per run in data/: learned probe accuracy vs timesteps, its own
kv_norm anchor as dashed line, half-means annotated. Regenerate any time:
    python plot_progress.py
Writes plots/e12_progress.png (grid) and prints a text summary.
"""
import csv
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

for _f in glob.glob("/usr/share/texmf/fonts/opentype/public/lm/lmroman*.otf"):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass

C = {"blue": "#457b9d", "slate": "#3d5a80", "teal": "#2a9d8f",
     "accent": "#e76f51", "ink": "#22303c"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
    "font.size": 11, "axes.titlesize": 12, "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e8eb", "axes.axisbelow": True,
})

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

# reference: the two finished E11 runs of the same config (for visual context)
P3 = HERE.parent / "phase3-dataset-causality" / "data"
REFS = {"s_e11_klC (ref seed0)": P3 / "e11_klC_probe.csv",
        "s_e11_klC_seed1 (ref seed1)": P3 / "e11_klC_seed1_probe.csv"}

runs = sorted(HERE.glob("data/s_e12_*_probe.csv"))
panels = [(p.stem.replace("_probe", ""), p) for p in runs]
panels += [(k, v) for k, v in REFS.items() if v.exists()]

if not panels:
    raise SystemExit("no probe CSVs yet in data/")

ncols = 3
nrows = (len(panels) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows),
                         squeeze=False)
for ax in axes.flat:
    ax.set_visible(False)

summary = []
for i, (name, path) in enumerate(panels):
    ax = axes[i // ncols][i % ncols]
    ax.set_visible(True)
    rows = list(csv.DictReader(open(path)))
    if not rows:
        ax.set_title(f"{name} (sin probes aun)")
        continue
    ts = np.array([int(r["timestep"]) for r in rows]) / 1e6
    acc = np.array([float(r["correct_learned"]) for r in rows])
    kv = float(np.mean([float(r["correct_kv_norm"]) for r in rows]))
    ax.plot(ts, acc, "o-", ms=3, lw=1.1, color=C["blue"], alpha=0.85)
    ax.axhline(kv, color=C["slate"], ls="--", lw=1.4)
    n = len(acc); h = n // 2
    if n >= 4:
        m1, m2 = acc[:h].mean(), acc[h:].mean()
        ax.set_title(f"{name}\n1a={m1:.2f} 2a={m2:.2f} (kv={kv:.2f}, n={n})")
        summary.append((name, ts[-1], m1, m2, kv))
    else:
        ax.set_title(f"{name} (n={n})")
        summary.append((name, ts[-1] if n else 0, float("nan"),
                        acc.mean() if n else float("nan"), kv))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("M pasos")

fig.tight_layout()
fig.savefig(OUT / "e12_progress.png")
print("plots/e12_progress.png")
for name, last_ts, m1, m2, kv in summary:
    print(f"  {name:32s} ts={last_ts:.2f}M  1a={m1:.2f}  2a={m2:.2f}  kv={kv:.2f}")
