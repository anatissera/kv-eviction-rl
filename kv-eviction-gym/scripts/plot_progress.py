"""Progress plots for the E12 stability sweep (phase 4).

One panel per run: learned probe accuracy vs timesteps, its own
kv_norm anchor as dashed line, half-means annotated. Regenerate any time:
    python scripts/plot_progress.py
Writes experiments/phase4-stability/plots/e12_progress.png and prints a text summary.

Continuation runs (s_e12_cont_*) are resumed from an E11 checkpoint and only
log their OWN new timesteps starting at the resume point (e.g. 3M onward for
s_e12_cont_klC). Their panel stitches that continuation CSV onto the original
E11 run's CSV (the 0..3M history) so the plotted line covers the whole
trajectory from step 0 to wherever the run currently is, not just the
continuation segment.
"""
import csv
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_style, PALETTE as C

apply_style(serif=True)

REPO = Path(__file__).resolve().parent.parent          # kv-eviction-gym/
PH4 = REPO / "experiments" / "phase4-stability"
OUT = PH4 / "plots"
OUT.mkdir(exist_ok=True)

P3 = REPO / "experiments" / "phase3-dataset-causality" / "data"

# continuation run name -> its origin E11 probe CSV (the 0..3M history to
# prepend). Any run NOT in this map is plotted from its own CSV alone.
CONT_ORIGINS = {
    "s_e12_cont_klC": P3 / "e11_klC_probe.csv",
    "s_e12_cont_klC_seed1": P3 / "e11_klC_seed1_probe.csv",
}
# Runs whose probe was built from 1-3 of 16 examples because kvp-ab carried a
# stale vendor/prompts.py (no raw_chat branch) that inflated passkey prompts
# past the eviction budget. Their kv_norm anchors (0.00 / 0.25 / 1.00) and every
# paired gap derived from them are meaningless, so they are excluded from the
# grid rather than plotted next to valid runs. See docs/runs/13.
INVALID_RUNS = {
    "s_e12_seed2", "s_e12_seed3", "s_e12_seed4", "s_e12_cont_klC_seed1",
}

# reference-only runs (finished E11 runs with no E12 continuation) still get
# their own panel for visual context.
REFS = {"s_e11_klC (ref seed0)": P3 / "e11_klC_probe.csv",
        "s_e11_klC_seed1 (ref seed1)": P3 / "e11_klC_seed1_probe.csv"}
# drop refs that are already embedded as the origin of a continuation panel
_embedded_origins = set(CONT_ORIGINS.values())
REFS = {k: v for k, v in REFS.items() if v not in _embedded_origins}


def _read_csv(path):
    # strip stray NUL bytes: an interrupted scp mid-write onto a CSV being
    # concurrently appended to remotely can leave a run of NUL bytes, which
    # crashes csv's C parser outright (_csv.Error: line contains NUL).
    text = Path(path).read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
    return list(csv.DictReader(io.StringIO(text)))


def load_rows(name, path):
    rows = _read_csv(path)
    origin = CONT_ORIGINS.get(name)
    if origin and origin.exists():
        rows = _read_csv(origin) + rows
    rows = [r for r in rows if r.get("timestep") not in (None, "")]
    rows.sort(key=lambda r: int(float(r["timestep"])))
    return rows


runs = sorted((PH4 / "data").glob("s_e12_*_probe.csv"))
panels = [(p.stem.replace("_probe", ""), p) for p in runs
          if p.stem.replace("_probe", "") not in INVALID_RUNS]
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
    rows = load_rows(name, path)
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
    ax.set_xlim(0, None)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("M pasos")

fig.tight_layout()
fig.savefig(OUT / "e12_progress.png")
print("plots/e12_progress.png")
for name, last_ts, m1, m2, kv in summary:
    print(f"  {name:32s} ts={last_ts:.2f}M  1a={m1:.2f}  2a={m2:.2f}  kv={kv:.2f}")
