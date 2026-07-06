"""Mirror every run's CSV curves into TensorBoard event files.

Reads probe_curve.csv (paired eval: full/random/kv_norm/learned + gap) and
learning_curve.csv (ep_rew_mean, correctness_rate, kl_step_mean, ...) from all
known result dirs and writes one TB run per experiment under a single logdir.

Idempotent: each run's event dir is cleared and rewritten from the current CSV
on every call, so a cron can re-run this after each keeper2 download and live
E12 curves keep growing without duplicating scalars.

    python csv_to_tb.py            # backfill + refresh all
Then (once):  tensorboard --logdir <REPO>/kv-eviction-gym/tb_all --port 6006
"""
import csv
import shutil
from pathlib import Path

from tensorboardX import SummaryWriter

REPO = Path(__file__).resolve().parents[2]        # kv-eviction-gym/
TB = REPO / "tb_all"
TB.mkdir(exist_ok=True)

# (glob dir, prefix, probe-suffix, learning-suffix). prefix groups runs in the
# TB sidebar: phase2/... phase3/... phase4/...
SOURCES = [
    (REPO / "ab_results", "phase2", "_probe_curve.csv", "_learning_curve.csv"),
    (REPO / "experiments/phase3-dataset-causality/data", "phase3",
     "_probe.csv", "_learning.csv"),
    (REPO / "experiments/phase4-stability/data", "phase4",
     "_probe.csv", "_learning.csv"),
]

PROBE_ARMS = ["correct_learned", "correct_kv_norm", "correct_random", "correct_full"]
# learning-curve scalars worth watching live
TRAIN_COLS = ["ep_rew_mean", "correctness_rate", "kl_step_mean",
              "truncation_rate", "ep_len_mean"]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path):
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def write_run(name, probe_csv, learn_csv):
    run_dir = TB / name
    if run_dir.exists():
        shutil.rmtree(run_dir)            # rewrite for idempotency
    w = SummaryWriter(logdir=str(run_dir))
    n_probe = 0

    for row in load(probe_csv):
        ts = _f(row.get("timestep"))
        if ts is None:
            continue
        step = int(ts)
        for arm in PROBE_ARMS:
            v = _f(row.get(arm))
            if v is not None:
                w.add_scalar(f"probe/{arm.replace('correct_', '')}", v, step)
        learned, kv = _f(row.get("correct_learned")), _f(row.get("correct_kv_norm"))
        if learned is not None and kv is not None:
            w.add_scalar("probe/gap_learned_minus_kvnorm", learned - kv, step)
        n_probe += 1

    for row in load(learn_csv):
        ts = _f(row.get("timestep"))
        if ts is None:
            continue
        step = int(ts)
        for col in TRAIN_COLS:
            v = _f(row.get(col))
            if v is not None:
                w.add_scalar(f"train/{col}", v, step)

    w.close()
    return n_probe


def main():
    total = 0
    for base, prefix, psuf, lsuf in SOURCES:
        if not base.exists():
            continue
        for pc in sorted(base.glob(f"*{psuf}")):
            run = pc.name[: -len(psuf)]
            lc = base / f"{run}{lsuf}"
            n = write_run(f"{prefix}/{run}", pc, lc)
            if n:
                total += 1
    print(f"csv_to_tb: wrote {total} runs -> {TB}")


if __name__ == "__main__":
    main()
