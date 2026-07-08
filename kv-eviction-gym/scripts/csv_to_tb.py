"""Mirror every run's CSV curves into TensorBoard event files.

Reads probe_curve.csv (paired eval: full/random/kv_norm/learned + gap) and
learning_curve.csv (ep_rew_mean, correctness_rate, kl_step_mean, ...) from all
known result dirs and writes one TB run per experiment under a single logdir.

Incremental / append-only: a JSON state file tracks the last timestep already
written per (run, source). Each call only emits NEW rows and never deletes or
rewrites existing event files. This matters because TensorBoard's directory
watcher treats a brand-new event file whose first step is <= the last step it
already saw as a "training restart" and PURGES everything after that step
(purge_orphaned_data) -- which is exactly what a naive rewrite-from-scratch on
every cron cycle triggers: the live view gets stuck at the point of the last
purge and never re-extends past it until a full page reload. Appending
monotonically (as a real SB3 tensorboard log does across checkpoints) avoids
that entirely.

    python csv_to_tb.py            # backfill + refresh all
Then (once):  tensorboard --logdir <REPO>/kv-eviction-gym/tb_all --port 6006

Lives in scripts/ because it processes ALL phases (phase2-4), not just one.
"""
import csv
import io
import json
from pathlib import Path

from tensorboardX import SummaryWriter

REPO = Path(__file__).resolve().parents[1]        # kv-eviction-gym/
TB = REPO / "tb_all"
TB.mkdir(exist_ok=True)
STATE_FILE = TB / ".tb_state.json"

P3 = REPO / "experiments/phase3-dataset-causality/data"
# continuation run (bare CSV stem) -> its origin E11 (probe, learn) CSVs. These
# runs resume from an E11 checkpoint and only log timesteps from the resume
# point onward (e.g. 3M+ for s_e12_cont_klC); prepending the origin's 0..3M
# history makes the TB curve span the whole trajectory, not just the new part.
CONT_ORIGINS = {
    "s_e12_cont_klC": (P3 / "e11_klC_probe.csv", P3 / "e11_klC_learning.csv"),
    "s_e12_cont_klC_seed1": (P3 / "e11_klC_seed1_probe.csv",
                             P3 / "e11_klC_seed1_learning.csv"),
}

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
# Numeric prefixes force TensorBoard's alphabetical sort into a useful order:
#   0_learned  (fig10 capstone metric)    ->  first
#   1_gap      (learned - kv_norm)        ->  second
#   8_full / 8_kv_norm                    ->  near end  (flat baselines)
#   9_random                              ->  last      (flat baseline)
PROBE_TAG = {
    "learned": "probe/0_learned",
    "kv_norm": "probe/8_kv_norm",
    "random":  "probe/9_random",
    "full":    "probe/8_full",
}
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
    # strip stray NUL bytes: an interrupted scp mid-write onto a CSV being
    # concurrently appended to remotely can leave a run of NUL bytes, which
    # crashes csv's C parser outright (_csv.Error: line contains NUL).
    text = path.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
    return list(csv.DictReader(io.StringIO(text)))


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state))


def _valid_ts(rows):
    # tolerate malformed rows (bad/missing header, stray NUL-corrupted line)
    # the same way the rest of this script tolerates bad cells: skip, don't crash.
    out = []
    for r in rows:
        v = _f(r.get("timestep"))
        if v is not None:
            out.append(r)
    return out


def all_probe_rows(run, own_csv):
    origin = CONT_ORIGINS.get(run)
    rows = (load(origin[0]) if origin else []) + load(own_csv)
    rows = _valid_ts(rows)
    rows.sort(key=lambda r: int(float(r["timestep"])))
    return rows


def all_learn_rows(run, own_csv):
    origin = CONT_ORIGINS.get(run)
    rows = (load(origin[1]) if origin else []) + load(own_csv)
    rows = _valid_ts(rows)
    rows.sort(key=lambda r: int(float(r["timestep"])))
    return rows


def append_run(name, run, probe_csv, learn_csv, state):
    last_probe = state.get(f"{name}:probe", -1)
    last_learn = state.get(f"{name}:learn", -1)

    probe_rows = [r for r in all_probe_rows(run, probe_csv)
                  if (_f(r.get("timestep")) or -1) > last_probe]
    learn_rows = [r for r in all_learn_rows(run, learn_csv)
                  if (_f(r.get("timestep")) or -1) > last_learn]
    if not probe_rows and not learn_rows:
        return 0

    run_dir = TB / name
    run_dir.mkdir(parents=True, exist_ok=True)
    w = SummaryWriter(logdir=str(run_dir))
    new_rows = 0

    for row in probe_rows:
        ts = _f(row.get("timestep"))
        if ts is None:
            continue
        step = int(ts)
        for arm in PROBE_ARMS:
            v = _f(row.get(arm))
            if v is not None:
                short = arm.replace("correct_", "")
                w.add_scalar(PROBE_TAG[short], v, step)
        learned, kv = _f(row.get("correct_learned")), _f(row.get("correct_kv_norm"))
        if learned is not None and kv is not None:
            w.add_scalar("probe/1_gap", learned - kv, step)
        new_rows += 1
        last_probe = max(last_probe, ts)

    for row in learn_rows:
        ts = _f(row.get("timestep"))
        if ts is None:
            continue
        step = int(ts)
        for col in TRAIN_COLS:
            v = _f(row.get(col))
            if v is not None:
                w.add_scalar(f"train/{col}", v, step)
        last_learn = max(last_learn, ts)

    w.close()
    state[f"{name}:probe"] = last_probe
    state[f"{name}:learn"] = last_learn
    return new_rows


def main():
    state = load_state()
    total_new = 0
    total_runs = 0
    for base, prefix, psuf, lsuf in SOURCES:
        if not base.exists():
            continue
        for pc in sorted(base.glob(f"*{psuf}")):
            run = pc.name[: -len(psuf)]
            lc = base / f"{run}{lsuf}"
            n = append_run(f"{prefix}/{run}", run, pc, lc, state)
            if n:
                total_new += n
                total_runs += 1
    save_state(state)
    print(f"csv_to_tb: appended {total_new} new probe rows across "
          f"{total_runs} run(s) -> {TB}")


if __name__ == "__main__":
    main()
