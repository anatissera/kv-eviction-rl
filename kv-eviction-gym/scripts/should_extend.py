"""Decide whether a finishing E12 run should be extended instead of stopped.

Usage: should_extend.py <probe_csv> <current_total_timesteps>
Prints "EXTEND:<new_total>" or "STOP" on stdout; keeper2.sh acts on it.

Criterion ("coming out well"): over ALL probes recorded since this run/segment
started (not an arbitrary first-half/second-half split, which is fragile for a
segment that is already a positive continuation and can dip briefly),
  - mean(learned) > mean(kv_norm)   [beats the heuristic on average]
  - mean(learned) >= 0.5            [absolute floor: guards against runs whose
                                      own probe split has a degenerate near-zero
                                      kv_norm anchor (seen with some seeds), where
                                      "beats kv_norm" would be trivially true]
  - fraction of probes above kv_norm >= 0.5
All three must hold, and there must be >= 6 probes to judge from.
Extension step: +3,000,000, capped at MAX_TOTAL. No extension if already there.
"""
import csv
import sys

MAX_TOTAL = 10_000_000
STEP = 3_000_000


def main():
    probe_csv, current_total = sys.argv[1], int(sys.argv[2])
    if current_total >= MAX_TOTAL:
        print("STOP")
        return

    rows = list(csv.DictReader(open(probe_csv)))
    n = len(rows)
    if n < 6:
        print("STOP")
        return

    learned = [float(r["correct_learned"]) for r in rows]
    kv = [float(r["correct_kv_norm"]) for r in rows]
    m_learned = sum(learned) / n
    m_kv = sum(kv) / n
    above = sum(1 for i in range(n) if learned[i] > kv[i])
    frac_above = above / n

    promising = (m_learned > m_kv) and (m_learned >= 0.5) and (frac_above >= 0.5)
    if promising:
        print(f"EXTEND:{min(current_total + STEP, MAX_TOTAL)}")
    else:
        print("STOP")


if __name__ == "__main__":
    main()
