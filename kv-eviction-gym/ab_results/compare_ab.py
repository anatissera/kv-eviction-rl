#!/usr/bin/env python3
"""Compare CONTROL (chat_full_v2) vs TREATMENT (chat_full_s4_kl) probe curves.

Usage: compare_ab.py <control_probe_curve.csv> <treat_probe_curve.csv> [out.png]
Headline metric: retention = correct_learned / correct_full. Reports the
last-N-probe average per arm and the treatment-minus-control delta. Stdlib only
(csv); matplotlib plot is best-effort.
"""
import csv, sys, statistics as st

def load(path):
    rows = []
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                rec = {}
                for k, v in row.items():
                    try: rec[k] = float(v)
                    except (TypeError, ValueError): rec[k] = float('nan')
                rows.append(rec)
    except FileNotFoundError:
        pass
    return rows

def col(rows, name):
    return [r.get(name, float('nan')) for r in rows]

def mean_clean(xs):
    xs = [x for x in xs if x == x]  # drop nan
    return st.mean(xs) if xs else float('nan')

def summary(label, rows, n=5):
    if not rows:
        print(f"== {label} ==\n  NO DATA\n")
        return None
    last = rows[-n:] if len(rows) >= n else rows
    ret = mean_clean(col(last, 'retention'))
    print(f"== {label} ==")
    print(f"  probes={len(rows)}  last_timestep={rows[-1].get('timestep', float('nan')):.0f}  "
          f"(averaging last {len(last)} probes)")
    print(f"  retention         {ret:.4f}")
    print(f"  correct_learned   {mean_clean(col(last,'correct_learned')):.4f}")
    print(f"  correct_random    {mean_clean(col(last,'correct_random')):.4f}")
    print(f"  correct_kv_norm   {mean_clean(col(last,'correct_kv_norm')):.4f}")
    print(f"  correct_full      {mean_clean(col(last,'correct_full')):.4f}")
    print(f"  evict_generated   {mean_clean(col(last,'evict_generated_frac')):.4f}")
    print(f"  truncation_rate   {mean_clean(col(last,'truncation_rate')):.4f}")
    print()
    return ret

def main():
    ctrl = load(sys.argv[1])
    treat = load(sys.argv[2])
    print("#" * 60)
    print("# S4 A/B RESULT — control vs S4-exact KL shaping")
    print("#" * 60 + "\n")
    rc = summary("CONTROL  chat_full_v2", ctrl)
    rt = summary("TREATMENT chat_full_s4_kl", treat)
    if rc is not None and rt is not None:
        d = rt - rc
        print(f"VERDICT: retention(treatment) - retention(control) = {d:+.4f}")
        if d > 0.02:
            print("  → S4-exact IMPROVES retention (H1 supported)")
        elif d < -0.02:
            print("  → S4-exact HURTS retention")
        else:
            print("  → S4 ~ no effect within noise → bottleneck likely PerTokenMLP capacity, not credit assignment")
    # best-effort plot
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for label, rows, c in [("control", ctrl, 'C0'), ("s4_kl", treat, 'C1')]:
            if rows:
                plt.plot(col(rows, 'timestep'), col(rows, 'retention'),
                         marker='.', color=c, label=label)
        plt.xlabel("timestep"); plt.ylabel("retention (learned/full)")
        plt.title("S4 A/B: retention vs steps"); plt.grid(alpha=0.3); plt.legend()
        out = sys.argv[3] if len(sys.argv) > 3 else "ab_retention.png"
        plt.savefig(out, dpi=120, bbox_inches='tight')
        print(f"\nplot → {out}")
    except Exception as e:
        print(f"\n(no plot: {e})")

if __name__ == "__main__":
    main()
