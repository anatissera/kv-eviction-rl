#!/usr/bin/env python3
"""Multi-arm Phase 2 comparison.

Usage: compare.py <label=probe_curve.csv> [<label=...> ...] [--out plot.png]

Headline metric = the PAIRED gap  (correct_learned − correct_kv_norm)  averaged
over the last-N probes. Because every arm shares the same seed/anchors, kv_norm is
identical across arms, so this gap is what actually tells us whether the policy
BEATS the heuristic (>0) or not (≤0). Also prints retention and reports whether
each arm's learning curve is FLAT (the failure signature from the invalid A/B) or
shows a real trajectory. Stdlib only; matplotlib best-effort.
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

def col(rows, name): return [r.get(name, float('nan')) for r in rows]
def clean(xs): return [x for x in xs if x == x]
def mean(xs):
    xs = clean(xs); return st.mean(xs) if xs else float('nan')

def summarize(label, rows, n=5):
    if not rows:
        print(f"== {label} ==  NO DATA"); return None
    last = rows[-n:] if len(rows) >= n else rows
    learned = mean(col(last, 'correct_learned'))
    kvn     = mean(col(last, 'correct_kv_norm'))
    rnd     = mean(col(last, 'correct_random'))
    full    = mean(col(last, 'correct_full'))
    ret     = mean(col(last, 'retention'))
    gap     = learned - kvn
    # flatness: std of retention across the WHOLE run vs across last-n
    ret_all = clean(col(rows, 'retention'))
    flat = (st.pstdev(ret_all) < 0.03) if len(ret_all) > 3 else None
    print(f"== {label} ==")
    print(f"  probes={len(rows)} last_ts={rows[-1].get('timestep',float('nan')):.0f} (avg last {len(last)})")
    print(f"  learned={learned:.3f}  kv_norm={kvn:.3f}  random={rnd:.3f}  full={full:.3f}  retention={ret:.3f}")
    print(f"  PAIRED  learned − kv_norm = {gap:+.3f}   "
          f"{'✓ BEATS kv_norm' if gap > 0.02 else ('≈ ties kv_norm' if gap > -0.02 else '✗ below kv_norm')}")
    if flat is not None:
        print(f"  curve: {'FLAT (no learning)' if flat else 'has trajectory'} "
              f"(retention pstd={st.pstdev(ret_all):.3f} over run)")
    print()
    return dict(label=label, rows=rows, gap=gap, ret=ret, learned=learned, kvn=kvn)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out = None
    if '--out' in sys.argv:
        out = sys.argv[sys.argv.index('--out') + 1]
    print("#" * 64)
    print("# PHASE 2 — capacity experiments: does the policy beat kv_norm?")
    print("#" * 64 + "\n")
    results = []
    for a in args:
        label, path = a.split('=', 1)
        r = summarize(label, load(path))
        if r: results.append(r)
    if results:
        print("SUMMARY (paired gap vs kv_norm, higher=better):")
        for r in sorted(results, key=lambda x: -x['gap']):
            print(f"  {r['label']:24s} learned−kv_norm={r['gap']:+.3f}  retention={r['ret']:.3f}")
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5))
        for r in results:
            plt.plot(col(r['rows'], 'timestep'), col(r['rows'], 'retention'),
                     marker='.', label=r['label'])
        # kv_norm reference line (identical across arms)
        if results:
            kvn = results[0]['kvn']
            plt.axhline(kvn / max(results[0]['rows'][-1].get('correct_full', 1) or 1, 1e-9),
                        ls='--', c='gray', alpha=0.5, label='kv_norm (retention)')
        plt.xlabel("timestep"); plt.ylabel("retention (learned/full)")
        plt.title("Phase 2: retention vs steps"); plt.grid(alpha=0.3); plt.legend()
        if out:
            plt.savefig(out, dpi=120, bbox_inches='tight'); print(f"\nplot → {out}")
    except Exception as e:
        print(f"\n(no plot: {e})")

if __name__ == "__main__":
    main()
