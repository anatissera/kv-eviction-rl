# 13 - E12: stability sweep on passkey (48 h, 3 VMs)

**Branch:** `exp/e12-stability` (isolated from main so the results are reportable as a
closed experiment).
**Status:** IN PROGRESS (launched 2026-07-06).
**Data:** `kv-eviction-gym/experiments/phase4-stability/data/`
**Progress plots:** `python experiments/phase4-stability/plot_progress.py`

## Motivation (what E11 left behind)

E11 tested the online method (sequential PPO + causal dense reward) in the passkey arena
with 3 hyperparameter configs. Result:

| run | config | result |
|---|---|---|
| s_e11_klC (seed 0) | LR 1e-4, ent 0.01, **n_epochs 10** | 1st half 0.45, 2nd half **0.66** vs kv_norm 0.56: **crosses and holds (+0.10)**, touches 1.0 |
| s_e11_klC_seed1 (seed 1) | same | thirds 0.33 / 0.35 / **0.47** vs kv_norm 0.62: **same direction, does not cross** |
| s_e11_klA / klB | n_epochs 4 | oscillate without a shift (klB cut at 64%, unambiguous verdict) |

Reading: the upward shift during training appears in both seeds of the winning config,
but only one crosses kv_norm. The effect is consistent in direction and fragile in
magnitude. E12 attacks exactly that.

## Questions and arms

| arm | run(s) | question | decision rule |
|---|---|---|---|
| A. Seeds | s_e12_seed2, s_e12_seed3 | with 4 total seeds (0,1,2,3), how many cross kv_norm in a sustained way? | report x/4; >=3/4 strengthens the positive, <=1/4 degrades it to anecdote |
| B. LR decay | s_e12_lrdecay_s0, s_e12_lrdecay_s1 | does linear LR decay turn the oscillation into sustained convergence? | if the last quarter stays above kv_norm with lower variance than klC: a stabilization result (new report headline) |
| C. Continuation | s_e12_cont_klC | does training klC from 3M to 6M stabilize it? (the user's direct question) | if the 3M-6M stretch sustains mean > kv_norm with fewer drops: "more steps help"; if it keeps oscillating the same: plateau |
| D. Epochs mechanism | s_e12_epochs4 | is n_epochs=10 THE driver? (klA/klB differ from klC in epochs AND ent_coef; this isolates epochs) | no shift => epochs is the driver; shift => it was ent_coef or something else |
| E. kl_weight | s_e12_klw15 | does raising the dense-reward weight (0.05 -> 0.15) improve the coupling with correctness? | compare trajectory vs klC |

All arms are exact clones of `e11_klC.yaml` changing ONLY the indicated knob
(`e12_*.yaml` configs). Metric: the usual one, paired contrast learned vs kv_norm on the
run's own probe.

**Discarded with justification: GAE / gamma < 1.** The episode buffer
(`episode_ppo._finalize_episode`) computes per-layer returns as suffix sums that assume
gamma=1; gamma<1 would be silently ignored (a guard was added to train.py that now exits
with an error). Implementing real GAE requires surgery on the buffer and would change the
objective, not just the optimization: it stays as future work, not for a 48 h window.

**Discarded: a new 4th L4 VM.** Uncertain quota, and a new software stack introduces the
"regime shift" problem the report documents. The plan fits in 48 h with the 3 existing
lanes, and the continuation is MORE comparable running on the same T4 that trained
s_e11_klC.

## Timeline (T0 = 2026-07-06 ~12:00 ART)

Estimaciones: 3M pasos ~ 16 h en L4, ~ 23 h en T4 (medido en E11).

```
kv-none-v2 (L4)   |smoke| lrdecay_s0 (16h) | lrdecay_s1 (16h) | epochs4 (16h) |  ~T+48.5h
kvp-ab (L4 SPOT)  | seed2 (16h + prempt) | seed3 (16h + prempt) | libre/slack |  ~T+36-40h
simcot-t4 (T4)    | cont_klC 3M->6M (23h)      | klw15 (23h)             |      ~T+46h
```

kvp-ab's slack absorbs preemptions (now cheap: keeper2 resumes from the latest checkpoint
instead of restarting from zero). If time is left on kvp-ab: seed4 as a bonus.

## Infrastructure (the two fixes that were needed)

1. **keeper2.sh** (`experiments/phase4-stability/`): per-lane queues, completion
   detection (remote final_model.zip -> local `.done` marker -> launch the next one), and
   relaunch with `--resume-from <latest checkpoint>` after a preemption or crash. The
   phase-3 keeper restarted already-finished runs from zero (klC and seed1 were
   relaunched redundantly; CSVs truncated to the real run, backups in scratchpad).
2. **train.py**: `lr_schedule: linear` support (SB3's native callable; progress is
   cumulative across resumes because train.py passes the remaining budget) + a guard that
   exits if gamma != 1 with the episode buffer.

Periodic checkpoints already existed (`checkpoint_freq: 900` ~ every 50k steps).

## Estimated cost

L4 on-demand ~USD 0.85/h, L4 spot ~0.25/h, T4 ~0.40/h. 48 h of the 3 lanes
~ USD 70-75 (edu credits). Window explicitly authorized; VMs are not stopped between
runs, each slot starts as soon as the previous one finishes.

## Live results

(filled in as runs finish; `plot_progress.py` generates the grid)

| run | estado | 1a mitad | 2a mitad | kv_norm | veredicto |
|---|---|---|---|---|---|
| s_e12_lrdecay_s0 | | | | | |
| s_e12_lrdecay_s1 | | | | | |
| s_e12_seed2 | | | | | |
| s_e12_seed3 | | | | | |
| s_e12_cont_klC | | | | | |
| s_e12_epochs4 | | | | | |
| s_e12_klw15 | | | | | |
