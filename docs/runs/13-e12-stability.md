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
| C. Continuation | s_e12_cont_klC | does training klC beyond 3M stabilize it? (the user's direct question) | see "automatic extension" below: while it keeps beating kv_norm over its whole history, it keeps extending (10M cap) instead of stopping at a fixed number |
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

**Automatic extension (added 2026-07-06 at the user's request).** When a run reaches its
configured `total_timesteps`, `should_extend.py` evaluates its WHOLE probe history (not a
segment): if it beats `kv_norm` on average (with an absolute floor of 0.5 so degenerate
anchors like kv_norm=0 in some seeds cannot fool it) and in >=50% of the probes,
`keeper2.sh` adds +3M steps (10M cap) and relaunches it with `--resume-from` the latest
checkpoint in the SAME cycle, instead of moving to the next queue item.
`s_e12_cont_klC` was extended automatically once (6M -> 9M) on 2026-07-07. At 9M the
automatic criterion said STOP by a hair (mean_learned 0.597 > kv_norm 0.562, but only 48%
of probes above the 50% floor); at the user's explicit request the last stretch was
forced manually up to the 10M cap (kill + relaunch with --resume-from the latest
checkpoint, config bumped by hand), without waiting for the automatic decision.
The "3M" numbers in the timeline below are each run's starting point, not necessarily
where it ends.

## Timeline (T0 = 2026-07-06 ~12:00 ART)

Estimates: 3M steps ~ 16 h on L4, ~ 23 h on T4 (measured in E11). With automatic
extension, a run can take longer than estimated here.

```
kv-none-v2 (L4)   |smoke| lrdecay_s0 (16h) | lrdecay_s1 (16h) | epochs4 (16h) |  ~T+48.5h
kvp-ab (L4 SPOT)  | seed2 (16h + prempt) | seed3 (16h + prempt) | free/slack  |  ~T+36-40h
simcot-t4 (T4)    | cont_klC 3M->6M, extended to 9M (23h+) | klw15 (23h)     |  ~T+46h+
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

| run | status | mean | kv_norm | % above | verdict |
|---|---|---|---|---|---|
| s_e12_lrdecay_s0 | DONE (3M) | 0.32 | 0.69 | 0/55 (0%) | never crossed; the LR-decay hypothesis is not confirmed with this seed |
| s_e12_lrdecay_s1 | DONE (3M) | 0.51 | 0.62 | 15/54 (28%) | strong final streak but insufficient, STOP by a small margin (0.5 floor) |
| s_e12_seed2 | DONE (3M) | 0.44 | 0.25 | 20/36 (56%) | positive on average but declines over time, not extended |
| s_e12_seed3 | DONE (3M) | 0.25 | 0.00 | 21/44 (48%) | degenerate kv_norm anchor (almost 0), not conclusive |
| s_e12_cont_klC | DONE (10M, extended 6M->9M auto + 9M->10M manual) | 0.58 (full 0->10M) | 0.56 | 89/186 (48%) | very slight positive in the aggregate (+0.02); real high peaks (touches 1.0) but they dilute over the full history, not sustained convergence |
| s_e12_cont_klC_seed1 | DONE (6M, extended 3M->6M auto) | own stretch (3M-6M): 0.43 vs kv 0.50 | 0.50-0.57 | own stretch: 8/36 (22%) | the extension did NOT reproduce cont_klC's improvement; it ends below |
| s_e12_epochs4 | DONE (3M) | 0.37 | 0.69 | 2/55 (4%) | **confirms n_epochs=10 is the driver**: with n_epochs=4 it behaves as poorly as lrdecay_s0/klA/klB |
| s_e12_klw15 | running | - | 0.56 | - | oscillates in a reasonable range, far too early for a verdict |
| s_e12_seed4 | KILLED (57%, unfinished) | - | 0.00 (degenerate anchor, like seed3) | - | a kv_norm=0 anchor made "beats kv_norm" pointless to evaluate; cut to free kvp-ab |
| s_e12_seed5 | running | - | 0.375 | - | mixed, no clear trend yet |
| s_e12_targetkl | running (just launched) | - | - | - | **new arm F**: clip_range 0.2->0.1 + target_kl=0.03 (the rest identical to klC), on kvp-ab (replaces seed4). Attacks head-on the "touches the optimum and falls" pattern: with n_epochs=10 confirmed as the driver, a more conservative policy update could keep 10 epochs of gradient from overshooting and undoing a good policy. Required threading `target_kl` into train.py's PPO constructor (it was not; only clip_range/ent_coef were). |

**Aggregate reading of the continuation arm (C):** extending training beyond 3M DOES
produce higher and more frequent peaks (cont_klC touches 1.0 several times), but in the
two seeds where it was tried, the final result over the full history is parity (+0.02) or
negative, not clear, sustained convergence above kv_norm. The "more steps stabilize"
hypothesis is confirmed partially: it improves the magnitude of the peaks, not the
consistency.
