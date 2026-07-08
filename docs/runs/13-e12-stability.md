# 13 - E12: stability sweep on passkey (48 h, 3 VMs)

**Branch:** `exp/e12-stability` (isolated from main so the results are reportable as a
closed experiment).
**Status:** CLOSED (launched 2026-07-06, VMs stopped 2026-07-08). See the conclusion at
the end.
**Data:** `kv-eviction-gym/experiments/phase4-stability/data/`
**Progress plots:** `python experiments/phase4-stability/plot_progress.py`
**Report figures:** `python scripts/figures_e12.py` -> `docs/imgs/fig11..14`

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
| s_e12_seed2 | **INVALID** (ran on kvp-ab) | - | 0.25 (broken probe) | - | see "kvp-ab probe bug" below: probe built on 1-3 of 16 examples |
| s_e12_seed3 | **INVALID** (ran on kvp-ab) | - | 0.00 (broken probe) | - | same |
| s_e12_cont_klC | DONE (10M, extended 6M->9M auto + 9M->10M manual) | 0.58 (full 0->10M) | 0.56 | 89/186 (48%) | VALID (simcot-t4, 16/16 probes). Very slight positive in the aggregate (+0.02); real high peaks (touches 1.0) but they dilute over the full history, not sustained convergence |
| s_e12_cont_klC_seed1 | **INVALID** (ran on kvp-ab) | - | 0.50 (broken probe, n=2) | - | the earlier conclusion ("the extension did NOT reproduce the improvement") is NOT sustainable: it compared against a kv_norm measured on 2 examples |
| s_e12_epochs4 | DONE (3M) | 0.37 | 0.69 | 2/55 (4%) | VALID (kv-none-v2, 16/16 probes). **Confirms n_epochs=10 is the driver**: with n_epochs=4 it behaves as poorly as lrdecay_s0/klA/klB |
| s_e12_klw15 | DONE (3M) | 0.494 | 0.562 | 19/55 (35%) | VALID (16/16). **Arm E negative**: raising kl_weight 0.05->0.15 never beat the heuristic. A stronger dense reward does not fix its decoupling from final correctness. |
| s_e12_seed4 | **INVALID / KILLED** | - | 0.00 (broken probe, n=1) | - | its "degenerate anchor" was the bug, not seed bad luck |
| s_e12_seed5 | DONE (3M) | 0.439 | 0.375 | 32/54 (59%) | VALID (16/16). Gap +0.064 but mean below the 0.5 floor -> STOP. FLAT trend (1st 0.444 -> 2nd 0.433), final stretch the weakest. Same pattern as seed2/cont_klC: real peaks, no sustained convergence. |
| s_e12_entcoef | KILLED at 33% (conclusive verdict) | 0.257 | 0.688 | 0/19 (0%) | **arm G negative, with a mechanism**: ent_coef 0.003 causes an entropy collapse (-5.5 -> -4.53) and `truncation_rate` 0 -> **1.0**. The policy becomes deterministic and converges to an eviction pattern that stops the model from emitting EOS: episodes never finish, correctness is 0 and PPO loses its gradient. Same failure mode as the chat-template bug, reached by another path. Same-window contrast: targetkl (ent_coef 0.01) sits at entropy -5.56 and truncation 0.06. |
| s_e12_entcoef6 | CUT at 0.55M (VM stopped) | 0.15 | 0.688 | 0/10 (0%) | **arm G' negative, mechanism DIFFERENT from entcoef's**: with ent_coef 0.006 entropy stays healthy (-5.3, does not collapse like 0.003 which fell to -4.53), but the policy still converges to a bad local optimum: it evicts ~80% freshly generated tokens, truncates 100% and correctness is 0. Two different paths (entropy collapse vs a local optimum with healthy exploration), same destination. Less exploration does not help. |
| s_e12_epochs15 | CUT at 1.15M (VM stopped, incomplete) | 0.63 | 0.562 | 14/21 (67% strict, 86% >=) | **arm H, the strongest signal in the sweep but the least mature**: n_epochs 15, 1st half 0.556 -> 2nd half 0.705, rising. Completes the monotonic story with epochs4/klC: 4 ep 4% above, 10 ep 48%, 15 ep 67%. BUT it ran only 1.15M steps (vs 3M for the others): a VERY promising early trend, not a comparable final value. **The number 1 candidate for more compute** if resumed. |
| s_e12_targetkl | CUT at 2.70M (VM stopped, ~90%) | 0.59 | 0.688 | 6/50 (12% strict, 26% >=) | **arm F negative**: clip_range 0.2->0.1 + target_kl=0.03. The more conservative update did NOT cross its anchor (the sweep's highest, 0.688): flat trend (1st 0.583 -> 2nd 0.605), oscillates below. Required threading `target_kl` into train.py's PPO constructor (it was not; it would have been a silent no-op). The first attempt ran with the broken probe and was discarded; this is the clean relaunch. |

### kvp-ab probe bug (found 2026-07-08, invalidates 4 runs)

`kvp-ab` carried an old copy of `src/kv_gym/vendor/prompts.py` **without the `raw_chat`
handling**. Passkey prompts received GSM8K's instruction wrapper ("solve this math
problem"), which inflated `T` from ~285 to ~316 tokens. Since the probe discards any
example with `T >= budget` (=300) and that `continue` logged nothing, **13-15 of the 16
probe examples disappeared silently**.

The symptom we had been misreading: "degenerate anchors" (kv_norm = 0.00, 0.25, 1.00)
that we attributed to seed bad luck. In reality `kv_norm` can only be 0.0/1.0 with n=1,
multiples of 0.5 with n=2, of 0.333 with n=3.

| VM | prompts.py | real probes | affected runs |
|---|---|---|---|
| simcot-t4 | correct | **16/16** | klC, cont_klC, klw15 -> **healthy** |
| kv-none-v2 | correct | **16/16** | lrdecay_s0/s1, epochs4, seed5 -> **healthy** |
| kvp-ab | **STALE** | 1-3/16 | seed2, seed3, seed4, cont_klC_seed1, targetkl(1st attempt) -> **invalid** |

**The report's central result (klC, cont_klC) is NOT compromised:** it ran on simcot-t4
with all 16 probes.

Fixes applied:
1. Full `src/` synced to kvp-ab (verified: `T` back to 267-286, 16/16 survive).
2. Prefill cache (`~/.kv_eviction_cache`) purged on that VM: it held captures with the
   malformed prompt.
3. **Guard in `probe.py`**: aborts with a `RuntimeError` if fewer than 50% of the probe
   examples survive, instead of emitting meaningless anchors. This failure mode can never
   be silent again.
4. Corrupt CSVs backed up to scratchpad; `targetkl` relaunched from scratch.

**Aggregate reading of the continuation arm (C):** extending training beyond 3M DOES
produce higher and more frequent peaks (cont_klC touches 1.0 several times), but in the
two seeds where it was tried, the final result over the full history is parity (+0.02) or
negative, not clear, sustained convergence above kv_norm. The "more steps stabilize"
hypothesis is confirmed partially: it improves the magnitude of the peaks, not the
consistency.

## Sweep conclusion (2026-07-08, VMs stopped)

13 arms were run; 4 were invalidated by the kvp-ab probe bug (seed2, seed3, seed4,
cont_klC_seed1), 9 are valid. The report figures (`docs/imgs/fig11..14`) reflect only the
valid ones.

**What was learned, ordered by strength of evidence:**

1. **`n_epochs` is the causal driver, and the effect looks monotonic.** With everything
   else fixed: 4 epochs cross kv_norm in 4% of the evaluations, 10 in 48%, 15 in 67%
   (fig13). It is the sweep's cleanest result. Honest caveat: the 15-epoch run was cut at
   1.15M steps (the others reached 3M), so that 67% is an early TREND, not a comparable
   final value. Even so, the direction (more optimization epochs per rollout -> more
   coupling with correctness) is consistent across three points.

2. **No configuration, on its own, crosses kv_norm in a stable, sustained way.**
   klC/cont_klC's positive is real but fragile: there are high, frequent peaks (touching
   1.0), but over the full history the aggregate is parity (+0.02 at 10M). Extending
   steps improves the HEIGHT of the peaks, not their CONSISTENCY. This is the central
   message the report already carries, and E12 reinforces it with more data rather than
   contradicting it.

3. **The PPO stabilization levers we tried did not help:**
   - Linear LR decay (seeds 0 and 1): below kv_norm, did not turn the oscillation into
     convergence.
   - A more conservative update (clip_range 0.1 + target_kl 0.03): flat trend, never
     crossed its anchor.
   - A stronger dense reward (kl_weight 0.05->0.15): never beat the heuristic; more
     weight on the KL proxy does not fix its decoupling from final correctness.
   - Less exploration (ent_coef 0.003 and 0.006): both collapse to zero correctness,
     via two distinct mechanisms (entropy collapse for 0.003; a bad local optimum with
     healthy entropy for 0.006). Confirms the base config's exploration (0.01) was not
     the bottleneck.

4. **Seeds:** with valid anchors (seed5 on kv-none-v2), the positive does not reproduce
   as a sustained crossing: gap +0.064 but a flat trend and a mean below the 0.5 floor.
   It remains "a single-seed positive that the other seeds do not reproduce as a stable
   crossing", just as before E12.

**What would remain for more compute (future work):**
- Resume **n_epochs=15 up to 3M+** and with >=2 seeds: it is the only arm with a clearly
  rising, unresolved trend. Candidate number 1.
- The **wide paired eval** (n~96-128 shared passkey examples) on klC's final checkpoint,
  to separate real policy instability from the 16-example probe's noise. Proposed, never
  executed.
- Real GAE/gamma<1, which requires surgery on the episode buffer (discarded for the 48h
  window, see above).

**Methodological note E12 made clear:** with probe_n=16, much of the "oscillation" we had
been reading as policy instability is measurement noise (1 example = 0.0625). The
analysis rests on the paired learned-vs-kv_norm contrast within each run and on
aggregates over dozens of evaluations, never on a single probe. The kvp-ab probe bug
(which shrank n to 1-3 without warning) was the most expensive reminder of this; now
`probe.py` aborts if fewer than 50% of the examples survive.
