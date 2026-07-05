# Phase 2: Findings log (chronological, honest)

Running log of what we tried, what it gave, and why we did the next thing. Numbers
are the real measured values. See `PLAN.md` for the up-front design/root-cause and
the git history for the code rationale per change. Last updated: 2026-07-01.

**One-line status:** rich features + a `kvz` feature bring the learned policy from
~random UP TO parity with `kv_norm` on a clean held-out probe, but do **not** clearly
beat it. Warm-start (running) and attention-at-scale (next) are the tests for whether
anything beats the heuristic.

---

## 0. Metric & baselines (so the numbers below mean something)

- Task: online KV-cache eviction on GSM8K, Qwen2.5-1.5B-Instruct, budget=256.
- Headline paired metric: **`correct_learned − correct_kv_norm`** on the SAME probe
  (kv_norm is a deterministic heuristic = evict lowest ‖K‖+‖V‖; it's the bar to beat).
- Baselines on a probe: `full` (no eviction = ceiling), `kv_norm` (best heuristic),
  `random`. `retention = correct_learned / correct_full`.

---

## 1. The overnight A/B looked like a win (+0.183) but was INVALID

- Control (no shaping) vs Treatment (S4 KL shaping), 5M steps, on TWO different VMs.
- Headline read: retention +0.183 for S4 → "S4 works".
- **Why invalid:** the policy-independent baselines differed between arms
  (`kv_norm` 0.625 vs 0.750; `random` 0.562 vs 0.750), impossible if the eval were
  comparable. Cause: the two VMs ran different SCP'd code → different 32-example probe
  sets. On the honest **paired** metric both arms were ≈ −0.037 below kv_norm, i.e.
  **S4 ≈ control and neither beat kv_norm**. The +0.183 was a probe-set artifact.
- **Decision:** re-run a CLEAN A/B (one VM, one seed, identical anchors).

## 2. Root cause of "learned never beats kv_norm" (code-level)

Four separable deficiencies (see PLAN §3): **D1 norm-blindness** (the PerTokenMLP
LayerNorms K/V at input → erases the magnitude kv_norm uses), **D2 position-blindness**,
**D3 no cross-token reasoning** (per-token independent), **D4 cold-start** (RL from
random never finds the kv_norm basin). D1 alone explains "≈ random".

## 3. Clean A/B → PIVOTED to the E0 screen (spot preemption + redundancy)

- Launched clean A/B (2M, 1 VM). It got spot-preempted and restarted from 0; its
  null was already established by the paired metric. Only 1 spot L4 exists
  (on-demand + spot STOCKOUT everywhere, 2 full zone hunts).
- **Decision:** killed the redundant A/B, pivoted the VM to the E0 capacity screen
  (higher value, short runs survive preemption).

## 4. E0 capacity screen: RESULT (16 examples, overfit, probe_on_train)

Question: on data it was trained on, can each variant beat kv_norm? (capacity ceiling)

Setup reality on these 16: `full`=0.81 (13/16); `kv_norm`=`random`=0.06 (1/16):
eviction is catastrophic for the heuristics here, so there's lots of headroom.

| variant | best learned | vs kv_norm | reading |
|---|---|---|---|
| baseline (no rich) | 0.06 | **+0.00 (ties)** | can't beat kv_norm, as expected (norm-blind) |
| **rich features** | 0.25 | **+0.19** | beats kv_norm 4× at peak, but SPIKY (0↔0.25) |
| **attention** | 0.25 | **+0.19** | same, spiky, trending up |
| rich + S4 | 0.125 | +0.06 | **< rich → S4 does NOT help** a capable policy |
| rich + warm-start | n/a | FAILED | BC bug (see §5) → inconclusive |

**Verdict:** representation (D1/D2) is the bottleneck DIRECTIONALLY, rich/attention
can beat kv_norm, baseline can't. BUT the signal is noisy/spiky and the absolute
recovery is modest. **S4 confirmed unhelpful even on a capable policy.**

## 5. Warm-start BC was broken: diagnosed and fixed (two bugs)

- Symptom: `match_kv_norm` plateaued at ~0.29; the cloned policy performed at 0.
- **Bug A (representability):** rich features gave `kz_mean` and `vz_mean`
  standardized SEPARATELY (σ_K≠σ_V), so the policy **cannot** reconstruct
  `argmin(‖K‖+‖V‖)` that kv_norm uses. → **Fix: added `kvz` feature** =
  standardized(‖K‖+‖V‖ mean-over-heads) = kv_norm's EXACT signal. Now the policy can
  represent kv_norm as `logit = −kvz`, and re-weight K vs V to try to beat it. Rich
  extra features: 2H+4 → **2H+5 = 9**. (test proves argmin(kvz)==kv_norm's choice.)
- **Bug B (optimization):** BC used cross-entropy to the argmin over ~220 slots with
  56-sample batches → high variance, bounced. → **Fix: dense MSE**: regress raw
  actor logits to `−kvz×3` on every valid slot (all slots supervised → stable;
  argmax=argmin(kvz)=kv_norm).
- Also fixed: probe CSV header (preempted runs left headerless CSVs → parsers saw
  "NO DATA"); run_screen wipes run_dir on fresh start (no concatenated CSVs).

## 6. Infra bug: checkpoints were never saved (spot resume was a no-op)

- `CheckpointCallback.save_freq` counts **on_step calls = wall-steps**, not timesteps.
  The batched env does n_parallel×n_layers = **56 timesteps per wall-step**, so the
  default `save_freq=50000` → first save at 2.8M timesteps > 2M total → **checkpoints
  dir stayed EMPTY** → every spot preemption restarted from 0 (s_rich lost ~660k
  twice). → **Fix: `checkpoint_freq=900`** (≈50k timesteps). Confirmed: 38 checkpoints
  now saved; future preemptions resume.

## 7. Scaled run s_rich (rich + kvz, 2M, full 1000-ex train, held-out 32 probe), RESULT

The clean, less-noisy setting. Here `kv_norm`=0.625 is a STRONG baseline (not 0.06).

`correct_learned − correct_kv_norm` across probes (both preemption attempts):
```
-0.125 -0.031 -0.094  0.000 -0.031 +0.031 -0.094  0.000 -0.031 -0.031
-0.156  0.000 -0.031 -0.094 +0.062 +0.062 -0.062 -0.156
```
- `full`=0.75, `kv_norm`=0.625, `random`=0.562. Learned oscillates AROUND kv_norm,
  mean ≈ **−0.02** (peaks +0.062, troughs −0.156).
- **FINAL (run completed 2026-07-02, 31 probes):** paired mean **−0.044**, last-5
  **−0.050**, max +0.094. Curves + final/best checkpoints downloaded to
  `ab_results/`. Definitive: s_rich = parity-to-slightly-below, does NOT beat.
- **Reading: PARITY, not a beat.** rich+kvz moved the policy from ~random (the old
  norm-blind level) UP TO kv_norm: real progress (representation closed the gap),
  but it does not reliably EXCEED the heuristic. With kvz as a feature the easiest
  thing to learn is "be kv_norm"; beating it needs something more (re-weight K/V,
  position, cross-token) that per-token PPO isn't finding.

## 8. Scaled run s_warm (BC→kv_norm + RL, 2M), RESULT: PARITY, exploration was NOT the wall

Completed 2026-07-02 ~07:25 UTC (survived two spot preemptions via checkpoint
resume; exposed + fixed a resume-budget bug, commit 8decafd). 10 probes:

```
ts:   162k  329k  525k  701k  932k  1086k 1283k 1416k 1624k 1959k
gap: +.031 +.031 -.031 -.062 -.094 +.000 -.031 +.031 -.031 -.062
```
- **The BC took:** first probe (162k) has learned=0.656 ≥ kv_norm=0.625, the
  policy STARTS at/above the heuristic (screen's BC failure is fixed).
- **RL does not push past it:** mean −0.022, avg-last-5 −0.019, max +0.031.
  If anything RL *degrades* the clone mid-run (932k: −0.094) then recovers to
  parity. Same online-PPO ceiling as s_rich (−0.050 last-5), slightly closer.
- **Verdict (D4):** exploration/cold-start was NOT the binding constraint.
  Starting AT kv_norm + 2M of PPO does not find anything better than kv_norm.
  Per-token policies = parity ceiling, warm or cold. Matches the 2026 literature
  (KVP/ForesightKV): online RL from terminal correctness can't beat the
  heuristic; their wins come from offline future-attention oracle supervision.

## 9. Scaled run s_attn (rich + cross-token attention, 2M), RESULT: BELOW kv_norm, D3 closed negative

Completed 2026-07-02 09:50 UTC on `kv-chat-v1` (Alex's project, on-demand L4, no
preemptions). 13 probes, within-arm paired gap:

```
ts:   149k  327k  474k  621k  762k  914k 1087k 1232k 1383k 1512k 1666k 1796k 1944k
gap: -.062 -.156 -.094 -.094 -.062 -.094 -.125 -.125 -.094 -.062 -.156 -.188 -.125
```
- **Never at or above kv_norm** (best probe −0.062). Mean **−0.111**, last-5 −0.125.
- **Environment-shift caveat (again!):** this VM's baselines are kv_norm=random=
  full=**0.750** vs kvp-ab's 0.625/0.562/0.750: same seed, same 32 examples, but a
  different torch/transformers stack changes generations. Two consequences:
  (a) cross-arm comparisons of RAW numbers between s_attn and s_rich/s_warm are
  NOT valid: only within-arm paired gaps are; (b) on this stack eviction barely
  hurts the heuristics (random ties full → ceiling effect, little headroom), yet
  the learned attention policy still lands BELOW random, it evicts actively worse.
- **Verdict (D3):** the cross-token architecture does not rescue online PPO either;
  at scale it is the worst of the three arms on its own probe. The screen's +0.19
  peak did not transfer (overfit-16 capacity ≠ generalization at 1000-train).
- All three deficiency-fix branches (D1/D2 rich, D4 warm, D3 attention) now
  closed: **online PPO from terminal correctness ⇒ parity with kv_norm at best.**
  The decisive cross-arm number comes from the wide eval (§10): n=128 fresh
  examples, all three checkpoints evaluated on ONE VM (one software stack).

## 10. WIDE EVAL (n=128 fresh, one VM/stack), the decisive table + a regime insight

Completed 2026-07-02 ~13:30 UTC (`scripts/wide_eval.py` on kv-chat-v1; 128 GSM8K
examples from the same seeded shuffle, slice [1032:1160] → disjoint from all
training/probe data; identical anchors for every row; VM auto-stopped after).

| arm | learned | paired vs kv_norm | trunc |
|---|---|---|---|
| s_rich_final | 0.695 | **−0.016** (≈2 ex.) | 0.03 |
| s_warm_final | 0.688 | **−0.023** (≈3 ex.) | 0.21 |
| s_attn_final | 0.648 | **−0.063** | 0.01 |
| s_attn_best  | 0.633 | **−0.078** | 0.02 |
| *kv_norm* | *0.711* | (baseline) | |
| *random*  | *0.703* | *−0.008 vs kv_norm* | |
| *full*    | *0.742* | *ceiling* | |

- **Per-token arms (rich, warm) = parity within noise** (1 example = 0.0078; they
  sit 2-3 examples below kv_norm). **Attention is genuinely worse** (8-10 ex.
  below, also below random): the D3 negative replicates out-of-probe.
- **THE REGIME INSIGHT (new, load-bearing):** `full − random = 0.039`, at
  budget=256 with these prompt lengths, random eviction only costs ~4pp vs no
  eviction, and kv_norm captures ~20% of that sliver (+0.008 over random). There
  is almost NOTHING for a learned policy to win in this regime: the terminal
  reward is nearly flat in the policy → PPO has ~no gradient signal, and even a
  perfect oracle could gain at most ~3pp over kv_norm. This is bounded by the
  ENV ARCHITECTURE: the batched env pre-allocates budget+1 slots and skips any
  example with T_prompt > budget → budget can never go below max prompt length
  (~232), i.e. the env cannot express aggressive compression. The literature's
  learned-eviction wins (KVP, ForesightKV) live at 50% budgets / 128K contexts
  where eviction actually hurts. Future work: redesign the env to allow prompt
  compression (SnapKV-style prefill selection) or move to long-generation tasks.

## 11. ORACLE-GAP experiment: the learnable margin EXISTS (+7pp) and is FUTURE information

Completed 2026-07-02 18:27 UTC (`scripts/oracle_eval.py`, 128 wide examples, all
arms paired per example under ONE eager-attention model instance; 0 errors).

| arm | acc | paired vs kv_norm |
|---|---|---|
| **oracle_fut** (golden eviction: min MAX-FUTURE attention from full trace) | **0.516** | **+0.070 ± 0.025 (2.8σ; 10W-1L, McNemar p≈0.01)** |
| full (no eviction) | 0.492 | +0.047 |
| attn_cur (H2O-family: min CURRENT accumulated attention) | 0.461 | +0.016 (0.5σ, 8W-6L, noise) |
| kv_norm | 0.445 | (baseline) |

Three findings:
1. **The learnable margin exists: ≈ +7pp, and it is specifically FUTURE
   information.** Present-attention (attn_cur) does NOT beat kv_norm. This
   coherently explains ALL of Phase 2: no online policy (learned or heuristic)
   can see the future → they all tie at the kv_norm level. kv_norm ≈ the online
   ceiling; the future-supervised ceiling is +7pp above it.
2. **The oracle beats even full-cache** (0.516 > 0.492): good eviction is not
   just harmless: dropping the right tokens *improves* reasoning (beneficial
   trajectory steering). "full" was never a true ceiling.
3. **Third environment shift, root-caused:** full = 0.49 under eager vs 0.74
   under sdpa on the SAME examples. Cause (loader.py): in bf16, eager computes
   QK^T in bf16 while sdpa accumulates in fp32 → eager inference is genuinely
   worse. Paired within-run gaps remain valid; the +7pp may shrink somewhat in
   the healthier sdpa regime. (Caveat for the report.)

**Decision (per GRAN_PLAN tree):** margin exists → **E5 Golden-BC launched**:
distill future-attention rankings into the online policy (ForesightKV recipe):
trace generation (~400 train examples) → dense-BC of actor logits to the
future-attention score → short RL → wide_eval. Target: capture a fraction of
the +7pp; how much is capturable by a policy that must PREDICT the future from
present features is exactly E5's question.

## 12. E5 Golden-BC: RESULT: future attention is UNPREDICTABLE from present features

Ran 2026-07-03 on kvp-ab (traces: 400 train examples, ~35 min; BC 3000 steps;
PPO 2M). Three findings:

1. **The BC never converged: `match_oracle` stuck at 0.018–0.036** for all 3000
   steps (loss 9.0→8.4, flat; vs match≈1.0 when cloning kv_norm with the same
   dense-MSE recipe). A per-token policy seeing raw K‖V + rich features CANNOT
   predict which slot the future-attention oracle evicts, barely 8× above
   chance (1/220). **This is the cleanest statement of the whole Phase-2 story:
   the +7pp oracle margin (§11) is made of information that does not exist in
   the present observation.** (ForesightKV gets around this with attention-
   history features: inputs our sdpa env doesn't expose, and even then needs
   pairwise-ranking, per-head scorers.)
2. **s_golden ≈ rich policy with a failed init:** train correctness DECLINED
   0.60→0.54 with truncation rising 0.10→0.40 (reward drift). On its own probe
   it posts mean **+0.074 over kv_norm**: but see the caveat below.
3. **Probe-slice heterogeneity (methodological trap, 2nd sighting):** e5's probe
   ([400:432], its n_examples=400 config) has kv_norm=0.06, random=0.09,
   full=0.78: eviction is CATASTROPHIC there, unlike [1000:1032] (kv_norm
   0.625) or the wide slice (0.71), same stack. GSM8K slices differ wildly in
   eviction sensitivity. s_golden's +0.074 is "beats a floor-level kv_norm on a
   hostile slice" (like the E0 screen), NOT oracle distillation. The decisive
   number is the shared wide slice [1032:1160] → §13.

## 13. wide2 + E6 v1: s_golden confirmed artifact; hard regime OVERSHOT; E6b launched

**wide2 (2026-07-03, shared slice [1032:1160], kvp-ab stack, paired anchors):**
- `s_rich_final` **−0.008**: parity, replicated on a 2nd stack.
- `s_golden_final` **−0.070**, trunc 0.29: WORSE than kv_norm on the benign
  slice. Confirms §12: its hostile-probe edge was regime, not oracle knowledge;
  its truncation drift actively hurts.
- Stack note: on kvp-ab, full=0.742 / kv_norm=0.570 / random=0.586, this stack
  has **15.6pp** of full−random headroom on the same examples where kv-chat-v1
  showed 3.9pp (4th environment-shift sighting), and rich STILL ties kv_norm →
  further evidence the ceiling is informational, not headroom-only.

**E6 v1 (s_long, budget 160 + min_answer_words 60), regime OVERSHOT:**
- Train correctness **0.000 on every rollout**, truncation 0.8→1.0: under this
  much cache pressure the model solves NOTHING → constant reward → zero signal
  (the mirror image of the benign regime's failure). Probe: kv_norm=0.037,
  random=0.000, full=0.778; learned 0.074-0.111 (+1-2 examples of noise).
- Lesson: the regime knob has a narrow discriminative band, task must remain
  SOLVABLE under eviction while eviction still hurts.

**E6b (s_long192) launched 2026-07-03 10:21 UTC:** budget 192 → ~215 evictions
(the oracle-data zone where kv_norm≈0.31 / full≈0.38), probes every 8 rollouts,
watcher with 45-min AUTO-ABORT if train correctness stays all-zero.

## 14. The exposure analysis: why online PPO was ALSO sample-starved (E7 design)

Computed from the training curves (2026-07-03): each episode costs ~4,900
timesteps (≈174 decode-steps × 28 layers), so a "2M-step" run completes only
**~287 episodes ≈ 57 distinct examples × 5 gradient passes**: 94% of the
"1000-example train set" was never touched. Statistically, with a terminal 0/1
reward, 5 samples per example give SE≈0.32 on the eviction effect (~0.05):
**the signal is 6× below the per-example noise floor.** ~50 samples/example
bring SE≈0.10. (Credit: Alex's suggestion to raise repetitions.)

**E7 (`configs/e7_repeat.yaml`, s_e7)**: three multiplying levers:
1. Pool of 32 long-gen examples (min_answer_words 60, budget 256, the
   validated live band); `repeats_per_problem: 50`; 8M timesteps → every
   example seen ~50×.
2. **Per-example regret baseline** (`per_example_baseline: true`, implemented
   in `batched_env._terminal_reward`): terminal reward centered by the running
   mean score of the same example. High repeats make the mean estimable; the
   centering removes the ±0.45 example-difficulty variance. Together they turn
   the terminal reward into a per-example paired contrast.
3. Probe: held-out filtered [32:48] (n=16, long-gen probes cost ~20 min).
   Final verdict: wide_eval on the shared filtered [1032:1160] slice, reusing
   eval_e6c's anchors.

Queued behind eval_e6c on kvp-ab (watcher_e7.sh chains automatically; curves
live-download to ab_results/ every 5 min). ETA ~20-24h of GPU.

## 15. eval_e6c: the long-gen regime is THE arena (45pp headroom) but s_rich doesn't win it

128 fresh filtered long-gen examples, budget 256, kvp-ab stack (2026-07-04):
full=0.680, **random=0.227, kv_norm=0.211**, s_rich=0.195 (paired −0.016).

- **full−random = 0.45**: in the long-gen regime the learnable margin is HUGE,
  the task remains solvable, and bad eviction is catastrophic, the
  discriminative arena we were looking for (vs 0.04 on the unfiltered slice).
- **kv_norm ≤ random here**: the per-token norm heuristic doesn't even beat
  chance on long generations. The bar E7 must clear is LOW and any real
  learning should be unmistakable.
- s_rich (trained on the general distribution, 5 exposures/example) does NOT
  transfer a win: parity again. Consistent with §14's exposure analysis.
- E7 (s_e7) auto-launched into exactly this arena with the statistical fix
  (50 reps/example + per-example regret baseline). Anchors for its final
  wide eval are these very numbers (reused, zero cost).

## 16. Long-gen oracle bound: UNMEASURABLE under eager (5th regime shift), oracle=parity there

Stopped at n=79/128 (conclusion locked, 2026-07-03 ~22:30 UTC): full=0.371,
kv_norm=0.371, attn_cur=0.214, oracle_fut gap **+0.013±0.038** (parity).

- **The eager stack has NO eviction damage on long-gen** (full−kv_norm = 0.000
  there, vs 0.47 on sdpa/eval_e6c: same slice, same seed). bf16-eager numerics
  change the generations enough to flip the regime itself (5th shift sighting).
- Consequence, methodological: **the golden-eviction bound cannot be measured
  for the sdpa regime at all**: capturing attention requires eager, and eager
  destroys the very damage we want the oracle to recover. Any oracle number is
  regime-relative. (attn_cur collapsing to 0.214 << kv_norm on long-gen also
  kills the "present attention" heuristic in this arena.)
- E7's premise is unaffected (an ONLINE policy adapts on-policy; it does not
  need trace labels), but no oracle upper bound exists for its arena.

## 17. E7 first probe: the arena's bar is ZERO; screening the pool (E7b prep)

First s_e7 probe (ts=1.48M): probe [32:48] filtered, budget 256 → full=0.875,
**kv_norm=0.000, random=0.000**, learned=0.000, trunc 0.19.
- The arena is maximally discriminative: ANY probe point > 0 = real learned
  eviction skill (bar at zero, headroom 0.875).
- But it is also a sparse-reward cliff: train corr=0 through 54 rollouts is
  partly group design (repeats=50 × n_parallel=2 → only examples 0-3 seen),
  partly the cliff (if no eviction policy reaches correctness, reward gradient
  toward correctness never appears: classic exploration chicken-and-egg).
- **Mitigation launched (kv-chat-v1, repurposed): pool screening**: full-cache
  sdpa correctness for filtered train [0:96] (`scripts/screen_pool.py`). If
  s_e7 shows no probe signal by ~4M, restart as **E7b** with the pool = first
  32 FULL-SOLVABLE examples (every group informative; ForesightKV-style
  trace filtering). Decision pending s_e7's next probes.

## 18. E7 (high-rep + regret, long-gen arena), RESULT: exploration cliff, PARITY-at-zero

s_e7 ran to ~4.5M/8M (spot-preempted, then repurposed). 3 held-out probes, all
**learned=0.000 = kv_norm=0.000** (full=0.875). Training found correct episodes
only by chance (~4 in 235 rollouts, no acceleration) and PPO never converted
them into policy. Verdict: **the sparse terminal reward cannot cross the
exploration cliff**: even with 50 exposures/example and the per-example regret
baseline, discovering a ~350-eviction sequence that preserves the answer (among
~220 choices/step) is not reachable by online exploration from scratch.

This is the load-bearing negative for the report: across THREE regimes
(benign=no headroom, hard=unsolvable, calibrated=solvable-but-cliff) online PPO
from the terminal reward never beats kv_norm. The bottleneck is not
representation (rich features), exploration-init (warm-start), architecture
(attention), sample count (50×), or difficulty variance (regret), it is the
**density/informativeness of the reward signal itself.**

## 19. E8: S4 "distill-to-full" DENSE reward (Alex's idea) in the long-gen arena

The exact fix the cliff calls for, and the first REAL test of S4: a per-step
reward = −kl_weight·clip(KL(p_full ‖ p_evict), 0, 5), where p_full is a shadow
never-evicted cache (batched_env kl_shaping=exact). The old S4 A/B was null
because the policy was norm-blind AND the regime had no headroom, both now
fixed (rich features + long-gen 45pp). S4 gives per-step credit so the policy
no longer needs to stumble onto a fully-correct episode; it is rewarded for
every eviction that keeps the next-token distribution close to full-cache.
= Alex's "destilar al full caché". kl_weight=0.03 (first-principles; the runs'
first-rollout kl_step_mean≈0.14 confirmed Σ≈1). E8 (mlp) on kvp-ab + E8-attn
(Alex's attention-classifier idea) on kv-chat-v1.

**RESULT (2026-07-04): PARITY, both arms: but the most informative null.**
- s_e8 (mlp+S4): 9 probes, paired `learned−kv_norm` = **+0.021 ± 0.014** (n.s.).
- s_e8attn (attn+S4): 12 probes, **+0.016 ± 0.018** (n.s.). The single +0.125
  final probe is one example; the mean is parity.
- **Mechanistic finding:** `kl_step_mean` DROPS (0.109→0.073 on s_e8), S4 IS
  being optimized: but probe correctness never follows. **Minimizing joint
  output-KL is a proxy that decouples from correctness** under the truncation
  cliff (evicting reasoning tokens → model rambles → truncates → wrong;
  `evict_generated_frac` 0.5-0.99 in the probes).

## 20. ROOT CAUSE (code-level): the layer credit-assignment wall

Reviewing the code after E8, we found the structural reason the ENTIRE series is
stuck at parity: and it is not any of the things we'd been fixing.

- The env (`batched_env.py`) exposes **`n_envs = N × n_layers`**: each
  `(episode, layer)` pair is a separate RL env. The policy emits **28 independent
  eviction actions per step**: one per transformer layer (`step_wait`,
  `actions.reshape(self.N, self.n_layers)`).
- But the reward is a **single scalar per episode, broadcast identically to all
  28 layer-slots** (`for l: all_rewards[base+l] = step_r[b]`; terminal
  `rewards = np.full(n_layers, score)`). Even S4's KL is measured on the **final
  output distribution**: a joint function of all 28 layers' evictions.

So PPO sees 28 different (state, action) pairs sharing ONE return. The marginal
effect of any single layer's eviction on the scalar is drowned by the other 27 →
**no per-layer gradient**. The only thing learnable from a joint scalar over a
220²⁸ joint action space is a "generically reasonable per-layer rule", which is
**approximately what kv_norm already is**. Hence parity, everywhere:

| we tried | it improved | touched layer credit? |
|---|---|---|
| rich features (s_rich) | representation | ✗ |
| warm-start (s_warm) | initialization | ✗ |
| attention (s_attn) | architecture | ✗ |
| S4 dense (s_e8) | **temporal** credit (per-step) | ✗ (KL is joint over layers) |
| 50 reps + regret (s_e7) | sample count + difficulty variance | ✗ |

None touched the binding constraint. This also explains s_e8 exactly: S4 improved
the *joint average* (kl_step drops) but per-layer decisions can't be refined, so
correctness stays at kv_norm level.

**Why we trained each arm to completion anyway (it was not wasted):** each arm
was a controlled ablation isolating ONE hypothesis (representation / exploration /
architecture / reward density / sampling). Running them to convergence is what
let us *rule each out* and, by elimination + code review, locate the real wall.
The null series IS the evidence base for the root cause, and a clean methodo-
logical contribution (apples-to-apples paired evaluation across a hypothesis
ladder).

## 21. E9: per-layer reward (the fix for the root cause), LAUNCHED 2026-07-04

**Architecture change (code):** a new training mode `per_layer_reward: true` in
`batched_env.step_wait`. Instead of the joint scalar, each layer-env gets its OWN
dense reward = the **marginal hidden-state divergence** that layer's eviction
causes vs a never-evicted shadow cache:
`div_k = 1 − cos(h_evict[k], h_full[k])` at each block output k (via
`output_hidden_states`); `damage_l = relu(div_{l+1} − div_l)` isolates layer l's
own contribution; reward `= −w · clip(damage_l, 0, 5)`. Terminal correctness
(+ regret baseline) stays broadcast (a joint outcome the critic absorbs); the
per-layer dense term supplies the differentiation.

**Why this and not "tie the layers":** tying eviction (same position all layers)
also fixes attributability but handicaps a *uniform* policy against a *per-layer*
heuristic (kv_norm evicts per-layer): a confounded, invasive change. Per-layer
reward keeps the fair per-layer paradigm, is contained (the buffer/GAE ALREADY
compute per-layer returns: `episode_ppo._finalize_episode` stacks rewards as
`[T, n_layers]`; the ONLY thing forcing identical returns was the broadcast), and
matches how the literature (KVP/ForesightKV) solves credit assignment (per-head
supervision).

**What we expect to see (the decisive difference from every prior run):** if
layer credit assignment was the wall, the paired probe gap `learned−kv_norm`
should now be able to **CLIMB above noise** (sustained > +1 example over ≥3
probes) as per-layer damage drops: something no joint-reward run ever did. If it
stays at parity while per-layer damage drops, per-layer *distributional* damage is
also decoupled from correctness → next is per-layer *correctness* attribution or
the env redesign (Phase B, GRAN_PLAN §4c).

**Experiments launched (2 VMs, parallel):**
- **s_e9** (per-layer reward, MLP policy) on kv-chat-v1.
- **s_e9attn** (per-layer reward, attention policy, Alex's classifier, now with
  fixed credit) on kvp-ab (after s_e8 finished).
Both: long-gen arena (min_answer_words 60, budget 256), 50 reps, regret baseline,
`per_layer_reward: true`, 4M steps, shared filtered probe, cron keeper +
live-download. Verification: the smoke printed the 28 per-layer rewards to confirm
they DIFFER (proof the fix is live), the old broadcast made them identical.

## 22. E9 RESULT: per-layer reward = PARITY too (fix works mechanically, second wall found)

Cut at 50% (2M/4M) once the verdict was locked, 2026-07-04; both VMs stopped.
- s_e9 (per-layer reward, MLP): 6 probes, paired gap mean **+0.021 ± 0.035** (n.s.).
  Trajectory [+0.06, 0, +0.19, -0.06, -0.06, 0]: the +0.19 was a single-probe
  outlier that reverted, not a sustained climb.
- s_e9attn (per-layer reward, attention): 5 probes, **+0.038 ± 0.022** (n.s.).

**What we learned (two nested walls, now both located):**
1. Layer credit assignment WAS a real wall: the fix changed the mechanism (the 28
   per-layer rewards now DIFFER, verified by the [E9] print, vs the old identical
   broadcast; per-layer damage is now an optimizable per-env signal).
2. But fixing it does NOT reach correctness. The per-layer distributional damage
   is optimized loosely (and not even cleanly: s_e9 damage went 0.00084, 0.00078,
   0.00097 by third, drifting back up), while the probe stays at parity. So there
   is a SECOND wall: every distributional proxy we can compute online (joint
   output-KL in S4, per-layer hidden-state divergence in E9) is DECOUPLED from
   task correctness under the truncation cliff (evicting reasoning tokens makes
   the model ramble past EOS, `evict_generated_frac` 0.5-0.99, `trunc` ~0.45).

**Consolidated conclusion of the whole series:** online PPO for KV eviction plateaus
at kv_norm because (a) the reward could not attribute credit per layer (fixed in
E9), and underneath (b) no online-computable proxy for "this eviction preserves
the answer" is tight enough, and the terminal correctness signal is too sparse to
learn from directly (the truncation cliff). This is exactly why the 2026 literature
(KVP, ForesightKV) uses OFFLINE future-attention/loss oracle labels, not online RL.

**Next step (Phase B, the principled attack on wall #2):** stop trying to make a
distributional proxy stand in for correctness. Options in GRAN_PLAN §4c, now
re-prioritized: (B1) block eviction (evict K contiguous tokens at once) to shrink
the horizon and let the sparse correctness reward carry; (B2) SnapKV-style
one-shot prompt-compression (turns the sequential-infinite problem into one
selection decision, the format where the literature wins); or per-layer CORRECTNESS
attribution (offline, ForesightKV recipe). Not launched; awaiting decision.

## 25. PASSKEY ARENA RESULT: CONFIRMED, the null was the DATASET (2026-07-05)

`scripts/eval_passkey.py`, n=96, budget=176, forced generation (count to 40) so
eviction pressure accumulates before the final answer must survive.

| arm | accuracy |
|---|---|
| full (no eviction) | 0.854 |
| **oracle_fut (future attention)** | **0.802** |
| random | 0.490 |
| **kv_norm** | **0.375** |
| attn_cur | 0.396 |

**PAIRED oracle_fut minus kv_norm = +0.427 +/- 0.059 (7.3 sigma; 45 wins / 4
losses / 47 ties out of 96).** Compare to GSM8K's oracle gap of +0.070 (FINDINGS
11) on the same model, same budget mechanics, same code path: a 6x larger,
statistically overwhelming margin.

**kv_norm is BELOW random here** (0.375 vs 0.490): norm-based eviction is
actively harmful when the thing worth keeping is a rare, content-distinguishable
needle rather than "whatever has been decoded most recently and forcefully".
This is the RULER-regime signature Apple's KVP paper reports, reproduced at our
scale with our own code: the null series was never a bug or an algorithm
failure, it was that GSM8K (short, dense, decisive content is GENERATED
reasoning) structurally lacks the kind of learnable eviction signal that RULER
(long, mostly-filler, needle-retrieval) has in abundance.

**Implication:** the passkey arena is where a LEARNED ranker (the actual KVP
recipe: offline, per-position utility regression, no online RL) should be built
and evaluated first, not GSM8K. This reframes the report's negative result from
"RL failed to learn eviction" to "we located, with paired evidence across two
regimes, exactly which regime supports learnable eviction, and it matches the
published literature's own regime choice". `scripts/rank_predictability.py` runs
the offline-ranker question (currently the GSM8K version; the natural next
target is the SAME ranker trained/evaluated on passkey traces, where the +0.43
ceiling gives it something real to capture).

## 24. Is it the DATASET? (analysis + the two decisive experiments, 2026-07-04)

Question from the user: is the null series GSM8K-specific or general? And can we
replicate what Apple reports? Investigation of Apple's actual setup
(github.com/apple/ml-learning-to-evict, the KVP paper arXiv 2602.10238):

| | Apple / KVP | ours |
|---|---|---|
| task | RULER: LONG-context retrieval (needle in mostly-filler prompts) | GSM8K: SHORT dense prompts, decisive content is GENERATED reasoning |
| training | OFFLINE, on pre-computed traces (Q/K/V extracted first) | online PPO in the decode loop |
| agents | per-HEAD rankers (112 agents for Qwen2.5-7B) | one layer-shared policy |
| reward | ranking of future utility across ALL budgets | terminal correctness (later: dense KL / per-layer divergence) |
| compute | 8 GPUs DDP per agent | 1 L4 per run |

**Why GSM8K is plausibly the worst case for learned eviction:**
1. Nothing in a GSM8K prompt is redundant (short, information-dense), and the
   tokens that matter most are the model's OWN reasoning, generated after the
   eviction decisions. Future utility of a reasoning token is unpredictable
   from its K/V (E5 measured this: 3% match). In RULER, most prompt tokens are
   provably useless filler and the needle is content-distinguishable: future
   utility IS predictable there.
2. Our oracle numbers agree: even with perfect hindsight the gap over kv_norm
   on GSM8K is only +7pp (and attn_cur <= kv_norm). Thin signal ceiling.
3. The truncation cliff (evict reasoning -> rambling -> truncate -> wrong) is a
   GSM8K-chain-of-thought phenomenon, not a retrieval phenomenon.

**The two experiments that settle it (launched, one per VM):**
- `scripts/eval_passkey.py` (kv-chat-v1): a RULER-style passkey arena at our
  scale (filler + buried secret code, T~230-270 > budget=176, eviction pressure
  on PROMPT tokens: Alex's prefill-eviction flavor). Paired arms full / random /
  kv_norm / attn_cur / oracle_fut. If the dataset hypothesis is right, this
  reproduces the Apple-regime signature: kv_norm ~ random, oracle >> kv_norm.
- `scripts/rank_predictability.py` (kvp-ab): KVP-lite, the Apple RECIPE offline:
  per-layer rankers trained on our 400 GSM8K traces (features = raw K/V + rich
  columns, label = future attention), held-out Spearman + evictable-set overlap
  vs the kv_norm ranking. If the learned ranker cannot beat kvz's own rank
  correlation on GSM8K, the dataset carries no learnable eviction signal beyond
  norms, and the null series is dataset-driven (not a bug, not the algorithm).

Component-by-component check (Alex's suggestion): pytest suite re-run on the VM
alongside; the E9 [E9]-print already verified per-layer rewards differ, the E5
test verified argmin(kvz)==kv_norm choice, test_train_eval_obs_identical verified
train/eval observation parity. No open bug candidates.

## 26. Building a dataset with signal, and re-running the ladder there (2026-07-05)

If the null is dataset-driven (25), the constructive move is a dataset that HAS
learnable eviction signal, then re-run the key experiments to confirm learned
eviction beats kv_norm there. Dataset checklist derived from the RULER-vs-GSM8K
contrast:
  1. real compression ratio >= 3-4x (context >> budget)
  2. large fraction of provably-useless tokens (distractors / filler)
  3. utility distinguishable FROM CONTENT (retrieval/QA, not dense reasoning)
  4. SHORT generation (answer, not long CoT), avoids the truncation cliff
  5. verifiable answer (exact/substring match) for clean reward + eval

Datasets that qualify: synthetic passkey (done, 25), and the REAL upgrade
HotpotQA-distractor (2 gold paragraphs among 8 distractors, T~900-1400, 4-5x
compression). scripts/eval_prefill_compress.py implements Alex's prefill-only
compression (SnapKV family): one-shot top-budget keep per layer after prefill,
then decode with no further eviction. Real ratios, much faster than the online
1-per-step loop.

Experiments queued/running (2 VMs):
  - GSM8K offline ranker (rank_predictability.py, kvp-ab): CONTROL. Is future
    utility predictable from K/V on GSM8K? Expected weak.
  - passkey learned ranker (passkey_ranker.py, kvp-ab, chained): the KVP recipe
    end-to-end (train per-layer future-attn rankers offline, evaluate AS an
    eviction policy vs kv_norm). v1 was TOO EASY (budget 176 + short answer =
    only ~15 evictions, needle always survived, everything=1.0): LESSON =
    eviction pressure scales with GENERATION length (1 evict/step), not with
    prompt excess. v2 uses the forced count-to-40 generation (~200 evictions).
  - HotpotQA prefill-compression arena (eval_prefill_compress.py, kv-chat-v1):
    real-dataset oracle gap at 4-5x compression. Smoke (n=3) ran clean.

Component check (Alex's request): pytest 272/273 pass on the VM; the 1 failure is
a config assertion (test_train_config_gamma_is_one, a stale expected value in a
new e6-e9 config), NOT a pipeline bug. The eviction/feature/obs paths are clean.

VM plan for the FULL ladder in 12-15h (answer to the user's question): the core
(oracle gaps + offline rankers on passkey/hotpot) fits on 2 L4s serialized. To
add the causal capstone, an RL re-run in a signal-bearing arena (does PPO, which
plateaued on GSM8K, now beat kv_norm?), plus per-head vs per-layer ablation and a
2nd seed for error bars, comfortably in 12-15h needs 3-4 on-demand L4s in
capacity-healthy regions (the 2 current spot/on-demand VMs hit repeated Asia
stockouts + preemptions tonight).

## 23. Current state / open questions / next

- **ALL THREE SCALED ARMS DONE:** s_rich −0.044 (§7), s_warm −0.019 (§8),
  s_attn −0.111 (§9). Answer to the headline question: **no, nothing beats
  kv_norm under online PPO from terminal correctness.** Both VMs stopped.
- **Wide eval DONE (§10):** parity confirmed for per-token arms, attention worse,
  and the regime-headroom insight (full−random ≈ 4pp). Both VMs STOPPED.
- **Nothing left running.** No new trainings pass the decision-tree bar: E5
  (Golden-BC) in THIS regime could gain ≤3pp by construction, the env redesign
  (prompt compression / harsher budgets) has to come first. Morning discussion.
- **Literature check (2026):** KVP (arXiv 2602.10238) and ForesightKV (2602.03203)
  DO beat H2O/SnapKV with learned eviction, but via OFFLINE supervision from
  future-attention oracle labels (+ attention features), not online PPO from the
  terminal reward. Our parity result replicates the failure mode that motivated
  their design. Full synthesis + decision tree: **GRAN_PLAN.md**.
- **Honest headline for the report (final):** "Rich, scale-aware features bring an
  RL KV-eviction policy from random-level to parity with the strongest per-token
  heuristic (kv_norm); reward shaping (S4), warm-starting AT the heuristic, and a
  cross-token attention policy all fail to exceed it. The wide eval explains why:
  in this env's regime (budget ≥ prompt length) the total headroom between random
  eviction and the full cache is ~4pp, the terminal reward is nearly flat, so
  online PPO has no signal to beat a good heuristic, and even an oracle is capped
  at ~3pp. Learned-eviction wins in the literature (KVP, ForesightKV 2026) come
  from offline future-attention supervision at aggressive budgets, both are
  future work here (env redesign first)."
