# Phase 2 — Findings log (chronological, honest)

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
  (`kv_norm` 0.625 vs 0.750; `random` 0.562 vs 0.750) — impossible if the eval were
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

## 4. E0 capacity screen — RESULT (16 examples, overfit, probe_on_train)

Question: on data it was trained on, can each variant beat kv_norm? (capacity ceiling)

Setup reality on these 16: `full`=0.81 (13/16); `kv_norm`=`random`=0.06 (1/16) —
eviction is catastrophic for the heuristics here, so there's lots of headroom.

| variant | best learned | vs kv_norm | reading |
|---|---|---|---|
| baseline (no rich) | 0.06 | **+0.00 (ties)** | can't beat kv_norm — as expected (norm-blind) |
| **rich features** | 0.25 | **+0.19** | beats kv_norm 4× at peak, but SPIKY (0↔0.25) |
| **attention** | 0.25 | **+0.19** | same, spiky, trending up |
| rich + S4 | 0.125 | +0.06 | **< rich → S4 does NOT help** a capable policy |
| rich + warm-start | — | FAILED | BC bug (see §5) → inconclusive |

**Verdict:** representation (D1/D2) is the bottleneck DIRECTIONALLY — rich/attention
can beat kv_norm, baseline can't. BUT the signal is noisy/spiky and the absolute
recovery is modest. **S4 confirmed unhelpful even on a capable policy.**

## 5. Warm-start BC was broken — diagnosed and fixed (two bugs)

- Symptom: `match_kv_norm` plateaued at ~0.29; the cloned policy performed at 0.
- **Bug A (representability):** rich features gave `kz_mean` and `vz_mean`
  standardized SEPARATELY (σ_K≠σ_V), so the policy **cannot** reconstruct
  `argmin(‖K‖+‖V‖)` that kv_norm uses. → **Fix: added `kvz` feature** =
  standardized(‖K‖+‖V‖ mean-over-heads) = kv_norm's EXACT signal. Now the policy can
  represent kv_norm as `logit = −kvz`, and re-weight K vs V to try to beat it. Rich
  extra features: 2H+4 → **2H+5 = 9**. (test proves argmin(kvz)==kv_norm's choice.)
- **Bug B (optimization):** BC used cross-entropy to the argmin over ~220 slots with
  56-sample batches → high variance, bounced. → **Fix: dense MSE** — regress raw
  actor logits to `−kvz×3` on every valid slot (all slots supervised → stable;
  argmax=argmin(kvz)=kv_norm).
- Also fixed: probe CSV header (preempted runs left headerless CSVs → parsers saw
  "NO DATA"); run_screen wipes run_dir on fresh start (no concatenated CSVs).

## 6. Infra bug — checkpoints were never saved (spot resume was a no-op)

- `CheckpointCallback.save_freq` counts **on_step calls = wall-steps**, not timesteps.
  The batched env does n_parallel×n_layers = **56 timesteps per wall-step**, so the
  default `save_freq=50000` → first save at 2.8M timesteps > 2M total → **checkpoints
  dir stayed EMPTY** → every spot preemption restarted from 0 (s_rich lost ~660k
  twice). → **Fix: `checkpoint_freq=900`** (≈50k timesteps). Confirmed: 38 checkpoints
  now saved; future preemptions resume.

## 7. Scaled run s_rich (rich + kvz, 2M, full 1000-ex train, held-out 32 probe) — RESULT

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
  norm-blind level) UP TO kv_norm — real progress (representation closed the gap) —
  but it does not reliably EXCEED the heuristic. With kvz as a feature the easiest
  thing to learn is "be kv_norm"; beating it needs something more (re-weight K/V,
  position, cross-token) that per-token PPO isn't finding.

## 8. Scaled run s_warm (BC→kv_norm + RL, 2M) — RESULT: PARITY, exploration was NOT the wall

Completed 2026-07-02 ~07:25 UTC (survived two spot preemptions via checkpoint
resume; exposed + fixed a resume-budget bug, commit 8decafd). 10 probes:

```
ts:   162k  329k  525k  701k  932k  1086k 1283k 1416k 1624k 1959k
gap: +.031 +.031 -.031 -.062 -.094 +.000 -.031 +.031 -.031 -.062
```
- **The BC took:** first probe (162k) has learned=0.656 ≥ kv_norm=0.625 — the
  policy STARTS at/above the heuristic (screen's BC failure is fixed).
- **RL does not push past it:** mean −0.022, avg-last-5 −0.019, max +0.031.
  If anything RL *degrades* the clone mid-run (932k: −0.094) then recovers to
  parity. Same online-PPO ceiling as s_rich (−0.050 last-5), slightly closer.
- **Verdict (D4):** exploration/cold-start was NOT the binding constraint.
  Starting AT kv_norm + 2M of PPO does not find anything better than kv_norm.
  Per-token policies = parity ceiling, warm or cold. Matches the 2026 literature
  (KVP/ForesightKV): online RL from terminal correctness can't beat the
  heuristic; their wins come from offline future-attention oracle supervision.

## 9. Scaled run s_attn (rich + cross-token attention, 2M) — RESULT: BELOW kv_norm, D3 closed negative

Completed 2026-07-02 09:50 UTC on `kv-chat-v1` (Alex's project, on-demand L4, no
preemptions). 13 probes, within-arm paired gap:

```
ts:   149k  327k  474k  621k  762k  914k 1087k 1232k 1383k 1512k 1666k 1796k 1944k
gap: -.062 -.156 -.094 -.094 -.062 -.094 -.125 -.125 -.094 -.062 -.156 -.188 -.125
```
- **Never at or above kv_norm** (best probe −0.062). Mean **−0.111**, last-5 −0.125.
- **Environment-shift caveat (again!):** this VM's baselines are kv_norm=random=
  full=**0.750** vs kvp-ab's 0.625/0.562/0.750 — same seed, same 32 examples, but a
  different torch/transformers stack changes generations. Two consequences:
  (a) cross-arm comparisons of RAW numbers between s_attn and s_rich/s_warm are
  NOT valid — only within-arm paired gaps are; (b) on this stack eviction barely
  hurts the heuristics (random ties full → ceiling effect, little headroom), yet
  the learned attention policy still lands BELOW random — it evicts actively worse.
- **Verdict (D3):** the cross-token architecture does not rescue online PPO either;
  at scale it is the worst of the three arms on its own probe. The screen's +0.19
  peak did not transfer (overfit-16 capacity ≠ generalization at 1000-train).
- All three deficiency-fix branches (D1/D2 rich, D4 warm, D3 attention) now
  closed: **online PPO from terminal correctness ⇒ parity with kv_norm at best.**
  The decisive cross-arm number comes from the wide eval (§10): n=128 fresh
  examples, all three checkpoints evaluated on ONE VM (one software stack).

## 10. WIDE EVAL (n=128 fresh, one VM/stack) — the decisive table + a regime insight

Completed 2026-07-02 ~13:30 UTC (`scripts/wide_eval.py` on kv-chat-v1; 128 GSM8K
examples from the same seeded shuffle, slice [1032:1160] → disjoint from all
training/probe data; identical anchors for every row; VM auto-stopped after).

| arm | learned | paired vs kv_norm | trunc |
|---|---|---|---|
| s_rich_final | 0.695 | **−0.016** (≈2 ex.) | 0.03 |
| s_warm_final | 0.688 | **−0.023** (≈3 ex.) | 0.21 |
| s_attn_final | 0.648 | **−0.063** | 0.01 |
| s_attn_best  | 0.633 | **−0.078** | 0.02 |
| *kv_norm* | *0.711* | — | |
| *random*  | *0.703* | *−0.008 vs kv_norm* | |
| *full*    | *0.742* | *ceiling* | |

- **Per-token arms (rich, warm) = parity within noise** (1 example = 0.0078; they
  sit 2-3 examples below kv_norm). **Attention is genuinely worse** (8-10 ex.
  below, also below random) — the D3 negative replicates out-of-probe.
- **THE REGIME INSIGHT (new, load-bearing):** `full − random = 0.039` — at
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

## 11. ORACLE-GAP experiment — the learnable margin EXISTS (+7pp) and is FUTURE information

Completed 2026-07-02 18:27 UTC (`scripts/oracle_eval.py`, 128 wide examples, all
arms paired per example under ONE eager-attention model instance; 0 errors).

| arm | acc | paired vs kv_norm |
|---|---|---|
| **oracle_fut** (golden eviction: min MAX-FUTURE attention from full trace) | **0.516** | **+0.070 ± 0.025 (2.8σ; 10W-1L, McNemar p≈0.01)** |
| full (no eviction) | 0.492 | +0.047 |
| attn_cur (H2O-family: min CURRENT accumulated attention) | 0.461 | +0.016 (0.5σ, 8W-6L — noise) |
| kv_norm | 0.445 | — |

Three findings:
1. **The learnable margin exists: ≈ +7pp — and it is specifically FUTURE
   information.** Present-attention (attn_cur) does NOT beat kv_norm. This
   coherently explains ALL of Phase 2: no online policy (learned or heuristic)
   can see the future → they all tie at the kv_norm level. kv_norm ≈ the online
   ceiling; the future-supervised ceiling is +7pp above it.
2. **The oracle beats even full-cache** (0.516 > 0.492): good eviction is not
   just harmless — dropping the right tokens *improves* reasoning (beneficial
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

## 12. E5 Golden-BC — RESULT: future attention is UNPREDICTABLE from present features

Ran 2026-07-03 on kvp-ab (traces: 400 train examples, ~35 min; BC 3000 steps;
PPO 2M). Three findings:

1. **The BC never converged: `match_oracle` stuck at 0.018–0.036** for all 3000
   steps (loss 9.0→8.4, flat; vs match≈1.0 when cloning kv_norm with the same
   dense-MSE recipe). A per-token policy seeing raw K‖V + rich features CANNOT
   predict which slot the future-attention oracle evicts — barely 8× above
   chance (1/220). **This is the cleanest statement of the whole Phase-2 story:
   the +7pp oracle margin (§11) is made of information that does not exist in
   the present observation.** (ForesightKV gets around this with attention-
   history features — inputs our sdpa env doesn't expose — and even then needs
   pairwise-ranking, per-head scorers.)
2. **s_golden ≈ rich policy with a failed init:** train correctness DECLINED
   0.60→0.54 with truncation rising 0.10→0.40 (reward drift). On its own probe
   it posts mean **+0.074 over kv_norm** — but see the caveat below.
3. **Probe-slice heterogeneity (methodological trap, 2nd sighting):** e5's probe
   ([400:432], its n_examples=400 config) has kv_norm=0.06, random=0.09,
   full=0.78 — eviction is CATASTROPHIC there, unlike [1000:1032] (kv_norm
   0.625) or the wide slice (0.71), same stack. GSM8K slices differ wildly in
   eviction sensitivity. s_golden's +0.074 is "beats a floor-level kv_norm on a
   hostile slice" (like the E0 screen), NOT oracle distillation. The decisive
   number is the shared wide slice [1032:1160] → §13.

## 13. Current state / open questions / next

- **ALL THREE SCALED ARMS DONE:** s_rich −0.044 (§7), s_warm −0.019 (§8),
  s_attn −0.111 (§9). Answer to the headline question: **no, nothing beats
  kv_norm under online PPO from terminal correctness.** Both VMs stopped.
- **Wide eval DONE (§10):** parity confirmed for per-token arms, attention worse,
  and the regime-headroom insight (full−random ≈ 4pp). Both VMs STOPPED.
- **Nothing left running.** No new trainings pass the decision-tree bar: E5
  (Golden-BC) in THIS regime could gain ≤3pp by construction — the env redesign
  (prompt compression / harsher budgets) has to come first. Morning discussion.
- **Literature check (2026):** KVP (arXiv 2602.10238) and ForesightKV (2602.03203)
  DO beat H2O/SnapKV with learned eviction — but via OFFLINE supervision from
  future-attention oracle labels (+ attention features), not online PPO from the
  terminal reward. Our parity result replicates the failure mode that motivated
  their design. Full synthesis + decision tree: **MASTER_PLAN.md**.
- **Honest headline for the report (final):** "Rich, scale-aware features bring an
  RL KV-eviction policy from random-level to parity with the strongest per-token
  heuristic (kv_norm); reward shaping (S4), warm-starting AT the heuristic, and a
  cross-token attention policy all fail to exceed it. The wide eval explains why:
  in this env's regime (budget ≥ prompt length) the total headroom between random
  eviction and the full cache is ~4pp — the terminal reward is nearly flat, so
  online PPO has no signal to beat a good heuristic, and even an oracle is capped
  at ~3pp. Learned-eviction wins in the literature (KVP, ForesightKV 2026) come
  from offline future-attention supervision at aggressive budgets — both are
  future work here (env redesign first)."
