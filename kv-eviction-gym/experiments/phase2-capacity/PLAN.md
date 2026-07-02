# Phase 2 — Closing the gap to `kv_norm`: why the learned policy doesn't beat a norm heuristic

**Status:** DRAFT for review (written 2026-06-30, clean A/B still running). Decision tree at the
end. Nothing in Phase 2 is launched yet — code is being staged so approval → immediate launch.

---

> **NOTE (2026-07-01):** this file is the up-front DESIGN + root-cause analysis.
> For the actual RESULTS and the chronological "what we tried → what it gave → why
> we changed course" narrative (screen results, the kvz fix, the checkpoint bug, the
> s_rich parity finding), see **[FINDINGS.md](FINDINGS.md)** — it supersedes the
> "Results so far" section below, which predates the experiments.

## 0. TL;DR (read this first)

- The overnight A/B **looked** like S4 won (+0.183 retention). **It's invalid** — the two arms ran
  on different VMs with different code, so they evaluated on **different probe sets**. Proof: the
  policy-independent baselines differ between arms (`kv_norm` = 0.625 vs 0.750), which is impossible
  under a comparable eval.
- The **robust** finding (valid *within* each arm, where the baseline IS comparable): the learned
  policy **never beats `kv_norm`** — in fact it sits at ~random. Its retention curve is **flat from
  0.15M steps**: it isn't learning anything a norm heuristic doesn't already do.
- We are re-running the A/B **clean** (one VM, one codebase, one seed → identical anchors). Predicted
  delta ≈ 0. That run confirms the null *rigorously* and is the first line of the report.
- **Root cause (found by reading the code, not guessing):** the policy is *structurally* unable to
  represent `kv_norm`. It `LayerNorm`s K and V on input, which **erases the norm** — the exact signal
  `kv_norm` uses. It also has no position feature and no cross-token interaction.
- **Phase 2** is a hypothesis-driven ablation that fixes each deficiency and measures which one
  unlocks *beating* `kv_norm`, and whether **S4 helps once the policy can actually act**. This turns
  an "informative null" into a real contribution.

---

## 1. Background — how we got here (for the report's narrative)

The environment is online KV-cache eviction on GSM8K with Qwen2.5-1.5B-Instruct: at each decode step,
once `cache_size > budget`, one token per layer is evicted before the next token is generated. The RL
policy chooses which slot to evict. Headline metric = **retention** = `correct_learned / correct_full`
(accuracy kept vs full cache), measured by a held-out 32-example probe alongside fixed baselines
(`random`, `kv_norm`, `streaming`, `full`).

Two things had to be fixed before any of this was measurable:

1. **Chat-template root cause (fixed, main `7c6d2d8`).** The prompt was fed raw (no chat template) →
   the Instruct model never emitted EOS → 100% truncation, 0% correctness on *every* run. Nothing
   could learn. After applying the chat template the task became learnable (correctness hits 1.0 on
   rollouts, truncation → 0). Every result before this fix is invalid.

2. **`protect_prompt` removed on both arms (user directive "sacá el protect prompt").** Budget is
   fixed at 256 (no curriculum) because the batched env pre-allocates a shared KV cache of `budget+1`
   slots; any prompt with `T > budget` is skipped (`batched_env.py:550`). Since `max(T)=232 < 256`,
   nothing is skipped. This is architectural and independent of `protect_prompt`.

**S4 (our novel contribution).** Training uses Monte-Carlo returns (`gamma=1, gae_lambda=1`): the
terminal reward (correctness ± length/trunc) is copied identically to every step and every one of the
28 layers. Credit assignment is therefore very weak — a catastrophic eviction at step 3 and a harmless
one at step 200 get the same credit. **S4** replaces this with a dense, per-step, causal signal: at
each decode step it compares the next-token distribution under the evicted cache vs a shadow
*full* cache and penalizes the divergence:

```
r_step[b] = − kl_weight · clip( KL( p_full(·) ‖ p_evict(·) ), 0, kl_clip )
```

`kl_weight=0.05`, `kl_clip=5.0`, `kl_mode=exact` (a real full-cache shadow forward per episode per
step, ~2× decode). It is label-free, works with `sdpa`, and directly measures the causal damage of
each eviction. The hypothesis (H1): S4 improves retention over pure correctness by fixing credit
assignment.

---

## 2. Results so far

### 2.1 Overnight A/B (5M steps/arm) — **INVALID (evaluation confound)**

| metric (avg last-5 probes) | CONTROL `chat_full_v2` | TREATMENT `chat_full_s4_kl` |
|---|---|---|
| retention | 0.633 | 0.817 |
| correct_learned | 0.588 | 0.713 |
| correct_random | **0.562** | **0.750** |
| correct_kv_norm | **0.625** | **0.750** |
| correct_full | 0.750 | 0.750 |
| truncation_rate | 0.163 | 0.025 |

Apparent verdict: retention(treat) − retention(ctrl) = **+0.183** → "S4 improves retention".

**Why it's invalid.** `correct_random` and `correct_kv_norm` are computed by *fixed heuristics* that
do not depend on the learned policy. At the **same budget on the same probe set** they MUST be
identical between arms. They are not (0.562 vs 0.750; 0.625 vs 0.750, each with std=0.000 across all
probes within a run). The only explanation: the two arms evaluated on **different probe sets**. Cause:
control ran on Alex's VM (older SCP'd snapshot) and treatment on ours (`~/repo`, newer code including
commit `c7fe3bf` that changed probe-anchor selection) → each picked a different 32-example held-out
set. `correct_full` matching at 0.750 is a coincidence (full cache solves the same fraction of two
different easy-ish sets). **The +0.183 is a probe-set artifact, not an S4 effect.**

**Second red flag (independent of the confound):** both retention curves are **flat from 0.15M
steps** (treatment oscillates ~0.79, control ~0.62 for the entire 5M run). The gap already exists when
the policy is essentially untrained. A reward-shaping method that fixed credit assignment would show a
*trajectory* (separation growing with training); a flat gap from step 0 is an eval-difficulty
difference, not learning.

### 2.2 The only valid comparison in that run: *within-arm* (baseline IS comparable there)

| arm | learned | random | kv_norm | reading |
|---|---|---|---|---|
| control | 0.572 | 0.562 | 0.625 | learned ≈ random, **< kv_norm** |
| treatment | 0.685 | 0.750 | 0.750 | learned **< random = kv_norm** |

**In neither arm does the learned policy beat `kv_norm`. It performs at ~random.** This is the real,
robust result and it drives everything below.

**The decisive number — the paired gap `learned − kv_norm` (identical baseline per arm):**

| arm | learned − kv_norm |
|---|---|
| control | **−0.038** |
| treatment (S4) | **−0.037** |

On the *honest* metric — how much the policy beats its OWN kv_norm baseline — the two arms are
**identical (−0.037 vs −0.038)** and both are **below zero**. The apparent **+0.183 "S4 advantage"
collapses to ≈ 0.001.** The entire headline gap was the difference in the kv_norm/random baseline
between the two (different) probe sets, not anything the policy learned. This is the single cleanest
statement of the result and essentially pre-confirms the clean A/B's prediction. (Computed by
`experiments/phase2-capacity/compare.py` on the two invalid probe curves.)

### 2.3 Clean A/B (2M steps/arm) — **RUNNING now, result pending**

One VM (`kvp-ab`, spot L4, asia-southeast1-a — the only capacity found after a full US/EU/Asia
on-demand+spot hunt), one codebase (current `main` SCP'd over the snapshot), one seed=0 → **identical
anchors and identical baselines** → a *paired* comparison. Configs differ ONLY in the `kl_*` block
(verified by diff). Horizon reduced to 2M because the curves plateau from 0.15M (2M is well past
steady state; cheaper; fits a spot window). Restart-safe driver + preemption-aware watcher.
**Prediction: clean delta ≈ 0.** Confirming this null rigorously is the first deliverable and the
methodological lesson of the report ("apples-to-apples evaluation matters").

---

## 3. Root-cause analysis — *why* learned ≤ kv_norm (code-level, not speculation)

`kv_norm` (`eval_core.py:331-339`) evicts the valid slot with the lowest `‖K‖ + ‖V‖` (mean over
heads). The learned policy sits at random. Four concrete, separable deficiencies explain it:

| # | deficiency | code evidence | consequence |
|---|---|---|---|
| **D1** | **Norm blindness.** The policy `LayerNorm`s K and V as its first op, erasing magnitude. | `policy.py:54-55` (`self.k_norm/v_norm`), `:84-86` applied before the MLP. | Cannot represent `‖K‖+‖V‖` → cannot even *match* `kv_norm`. |
| **D2** | **Position blindness.** No explicit position feature; relies on RoPE-in-K. | `features.py:15-16` ("Position is already encoded in K via RoPE"). | RoPE is a norm-invariant rotation; a per-token MLP can't decode absolute/relative position, esp. post-LayerNorm. Can't learn recency/positional structure. |
| **D3** | **No cross-token reasoning.** Per-token MLP is independent; only a fixed `Linear(max_len→max_len)` head mixes positions, by absolute slot index, not content. | `policy.py:34-91` (per-token), SB3 default actor head. | Can't represent "token A is redundant *given* B is kept" (set-level reasoning). And absolute-index mixing is fragile as slots shift under eviction/append. |
| **D4** | **Cold-start optimization.** RL explores from random init; `kv_norm` is a strong basin the policy never finds. | learned ≈ random after 5M steps; flat curves. | Even with the right features, exploration may not reach a good policy without a warm start. |

D1 alone is sufficient to explain "≈ random": the single most predictive feature for this task is
destroyed at the input. This is the highest-value, cheapest fix.

---

## 4. Phase 2 — the experiments

### E0 — capacity screen (THE centerpiece, run FIRST, cheap)

Before spending GPU-days scaling anything, run every candidate on the SAME tiny
fixed set (16 examples) and **probe on those same examples** (`probe_on_train`) —
a pure overfit/capacity test: *can this variant beat kv_norm on data it was trained
on?* This is the highest-information-per-GPU-hour experiment we have (~1h/variant),
and it disambiguates which of the four deficiencies is the real bottleneck in ONE
shot. Five variants, each changing ONE factor:

| variant | what it adds | isolates |
|---|---|---|
| `e0_baseline` | nothing (K‖V, MLP) | reference — should NOT beat kv_norm |
| `e0_rich` | rich features, MLP | representation (D1+D2) |
| `e0_rich_s4` | rich + S4 KL shaping | reward on a capable policy |
| `e0_rich_warm` | rich + BC warm-start from kv_norm | exploration (D4) |
| `e0_attn` | rich + cross-token attention policy | per-token architecture ceiling (D3) |

**Reading (the disambiguation):**
- `e0_rich` beats kv_norm, `e0_baseline` doesn't → **representation** was the wall. Scale rich.
- `e0_rich` ties but `e0_rich_warm` beats → **exploration** was the wall. Scale warm-start.
- only `e0_attn` beats → **per-token architecture** was the ceiling. Scale attention.
- nothing beats but rich variants tie → kv_norm ≈ optimal per-token; the *objective* is
  the ceiling → rethink the reward before scaling.

**Note on warm-start:** it is NOT an alternative to rich features — you cannot
behavior-clone kv_norm if the policy can't see the norm (D1). So `e0_rich_warm` =
rich features + a kv_norm initialization; the screen tests whether that init (fixing
D4) buys anything beyond features alone. `screen_verdict.py` prints this verdict
automatically; `warm_start.py` prints a `match_kv_norm` accuracy that must approach
~1.0 for the clone to have taken.

Only the screen winner(s) get scaled to the full 2M-step runs below.

---

### Scaled runs (only the screen winner)

Goal: determine **which deficiency is binding**, get the learned policy to **beat `kv_norm`**, and
re-test **whether S4 helps once the policy can act**. All runs: same data (n=1000, seed=0), same 32
probe anchors, same budget=256, 2M steps, same VM/code family → directly comparable to §2.3.

### Experiment E1 — Rich features (fixes D1 + D2) ⭐ highest priority, cheapest

Add explicit, scale-preserving features per token, concatenated to the existing (LayerNorm'd) K‖V:
- `‖K‖` per KV-head and mean, `‖V‖` per head and mean (the exact `kv_norm` signal),
- normalized position: `slot / cache_size`, `slot / max_len`, and distance-from-end `(cache_size-slot)/max_len` (recency),
- optionally `log(cache_size)` as a global scalar broadcast per token.

Implementation is a flagged addition (`rich_features: true`) so it's opt-in and byte-comparable to the
baseline. Only `features.py` (append columns) + `policy.py` (feed the raw norm/pos scalars *around*
the LayerNorm, not through it) change.

**Hypothesis H-E1:** a norm+position-aware per-token policy **matches or beats `kv_norm`**. If true,
the bottleneck was representation, not the reward and not the algorithm — a clean, likely-positive
headline finding.

### Experiment E2 — S4 on the capable policy (crossed factor)

Re-run E1 **+ S4 KL shaping**. This is the resurrection of the S4 question: does the causal per-step
reward add value *once the policy has the features to act on it*? The overnight/clean A/B tested S4 on
a norm-blind policy (which can't act on any signal); E2 tests it on a policy that can.

**Hypothesis H-E2:** if E2 > E1, S4 helps a capable policy (rescues the contribution). If E2 ≈ E1,
S4 is redundant with good features (also a clean finding: "good features > clever reward").

**Round 1 = {E1, E2} in parallel on the 2 VMs.** ~2M each, ~4–7h. One decision point after.

### Experiment E3 — Warm-start from `kv_norm` (fixes D4)

Behavior-clone the policy to imitate `kv_norm`'s slot choice (supervised, cheap: generate `kv_norm`
decisions on rollouts, cross-entropy to the policy logits), then RL fine-tune (with and without S4).
Guarantees the policy starts *at* `kv_norm` and can only improve. Isolates optimization/exploration
from representation.

**Hypothesis H-E3:** warm-start + RL > `kv_norm` even if E1 didn't reach it → the bottleneck was
exploration. If E1 already beats `kv_norm`, E3 is a robustness/accelerator, not a necessity.

### Experiment E4 — Cross-token attention head (fixes D3) — only if E1/E3 plateau at `kv_norm`

Replace the per-token-independent extractor with a small self-attention encoder over the resident
slots (1–2 layers, 2–4 heads, dim 64), producing content-aware per-slot logits (permutation-equivariant,
no absolute-index head). This is the representational upgrade needed to *beat* (not just match)
`kv_norm` by reasoning about redundancy across the kept set. Most engineering; deferred until the
cheap fixes are exhausted, so we only pay for it if needed.

---

## 5. Decision tree (what I run, autonomously, based on results)

```
clean A/B (§2.3) finishes → confirms null (VM auto-stops).

E0 SCREEN: {baseline, rich, rich+S4, rich+warm, attn} on 16 examples, probe_on_train.
           Split across the 2 VMs. ~1h/variant. screen_verdict.py prints the winner.
   ├─ rich beats kv_norm, baseline doesn't  → REPRESENTATION (D1/D2). Scale rich (+S4 if it adds).
   ├─ rich ties, rich+warm beats            → EXPLORATION (D4). Scale warm-start.
   ├─ only attn beats                       → ARCHITECTURE ceiling (D3). Scale attention.
   └─ nothing beats, rich ties kv_norm      → OBJECTIVE is the ceiling → rethink reward, don't scale.

SCALE the winner to 2M (full 1000-example train, held-out probe) on 1–2 VMs.
   + 2–3 seeds for error bars ONCE an effect is confirmed (don't pay for CI on a null).
   + the losing hypotheses stay unscaled (compute saved).
```

At every branch: no VM is ever left running idle (stopped = fine), results are downloaded and the
comparison auto-run, and each result is documented in this folder.

---

## 6. Ops / VM plan

- **VMs:** up to 2 in parallel (user-authorized). Reality: on-demand L4 is broadly STOCKOUT; we
  currently hold **one spot L4** (`kvp-ab`). A 2nd VM requires another hunt (may only yield spot).
  Fallback if only 1 VM: run the two Round-1 arms sequentially (driver already supports chaining).
- **Spot resilience:** restart-safe driver (per-arm marker files) + preemption-aware watcher
  (auto restart + resume). Already built and in use for the clean A/B.
- **Comparability:** every Phase 2 arm reuses the §2.3 config skeleton (seed=0, same anchors, 2M) so
  its probe's `kv_norm`/`random` columns are identical to the clean A/B → all arms live on one axis.
- **Cost discipline:** 2M not 5M (plateau); short arms; stop VMs the instant a run ends; seeds only
  after an effect is confirmed (don't pay for error bars on a null).
- **Old stopped VMs** `ppo-kvp` (us-central1-b) and `kvp-s4` (us-west4-a): disk-only billing;
  `kvp-s4` holds the now-invalid treatment checkpoint. Delete on request (frees ~$16/mo of disk).

---

## 7. Metrics & documentation (every arm)

- Primary: `correct_learned` vs `correct_kv_norm` on identical anchors (paired), and retention.
- Learning dynamics: retention/learned vs steps (is there a trajectory now, vs the old flat line?).
- Collapse diagnostics: `evict_generated_frac`, `truncation_rate`.
- S4 arms: `kl_step_mean` vs steps (should decrease as the policy learns low-damage evictions).
- Each experiment writes `experiments/phase2-capacity/<exp>/` with probe/learning curves + a
  one-paragraph result note. A `compare.py` overlays all arms' retention on one plot.

---

## 7b. Code status — E1/E2 STAGED & TESTED (ready to launch on approval)

Implemented on branch `phase2-rich-features` (working tree, not committed — awaiting your review):

- **`src/kv_gym/features.py`** — `build_extra_columns()` (single source of truth for the 4 rich
  columns: within-observation z-scored ‖K‖/‖V‖ + pos + recency), `feature_dim(..., rich)`,
  `build_obs(..., rich)`.
- **`src/kv_gym/policy.py`** — `PerTokenMLP(n_extra=…)`: extra columns bypass the K/V LayerNorm
  (the whole point — preserve magnitude).
- **`src/kv_gym/batched_env.py`** — ctor `rich_features` flag; `_obs_episode` appends the same
  columns via the shared helper.
- **`scripts/train.py` / `probe.py` / `eval_core.py`** — thread `rich_features` through training,
  the policy's `n_extra`, and the eval probe → `build_obs`.
- **`configs/e1_rich.yaml`** (rich, correctness) and **`configs/e2_rich_s4.yaml`** (rich + S4);
  diffs confirmed minimal (E1 vs control = `rich_features` + 2M; E1 vs E2 = the `kl_*` block only).
- **`tests/test_rich_features.py`** — 6 pass / 1 skip (the skip = PerTokenMLP forward, needs sb3,
  runs on the VM). The load-bearing test `test_train_eval_obs_identical` proves training and eval
  build byte-identical observations (no train/eval confound), and `test_kz_recovers_kv_norm_ordering`
  proves the z-scored norm feature reproduces kv_norm's exact slot ranking → the policy CAN now
  represent (and beat) the heuristic. With `rich_features:false` everything is byte-identical to the
  old code, so the running clean A/B is unaffected.
- **`experiments/phase2-capacity/run_phase2.sh`** — restart-safe VM-side driver (1 or 2 VMs).
- **`experiments/phase2-capacity/compare.py`** — multi-arm compare on the PAIRED `learned − kv_norm`.

**Also staged for the screen (commit 2):**
- `src/kv_gym/policy.py` — `PerTokenAttention` (E4): cross-token self-attention extractor,
  permutation-equivariant, same I/O as PerTokenMLP (drop-in via `policy_arch: attention`).
- `src/kv_gym/warm_start.py` — `behavior_clone_kv_norm` (E3): supervised BC pre-phase, kv_norm
  target computed exactly from the obs's raw K‖V; prints `match_kv_norm` sanity accuracy.
- `scripts/train.py` — `policy_arch` (mlp|attention) selection, `warm_start_bc` pre-phase,
  `probe_on_train` (probe the training examples = overfit metric).
- Full E1 feature set now 2H+4 = 8 columns: per-head + mean z-scored ‖K‖/‖V‖ + recency +
  **original position** (D2). (is_prompt/is_numeric = `semantic_features` fast-follow, deferred:
  they need signature-touching plumbing and the screen will say if they're worth it.)
- `experiments/phase2-capacity/screen_configs/e0_*.yaml` (5 variants), `run_screen.sh`,
  `screen_verdict.py`.

**Launch playbook (executed autonomously):**
1. SCP current `src/scripts/configs/experiments` to the VM + `pytest tests/test_rich_features.py`
   there (unskips the PerTokenMLP/attention forward tests).
2. E0 screen: split the 5 variants across the 2 VMs, `bash run_screen.sh <variants...>`
   (~1h each). Preemption-aware watcher downloads `screen/*/probe_curve.csv`, stops VMs,
   runs `screen_verdict.py`.
3. Scale the winner to 2M (`e1_rich`/`e2_rich_s4` or the attn/warm equivalent) → `compare.py`.

## 8. Risks

- **GPU capacity** is the real blocker; spot preemption could stretch wall-clock. Mitigated by
  restart-safe infra; worst case runs go sequential on 1 VM.
- **E1 might overfit the probe** (32 examples is small). Mitigation: the paired `learned − kv_norm`
  on identical anchors cancels much of the probe noise; add seeds once an effect appears; optionally
  widen the probe to 64.
- **E4 (attention) is more code** and could regress if not permutation-equivariant; only attempt if
  the cheap fixes plateau.
