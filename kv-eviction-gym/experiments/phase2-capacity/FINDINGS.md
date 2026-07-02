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

## 8. Current state / open questions / next

- **PROGRESS (2026-07-02 03:07 UTC):** `s_rich` (rich+kvz, 2M) **completed** at
  2,000,488 steps. `s_warm` (warm-start: BC-clone kv_norm then RL) **running**,
  ~695k/2M steps (started 01:18 UTC, ~370k steps/h → ETA ~3.5h). Watcher relaunched
  after it had died silently ~00:44 UTC; on `s_warm` finish it auto-downloads both
  arms' curves, STOPS the VM, and runs compare.py (rich vs warm_start vs screen base).
- **RUNNING:** `s_warm` (warm-start: BC-clone kv_norm then RL) — does starting AT
  kv_norm + RL push past it? Then done.
- **Open:** does ANY method beat kv_norm? Per-token (rich, warm) look like parity.
- **LAUNCHED (2026-07-02 03:41 UTC):** `s_attn` (E4: rich+kvz+PerTokenAttention,
  2M, same seed/anchors) on `kv-chat-v1` (Alex's project, on-demand L4) — running
  in PARALLEL with s_warm. Cross-token redundancy reasoning is the only candidate
  with a ceiling above a per-token heuristic. Watcher auto-downloads + stops VM.
- **Literature check (2026):** KVP (arXiv 2602.10238) and ForesightKV (2602.03203)
  DO beat H2O/SnapKV with learned eviction — but via OFFLINE supervision from
  future-attention oracle labels (+ attention features), not online PPO from the
  terminal reward. Our parity result replicates the failure mode that motivated
  their design. Full synthesis + decision tree: **GRAN_PLAN.md**.
- **Honest headline for the report so far:** "rich, scale-aware features bring an
  RL KV-eviction policy from random-level up to parity with the strongest per-token
  heuristic (kv_norm); reward shaping (S4) does not help; beating the heuristic
  appears to require cross-token reasoning (in progress)."
