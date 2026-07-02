# MASTER PLAN — synthesis of everything + what we run with the remaining compute

Written 2026-07-02 (early morning, autonomous execution). Synthesises ALL the results
(FINDINGS.md), the original design (PLAN.md), and the 2026 literature on learned
KV-cache eviction. Defines the decision tree that runs tonight without
intervention, and the closing plan for the report.

---

## 1. Where we are — the full map of results

| experiment | what it isolates | result (paired learned−kv_norm) | verdict |
|---|---|---|---|
| A/B overnight 5M (S4 vs control) | reward shaping | ctrl −0.038 / S4 −0.037 (the +0.183 was a probe-set artifact) | **S4 null**; methodological lesson |
| E0 screen `baseline` | — | +0.00 (ties) | norm-blind policy ≈ random |
| E0 screen `rich` | D1/D2 representation | **+0.19 peak** (overfit, 16 ex.) | representation WAS the bottleneck |
| E0 screen `rich+S4` | reward on a capable policy | +0.06 < rich alone | **S4 does not help even with features** |
| E0 screen `attn` | D3 architecture | **+0.19 peak**, still rising at the cutoff | candidate with a higher ceiling |
| `s_rich` 2M at scale (1000 train, 32 held-out) | D1/D2 at scale | **mean −0.044, last5 −0.050** (31 probes) | **PARITY/barely-below — it does not beat it** |
| `s_warm` 2M (BC→kv_norm + RL) | D4 exploration | RUNNING (ETA ~06:40 UTC) | does starting AT kv_norm push further? |
| `s_attn` 2M (rich + cross-token attention) | D3 architecture | RUNNING on kv-chat-v1 (ETA ~09:30 UTC) | does cross-token reasoning beat the heuristic? |

**The pattern:** rich features close the gap from ~random → kv_norm (real progress on
representation), but online per-token PPO finds nothing BETTER than the heuristic.
With `kvz` as a feature, "being kv_norm" is the easiest thing to learn and that is
where it stays.

## 2. What the literature says (2026) — and how it positions our result

Three papers directly on our problem:

- **KVP — "Learning to Evict from Key-Value Cache"** (arXiv 2602.10238): formulates
  eviction as *learning-to-rank* of tokens by future utility. Lightweight per-head RL
  agents, **trained OFFLINE on precomputed generation traces**, with a holistic reward
  derived from each token's future utility across all budgets. Beats strong baselines
  on RULER/OASST2.
- **ForesightKV** (arXiv 2602.03203): two stages — (1) supervised with
  **"Golden Eviction"** labels: for each step, compute the real future attention over
  the full trace and mark as evictable the KV with the lowest maximum future attention;
  ranking loss. (2) RL (GRPO) with a dense reward = post-eviction loss spike on
  low-entropy tokens. MLP scorer over K, V **and attention features** (recent windows
  8/16/32 + accumulated history with decay). 92-99% of the performance with 50% of the
  cache; beats SnapKV/H2O/R-KV.
- **LKV** (arXiv 2605.06676): per-head budgets + end-to-end learned token selection;
  same thesis ("optimal compression must be learned, not heuristic").

**All three agree on three design decisions we do NOT have:**
1. **Supervision from future utility** (future attention / causal loss-spike), not a
   terminal correctness reward with weak credit assignment.
2. **Supervised pretraining with oracle labels** derived from full traces (our
   warm-start clones kv_norm — their oracle is BETTER than kv_norm).
3. **Attention features** (recent/accumulated scores, the H2O signal) in the
   observation. Our rich features are norm+position; we do not see attention
   (sdpa does not expose it — ForesightKV captures it while generating the offline trace).

**Honest positioning of our result:** our parity with kv_norm under online PPO from a
terminal reward is exactly the *failure mode* that pushed the literature towards
offline supervision by a future-attention oracle. It is not a failed result: it is an
independent replication of the "why" behind the KVP/ForesightKV design, with our own
causal diagnosis (D1-D4 + capacity screen).

## 3. Hypotheses still open (tonight closes them)

- **H-E3 (s_warm):** if the problem was exploration (D4), BC→kv_norm + RL should beat
  kv_norm. The literature predicts it will NOT be enough: the cloned oracle (kv_norm)
  is the wrong ceiling; RL from there still has the same weak credit assignment.
- **H-E4 (s_attn):** if the per-token ceiling IS kv_norm (nothing per-token can beat a
  near-optimal per-token heuristic), cross-token attention is the only way. The screen
  supports it (+0.19, still rising at the cutoff). The literature is ambiguous:
  ForesightKV uses a per-token MLP but with attention features — that is "cross-token
  through features" rather than "cross-token through architecture". s_attn tests the
  second route.

## 4. Decision tree (runs autonomously as each result lands)

```
s_warm finishes (~06:40 UTC; the watcher pulls the curves and SHUTS DOWN kvp-ab)
  ├─ mean(last5 paired) > +0.03  → BEAT: relaunch kvp-ab with e3_warm seed=1
  │                                 (confirmation; spot may STOCKOUT → best effort)
  └─ |paired| ≤ 0.03 or negative → PARITY (predicted): VM stays off, $0 extra.

s_attn finishes (~09:30 UTC; the watcher pulls the curves and SHUTS DOWN kv-chat-v1)
  ├─ BEAT (> +0.03 sustained)    → replicate seed=1 on kv-chat-v1 (restart);
  │                                 report headline = "cross-token beats the heuristic".
  └─ PARITY                      → NO more 2M runs on these branches. Close with:
                                    (a) a final WIDE eval (n=128 held-out) of the 3
                                        final checkpoints (s_rich/s_warm/s_attn) →
                                        the report's definitive number;
                                    (b) staging E5 (below) for tomorrow's discussion.

E5 (contingency, the "right next step" per the literature — NOT launched tonight
    without review, only left designed):
    Golden-BC: generate full-cache traces offline (eager attention to capture
    scores), compute Golden Eviction labels (min max-future-attention), BC the
    rich/attn policy to that oracle (better than kv_norm BY CONSTRUCTION on the
    traces), then a short RL on top. It is the ForesightKV recipe adapted to our gym.
    Estimated cost: ~1 day of implementation + ~6-8 GPU-h.
```

**Cost rule (in force):** no VM on without an active run; seeds only to confirm an
EFFECT (never for a null); 2M steps max per run.

## 5. Closing the report (independent of tonight's results)

The story is already publishable as a TP with what we have:

1. **Methodology:** the invalid A/B → the paired metric `learned − kv_norm` over
   identical anchors (the "apples-to-apples" lesson).
2. **Causal diagnosis:** D1-D4 by reading the code; the E0 screen as a cheap
   disambiguation instrument (baseline cannot / rich+attn can, overfit).
3. **Result at scale:** rich features take the policy from ~random to parity with
   kv_norm; S4 null twice (clean paired A/B + screen).
4. **s_warm / s_attn:** close D4 and D3 respectively (tonight).
5. **Positioning:** our parity replicates the failure mode that motivated the
   offline-oracle design of KVP/ForesightKV → E5 is the concrete future work.

Sources: [KVP](https://arxiv.org/abs/2602.10238) ·
[ForesightKV](https://arxiv.org/html/2602.03203v1) ·
[LKV](https://arxiv.org/html/2605.06676) ·
[KV-cache survey 2026](https://arxiv.org/html/2603.20397v1)

## 6. Autonomous execution log (tonight)

- 03:41 UTC — `s_attn` (e4_attn.yaml: rich+kvz+PerTokenAttention 2L/4H, 2M, seed=0,
  mismos anchors) lanzado en `kv-chat-v1` (proyecto de Alex, L4 on-demand → sin
  preemption). Venv de Alex reutilizado read-only + `PYTHONPATH` a nuestro código
  SCPeado (verificado: `kv_gym` resuelve a `~/repo/src`, GPU 45%, curvas escribiendo).
- 03:45 UTC — watcher_attn armado (baja curvas + apaga VM al terminar + compare.py).
- 03:50 UTC — `s_rich` asegurado: curvas + final/best checkpoints descargados a
  `ab_results/` (la VM spot podría no volver a arrancar). Stats finales:
  **mean −0.044 / last5 −0.050** sobre 31 probes → paridad/apenas-abajo, no supera.
- (se completa al llegar cada resultado)
