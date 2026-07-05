# 10 · The per-layer credit wall and E9 (per-layer reward)

**Report sections:** §4.13 "Per-layer credit" and §4.14 "Per-layer reward and a second
bottleneck".
**Configs:** `kv-eviction-gym/configs/e9_perlayer.yaml`, `e9_perlayer_attn.yaml`, `e9_smoke.yaml`.
Implementation: `per_layer_reward: true` in `src/kv_gym/batched_env.py` (`step_wait`).

## The root cause (found by reading the code after E8)

The environment exposes `n_envs = N x 28`: the policy makes **28 eviction decisions per
step**, one per layer. But the reward is **a single scalar per episode, broadcast
identically to the 28 layers**. Even E8's dense KL is computed on the model's FINAL
distribution, which is a joint function of all 28 layers' evictions.

PPO sees 28 distinct (state, action) pairs sharing ONE return: a layer's marginal effect
is drowned out by the other 27. The only thing learnable from a joint scalar over a
220^28 action space is a rule that is reasonable on average for all layers, which is
approximately what kv_norm already is. That explains why the WHOLE series ended at
parity:

| we tried | improved | did it touch per-layer credit? |
|---|---|---|
| rich features (s_rich) | representation | no |
| warm-start (s_warm) | initialization | no |
| cross-token attention (s_attn) | architecture | no |
| dense KL (s_e8) | TEMPORAL (per-step) credit | no (the KL is joint) |
| 50 reps + regret (s_e7) | samples + variance | no |

(It is the table in §4.13 of the report.) Running each arm to completion was not waste:
each one was a controlled ablation that discarded a hypothesis with evidence, and the
systematic elimination is what made it possible to locate the real wall in the
environment.

## E9: the fix (per-layer dense reward)

Each layer-env receives ITS own dense reward = the marginal hidden-state divergence its
decision causes, against a shadow cache that never evicts:

```
div_k    = 1 - cos(h_evict[k], h_full[k])        (output of each block k)
damage_l = relu(div_{l+1} - div_l)               (isolates layer l's contribution)
r_l      = -w * clip(damage_l, 0, 5)
```

The terminal reward (correctness + regret) remains shared; the per-layer dense term
supplies the differentiation. Mechanical verification: the smoke prints the 28 per-layer
rewards and confirms they DIFFER (previously identical due to the broadcast). This was
chosen over "tying" the eviction of the 28 layers because tying handicaps a uniform
policy against a per-layer heuristic (kv_norm evicts per layer): a confounded and
invasive change.

## Result: parity again (a second, nested wall)

| arm | probes | paired learned - kv_norm |
|---|---|---|
| s_e9 (per-layer, MLP) | 6 | **+0.021 ± 0.035** (n.s.; an isolated +0.19 that reverted) |
| s_e9attn (per-layer, attention) | 5 | **+0.038 ± 0.022** (n.s.) |

Two nested walls, both located:
1. Per-layer credit WAS a real wall (the fix changes the mechanism, verified).
2. But fixing it is not enough: **every dense reward computable online (E8's joint KL,
   E9's per-layer hidden-state divergence) decouples from correctness under truncation**,
   and the terminal reward remains too sparse. Exactly the reason KVP and ForesightKV use
   offline supervision with future information.

## What the report says

§4.13 presents the root cause and the table; §4.14 the fix, the verification, the parity
result and the consolidated conclusion: "online PPO remains limited by the training
signal".

## Raw data

- `kv-eviction-gym/ab_results/s_e9_{learning,probe}_curve.csv`, `s_e9attn_{learning,probe}_curve.csv`

## Status

Valid; closes the GSM8K phase. The remaining question (is it the method or the dataset?)
is answered in [11](11-dataset-causality.md).
