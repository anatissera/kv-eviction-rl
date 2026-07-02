"""Phase 2 · E3 — warm-start the policy by behavior-cloning kv_norm.

RL from a random init never finds the kv_norm basin (learned ≈ random after 5M
steps). This supervised pre-phase makes the policy imitate kv_norm's slot choice
so PPO STARTS at the best heuristic and can only improve. It isolates the
optimization/exploration bottleneck (D4) from representation (D1/D2).

The kv_norm target is computed EXACTLY from the raw K||V already in the observation
(the first kv_dim columns, before the policy's LayerNorm) — no env internals, no
dependence on the rich columns. But cloning only *succeeds* (loss → 0) when the
policy can SEE the norm, i.e. with rich_features; the printed `match_kv_norm`
accuracy is the sanity check (should approach ~1.0).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import obs_as_tensor


def behavior_clone_kv_norm(
    ppo,
    env,
    n_bc_steps: int,
    n_kv_heads: int,
    head_dim: int,
    device,
    lr: float = 1e-3,
    log_every: int = 50,
) -> float:
    """Run n_bc_steps of supervised cloning of kv_norm onto ppo.policy in place.
    Returns the final match_kv_norm accuracy. The env auto-resets finished episodes
    inside step(), so stepping continuously traverses realistic eviction states."""
    policy  = ppo.policy
    policy.set_training_mode(True)
    opt     = torch.optim.Adam(policy.parameters(), lr=lr)
    kv_half = n_kv_heads * head_dim
    # column of kvz (the exact standardized ||K||+||V|| kv_norm signal) in the obs:
    # after the kv_dim K||V block come the extra cols [kz_h(H), vz_h(H), kz_mean,
    # vz_mean, kvz, ...]; kvz is at extra index 2H+2.
    kvz_col = 2 * kv_half + 2 * n_kv_heads + 2
    TARGET_SCALE = 3.0     # peak the cloned policy at kv_norm's choice (PPO sharpens further)

    obs  = env.reset()
    last_loss = float("nan")
    last_acc  = 0.0
    for step in range(n_bc_steps):
        masks   = get_action_masks(env)                       # [B, max_len] bool
        obs_t   = obs_as_tensor(obs, device).float()
        masks_t = torch.as_tensor(np.asarray(masks), device=device, dtype=torch.bool)

        # DENSE target: regress the actor logits to -kvz on every valid slot, so
        # argmax(logits) = argmin(kvz) = kv_norm's exact choice. Dense (all slots
        # supervised) → stable/fast, unlike the noisy 220-way argmin cross-entropy.
        kvz    = obs_t[..., kvz_col]                          # [B, max_len]
        target = (-kvz * TARGET_SCALE)

        features  = policy.extract_features(obs_t)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        logits    = policy.action_net(latent_pi)             # [B, max_len] raw actor logits

        diff = (logits - target) * masks_t.float()
        loss = (diff.pow(2).sum(dim=-1) / masks_t.float().sum(dim=-1).clamp(min=1)).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        last_loss = float(loss.item())
        with torch.no_grad():
            # match = does the greedy (masked) action equal kv_norm's argmin(kvz)?
            kvz_masked  = kvz.masked_fill(~masks_t, float("inf"))
            kvn_choice  = kvz_masked.argmin(dim=-1)
            pol_choice  = logits.masked_fill(~masks_t, float("-inf")).argmax(dim=-1)
            last_acc    = float((pol_choice == kvn_choice).float().mean().item())
        if step % log_every == 0:
            print(f"  [BC] step {step}/{n_bc_steps} loss={last_loss:.4f} "
                  f"match_kv_norm={last_acc:.3f}", flush=True)

        # advance along kv_norm's own trajectory (finished episodes auto-reset)
        obs, _, _, _ = env.step(kvn_choice.detach().cpu().numpy())

    policy.set_training_mode(False)
    print(f"  [BC] done: final loss={last_loss:.4f} match_kv_norm={last_acc:.3f}", flush=True)
    return last_acc


# ── E5 — Golden-BC: clone the FUTURE-attention oracle (ForesightKV recipe) ────

class _TraceStore:
    """Lazy LRU over trace_gen.py npz files (~28MB each; env visits examples
    sequentially via its cursor, so a small cache has ~perfect hit rate)."""

    def __init__(self, traces_dir, max_loaded: int = 40):
        from pathlib import Path
        self.dir = Path(traces_dir)
        self.max_loaded = max_loaded
        self._cache: dict[int, object] = {}
        self._order: list[int] = []

    def get(self, ex_idx: int):
        """Returns dict with 'suf' list[L] of [steps, keys] arrays, or None."""
        if ex_idx in self._cache:
            return self._cache[ex_idx]
        path = self.dir / f"ex{ex_idx:05d}.npz"
        if not path.exists():
            entry = None
        else:
            z = np.load(path)
            if int(z["skip"]) == 1:
                entry = None
            else:
                layers = sorted(k for k in z.files if k.startswith("l"))
                entry = {"suf": [z[k] for k in layers], "n_steps": int(z["n_steps"])}
        self._cache[ex_idx] = entry
        self._order.append(ex_idx)
        if len(self._order) > self.max_loaded:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return entry


def behavior_clone_golden(
    ppo,
    env,
    n_bc_steps: int,
    traces_dir: str,
    device,
    lr: float = 1e-3,
    target_scale: float = 3.0,
    log_every: int = 50,
) -> float:
    """Supervised cloning of the GOLDEN EVICTION oracle onto ppo.policy.

    Target per (episode b, layer l) stream: rank resident slots by their max
    FUTURE attention SUF[l][t, orig_pos] from the example's full-cache trace
    (scripts/trace_gen.py). Actor logits regress to -z(log SUF) on all valid
    slots (dense, like the kv_norm BC) → argmax(logits) = the oracle's evictee.
    The oracle beat kv_norm by +7pp (FINDINGS §11); this measures how much of
    that a policy that must PREDICT the future from present features captures.

    The env is advanced along the GOLDEN action (on-oracle-policy states).
    Streams whose example has no trace are masked out of the loss and advance
    with a safe fallback (their masked argmax).
    """
    policy = ppo.policy
    policy.set_training_mode(True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    store = _TraceStore(traces_dir)

    L = env.n_layers
    max_len = env.observation_space.shape[0]
    obs = env.reset()
    last_loss, last_acc = float("nan"), 0.0

    for step in range(n_bc_steps):
        masks = get_action_masks(env)                        # [N*L, max_len]
        obs_t = obs_as_tensor(obs, device).float()
        masks_t = torch.as_tensor(np.asarray(masks), device=device, dtype=torch.bool)
        B = obs_t.shape[0]

        # Build golden score rows (+inf = never evict / unknown key).
        score = np.full((B, max_len), np.inf, dtype=np.float32)
        row_ok = np.zeros(B, dtype=bool)
        for b in range(env.N):
            ep = env.episodes[b]
            tr = store.get(env._current_example_indices[b])
            if tr is None or ep.past_kv is None:
                continue
            t = min(ep.step_count, tr["n_steps"] - 1)
            for l in range(L):
                i = b * L + l
                suf = tr["suf"][l]
                n_keys = suf.shape[1]
                pos = ep.slot_to_pos[l]
                vals = [float(suf[t, p]) if p < n_keys else np.inf for p in pos]
                score[i, :len(pos)] = vals
                row_ok[i] = True

        score_t = torch.as_tensor(score, device=device)
        valid = masks_t & torch.isfinite(score_t)
        ok_t = torch.as_tensor(row_ok, device=device) & (valid.sum(-1) > 1)

        # z(log) over valid slots per row → scale-free ranking target.
        logs = torch.where(valid, torch.log(score_t.clamp(min=1e-8)),
                           torch.zeros_like(score_t))
        cnt = valid.float().sum(-1).clamp(min=1)
        mu = (logs * valid.float()).sum(-1) / cnt
        var = (((logs - mu[:, None]) ** 2) * valid.float()).sum(-1) / cnt
        z = (logs - mu[:, None]) / var.sqrt().clamp(min=1e-6)[:, None]
        target = -z * target_scale

        features = policy.extract_features(obs_t)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        logits = policy.action_net(latent_pi)

        diff = (logits - target) * valid.float()
        per_row = diff.pow(2).sum(-1) / valid.float().sum(-1).clamp(min=1)
        loss = (per_row * ok_t.float()).sum() / ok_t.float().sum().clamp(min=1)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        last_loss = float(loss.item())
        with torch.no_grad():
            golden = score_t.masked_fill(~valid, float("inf")).argmin(-1)
            pol = logits.masked_fill(~masks_t, float("-inf")).argmax(-1)
            m = ok_t.float()
            last_acc = float(((pol == golden).float() * m).sum() / m.sum().clamp(min=1))
        if step % log_every == 0:
            print(f"  [golden-BC] step {step}/{n_bc_steps} loss={last_loss:.4f} "
                  f"match_oracle={last_acc:.3f} rows_ok={float(m.mean()):.2f}", flush=True)

        # advance along the oracle's own action (fallback: policy argmax where no trace)
        act = torch.where(ok_t, golden, pol)
        obs, _, _, _ = env.step(act.detach().cpu().numpy())

    policy.set_training_mode(False)
    print(f"  [golden-BC] done: loss={last_loss:.4f} match_oracle={last_acc:.3f}", flush=True)
    return last_acc
