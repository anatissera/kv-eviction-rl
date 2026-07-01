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
