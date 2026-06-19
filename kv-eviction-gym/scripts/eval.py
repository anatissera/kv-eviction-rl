"""
Evaluate a trained MaskablePPO policy against two baselines on GSM8K.

Baselines:
  oracle  — keep the top-k tokens by future_attn (best achievable AUC)
  random  — keep a uniformly random subset of k tokens

Usage:
    python scripts/eval.py --checkpoint checkpoints/phase1 --n_eval 50
    python scripts/eval.py --checkpoint checkpoints/phase1 --n_eval 50 --device cuda
"""

import argparse
import math
import numpy as np
import torch

from sb3_contrib import MaskablePPO

from kv_gym.env import SharedKVVecEnv
from kv_gym.policy import PerTokenMLP
from kv_gym.rewards.auc import future_attention_auc
from kv_gym.vendor.loader import load_model_and_tokenizer
from kv_gym.vendor.gsm8k import load_gsm8k


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to saved MaskablePPO (.zip)")
    p.add_argument("--model_name",  default="qwen-1.5b")
    p.add_argument("--n_eval",      type=int, default=50)
    p.add_argument("--budget",      type=int, default=32)
    p.add_argument("--max_len",     type=int, default=256)
    p.add_argument("--device",      default=None)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def run_episode_learned(env, policy) -> np.ndarray:
    """Run one full episode with the learned policy. Returns per-env AUC rewards."""
    obs = env.reset()
    done = False
    while not done:
        masks = env.action_masks()
        actions, _ = policy.predict(obs, action_masks=masks, deterministic=True)
        obs, rewards, dones, _ = env.step(actions)
        done = bool(dones[0])
    return rewards  # [n_envs]


def oracle_auc(future_attn: torch.Tensor, budget: int) -> torch.Tensor:
    """AUC when keeping the top-k tokens by future_attn (reward = 1.0 by definition)."""
    topk_vals, topk_idx = future_attn.topk(k=min(budget, future_attn.shape[-1]), dim=-1)
    n_envs, T = future_attn.shape
    resident = torch.zeros(n_envs, T, dtype=torch.bool)
    for i in range(n_envs):
        resident[i, topk_idx[i]] = True
    return future_attention_auc(future_attn, resident, budget)


def random_auc(future_attn: torch.Tensor, budget: int, rng: np.random.Generator) -> torch.Tensor:
    """AUC when keeping a uniformly random subset of k tokens."""
    n_envs, T = future_attn.shape
    k = min(budget, T)
    resident = torch.zeros(n_envs, T, dtype=torch.bool)
    for i in range(n_envs):
        chosen = rng.choice(T, size=k, replace=False)
        resident[i, chosen] = True
    return future_attention_auc(future_attn, resident, budget)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device) if args.device else None
    model, tokenizer, device = load_model_and_tokenizer(
        name=args.model_name,
        device=device,
        attn_implementation="eager",
    )

    examples = load_gsm8k(n=args.n_eval, seed=args.seed)

    env = SharedKVVecEnv(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        budget=args.budget,
        max_len=args.max_len,
        reward_mode="auc",
        device=device,
    )

    policy = MaskablePPO.load(args.checkpoint, env=env, device=device)

    learned_aucs, oracle_aucs, random_aucs = [], [], []

    for ep_idx, example in enumerate(examples):
        print(f"\rEpisode {ep_idx+1}/{args.n_eval}", end="", flush=True)

        # Run the learned policy episode
        env._example_iter = iter([example] * 1 + list(examples))  # force this example next
        learned_rew = run_episode_learned(env, policy)
        learned_aucs.append(float(learned_rew.mean()))

        # Oracle and random don't need a full episode — compute directly from capture
        cap = env.capture
        L, H = env.n_layers, env.n_heads
        fa_flat = cap.future_attn.view(L * H, cap.prompt_len)

        oracle_aucs.append(float(oracle_auc(fa_flat, args.budget).mean()))
        random_aucs.append(float(random_auc(fa_flat, args.budget, rng).mean()))

    print()
    print(f"\n{'Policy':<12} {'Mean AUC':>10}  {'Std':>8}")
    print("-" * 34)
    for name, vals in [("learned", learned_aucs), ("oracle", oracle_aucs), ("random", random_aucs)]:
        print(f"{name:<12} {np.mean(vals):>10.4f}  {np.std(vals):>8.4f}")

    print(f"\nLearned / Oracle ratio: {np.mean(learned_aucs)/np.mean(oracle_aucs):.4f}")
    print(f"Learned / Random ratio: {np.mean(learned_aucs)/np.mean(random_aucs):.4f}")


if __name__ == "__main__":
    main()
