"""
Train the KV-eviction policy using MaskablePPO (Phase 1: AUC reward).

Usage:
    python scripts/train.py --config configs/phase1.yaml

Requirements:
    pip install stable-baselines3 sb3-contrib gymnasium torch transformers datasets
"""

import argparse
import yaml
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from kv_gym.env import SharedKVVecEnv
from kv_gym.policy import PerTokenMLP
from kv_gym.vendor.loader import load_model_and_tokenizer
from kv_gym.vendor.gsm8k import load_gsm8k


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/phase1.yaml")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    device = torch.device(cfg.get("device", "cpu"))
    model, tokenizer, device = load_model_and_tokenizer(
        name=cfg.get("model_name", "qwen-1.5b"),
        device=device,
        attn_implementation="eager",
    )

    examples = load_gsm8k(n=cfg.get("n_examples", 200), seed=cfg.get("seed", 0))

    env = SharedKVVecEnv(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        budget=cfg.get("budget", 32),
        max_len=cfg.get("max_len", 256),
        reward_mode=cfg.get("reward_mode", "auc"),
        device=device,
    )

    # MaskablePPO needs the env wrapped so it can call action_masks()
    # We subclass the env itself with the masks method, so we pass a lambda.
    def mask_fn(e):
        return e.action_masks()

    # For VecEnv, MaskablePPO reads masks directly from the env via
    # the action_masks() method — no extra wrapper needed when using
    # the VecEnv interface directly.

    ppo = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": PerTokenMLP,
            "features_extractor_kwargs": {"hidden": cfg.get("hidden", 64)},
        },
        n_steps=cfg.get("n_steps", 560),        # 10 full episodes × 56 envs
        batch_size=cfg.get("batch_size", 560),
        n_epochs=cfg.get("n_epochs", 4),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.0),
        verbose=1,
        device=device,
    )

    ppo.learn(total_timesteps=cfg.get("total_timesteps", 500_000))
    ppo.save(cfg.get("save_path", "ppo_kv_eviction"))
    print("Done. Model saved.")


if __name__ == "__main__":
    main()
