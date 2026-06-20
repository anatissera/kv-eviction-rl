"""
Train the KV-eviction policy using MaskablePPO (Phase 1: AUC reward).

Usage:
    python scripts/train.py --config configs/phase1.yaml

Requirements:
    pip install stable-baselines3 sb3-contrib gymnasium torch transformers datasets
"""

import argparse
import json
import yaml
import torch
from pathlib import Path

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
        cfg = yaml.safe_load(f)

    # Merge model-specific defaults (configs/model_defaults.json).
    # Values in the yaml always take precedence over the defaults.
    defaults_path = Path(path).parent / "model_defaults.json"
    if defaults_path.exists():
        with open(defaults_path) as f:
            all_defaults = json.load(f)
        model_key = cfg.get("model_name", "")
        model_defaults = all_defaults.get(model_key, {})
        # Only fill in keys not already present in the yaml
        for k, v in model_defaults.items():
            if not k.startswith("_") and k not in cfg:
                cfg[k] = v

    return cfg


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    device_cfg = cfg.get("device", "auto")
    device = None if device_cfg == "auto" else torch.device(device_cfg)

    # Attention shaping requires output_attentions=True, which only works with
    # eager attention.  Auto-select: eager when shaping is on, sdpa otherwise.
    # An explicit attn_implementation in the config always takes precedence.
    use_shaping = cfg.get("use_attention_shaping", True)
    attn_impl   = cfg.get("attn_implementation", "eager" if use_shaping else None)

    model, tokenizer, device = load_model_and_tokenizer(
        name=cfg.get("model_name", "qwen-1.5b"),
        device=device,
        attn_implementation=attn_impl,
    )
    print(f"attn_implementation: {attn_impl or 'default'}")
    print(f"Using device: {device}")

    examples = load_gsm8k(n=cfg.get("n_examples", 200), seed=cfg.get("seed", 0), split="train")

    print(f"budget_min={cfg.get('budget_min', 32)}  budget_max={cfg.get('budget_max', 256)}"
          f"  max_new_tokens={cfg.get('max_new_tokens', 512)}")

    env = SharedKVVecEnv(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        budget_min=cfg.get("budget_min", 32),
        budget_max=cfg.get("budget_max", 256),
        max_len=cfg.get("max_len", 256),
        max_new_tokens=cfg.get("max_new_tokens", 512),
        device=device,
        use_attention_shaping=cfg.get("use_attention_shaping", True),
        attention_weight=cfg.get("attention_weight", 0.3),
        seed=cfg.get("seed", 0),
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
