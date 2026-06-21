"""
Train the KV-eviction policy using MaskablePPO.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --run-name my_run

All outputs are written to runs/<run-name>/:
    best_model.zip       — checkpoint with highest mean episode reward
    checkpoints/         — periodic checkpoints every 50 k steps
    learning_curve.csv   — (timestep, ep_rew_mean, ep_len_mean) per rollout
    tb/                  — TensorBoard event files
"""

import argparse
import csv
import json
import yaml
import torch
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from sb3_contrib import MaskablePPO

from kv_gym.env import SharedKVVecEnv
from kv_gym.policy import PerTokenMLP
from kv_gym.vendor.loader import load_model_and_tokenizer
from kv_gym.vendor.gsm8k import load_gsm8k


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="configs/train.yaml")
    p.add_argument("--run-name", default=None,
                   help="Sub-directory under runs/ for all outputs. "
                        "Defaults to a timestamp.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    defaults_path = Path(path).parent / "model_defaults.json"
    if defaults_path.exists():
        with open(defaults_path) as f:
            all_defaults = json.load(f)
        model_key = cfg.get("model_name", "")
        model_defaults = all_defaults.get(model_key, {})
        for k, v in model_defaults.items():
            if not k.startswith("_") and k not in cfg:
                cfg[k] = v

    return cfg


class TrainingLogger(BaseCallback):
    """Writes one CSV row per rollout and saves the best model by mean episode reward."""

    def __init__(self, run_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.run_dir   = run_dir
        self.csv_path  = run_dir / "learning_curve.csv"
        self.best_path = run_dir / "best_model"
        self.best_mean = float("-inf")
        self._writer   = None
        self._file     = None

    def _on_training_start(self) -> None:
        self._file   = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestep", "ep_rew_mean", "ep_len_mean"])

    def _on_rollout_end(self) -> None:
        buf = self.model.ep_info_buffer
        if not buf:
            return
        mean_rew = float(sum(ep["r"] for ep in buf) / len(buf))
        mean_len = float(sum(ep["l"] for ep in buf) / len(buf))
        self._writer.writerow([self.num_timesteps, f"{mean_rew:.6f}", f"{mean_len:.1f}"])
        self._file.flush()

        if mean_rew > self.best_mean:
            self.best_mean = mean_rew
            self.model.save(str(self.best_path))
            if self.verbose:
                print(f"  [best] t={self.num_timesteps}  ep_rew_mean={mean_rew:.4f}  → {self.best_path}")

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self._file:
            self._file.close()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = Path("runs") / run_name
    ckpt_dir = run_dir / "checkpoints"
    tb_dir   = run_dir / "tb"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    # Copy config into run dir for reproducibility
    (run_dir / "config.yaml").write_text(Path(args.config).read_text())

    device_cfg = cfg.get("device", "auto")
    device = None if device_cfg == "auto" else torch.device(device_cfg)

    use_shaping = cfg.get("use_attention_shaping", True)
    attn_impl   = cfg.get("attn_implementation", "eager" if use_shaping else None)

    model, tokenizer, device = load_model_and_tokenizer(
        name=cfg.get("model_name", "qwen-1.5b"),
        device=device,
        attn_implementation=attn_impl,
    )
    print(f"run_name: {run_name}")
    print(f"run_dir:  {run_dir}")
    print(f"attn_implementation: {attn_impl or 'default'}")
    print(f"Using device: {device}")

    examples = load_gsm8k(n=cfg.get("n_examples", 200), seed=cfg.get("seed", 0), split="train")

    print(f"budget_min={cfg.get('budget_min', 128)}  budget_max={cfg.get('budget_max', 256)}"
          f"  max_new_tokens={cfg.get('max_new_tokens', 524)}")

    env = SharedKVVecEnv(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        budget_min=cfg.get("budget_min", 128),
        budget_max=cfg.get("budget_max", 256),
        max_len=cfg.get("max_len", 512),
        max_new_tokens=cfg.get("max_new_tokens", 524),
        device=device,
        use_attention_shaping=cfg.get("use_attention_shaping", True),
        attention_weight=cfg.get("attention_weight", 0.3),
        seed=cfg.get("seed", 0),
    )

    checkpoint_freq = cfg.get("checkpoint_freq", 50_000)
    callbacks = CallbackList([
        TrainingLogger(run_dir, verbose=1),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(ckpt_dir),
            name_prefix="ckpt",
            verbose=1,
        ),
    ])

    ppo = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": PerTokenMLP,
            "features_extractor_kwargs": {"hidden": cfg.get("hidden", 64)},
        },
        n_steps=cfg.get("n_steps", 524),
        batch_size=cfg.get("batch_size", 1024),
        n_epochs=cfg.get("n_epochs", 4),
        gamma=cfg.get("gamma", 1.0),
        gae_lambda=cfg.get("gae_lambda", 1.0),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.01),
        tensorboard_log=str(tb_dir),
        verbose=1,
        device=device,
    )

    ppo.learn(total_timesteps=cfg.get("total_timesteps", 500_000), callback=callbacks)

    final_path = run_dir / "final_model"
    ppo.save(str(final_path))
    print(f"\nTraining complete.")
    print(f"  Final model : {final_path}.zip")
    print(f"  Best model  : {run_dir / 'best_model'}.zip")
    print(f"  Curves CSV  : {run_dir / 'learning_curve.csv'}")
    print(f"  TensorBoard : tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    main()
