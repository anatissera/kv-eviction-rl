# Added for this project (UdeSA RL final). Not part of Apple's original ml-learning-to-evict release.
"""
Compare RLOO vs PPO learning curves from saved checkpoint metrics_history.

Usage:
    python scripts/plot_rloo_vs_ppo.py \
        --ppo  agents/grouped/qwen1b_ppo_validation/layer_000000/kv_head_000/2026-06-15_09-44-22/checkpoints/000000000050.pth \
        --rloo agents/grouped/qwen1b_rloo_cpu/layer_000000/kv_head_000/<RUN_DIR>/checkpoints/000000000050.pth \
        --out  learning_curves.png

Or let it auto-discover the latest checkpoint for each sweep:
    python scripts/plot_rloo_vs_ppo.py
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; works on Mac too
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def latest_checkpoint(sweep_dir: str) -> str | None:
    pattern = os.path.join(sweep_dir, "**", "checkpoints", "*.pth")
    ckpts = sorted(glob.glob(pattern, recursive=True))
    return ckpts[-1] if ckpts else None


def load_metrics(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    history = ckpt.get("metrics_history", {})
    # normalise: strip empty lists so callers don't have to guard
    return {k: v for k, v in history.items() if isinstance(v, list) and len(v) > 0}


def smooth(values, w=3):
    if len(values) < w:
        return np.array(values, dtype=float)
    kernel = np.ones(w) / w
    padded = np.pad(values, (w // 2, w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def plot_series(ax, steps, values, label, color, alpha_raw=0.25, smooth_w=3):
    arr = np.array(values, dtype=float)
    steps_arr = np.array(steps)
    ax.plot(steps_arr, arr, color=color, alpha=alpha_raw, linewidth=0.8)
    ax.plot(steps_arr, smooth(arr, smooth_w), color=color, linewidth=1.8, label=label)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    repo = Path(__file__).parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo",  default=None, help="Path to PPO  step-50 checkpoint (.pth)")
    parser.add_argument("--rloo", default=None, help="Path to RLOO step-50 checkpoint (.pth)")
    parser.add_argument("--out",  default=str(repo / "learning_curves.png"))
    args = parser.parse_args()

    ppo_path = args.ppo or latest_checkpoint(
        str(repo / "agents/grouped/qwen1b_ppo_validation")
    )
    rloo_path = args.rloo or latest_checkpoint(
        str(repo / "agents/grouped/qwen1b_rloo_cpu")
    )

    if not ppo_path or not os.path.exists(ppo_path):
        sys.exit(f"ERROR: PPO checkpoint not found. Got: {ppo_path}")
    if not rloo_path or not os.path.exists(rloo_path):
        sys.exit(f"ERROR: RLOO checkpoint not found. Got: {rloo_path}")

    print(f"PPO  checkpoint : {ppo_path}")
    print(f"RLOO checkpoint : {rloo_path}")

    ppo  = load_metrics(ppo_path)
    rloo = load_metrics(rloo_path)

    print("\nPPO  metrics available :", sorted(ppo.keys()))
    print("RLOO metrics available :", sorted(rloo.keys()))

    # ── build step axes ──────────────────────────────────────────────────────
    def steps_for(history, key):
        vals = history[key]
        n = len(vals)
        # metrics are logged every train step; eval every eval_interval
        # heuristic: if it looks like eval cadence (len << 50), space by 10
        if n <= 10:
            return list(range(10, n * 10 + 1, 10))
        return list(range(1, n + 1))

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), dpi=120)
    fig.suptitle("RLOO vs PPO — 50 training steps (layer 0, head 0)", fontsize=13, y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    PPO_COLOR  = "#e05c2a"   # orange-red
    RLOO_COLOR = "#2a7ae0"   # blue

    # ── Panel 1 : episode_reward (train) ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_title("Training reward (episode_reward)", fontsize=10)
    if "episode_reward" in ppo:
        s = steps_for(ppo,  "episode_reward")
        plot_series(ax1, s, ppo["episode_reward"],  "PPO",  PPO_COLOR)
    if "episode_reward" in rloo:
        s = steps_for(rloo, "episode_reward")
        plot_series(ax1, s, rloo["episode_reward"], "RLOO", RLOO_COLOR)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Reward"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # ── Panel 2 : eval_reward ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("Eval reward", fontsize=10)
    if "eval_reward" in ppo:
        s = steps_for(ppo,  "eval_reward")
        plot_series(ax2, s, ppo["eval_reward"],  "PPO",  PPO_COLOR,  smooth_w=1)
    if "eval_reward" in rloo:
        s = steps_for(rloo, "eval_reward")
        plot_series(ax2, s, rloo["eval_reward"], "RLOO", RLOO_COLOR, smooth_w=1)
    ax2.set_xlabel("Step"); ax2.set_ylabel("Reward"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # ── Panel 3 : gradient_norm ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_title("Gradient norm", fontsize=10)
    if "gradient_norm" in ppo:
        s = steps_for(ppo,  "gradient_norm")
        plot_series(ax3, s, ppo["gradient_norm"],  "PPO",  PPO_COLOR)
    if "gradient_norm" in rloo:
        s = steps_for(rloo, "gradient_norm")
        plot_series(ax3, s, rloo["gradient_norm"], "RLOO", RLOO_COLOR)
    ax3.set_xlabel("Step"); ax3.set_ylabel("‖∇‖"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # ── Panel 4 : PPO value loss ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_title("PPO value loss (critic convergence)", fontsize=10)
    if "ppo_value_loss" in ppo:
        s = steps_for(ppo, "ppo_value_loss")
        plot_series(ax4, s, ppo["ppo_value_loss"], "PPO value loss", PPO_COLOR)
    ax4.set_xlabel("Step"); ax4.set_ylabel("MSE loss"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    # ── Panel 5 : policy losses ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.set_title("Policy loss", fontsize=10)
    if "ppo_policy_loss" in ppo:
        s = steps_for(ppo,  "ppo_policy_loss")
        plot_series(ax5, s, ppo["ppo_policy_loss"],  "PPO policy loss",  PPO_COLOR)
    # RLOO uses "policy_loss" or "total_loss"
    for key in ("policy_loss", "total_loss"):
        if key in rloo:
            s = steps_for(rloo, key)
            plot_series(ax5, s, rloo[key], f"RLOO {key}", RLOO_COLOR)
            break
    ax5.set_xlabel("Step"); ax5.set_ylabel("Loss"); ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    # ── Panel 6 : PPO clip fraction & approx KL ──────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_title("PPO clip fraction & approx KL", fontsize=10)
    if "ppo_clip_fraction" in ppo:
        s = steps_for(ppo, "ppo_clip_fraction")
        plot_series(ax6, s, ppo["ppo_clip_fraction"], "clip fraction", PPO_COLOR, smooth_w=1)
    if "ppo_approx_kl" in ppo:
        s = steps_for(ppo, "ppo_approx_kl")
        plot_series(ax6, s, ppo["ppo_approx_kl"], "approx KL", "#c05090", smooth_w=1)
    ax6.set_xlabel("Step"); ax6.legend(fontsize=8); ax6.grid(alpha=0.3)

    # ── health annotation ────────────────────────────────────────────────────
    notes = []
    if "eval_reward" in ppo and len(ppo["eval_reward"]) >= 2:
        delta = ppo["eval_reward"][-1] - ppo["eval_reward"][0]
        notes.append(f"PPO eval Δ = {delta:+.4f}")
    if "eval_reward" in rloo and len(rloo["eval_reward"]) >= 2:
        delta = rloo["eval_reward"][-1] - rloo["eval_reward"][0]
        notes.append(f"RLOO eval Δ = {delta:+.4f}")
    if "ppo_value_loss" in ppo and len(ppo["ppo_value_loss"]) >= 2:
        notes.append(f"Value loss {ppo['ppo_value_loss'][0]:.3f} → {ppo['ppo_value_loss'][-1]:.3f}")
    if notes:
        fig.text(0.5, 0.005, "  |  ".join(notes), ha="center", fontsize=8, color="#444")

    plt.savefig(args.out, bbox_inches="tight")
    print(f"\nSaved → {args.out}")

    # ── text summary ─────────────────────────────────────────────────────────
    print("\n── Health check ─────────────────────────────────────────────────")
    for algo, h in [("PPO", ppo), ("RLOO", rloo)]:
        if "eval_reward" in h:
            er = h["eval_reward"]
            trend = "↑ improving" if er[-1] > er[0] else "↓ degrading" if er[-1] < er[0] else "→ flat"
            print(f"  {algo:4s}  eval_reward: {er[0]:.4f} → {er[-1]:.4f}  ({trend})")
        if "gradient_norm" in h:
            gn = h["gradient_norm"]
            print(f"  {algo:4s}  grad_norm:   {gn[0]:.3f} → {gn[-1]:.3f}")
    if "ppo_value_loss" in ppo:
        vl = ppo["ppo_value_loss"]
        print(f"  PPO   value_loss:  {vl[0]:.4f} → {vl[-1]:.4f}  (critic converged: {vl[-1] < 0.05})")
    if "ppo_clip_fraction" in ppo:
        cf = ppo["ppo_clip_fraction"]
        avg_cf = np.mean(cf)
        print(f"  PPO   clip_frac:   mean={avg_cf:.4f}  (healthy if < 0.3)")
    print("─────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
