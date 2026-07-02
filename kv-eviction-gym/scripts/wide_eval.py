"""One-shot WIDE paired eval of trained checkpoints on FRESH GSM8K examples.

Purpose: the training probe uses 32 held-out examples — too noisy (1 example =
±0.031) to distinguish parity from a small effect. This evaluates final
checkpoints on N (default 128) examples drawn from the SAME seeded GSM8K
shuffle, *after* the train+probe prefix → guaranteed disjoint from anything
any 2M run saw (train = all[:n_examples], probe = all[n_examples:n_examples+
probe_n], wide = all[n_examples+probe_n : n_examples+probe_n+N]).

It reuses ProbeCallback — the exact code path of the training probe (same
obs building, same episode runner) — so numbers are directly comparable and
there is no train/eval confound. Anchors (full/random/kv_norm baselines) are
cached in <out-dir>/probe_anchors.pkl and SHARED across checkpoints evaluated
into the same --out-dir: baselines are paid once, each checkpoint adds one
learned-pass row to probe_curve.csv.

Usage (on a GPU VM, from the repo root):
  python scripts/wide_eval.py --config configs/e1_rich.yaml \
      --model runs/s_rich/final_model.zip --label s_rich \
      --n 128 --out-dir runs/wide_eval
Each invocation appends ONE row; the `label → timestep` mapping is written to
<out-dir>/labels.csv (ProbeCallback's CSV keys rows by num_timesteps).
"""
import argparse
import csv
import sys
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure

from kv_gym.episode_ppo import EpisodeMaskablePPO
from kv_gym.probe import EvalProbeCallback
from kv_gym.vendor.loader import load_model_and_tokenizer
from kv_gym.vendor.gsm8k import load_gsm8k
from kv_gym.free_growth_cache import FreeGrowthCache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Training config of the checkpoint")
    p.add_argument("--model", required=True, help="Path to MaskablePPO checkpoint .zip")
    p.add_argument("--label", required=True, help="Run label (recorded in labels.csv)")
    p.add_argument("--n", type=int, default=128, help="Wide-eval example count")
    p.add_argument("--out-dir", default="runs/wide_eval")
    p.add_argument("--slice-start", type=int, default=None,
                   help="Override the wide-slice start (default: n_examples+probe_n "
                        "from the config). Use 1032 to evaluate arms trained with a "
                        "different n_examples on the SHARED [1032:1160] slice.")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device_cfg = cfg.get("device", "auto")
    device = torch.device("cuda" if (device_cfg == "auto" and torch.cuda.is_available())
                          else device_cfg if device_cfg != "auto" else "cpu")

    llm, tokenizer, device = load_model_and_tokenizer(
        name=cfg.get("model_name", "qwen-1.5b"),
        device=device,
        attn_implementation=cfg.get("attn_implementation"),
    )

    n_examples = cfg.get("n_examples", 200)
    probe_n    = cfg.get("probe_n", 32)
    seed       = cfg.get("seed", 0)
    # Same seeded shuffle as train.py, extended: the wide slice starts after
    # the train+probe prefix → disjoint from anything the checkpoint saw.
    start = args.slice_start if args.slice_start is not None else n_examples + probe_n
    all_examples = load_gsm8k(n=start + args.n, seed=seed, split="train")
    wide = all_examples[start:]
    print(f"wide eval: {len(wide)} fresh examples "
          f"(slice [{start}:{start + args.n}], seed={seed})")

    # Load the policy (custom extractor classes resolve via kv_gym.policy imports
    # inside the pickled policy_kwargs). env=None: we only use the policy.
    ppo_cls = EpisodeMaskablePPO if cfg.get("n_parallel", 1) > 1 else MaskablePPO
    ppo = ppo_cls.load(args.model, env=None, device=device)
    print(f"loaded {args.label}: {args.model}  (num_timesteps={ppo.num_timesteps:,})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fgc_dir = cfg.get("free_growth_cache_dir")
    fgc = FreeGrowthCache(fgc_dir, cfg.get("model_name", "qwen-1.5b")) if fgc_dir else None

    probe = EvalProbeCallback(
        llm=llm, tokenizer=tokenizer, probe_examples=wide,
        budget_min=cfg.get("budget_min", 256),
        budget_max=cfg.get("budget_max", 256),
        max_new_tokens=cfg.get("max_new_tokens", 600),
        max_len=cfg.get("max_len", 288),
        every_n_rollouts=1,
        n_sinks=cfg.get("n_sinks", 4),
        n_recent=cfg.get("n_recent", 32),
        run_dir=out_dir,
        device=device,
        free_growth_cache=fgc,
        protect_prompt=cfg.get("protect_prompt", False),
        rich_features=cfg.get("rich_features", False),
    )
    # Minimal SB3 callback wiring (no training loop): .model for the policy,
    # .logger for the record() calls, num_timesteps keys the CSV row.
    probe.model = ppo
    ppo.set_logger(configure(str(out_dir / "tb_null"), ["stdout"]))
    # (BaseCallback.logger is a read-only property → resolves via probe.model)
    # BaseCallback.num_timesteps is normally synced in on_step(); we bypass the
    # training loop, so sync it once — it keys the CSV row per checkpoint.
    probe.num_timesteps = ppo.num_timesteps

    probe._on_training_start()      # builds/loads anchors + baselines (cached)
    probe._rollout_count = 0        # so count=1 % every_n_rollouts(1) == 0
    probe._on_rollout_end()         # one learned pass → one CSV row
    probe._on_training_end()

    labels_path = out_dir / "labels.csv"
    new = not labels_path.exists()
    with open(labels_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "model", "num_timesteps", "n_wide"])
        w.writerow([args.label, args.model, ppo.num_timesteps, len(wide)])
    print(f"wide eval row appended for {args.label} → {out_dir/'probe_curve.csv'}")


if __name__ == "__main__":
    main()
