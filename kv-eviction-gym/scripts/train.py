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
import sys
import yaml
import torch
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor
from sb3_contrib import MaskablePPO

from kv_gym.env import SharedKVVecEnv
from kv_gym.batched_env import BatchedSharedKVVecEnv
from kv_gym.policy import PerTokenMLP, PerTokenAttention
from kv_gym.features import extra_feature_dim
from kv_gym.buffer import FP16ObsMaskableRolloutBuffer
from kv_gym.episode_ppo import EpisodeMaskablePPO
from kv_gym.free_growth_cache import FreeGrowthCache
from kv_gym.vendor.loader import load_model_and_tokenizer
from kv_gym.vendor.gsm8k import load_gsm8k
from kv_gym.probe import EvalProbeCallback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="configs/train.yaml")
    p.add_argument("--run-name", default=None,
                   help="Sub-directory under runs/ for all outputs. "
                        "Defaults to a timestamp.")
    p.add_argument("--resume-from", default=None,
                   help="Path to a saved MaskablePPO checkpoint to resume training from.")
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


class BudgetCurriculumCallback(BaseCallback):
    """Linearly anneals budget_min, budget_max, and eviction_k over training.

    budget_min: budget_min_start → budget_min_end (shared budget floor)
    budget_max: budget_max_start → budget_max_end (shared budget ceiling, optional)
    eviction_k: eviction_k_start → eviction_k_end (per-example budget difficulty, optional)
                Small K → budget high → easy (EOS nearby, few evictions).
                Large K → budget low → hard (strong compression required).

    All values updated at the start of each rollout.
    curriculum_fraction: fraction of total_timesteps the annealing spans.
    """

    def __init__(self, budget_min_start: int, budget_min_end: int,
                 total_timesteps: int, curriculum_fraction: float = 1.0,
                 budget_max_start: int | None = None, budget_max_end: int | None = None,
                 eviction_k_start: int | None = None, eviction_k_end: int | None = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.budget_min_start  = budget_min_start
        self.budget_min_end    = budget_min_end
        self.budget_max_start  = budget_max_start
        self.budget_max_end    = budget_max_end
        self.eviction_k_start  = eviction_k_start
        self.eviction_k_end    = eviction_k_end
        self.curriculum_end_ts = int(total_timesteps * max(curriculum_fraction, 1e-6))

    def _on_rollout_start(self) -> None:
        frac = min(self.num_timesteps / self.curriculum_end_ts, 1.0)
        env  = self.training_env.venv   # VecMonitor wraps the underlying env

        new_min = int(self.budget_min_start
                      + (self.budget_min_end - self.budget_min_start) * frac)
        env.budget_min = new_min

        if self.budget_max_start is not None and self.budget_max_end is not None:
            new_max = int(self.budget_max_start
                          + (self.budget_max_end - self.budget_max_start) * frac)
            env.budget_max = new_max
        else:
            new_max = None

        if self.eviction_k_start is not None and self.eviction_k_end is not None:
            new_k = int(self.eviction_k_start
                        + (self.eviction_k_end - self.eviction_k_start) * frac)
            env._eviction_k = new_k
        else:
            new_k = None

        if self.verbose:
            parts = [f"budget_min={new_min}"]
            if new_max is not None: parts.append(f"budget_max={new_max}")
            if new_k   is not None: parts.append(f"eviction_k={new_k}")
            print(f"  [curriculum] {', '.join(parts)}  (frac={frac:.2f})")

    def _on_step(self) -> bool:
        return True


class TrainingLogger(BaseCallback):
    """Writes one CSV row per rollout; tracks correctness rate and saves the best model.

    Correctness and alignment are read from the info dicts that step_wait() sets
    on episode end: info["correct"] (bool) and info["alignment"] (float).
    These are accumulated within each rollout and averaged before being written.
    """

    def __init__(self, run_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.run_dir   = run_dir
        self.csv_path  = run_dir / "learning_curve.csv"
        self.best_path = run_dir / "best_model"
        self.best_mean = float("-inf")
        self._writer   = None
        self._file     = None
        self._correct_buf:   list[bool]  = []
        self._align_buf:     list[float] = []
        self._episode_seconds_buf: list[float] = []
        self._truncated_buf: list[bool] = []
        self._context_size_buf: list[int] = []
        self._kl_buf: list[float] = []              # per-episode kl_step_mean (S4 shaping)
        self._example_counts: dict[int, int] = {}   # example_idx → total times seen
        self._dataset_pass: int = 0                 # latest dataset_pass seen

    def _on_training_start(self) -> None:
        self._file   = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "timestep", "ep_rew_mean", "ep_len_mean",
            "correctness_rate", "alignment_mean", "episode_seconds_mean",
            "episodes", "truncation_rate", "context_size_mean",
            "examples_seen", "dataset_pass", "kl_step_mean",
        ])

    def _on_step(self) -> bool:
        # SB3 passes locals["infos"] — a list of info dicts, one per env.
        # Only env 0 carries "correct" / "alignment" (all envs share the same episode).
        infos = self.locals.get("infos", [])
        if infos and "correct" in infos[0]:
            self._correct_buf.append(infos[0]["correct"])
            align = infos[0].get("alignment", float("nan"))
            if align == align:  # not NaN
                self._align_buf.append(align)
            episode_seconds = infos[0].get("episode_seconds", float("nan"))
            if episode_seconds == episode_seconds:  # not NaN
                self._episode_seconds_buf.append(episode_seconds)
            self._truncated_buf.append(infos[0].get("truncated", False))
            self._context_size_buf.append(infos[0].get("context_size", 0))
            kl_sm = infos[0].get("kl_step_mean")
            if kl_sm is not None:
                self._kl_buf.append(kl_sm)
            idx = infos[0].get("example_idx")
            if idx is not None:
                self._example_counts[idx] = self._example_counts.get(idx, 0) + 1
            dp = infos[0].get("dataset_pass")
            if dp is not None:
                self._dataset_pass = max(self._dataset_pass, dp)
        return True

    def _on_rollout_end(self) -> None:
        # ep_info_buffer is only populated by the standard SB3 PPO collect_rollouts
        # loop, which EpisodeMaskablePPO does not use. Fall back to NaN so the CSV
        # row is still written with all other metrics intact.
        buf = self.model.ep_info_buffer
        if buf:
            mean_rew = float(sum(ep["r"] for ep in buf) / len(buf))
            mean_len = float(sum(ep["l"] for ep in buf) / len(buf))
        else:
            mean_rew = float("nan")
            mean_len = float("nan")
        if not self._episode_seconds_buf:
            return

        corr_rate = (sum(self._correct_buf) / len(self._correct_buf)
                     if self._correct_buf else float("nan"))
        align_mean = (sum(self._align_buf) / len(self._align_buf)
                      if self._align_buf else float("nan"))
        episode_seconds_mean = (
            sum(self._episode_seconds_buf) / len(self._episode_seconds_buf)
            if self._episode_seconds_buf else float("nan")
        )
        episodes = len(self._episode_seconds_buf)
        trunc_rate = (sum(self._truncated_buf) / len(self._truncated_buf)
                      if self._truncated_buf else float("nan"))
        context_size_mean = (sum(self._context_size_buf) / len(self._context_size_buf)
                             if self._context_size_buf else float("nan"))
        examples_seen = len(self._example_counts)
        max_seen      = max(self._example_counts.values()) if self._example_counts else 0
        dataset_pass  = self._dataset_pass
        kl_step_mean  = (sum(self._kl_buf) / len(self._kl_buf)
                         if self._kl_buf else float("nan"))
        self._correct_buf.clear()
        self._align_buf.clear()
        self._episode_seconds_buf.clear()
        self._truncated_buf.clear()
        self._context_size_buf.clear()
        self._kl_buf.clear()

        self._writer.writerow([
            self.num_timesteps,
            f"{mean_rew:.6f}", f"{mean_len:.1f}",
            f"{corr_rate:.4f}", f"{align_mean:.4f}",
            f"{episode_seconds_mean:.3f}", episodes,
            f"{trunc_rate:.4f}", f"{context_size_mean:.1f}",
            examples_seen, dataset_pass, f"{kl_step_mean:.5f}",
        ])
        self._file.flush()

        if self.model is not None and hasattr(self.model, "logger") and kl_step_mean == kl_step_mean:
            self.model.logger.record("reward/kl_step_mean", kl_step_mean)

        if self.verbose:
            corr_str  = f"{corr_rate:.1%}" if corr_rate == corr_rate else "n/a"
            align_str = f"{align_mean:.3f}" if align_mean == align_mean else "n/a"
            sec_str   = (f"{episode_seconds_mean:.1f}s"
                         if episode_seconds_mean == episode_seconds_mean else "n/a")
            trunc_str = f"{trunc_rate:.1%}" if trunc_rate == trunc_rate else "n/a"
            print(f"  t={self.num_timesteps:>9,}  rew={mean_rew:.4f}"
                  f"  correct={corr_str}  align={align_str}"
                  f"  ep_time={sec_str}  episodes={episodes}"
                  f"  trunc={trunc_str}"
                  f"  seen={examples_seen} (max {max_seen}x)  pass={dataset_pass}")

        if mean_rew > self.best_mean:
            self.best_mean = mean_rew
            self.model.save(str(self.best_path))
            if self.verbose:
                print(f"  [best] → {self.best_path}")

    def _on_training_end(self) -> None:
        if self._file:
            self._file.close()


def _fix_glibc_malloc_threshold() -> None:
    # glibc dynamically raises M_MMAP_THRESHOLD after large frees (up to 32 MB).
    # Each obs array is ~17.8 MB (28 layers × 620 × 512 fp16). After rollout 1
    # frees ~14 GB of obs, glibc raises the threshold past 17.8 MB so R2+ obs
    # land in the brk arena instead of mmap — arena pages are not returned to the
    # OS on free, causing +6 GB/ep duplication during buffer assembly → OOM.
    # Pinning the threshold to 128 KB forces every obs alloc through mmap so
    # freeing it does munmap → immediate OS reclaim. No-op on non-Linux.
    try:
        import ctypes, ctypes.util
        _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        _libc.mallopt(-3, 131072)   # M_MMAP_THRESHOLD = 128 KB (disable dynamic growth)
        _libc.mallopt(-1, 131072)   # M_TRIM_THRESHOLD = 128 KB (trim arena aggressively)
    except Exception:
        pass


def main():
    _fix_glibc_malloc_threshold()
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

    n_examples = cfg.get("n_examples", 200)
    probe_n    = cfg.get("probe_n", 32)

    # Load training + probe examples together from the train split so the probe
    # uses left-out training examples with guaranteed no overlap.
    if cfg.get("dataset", "gsm8k") == "passkey":
        # The RULER-style needle arena (FINDINGS 25): oracle-kv_norm = +0.43 vs
        # GSM8K's +0.07 -- the regime with real learnable eviction signal. This
        # re-runs online PPO (the exact same pipeline that plateaued on GSM8K)
        # here, to test causally whether the dataset was the whole story.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_passkey import make_passkey_examples  # noqa: E402
        all_examples = make_passkey_examples(n_examples + probe_n, seed=cfg.get("seed", 0),
                                             n_filler=cfg.get("passkey_n_filler", 14))
    else:
        all_examples = load_gsm8k(n=n_examples + probe_n, seed=cfg.get("seed", 0), split="train",
                                  min_answer_words=cfg.get("min_answer_words", 0))
    examples       = all_examples[:n_examples]
    # Phase 2 · E0 screen — probe_on_train makes the probe evaluate the TRAINING
    # examples themselves (pure overfit/capacity check: can the policy beat kv_norm
    # on data it was trained on?). Default keeps the held-out left-out tail.
    if cfg.get("probe_on_train", False):
        probe_examples = examples[:probe_n]
        print(f"probe_on_train=ON → probing the first {len(probe_examples)} TRAINING examples")
    else:
        probe_examples = all_examples[n_examples:]

    print(f"train examples: {len(examples)}  probe examples: {len(probe_examples)} "
          f"(left-out train split, seed={cfg.get('seed', 0)})")
    print(f"budget_min={cfg.get('budget_min', 128)}  budget_max={cfg.get('budget_max', 256)}"
          f"  max_new_tokens={cfg.get('max_new_tokens', 524)}")

    n_parallel = cfg.get("n_parallel", 1)

    # Per-example budget calibration: optionally restrict training to examples
    # with pre-computed base budgets (T + cached_len) from the FreeGrowthCache.
    # Effective budget at runtime = base_budget - eviction_k (annealed by curriculum).
    per_example_base_budgets = None
    per_example_budgets_path = cfg.get("per_example_budgets_path", None)
    if per_example_budgets_path:
        with open(per_example_budgets_path) as f:
            raw_budgets = json.load(f)           # {str(original_idx): base_budget}
        calibrated_set = {int(k) for k in raw_budgets}
        filtered_examples    = [ex for i, ex in enumerate(examples) if i in calibrated_set]
        new_idx_map          = {old_i: new_i
                                for new_i, old_i in enumerate(
                                    i for i in range(len(examples)) if i in calibrated_set)}
        per_example_base_budgets = {new_idx_map[int(k)]: v for k, v in raw_budgets.items()
                                    if int(k) in new_idx_map}
        k_start = cfg.get("eviction_k_start", cfg.get("eviction_k", 100))
        k_end   = cfg.get("eviction_k_end",   cfg.get("eviction_k", 100))
        base_vals = list(per_example_base_budgets.values())
        print(f"per_example_base_budgets: {len(filtered_examples)}/{len(examples)} calibrated "
              f"examples  base_budget=[{min(base_vals)}, {max(base_vals)}]  "
              f"effective_budget=[{min(base_vals)-k_start}, {max(base_vals)-k_end}] "
              f"(eviction_k: {k_start}→{k_end})")
        examples = filtered_examples

    fg_cache = None
    fg_cache_dir = cfg.get("free_growth_cache_dir", None)
    if fg_cache_dir:
        model_key = cfg.get("model_name", "unknown")
        fg_cache  = FreeGrowthCache(fg_cache_dir, model_key)
        n_cached, _ = fg_cache.stats()
        print(f"free_growth_cache: {fg_cache_dir}/{model_key}  ({n_cached} cached examples)")
    else:
        print("free_growth_cache: disabled")

    env_kwargs = dict(
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
        shaping_mode=cfg.get("shaping_mode", "none"),
        n_sinks=cfg.get("n_sinks", 4),
        n_recent=cfg.get("n_recent", 8),
        length_penalty_weight=cfg.get("length_penalty_weight", 0.0),
        truncation_penalty=cfg.get("truncation_penalty", 0.0),
        entropy_reward_weight=cfg.get("entropy_reward_weight", 0.0),
        seed=cfg.get("seed", 0),
        rich_features=cfg.get("rich_features", False),
    )
    if n_parallel > 1:
        kl_shaping = cfg.get("kl_shaping", False)
        print(f"env: BatchedSharedKVVecEnv  n_parallel={n_parallel}"
              + (f"  kl_shaping={cfg.get('kl_mode', 'exact')} "
                 f"w={cfg.get('kl_weight', 0.0)} clip={cfg.get('kl_clip', 5.0)}"
                 if kl_shaping else "")
              + (f"  protect_prompt=ON" if cfg.get("protect_prompt", False) else ""))
        env = BatchedSharedKVVecEnv(
            n_parallel=n_parallel,
            free_growth_cache=fg_cache,
            kl_shaping=kl_shaping,
            kl_mode=cfg.get("kl_mode", "exact"),
            kl_weight=cfg.get("kl_weight", 0.0),
            kl_clip=cfg.get("kl_clip", 5.0),
            per_example_base_budgets=per_example_base_budgets,
            eviction_k=cfg.get("eviction_k_start", cfg.get("eviction_k", 100)),
            protect_prompt=cfg.get("protect_prompt", False),
            per_example_baseline=cfg.get("per_example_baseline", False),
            per_layer_reward=cfg.get("per_layer_reward", False),
            layer_reward_weight=cfg.get("layer_reward_weight", 0.0),
            **env_kwargs,
        )
    else:
        env = SharedKVVecEnv(**env_kwargs)
    env = VecMonitor(env)

    checkpoint_freq = cfg.get("checkpoint_freq", 50_000)
    callback_list = [
        TrainingLogger(run_dir, verbose=1),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(ckpt_dir),
            name_prefix="ckpt",
            verbose=1,
        ),
    ]

    budget_min_start  = cfg.get("budget_min_start",  None)
    budget_min_end    = cfg.get("budget_min_end",    None)
    budget_max_start  = cfg.get("budget_max_start",  None)
    budget_max_end    = cfg.get("budget_max_end",    None)
    eviction_k_start  = cfg.get("eviction_k_start",  None)
    eviction_k_end    = cfg.get("eviction_k_end",    None)
    has_curriculum = (budget_min_start is not None and budget_min_end is not None) or \
                     (eviction_k_start is not None and eviction_k_end is not None)
    if has_curriculum:
        total_ts          = cfg.get("total_timesteps", 500_000)
        curriculum_frac   = cfg.get("curriculum_fraction", 1.0)
        curriculum_end_ts = int(total_ts * curriculum_frac)
        parts = []
        if budget_min_start is not None:
            parts.append(f"budget_min: {budget_min_start}→{budget_min_end}")
        if budget_max_start is not None:
            parts.append(f"budget_max: {budget_max_start}→{budget_max_end}")
        if eviction_k_start is not None:
            parts.append(f"eviction_k: {eviction_k_start}→{eviction_k_end}")
        print(f"curriculum: {', '.join(parts)} "
              f"over first {curriculum_frac:.0%} of training ({curriculum_end_ts:,} steps)")
        callback_list.append(BudgetCurriculumCallback(
            budget_min_start=budget_min_start or cfg.get("budget_min", 256),
            budget_min_end=budget_min_end or cfg.get("budget_min", 256),
            total_timesteps=total_ts,
            curriculum_fraction=curriculum_frac,
            budget_max_start=budget_max_start,
            budget_max_end=budget_max_end,
            eviction_k_start=eviction_k_start,
            eviction_k_end=eviction_k_end,
            verbose=1,
        ))

    # Fixed held-out probe: denoised periodic monitoring (retention + eviction behavior).
    # probe_examples are the left-out tail of the training draw (loaded above).
    # Disable with probe_n: 0.
    if probe_n and probe_n > 0:
        print(f"probe: n={probe_n} budget=[{cfg.get('budget_min', 128)},{cfg.get('budget_max', 256)}] "
              f"every_n_rollouts={cfg.get('probe_every_n_rollouts', 5)} (left-out train split)")
        callback_list.append(EvalProbeCallback(
            llm=model,
            tokenizer=tokenizer,
            probe_examples=probe_examples,
            budget_min=cfg.get("budget_min", 128),
            budget_max=cfg.get("budget_max", 256),
            max_new_tokens=cfg.get("max_new_tokens", 524),
            max_len=cfg.get("max_len", 512),
            every_n_rollouts=cfg.get("probe_every_n_rollouts", 5),
            n_sinks=cfg.get("n_sinks", 4),
            n_recent=cfg.get("n_recent", 8),
            run_dir=run_dir,
            device=device,
            verbose=1,
            free_growth_cache=fg_cache,
            protect_prompt=cfg.get("protect_prompt", False),
            rich_features=cfg.get("rich_features", False),
        ))

    callbacks = CallbackList(callback_list)

    # EpisodeMaskablePPO is used when n_parallel > 1: collects complete episodes
    # per rollout (no partial-episode contamination). n_steps acts as a safety cap.
    # For n_parallel == 1 (sequential), standard MaskablePPO is used.
    ppo_class = EpisodeMaskablePPO if n_parallel > 1 else MaskablePPO
    print(f"ppo: {ppo_class.__name__}")

    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        ppo = ppo_class.load(args.resume_from, env=env, device=device)
        # Point to the correct tensorboard directory
        ppo.tensorboard_log = str(tb_dir)
    else:
        _n_extra = extra_feature_dim(
            cfg.get("rich_features", False),
            getattr(model.config, "num_key_value_heads", model.config.num_attention_heads),
        )
        _arch = cfg.get("policy_arch", "mlp")
        if _arch == "attention":
            _extractor_cls = PerTokenAttention
            _extractor_kwargs = {
                "hidden":   cfg.get("hidden", 64),
                "n_extra":  _n_extra,
                "n_heads":  cfg.get("attn_policy_heads", 4),
                "n_layers": cfg.get("attn_policy_layers", 2),
            }
        else:
            _extractor_cls = PerTokenMLP
            _extractor_kwargs = {"hidden": cfg.get("hidden", 64), "n_extra": _n_extra}
        print(f"policy_arch: {_arch}  (extractor={_extractor_cls.__name__}, n_extra={_n_extra})")
        ppo = ppo_class(
            "MlpPolicy",
            env,
            policy_kwargs={
                "features_extractor_class": _extractor_cls,
                "features_extractor_kwargs": _extractor_kwargs,
            },
            learning_rate=cfg.get("learning_rate", 3e-4),
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

    if n_parallel > 1:
        # EpisodeMaskablePPO creates its own FP16 buffer dynamically each rollout.
        # The initial buffer is never used for data — just needs to exist.
        ppo.repeats_per_problem = cfg.get("repeats_per_problem", 1)
        print(f"rollout buffer: dynamic FP16ObsMaskableRolloutBuffer (episode-based, "
              f"n_parallel={n_parallel}, n_steps_cap={ppo.n_steps}, "
              f"repeats_per_problem={ppo.repeats_per_problem})")
    else:
        # Static fp16 buffer for single-episode mode.
        ppo.rollout_buffer = FP16ObsMaskableRolloutBuffer(
            buffer_size=ppo.n_steps,
            observation_space=ppo.observation_space,
            action_space=ppo.action_space,
            device=ppo.device,
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
            n_envs=ppo.n_envs,
        )
        print(f"rollout buffer: FP16ObsMaskableRolloutBuffer "
              f"(n_steps={ppo.n_steps}, n_envs={ppo.n_envs})")

    # Phase 2 · E3 — optional warm-start: behavior-clone kv_norm before PPO so the
    # policy STARTS at the best heuristic (isolates exploration D4). Needs rich
    # features to succeed (the printed match_kv_norm should approach ~1.0).
    n_bc = int(cfg.get("warm_start_bc", 0))
    if n_bc > 0 and not args.resume_from:
        from kv_gym.warm_start import behavior_clone_kv_norm
        n_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        head_dim   = model.config.hidden_size // model.config.num_attention_heads
        print(f"warm_start_bc: cloning kv_norm for {n_bc} steps before PPO")
        behavior_clone_kv_norm(ppo, env, n_bc, n_kv_heads, head_dim, device,
                               lr=cfg.get("warm_start_lr", 1e-3))

    # Phase 2 · E5 — Golden-BC: clone the FUTURE-attention oracle (traces from
    # scripts/trace_gen.py). The oracle beats kv_norm by +7pp (FINDINGS §11);
    # this pre-phase distills its ranking into the policy before PPO.
    n_gbc = int(cfg.get("golden_bc", 0))
    if n_gbc > 0 and not args.resume_from:
        from kv_gym.warm_start import behavior_clone_golden
        traces_dir = cfg.get("golden_traces_dir", "traces")
        print(f"golden_bc: cloning the future-attention oracle for {n_gbc} steps "
              f"(traces: {traces_dir})")
        behavior_clone_golden(ppo, env, n_gbc, traces_dir, device,
                              lr=cfg.get("warm_start_lr", 1e-3))

    # On resume, SB3's _setup_learn ADDS num_timesteps to the passed budget when
    # reset_num_timesteps=False → passing the full total again would grant a fresh
    # 2M PER RESUME (s_warm ran past 2M after two spot preemptions). Pass only the
    # REMAINING budget so the cumulative cap is honored across any number of resumes.
    _total = cfg.get("total_timesteps", 500_000)
    _budget = _total - ppo.num_timesteps if args.resume_from else _total
    if _budget > 0:
        ppo.learn(
            total_timesteps=_budget,
            callback=callbacks,
            reset_num_timesteps=False if args.resume_from else True,
        )
    else:
        print(f"resume: num_timesteps={ppo.num_timesteps} >= total_timesteps={_total} "
              f"— nothing left to train, saving final model.")

    final_path = run_dir / "final_model"
    ppo.save(str(final_path))
    print(f"\nTraining complete.")
    print(f"  Final model : {final_path}.zip")
    print(f"  Best model  : {run_dir / 'best_model'}.zip")
    print(f"  Curves CSV  : {run_dir / 'learning_curve.csv'}")
    print(f"  TensorBoard : tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    main()
