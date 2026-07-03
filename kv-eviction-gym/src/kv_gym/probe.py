"""
EvalProbeCallback — denoised training-time monitoring on a FIXED held-out problem set.

The per-rollout `correctness_rate` logged by TrainingLogger is noisy: each rollout sees a
different mix of GSM8K problems (different intrinsic difficulty) AND a stochastic policy, so
problem-difficulty variance is tangled with policy learning.  This callback removes that:
every `every_n_rollouts` rollouts it runs the CURRENT policy (deterministic) over the SAME
held-out problems, so the only thing changing between probes is the policy.

Two design choices denoise it further:

  * Retention, not raw correctness.  `probe/retention = correct_learned / correct_full`
    computed only over the problems the FULL cache already solves — intrinsically-failed
    problems (which eviction can't fix) are excluded.  full / random / kv_norm correctness
    are policy-independent and computed ONCE as fixed anchors.

  * Continuous behavior metrics that move smoothly and lead correctness:
      probe/evict_attn_percentile — attention-rank percentile of evicted tokens
                                    (0 = evicts least-attended like the oracle, 0.5 = random)
      probe/evict_mean_pos_frac   — recency of evictions (orig_pos / true_pos, 0=old, 1=recent)
      probe/evict_sink_frac       — fraction of evictions hitting sink positions (<n_sinks)

Everything is logged to TensorBoard (`probe/*`) and appended to `<run_dir>/probe_curve.csv`.
The probe builds its own caches via fresh prefills, so it never touches the training env's
episode state.  Gradients never flow (torch.no_grad via run_online_episode / generate).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from kv_gym.capture import capture
from kv_gym.eval_core import (
    capture_per_layer_attention,
    make_kv_norm_evict_fn,
    make_learned_evict_fn,
    make_random_evict_fn,
    run_online_episode,
    score_full_cache,
)
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract

logger = logging.getLogger(__name__)


class EvalProbeCallback(BaseCallback):
    # Probe examples use cache keys offset by this value so they never collide
    # with training example keys (training has at most a few thousand examples).
    PROBE_CACHE_OFFSET = 1_000_000

    def __init__(
        self,
        llm,
        tokenizer,
        probe_examples: list[dict],
        budget_min:       int,
        budget_max:       int,
        max_new_tokens:   int,
        max_len:          int,
        every_n_rollouts: int = 5,
        n_sinks:          int = 4,
        n_recent:         int = 8,
        run_dir:          Path | str = ".",
        device:           torch.device | None = None,
        ref_max_new_tokens: int = 64,
        verbose:          int = 1,
        free_growth_cache = None,  # FreeGrowthCache | None
        protect_prompt:   bool = False,
        rich_features:    bool = False,
    ):
        super().__init__(verbose)
        self.llm              = llm
        self.tokenizer        = tokenizer
        self.probe_examples   = probe_examples
        self.budget_min       = budget_min
        self.budget_max       = budget_max
        self.max_new_tokens   = max_new_tokens
        self.max_len          = max_len
        self.every_n_rollouts = max(1, every_n_rollouts)
        self.n_sinks          = n_sinks
        self.n_recent         = n_recent
        self.run_dir          = Path(run_dir)
        self.device           = device or next(llm.parameters()).device
        self.ref_max_new_tokens = ref_max_new_tokens
        self.free_growth_cache  = free_growth_cache
        self.protect_prompt     = protect_prompt
        self.rich_features      = rich_features

        self.L = llm.config.num_hidden_layers

        # Filled in _on_training_start (the fixed probe set + anchors)
        self._probes: list[dict] = []      # one dict per usable probe example
        self._correct_full_mean   = float("nan")
        self._correct_random_mean = float("nan")
        self._correct_kvnorm_mean = float("nan")
        self._attn_available      = False

        self._rollout_count = 0
        self._best_retention = float("-inf")
        self._csv_file = None
        self._csv_writer = None

    # ------------------------------------------------------------------
    def _on_training_start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.run_dir / "probe_curve.csv"
        # Write the header unless the file already has a VALID one (resume append).
        # Guards the case where a preempted run created/append-opened the file but
        # never wrote a header → parsers saw a headerless CSV (the warm-start bug).
        has_valid_header = False
        if csv_path.exists() and csv_path.stat().st_size > 0:
            with open(csv_path) as _f:
                has_valid_header = _f.readline().startswith("timestep")
        self._csv_file = open(csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if not has_valid_header:
            self._csv_writer.writerow([
            "timestep",
            "correct_full", "correct_random", "correct_kv_norm",
            "correct_learned", "retention",
            "evict_attn_percentile", "evict_mean_pos_frac", "evict_sink_frac",
            "evict_generated_frac", "truncation_rate",
        ])

        
        import pickle
        anchors_path = self.run_dir / "probe_anchors.pkl"
        if anchors_path.exists():
            logger.info(f"Loading existing probe anchors from {anchors_path}")
            with open(anchors_path, "rb") as f:
                data = pickle.load(f)
            self._probes = data["probes"]
            self._correct_full_mean = data["full_mean"]
            self._correct_random_mean = data["random_mean"]
            self._correct_kvnorm_mean = data["kvnorm_mean"]
            self._attn_available = data["attn_available"]
            return

        rng = np.random.default_rng(0)   # fixed → random anchor is reproducible
        full_vals, rand_vals, kvn_vals = [], [], []

        # Partial-anchor checkpointing: anchors on a wide/long probe take ~2h and
        # were only cached when COMPLETE, so on churny spot capacity a preemption
        # mid-anchors restarted the whole phase (observed 3× on 2026-07-03).
        # Persist progress every 8 examples; a restart loses minutes, not hours.
        partial_path = self.run_dir / "probe_anchors_partial.pkl"
        start_idx = 0
        if partial_path.exists():
            try:
                with open(partial_path, "rb") as f:
                    part = pickle.load(f)
                start_idx = part["done_idx"]
                self._probes = part["probes"]
                full_vals, rand_vals, kvn_vals = part["full_vals"], part["rand_vals"], part["kvn_vals"]
                self._attn_available = part["attn_available"]
                rng.bit_generator.state = part["rng_state"]
                logger.info(f"Resuming anchor build from example {start_idx} "
                            f"({len(self._probes)} anchors so far)")
            except Exception as _e:
                logger.warning(f"partial anchors unreadable ({_e}); rebuilding from 0")
                start_idx = 0
                self._probes, full_vals, rand_vals, kvn_vals = [], [], [], []

        for ex_i, ex in enumerate(self.probe_examples):
            if ex_i < start_idx:
                continue
            if ex_i > start_idx and (ex_i - start_idx) % 8 == 0:
                with open(partial_path, "wb") as f:
                    pickle.dump({"done_idx": ex_i, "probes": self._probes,
                                 "full_vals": full_vals, "rand_vals": rand_vals,
                                 "kvn_vals": kvn_vals,
                                 "attn_available": self._attn_available,
                                 "rng_state": rng.bit_generator.state}, f)
            try:
                cap = capture(self.llm, self.tokenizer, ex, self.device)
                T = cap.prompt_len

                ex_budget = int(rng.integers(self.budget_min, self.budget_max + 1))

                # Skip prompts that can't be evicted (too long, or already <= budget).
                if T > self.max_len or T >= ex_budget:
                    continue

                per_layer_imp = capture_per_layer_attention(
                    self.llm, cap.input_ids, self.device, self.ref_max_new_tokens,
                )
                if per_layer_imp is not None:
                    self._attn_available = True

                full_val, _, _ = score_full_cache(
                    self.llm, self.tokenizer, cap.input_ids, cap.gold_answer,
                    self.device, self.max_new_tokens,
                )
                full = float(full_val)
                probe_idx = len(self._probes)
                cache_key = self.PROBE_CACHE_OFFSET + probe_idx
                rand_text, _, _, _, _, _ = run_online_episode(
                    self.llm, self.tokenizer, cap.input_ids, ex_budget,
                    self.max_new_tokens, self.device,
                    make_random_evict_fn(self.L, np.random.default_rng(rng.integers(1 << 30)),
                                         self.n_sinks, self.n_recent),
                    free_growth_cache=self.free_growth_cache,
                    example_idx=cache_key,
                    protect_prompt=self.protect_prompt,
                )
                rand = float(flexible_extract(rand_text, [cap.gold_answer]))
                kvn_text, _, _, _, _, _ = run_online_episode(
                    self.llm, self.tokenizer, cap.input_ids, ex_budget,
                    self.max_new_tokens, self.device,
                    make_kv_norm_evict_fn(self.L, self.n_sinks, self.n_recent),
                    free_growth_cache=self.free_growth_cache,
                    example_idx=cache_key,
                    protect_prompt=self.protect_prompt,
                )
                kvn = float(flexible_extract(kvn_text, [cap.gold_answer]))

                self._probes.append({
                    "input_ids":     cap.input_ids,
                    "gold":          cap.gold_answer,
                    "per_layer_imp": per_layer_imp,
                    "full":          full,
                    "T":             T,
                    "budget":        ex_budget,
                })
                full_vals.append(full); rand_vals.append(rand); kvn_vals.append(kvn)
            except Exception as _probe_ex:
                logger.warning(f"EvalProbeCallback: skipping probe example {len(self._probes)} "
                               f"due to error: {_probe_ex}")

        if not self._probes:
            logger.warning("EvalProbeCallback: no usable probe examples "
                           "(all skipped for prompt_len vs budget). Probe disabled.")
            return

        self._correct_full_mean   = float(np.mean(full_vals))
        self._correct_random_mean = float(np.mean(rand_vals))
        self._correct_kvnorm_mean = float(np.mean(kvn_vals))

        with open(anchors_path, "wb") as f:
            pickle.dump({
                "probes": self._probes,
                "full_mean": self._correct_full_mean,
                "random_mean": self._correct_random_mean,
                "kvnorm_mean": self._correct_kvnorm_mean,
                "attn_available": self._attn_available
            }, f)
        partial_path.unlink(missing_ok=True)


        if self.verbose:
            print(f"[probe] {len(self._probes)} fixed examples | "
                  f"anchors: full={self._correct_full_mean:.3f} "
                  f"random={self._correct_random_mean:.3f} "
                  f"kv_norm={self._correct_kvnorm_mean:.3f} | "
                  f"attn_oracle={'on' if self._attn_available else 'off'}")

    # ------------------------------------------------------------------
    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if not self._probes or (self._rollout_count % self.every_n_rollouts != 0):
            return

        evict_fn = make_learned_evict_fn(self.model, self.L, self.max_len,
                                         self.n_sinks, self.n_recent,
                                         rich_features=self.rich_features)
        learned, corrs, evicted = [], [], []
        gen_flags: list[float] = []   # 1.0 if the evicted token was a generated token
        truncs = []

        for pi, p in enumerate(self._probes):
            text, corr, ev, trunc, _, _ = run_online_episode(
                self.llm, self.tokenizer, p["input_ids"], p["budget"],
                self.max_new_tokens, self.device, evict_fn,
                per_layer_imp=p["per_layer_imp"],
                free_growth_cache=self.free_growth_cache,
                example_idx=self.PROBE_CACHE_OFFSET + pi,
                protect_prompt=self.protect_prompt,
            )
            learned.append(float(flexible_extract(text, [p["gold"]])))
            corrs.extend(corr)
            evicted.extend(ev)
            truncs.append(float(trunc))
            # generated = original position >= this example's prompt length
            gen_flags.extend(1.0 if op >= p["T"] else 0.0 for op, _ in ev)

        correct_learned = float(np.mean(learned))
        trunc_rate = float(np.mean(truncs))

        full_correct_idx = [i for i, p in enumerate(self._probes) if p["full"] > 0.5]
        retention = (float(np.mean([learned[i] for i in full_correct_idx]))
                     if full_correct_idx else float("nan"))

        attn_pct = float(np.mean(corrs)) if corrs else float("nan")
        if evicted:
            mean_pos_frac = float(np.mean([op / max(tp, 1) for op, tp in evicted]))
            sink_frac     = float(np.mean([1.0 if op < self.n_sinks else 0.0
                                           for op, _ in evicted]))
            gen_frac      = float(np.mean(gen_flags))
        else:
            mean_pos_frac = float("nan")
            sink_frac     = float("nan")
            gen_frac      = float("nan")

        # TensorBoard
        rec = self.logger.record
        rec("probe/retention", retention)
        rec("probe/correct_learned", correct_learned)
        rec("probe/correct_full", self._correct_full_mean)
        rec("probe/correct_random", self._correct_random_mean)
        rec("probe/correct_kv_norm", self._correct_kvnorm_mean)
        rec("probe/evict_attn_percentile", attn_pct)
        rec("probe/evict_mean_pos_frac", mean_pos_frac)
        rec("probe/evict_sink_frac", sink_frac)
        rec("probe/evict_generated_frac", gen_frac)
        rec("probe/truncation_rate", trunc_rate)

        # CSV
        self._csv_writer.writerow([
            self.num_timesteps,
            f"{self._correct_full_mean:.4f}", f"{self._correct_random_mean:.4f}",
            f"{self._correct_kvnorm_mean:.4f}",
            f"{correct_learned:.4f}", f"{retention:.4f}",
            f"{attn_pct:.4f}", f"{mean_pos_frac:.4f}", f"{sink_frac:.4f}",
            f"{gen_frac:.4f}", f"{trunc_rate:.4f}",
        ])
        self._csv_file.flush()

        if self.verbose:
            ret_str = f"{retention:.3f}" if retention == retention else "n/a"
            pct_str = f"{attn_pct:.3f}" if attn_pct == attn_pct else "n/a"
            print(f"[probe] t={self.num_timesteps:>9,}  retention={ret_str}  "
                  f"learned={correct_learned:.3f} (full={self._correct_full_mean:.3f}, "
                  f"rand={self._correct_random_mean:.3f})  evict_attn_pct={pct_str}  "
                  f"gen_frac={gen_frac:.2f}  pos_frac={mean_pos_frac:.2f}  "
                  f"trunc={trunc_rate:.1%}")

        # Save best-by-retention (ignore NaN)
        if retention == retention and retention > self._best_retention:
            self._best_retention = retention
            self.model.save(str(self.run_dir / "best_probe_model"))

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()
