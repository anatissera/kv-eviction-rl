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
        self._csv_file = open(self.run_dir / "probe_curve.csv", "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestep",
            "correct_full", "correct_random", "correct_kv_norm",
            "correct_learned", "retention",
            "evict_attn_percentile", "evict_mean_pos_frac", "evict_sink_frac",
            "evict_generated_frac", "truncation_rate",
        ])

        rng = np.random.default_rng(0)   # fixed → random anchor is reproducible
        full_vals, rand_vals, kvn_vals = [], [], []

        for ex in self.probe_examples:
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
            rand_text, _, _, _, _, _ = run_online_episode(
                self.llm, self.tokenizer, cap.input_ids, ex_budget,
                self.max_new_tokens, self.device,
                make_random_evict_fn(self.L, np.random.default_rng(rng.integers(1 << 30)),
                                     self.n_sinks, self.n_recent),
            )
            rand = float(flexible_extract(rand_text, [cap.gold_answer]))
            kvn_text, _, _, _, _, _ = run_online_episode(
                self.llm, self.tokenizer, cap.input_ids, ex_budget,
                self.max_new_tokens, self.device,
                make_kv_norm_evict_fn(self.L, self.n_sinks, self.n_recent),
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

        if not self._probes:
            logger.warning("EvalProbeCallback: no usable probe examples "
                           "(all skipped for prompt_len vs budget). Probe disabled.")
            return

        self._correct_full_mean   = float(np.mean(full_vals))
        self._correct_random_mean = float(np.mean(rand_vals))
        self._correct_kvnorm_mean = float(np.mean(kvn_vals))

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
                                         self.n_sinks, self.n_recent)
        learned, corrs, evicted = [], [], []
        gen_flags: list[float] = []   # 1.0 if the evicted token was a generated token
        truncs = []

        for p in self._probes:
            text, corr, ev, trunc, _, _ = run_online_episode(
                self.llm, self.tokenizer, p["input_ids"], p["budget"],
                self.max_new_tokens, self.device, evict_fn,
                per_layer_imp=p["per_layer_imp"],
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
