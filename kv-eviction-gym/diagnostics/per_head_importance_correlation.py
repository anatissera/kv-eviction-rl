"""
Diagnostic: do the 2 KV-heads per layer agree on which tokens are important?

Uses actual decode-time attention weights (not K-norm proxy):
  - Run model.generate() with output_attentions=True (requires eager attention)
  - For each decode step and each layer, attention is [1, H_q, 1, seq]
  - Apply GQA grouping: max over Q-heads within each KV group → [H_kv, T]
  - Accumulate over decode steps, normalise → per-head token importance [L, H_kv, T]
  - Compute Spearman rank correlation between KV-head 0 and KV-head 1 per layer

High correlation (>0.8) → heads agree → per-layer eviction (28 agents) is justified.
Low correlation        → heads specialise → per-head eviction (56 agents) is warranted.

Usage:
    cd kv-eviction-gym
    python diagnostics/per_head_importance_correlation.py [--n N] [--max_new_tokens K]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from kv_gym.capture import capture
from kv_gym.vendor.gsm8k import load_gsm8k
from kv_gym.vendor.loader import load_model_and_tokenizer


def per_head_importance_from_attentions(
    attentions,   # tuple[n_steps] of tuple[n_layers] of Tensor[1, H_q, 1, seq]
    T: int,
    L: int,
    H_kv: int,
    H_q: int,
) -> torch.Tensor:
    """Accumulate per-KV-head token importance from generate() attention output.

    Returns [L, H_kv, T] tensor, normalised per (layer, head) to sum to 1.
    """
    group_size = H_q // H_kv
    importance = torch.zeros(L, H_kv, T)
    n_terms = 0

    for step_attns in attentions:
        for l, layer_attn in enumerate(step_attns):
            if layer_attn is None or l >= L:
                continue
            # layer_attn: [1, H_q, 1, seq_len]  — query is the new decode token
            seq_len  = layer_attn.shape[-1]
            n_prompt = min(T, seq_len)

            per_q = layer_attn[0, :, 0, :n_prompt].cpu()  # [H_q, T]

            # Max within each KV-head's Q-group — same as SnapKV / attention_shaping.py
            # but kept per KV-head instead of averaged.
            per_kv = per_q.view(H_kv, group_size, n_prompt).amax(dim=1)  # [H_kv, T]
            importance[l, :, :n_prompt] += per_kv

        n_terms += 1

    if n_terms > 0:
        importance /= n_terms

    # Normalise each (layer, head) distribution to sum to 1
    totals = importance.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return importance / totals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",              type=int, default=20, help="Number of GSM8K examples")
    p.add_argument("--max_new_tokens", type=int, default=64, help="Decode steps per example")
    args = p.parse_args()

    print("Loading model (eager attention — required for output_attentions=True)...")
    model, tokenizer, device = load_model_and_tokenizer(
        name="qwen-1.5b",
        attn_implementation="eager",
    )
    model.eval()
    print(f"Device: {device}")

    L    = model.config.num_hidden_layers
    H_q  = model.config.num_attention_heads
    H_kv = getattr(model.config, "num_key_value_heads", H_q)
    print(f"Model: {L} layers, {H_q} Q-heads, {H_kv} KV-heads (group size {H_q // H_kv})\n")

    examples = load_gsm8k(n=args.n, seed=0, split="train")

    layer_corrs: list[list[float]] = [[] for _ in range(L)]

    for i, ex in enumerate(examples):
        cap = capture(model, tokenizer, ex, device)
        T   = cap.prompt_len

        with torch.no_grad():
            out = model.generate(
                input_ids=cap.input_ids.to(device),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        if not (hasattr(out, "attentions") and out.attentions):
            print(f"  Example {i+1}: no attention weights returned — check attn_implementation.")
            continue

        importance = per_head_importance_from_attentions(
            out.attentions, T=T, L=L, H_kv=H_kv, H_q=H_q,
        )  # [L, H_kv, T]

        for l in range(L):
            head0 = importance[l, 0, :].numpy()
            head1 = importance[l, 1, :].numpy()
            rho, _ = spearmanr(head0, head1)
            layer_corrs[l].append(float(rho))

        print(f"Example {i+1}/{args.n} (T={T} tokens, {out.sequences.shape[1] - T} new) — done")

    mean_per_layer = np.array([np.mean(c) for c in layer_corrs])
    std_per_layer  = np.array([np.std(c)  for c in layer_corrs])

    print("\n--- Per-layer Spearman ρ (KV-head 0 vs KV-head 1, actual attention weights) ---")
    print(f"{'Layer':>6}  {'mean ρ':>8}  {'std ρ':>7}")
    for l in range(L):
        print(f"{l:>6}  {mean_per_layer[l]:>8.3f}  {std_per_layer[l]:>7.3f}")

    overall_mean = mean_per_layer.mean()
    overall_std  = mean_per_layer.std()
    print(f"\nOverall mean ρ across all layers: {overall_mean:.3f} ± {overall_std:.3f}")

    if overall_mean > 0.8:
        print("\nConclusion: heads are highly correlated — per-layer eviction (28 agents) is justified.")
    elif overall_mean > 0.5:
        print("\nConclusion: moderate correlation — borderline, could go either way.")
    else:
        print("\nConclusion: low correlation — heads specialise, per-head eviction (56 agents) is warranted.")


if __name__ == "__main__":
    main()
