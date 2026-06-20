"""
Evaluate a trained eviction policy against random and oracle baselines.

Three strategies compared on a held-out set of GSM8K examples:

  learned  — the trained MaskablePPO policy
  oracle   — keep the top-budget tokens by ||K|| + ||V|| norm (greedy heuristic)
  random   — keep a random subset of budget tokens

All strategies scored by GSM8K answer correctness (flexible_extract).

Usage:
    python scripts/eval.py --model checkpoints/quickstart --config configs/quickstart.yaml
"""

import argparse
import numpy as np
import torch
import yaml

from sb3_contrib import MaskablePPO

from kv_gym.capture import capture
from kv_gym.features import build_obs, feature_dim
from kv_gym.vendor.loader import load_model_and_tokenizer
from kv_gym.vendor.gsm8k import load_gsm8k
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, help="Path to saved MaskablePPO checkpoint")
    p.add_argument("--config", default="configs/quickstart.yaml")
    p.add_argument("--n",      type=int, default=50, help="Number of eval examples")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


def score_with_mask(model, tokenizer, input_ids, attn_mask, gold, device, max_new_tokens):
    """Generate with an eviction mask and return correctness score."""
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attn_mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    T    = input_ids.shape[1]
    text = tokenizer.decode(out[0, T:], skip_special_tokens=True)
    return flexible_extract(text, [gold])


def oracle_mask(K, V, budget, T):
    """Keep top-budget tokens by combined ||K||+||V|| norm averaged across heads."""
    score = (K.norm(dim=-1) + V.norm(dim=-1)).mean(dim=(0, 1))  # [T]
    _, topk = score.topk(min(budget, T))
    mask = torch.zeros(1, T, dtype=torch.long)
    mask[0, topk] = 1
    return mask


def random_mask(budget, T, rng):
    kept = rng.choice(T, size=min(budget, T), replace=False)
    mask = torch.zeros(1, T, dtype=torch.long)
    mask[0, kept] = 1
    return mask


def main():
    args   = parse_args()
    cfg    = yaml.safe_load(open(args.config))
    rng    = np.random.default_rng(args.seed)

    device_cfg = cfg.get("device", "auto")
    device = None if device_cfg == "auto" else torch.device(device_cfg)
    llm, tokenizer, device = load_model_and_tokenizer(
        name=cfg.get("model_name", "qwen-1.5b"),
        device=device,
        attn_implementation=cfg.get("attn_implementation", None),
    )

    budget         = cfg.get("budget", 32)
    max_new_tokens = cfg.get("max_new_tokens", 64)
    max_len        = cfg.get("max_len", 256)
    examples       = load_gsm8k(n=args.n, seed=args.seed)

    L      = llm.config.num_hidden_layers
    H      = getattr(llm.config, "num_key_value_heads", llm.config.num_attention_heads)
    D      = llm.config.hidden_size // llm.config.num_attention_heads
    n_envs = L * H
    fdim   = feature_dim(D)
    head_indices = [(l, h) for l in range(L) for h in range(H)]

    policy = MaskablePPO.load(args.model, device=device)

    learned_scores, oracle_scores, random_scores = [], [], []

    for ex in examples:
        cap = capture(llm, tokenizer, ex, device)
        T   = cap.prompt_len

        K_flat = cap.K.view(n_envs, T, D)
        V_flat = cap.V.view(n_envs, T, D)

        # ---------- learned ----------
        resident = torch.ones(L, H, T, dtype=torch.bool)
        for _ in range(T - budget):
            obs   = build_obs(K_flat, V_flat, T, max_len)
            masks = np.zeros((n_envs, max_len), dtype=bool)
            masks[:, :T] = resident.view(n_envs, T).numpy()
            actions, _ = policy.predict(obs, action_masks=masks, deterministic=True)
            for env_idx, (l, h) in enumerate(head_indices):
                tok = int(actions[env_idx])
                if tok < T and resident[l, h, tok]:
                    resident[l, h, tok] = False

        token_scores = resident.float().mean(dim=(0, 1))
        _, topk = token_scores.topk(min(budget, T))
        learned_mask = torch.zeros(1, T, dtype=torch.long)
        learned_mask[0, topk] = 1

        learned_scores.append(score_with_mask(
            llm, tokenizer, cap.input_ids, learned_mask,
            cap.gold_answer, device, max_new_tokens,
        ))
        oracle_scores.append(score_with_mask(
            llm, tokenizer, cap.input_ids,
            oracle_mask(cap.K, cap.V, budget, T),
            cap.gold_answer, device, max_new_tokens,
        ))
        random_scores.append(score_with_mask(
            llm, tokenizer, cap.input_ids,
            random_mask(budget, T, rng),
            cap.gold_answer, device, max_new_tokens,
        ))

    print(f"\nResults over {len(examples)} examples  (budget={budget}):")
    print(f"{'strategy':<10} {'mean':>6}  {'std':>6}")
    print("-" * 26)
    for name, vals in [("learned", learned_scores), ("oracle", oracle_scores), ("random", random_scores)]:
        print(f"{name:<10} {np.mean(vals):>6.3f}  {np.std(vals):>6.3f}")
    print(f"\nLearned / Oracle: {np.mean(learned_scores) / max(np.mean(oracle_scores), 1e-8):.3f}")
    print(f"Learned / Random: {np.mean(learned_scores) / max(np.mean(random_scores), 1e-8):.3f}")


if __name__ == "__main__":
    main()
