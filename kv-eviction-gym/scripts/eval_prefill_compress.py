"""REAL-dataset arena: one-shot PREFILL compression on HotpotQA-distractor.

The passkey arena proved the regime hypothesis with synthetic text (FINDINGS
25: oracle-kv_norm = +0.43). This is the upgrade to a REAL dataset with the
same structural properties (long context, mostly-irrelevant distractors,
content-distinguishable needle): HotpotQA-distractor. Each item: a question
whose answer lives in 2 gold paragraphs mixed among 8 distractors
(T ~ 900-1500 tokens).

Compression mode (Alex's "eviccion en el prefill solo", = SnapKV family):
after the prefill, each layer keeps only the top-`budget` slots ranked by a
scoring function, in ONE shot (real 3-6x compression, unlike the 1-per-step
online loop). Decode then runs with no further eviction (all arms grow
equally by the short generation).

Arms (paired per example, shared prefill, shared full-cache trace):
  full        no compression
  random      keep random slots
  kv_norm     keep highest ||K||+||V||
  attn_pre    keep highest prefill-attention-received (H2O-at-prefill)
  oracle_fut  keep highest FUTURE attention (from the full trace) = ceiling

Scoring: normalized substring match of the gold answer in the generation
(HotpotQA answers are entities/short spans; yes/no items are filtered out).

Resume-safe JSONL. Usage:
  python scripts/eval_prefill_compress.py --n 96 --budget 256 \
      --max-ctx 1400 --out-dir runs/hotpot_eval
"""
import argparse
import json
import re
import string
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_eval import full_trace_with_future_attention  # noqa: E402
from kv_gym.vendor.loader import load_model_and_tokenizer  # noqa: E402


def norm_text(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def answer_in(pred: str, gold: str) -> float:
    return 1.0 if norm_text(gold) in norm_text(pred) else 0.0


def load_hotpot(n: int, seed: int, tokenizer, max_ctx: int):
    """n distractor-mode items, yes/no filtered, context capped at max_ctx tokens."""
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    ds = ds.shuffle(seed=seed)
    out = []
    for row in ds:
        ans = row["answer"].strip()
        if ans.lower() in ("yes", "no") or len(ans) < 3:
            continue
        paras = ["".join(s) for s in row["context"]["sentences"]]
        ctx = "\n".join(f"Paragraph {i+1}: {p}" for i, p in enumerate(paras))
        prompt = (f"{ctx}\n\nQuestion: {row['question']}\n"
                  f"Answer with the exact short answer only.")
        n_tok = len(tokenizer(prompt)["input_ids"])
        if n_tok > max_ctx or n_tok < 500:
            continue
        out.append({"prompt_text": prompt, "gold": ans, "T_est": n_tok})
        if len(out) >= n:
            break
    return out


def _get_kv(cache, l):
    """Version-portable DynamicCache access (same pattern as batched_env)."""
    if hasattr(cache, "layers"):
        return cache.layers[l].keys, cache.layers[l].values
    return cache.key_cache[l], cache.value_cache[l]


@torch.no_grad()
def compress_cache(past_kv, keep_idx_per_layer):
    """Return a new DynamicCache with only the kept slots per layer."""
    from transformers import DynamicCache
    new = DynamicCache()
    L = len(keep_idx_per_layer)
    for l in range(L):
        K, V = _get_kv(past_kv, l)   # [1, H, T, D]
        idx = keep_idx_per_layer[l]
        new.update(K[:, :, idx, :].contiguous(), V[:, :, idx, :].contiguous(), l)
    return new


@torch.no_grad()
def decode(model, tokenizer, cache, next_tok, start_pos, max_new):
    eos = tokenizer.eos_token_id
    toks = []
    pos = start_pos
    for _ in range(max_new):
        out = model(input_ids=torch.tensor([[next_tok]], device=model.device),
                    past_key_values=cache,
                    position_ids=torch.tensor([[pos]], device=model.device),
                    use_cache=True)
        cache = out.past_key_values
        toks.append(next_tok)
        next_tok = int(out.logits[0, -1].argmax())
        pos += 1
        if next_tok == eos:
            break
    return tokenizer.decode(toks, skip_special_tokens=True)


def main():
    torch.set_grad_enabled(False)   # pure-eval script
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=96)
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--max-ctx", type=int, default=1400)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-sinks", type=int, default=4)
    p.add_argument("--n-recent", type=int, default=32)
    p.add_argument("--out-dir", default="runs/hotpot_eval")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm, tokenizer, device = load_model_and_tokenizer(
        name="qwen-1.5b", device=device, attn_implementation="eager")
    L = llm.config.num_hidden_layers

    examples = load_hotpot(args.n, args.seed, tokenizer, args.max_ctx)
    print(f"hotpot: {len(examples)} items, budget={args.budget} "
          f"(compression ~{np.mean([e['T_est'] for e in examples])/args.budget:.1f}x)")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "hotpot_results.jsonl"
    done = set()
    if results_path.exists():
        for line in open(results_path):
            try: done.add(json.loads(line)["idx"])
            except Exception: pass
        print(f"resume: {len(done)}")

    rng = np.random.default_rng(args.seed)
    for idx, ex in enumerate(examples):
        if idx in done:
            continue
        try:
            # chat-template the prompt the same way capture() does for GSM8K
            msgs = [{"role": "user", "content": ex["prompt_text"]}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
            ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            T = ids.shape[1]

            # one full-cache trace: gives full-arm answer + future attention
            full_text, suf, n_steps = full_trace_with_future_attention(
                llm, tokenizer, ids, device, args.max_new_tokens)
            c_full = answer_in(full_text, ex["gold"])

            # prefill once more to get a clean cache + prefill attentions
            pre = llm(input_ids=ids, use_cache=True, output_attentions=True)
            att_pre = [a[0].mean(0).sum(0).float().cpu().numpy()
                       for a in pre.attentions]        # [T] received attn per layer
            first_tok = int(pre.logits[0, -1].argmax())

            protected = set(range(args.n_sinks)) | set(range(T - args.n_recent, T))
            k_free = args.budget - len(protected)

            def keep_from_scores(score_per_layer):
                keeps = []
                for l in range(L):
                    s = np.asarray(score_per_layer[l], dtype=np.float64).copy()
                    s[list(protected)] = np.inf     # always keep sinks+recent
                    order = np.argsort(-s)          # highest score kept
                    keep = np.sort(order[:args.budget])
                    keeps.append(torch.tensor(keep, device=device, dtype=torch.long))
                return keeps

            K_norms = []
            for l in range(L):
                Kl, Vl = _get_kv(pre.past_key_values, l)
                K_norms.append((Kl[0].norm(dim=-1) + Vl[0].norm(dim=-1))
                               .mean(0).float().cpu().numpy())
            scores = {
                "random":     [rng.random(T) for _ in range(L)],
                "kv_norm":    K_norms,
                "attn_pre":   att_pre,
                "oracle_fut": [suf[l][0, :T].astype(np.float64) for l in range(L)],
            }
            row = {"idx": idx, "T": T, "full": c_full, "steps_full": n_steps}
            for name, sc in scores.items():
                cache = compress_cache(pre.past_key_values, keep_from_scores(sc))
                txt = decode(llm, tokenizer, cache, first_tok, T, args.max_new_tokens)
                row[name] = answer_in(txt, ex["gold"])
            del suf, pre
            print(f"[{idx}] T={T} full={row['full']:.0f} rnd={row['random']:.0f} "
                  f"kv={row['kv_norm']:.0f} attn={row['attn_pre']:.0f} "
                  f"orc={row['oracle_fut']:.0f}")
        except Exception as e:  # noqa: BLE001
            row = {"idx": idx, "error": str(e)[:200]}
            print(f"[{idx}] ERROR {e}")
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    rows = [json.loads(l) for l in open(results_path)]
    ok = [r for r in rows if "error" not in r]
    if ok:
        n = len(ok)
        print(f"\n==== HOTPOT PREFILL-COMPRESSION ({n} items, budget={args.budget}) ====")
        for k in ("full", "oracle_fut", "attn_pre", "kv_norm", "random"):
            print(f"  {k:10s} {sum(r[k] for r in ok)/n:.4f}")
        d = sum(r["oracle_fut"] - r["kv_norm"] for r in ok) / n
        print(f"  PAIRED oracle - kv_norm = {d:+.4f}")


if __name__ == "__main__":
    main()
