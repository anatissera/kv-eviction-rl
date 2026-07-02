"""Oracle-gap experiment: how much CAN any eviction policy beat kv_norm here?

Measures, on the same fresh wide slice as scripts/wide_eval.py and with all
arms paired per example under ONE model instance (eager attention):

  full         — no eviction (ceiling; from the trace pass)
  kv_norm      — evict lowest ||K||+||V|| (the heuristic to beat)
  attn_cur     — evict lowest CURRENT accumulated attention (H2O-family
                 present-information heuristic; uses eval_core's scorer)
  oracle_fut   — GOLDEN EVICTION (ForesightKV, arXiv 2602.03203): evict the slot
                 with the lowest MAX FUTURE attention, computed from the
                 full-cache trace (uses information no online policy can have)

Reading:
  oracle_fut − kv_norm ≈ 0  → kv_norm is near-optimal in this regime: the
    parity results (FINDINGS §7-§10) are not an RL failure — there is provably
    almost nothing to learn. Case closed.
  oracle_fut − kv_norm >> 0 → that margin is learnable in principle → invest
    in offline oracle supervision (E5 Golden-BC).

Caveat (inherent to golden eviction, same as the literature): future attention
comes from the FULL-cache trajectory; once evictions change the generated text
the labels are approximate. In this regime divergence is small (full−random
≈ 4pp). Keys created beyond the trace (divergent generation) get +inf score
(never evicted — they are recent and mostly inside the n_recent window anyway).

Resume-safe: appends one JSON line per example to <out>/oracle_results.jsonl
and skips already-done indices on restart.

Usage:
  python scripts/oracle_eval.py --config configs/e1_rich.yaml --n 128 \
      --out-dir runs/oracle_eval
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kv_gym.capture import capture
from kv_gym.eval_core import (
    PositionTracker, _valid_slots, run_online_episode,
    make_kv_norm_evict_fn, make_attention_evict_fn, score_full_cache,
)
from kv_gym.vendor.loader import load_model_and_tokenizer
from kv_gym.vendor.gsm8k import load_gsm8k
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract


@torch.no_grad()
def full_trace_with_future_attention(model, tokenizer, input_ids, device,
                                     max_new_tokens):
    """Greedy full-cache decode capturing per-layer head-max attention per step.

    Returns (text, per_layer_suffix_max, n_steps) where per_layer_suffix_max[l]
    is a [n_steps, T+n_steps] float16 array: SUF[t, j] = max over decode steps
    t' >= t (and heads) of attention from query t' to key j. Score for evicting
    key j when the episode is at generated-step t is SUF[t, j].
    """
    L = model.config.num_hidden_layers
    T = input_ids.shape[1]
    eos = tokenizer.eos_token_id

    out = model(input_ids=input_ids.to(device), use_cache=True, output_attentions=True)
    past = out.past_key_values
    next_tok = int(out.logits[0, -1].argmax())

    # step_attn[l][t] = np[keys_at_step_t] head-max attention of decode step t
    step_attn: list[list[np.ndarray]] = [[] for _ in range(L)]
    generated: list[int] = []

    for _t in range(max_new_tokens):
        ids = torch.tensor([[next_tok]], device=device)
        out = model(input_ids=ids, past_key_values=past, use_cache=True,
                    output_attentions=True)
        past = out.past_key_values
        for l in range(L):
            # attentions[l]: [1, H, 1, keys] → head-max → [keys]
            a = out.attentions[l][0, :, 0, :].amax(dim=0).float().cpu().numpy()
            step_attn[l].append(a.astype(np.float16))
        generated.append(next_tok)
        next_tok = int(out.logits[0, -1].argmax())
        if next_tok == eos:
            break

    n_steps = len(generated)
    n_keys = T + n_steps
    suf = []
    for l in range(L):
        M = np.zeros((n_steps, n_keys), dtype=np.float16)
        for t, a in enumerate(step_attn[l]):
            M[t, :a.shape[0]] = a
        # suffix max over steps (backwards)
        for t in range(n_steps - 2, -1, -1):
            np.maximum(M[t], M[t + 1], out=M[t])
        suf.append(M)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, suf, n_steps


def make_oracle_evict_fn(suf, L, T, n_sinks, n_recent):
    """Golden eviction: lowest max-FUTURE-attention among valid slots.

    Current generated-step index is inferred from the newest resident original
    position (the newest token is inside the protected n_recent window, so it
    is always resident). Keys beyond the trace get +inf (never evicted).
    """
    n_steps = suf[0].shape[0]
    n_keys = suf[0].shape[1]

    def evict(K, V, tracker, scores=None, n_prompt=0):
        slots = []
        for l in range(L):
            pos = tracker._pos[l]
            cur_step = min(max(max(pos) - T, 0), n_steps - 1)
            cand = _valid_slots(len(pos), n_sinks, n_recent, n_prompt)
            row = suf[l][cur_step]
            vals = [float(row[pos[s]]) if pos[s] < n_keys else float("inf")
                    for s in cand]
            slots.append(cand[int(np.argmin(vals))])
        return slots
    evict.needs_kv = False
    return evict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--out-dir", default="runs/oracle_eval")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EAGER attention: required for output_attentions (trace + attn_cur scorer).
    llm, tokenizer, device = load_model_and_tokenizer(
        name=cfg.get("model_name", "qwen-1.5b"),
        device=device,
        attn_implementation="eager",
    )
    L = llm.config.num_hidden_layers

    n_examples = cfg.get("n_examples", 1000)
    probe_n    = cfg.get("probe_n", 32)
    seed       = cfg.get("seed", 0)
    budget     = cfg.get("budget_min", 256)
    max_new    = cfg.get("max_new_tokens", 600)
    max_len    = cfg.get("max_len", 288)
    n_sinks    = cfg.get("n_sinks", 4)
    n_recent   = cfg.get("n_recent", 32)

    all_examples = load_gsm8k(n=n_examples + probe_n + args.n, seed=seed, split="train")
    wide = all_examples[n_examples + probe_n:]
    print(f"oracle eval: {len(wide)} examples, budget={budget}, "
          f"sinks={n_sinks} recent={n_recent} (eager)")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "oracle_results.jsonl"
    done = set()
    if results_path.exists():
        for line in open(results_path):
            try: done.add(json.loads(line)["idx"])
            except Exception: pass
        print(f"resume: {len(done)} examples already done")

    kv_fn   = make_kv_norm_evict_fn(L, n_sinks, n_recent)
    attn_fn = make_attention_evict_fn(L, n_sinks, n_recent)

    for idx, ex in enumerate(wide):
        if idx in done:
            continue
        try:
            cap = capture(llm, tokenizer, ex, device)
            T = cap.prompt_len
            if T > max_len or T >= budget:
                row = {"idx": idx, "skip": True, "T": T}
                print(f"[{idx}] skip T={T}")
            else:
                gold = cap.gold_answer
                # 1) full-cache trace (also the `full` arm) + future attention
                full_text, suf, n_steps = full_trace_with_future_attention(
                    llm, tokenizer, cap.input_ids, device, max_new)
                c_full = float(flexible_extract(full_text, [gold]))
                orc_fn = make_oracle_evict_fn(suf, L, T, n_sinks, n_recent)

                def run(fn):
                    text, _, _, trunc, _, _ = run_online_episode(
                        llm, tokenizer, cap.input_ids, budget, max_new,
                        device, fn)
                    return float(flexible_extract(text, [gold])), float(trunc)

                c_kv,  tr_kv  = run(kv_fn)
                c_at,  tr_at  = run(attn_fn)
                c_orc, tr_orc = run(orc_fn)
                del suf
                row = {"idx": idx, "skip": False, "T": T, "steps_full": n_steps,
                       "full": c_full, "kv_norm": c_kv, "attn_cur": c_at,
                       "oracle_fut": c_orc,
                       "trunc_kv": tr_kv, "trunc_attn": tr_at, "trunc_orc": tr_orc}
                print(f"[{idx}] full={c_full:.0f} kv={c_kv:.0f} "
                      f"attn={c_at:.0f} oracle={c_orc:.0f}")
        except Exception as e:  # noqa: BLE001 — log and continue, resume-safe
            row = {"idx": idx, "error": str(e)[:200]}
            print(f"[{idx}] ERROR {e}")
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    # aggregate
    rows = [json.loads(l) for l in open(results_path)]
    ok = [r for r in rows if not r.get("skip") and "error" not in r]
    if ok:
        import statistics
        print(f"\n==== ORACLE GAP ({len(ok)} examples) ====")
        for k in ("full", "kv_norm", "attn_cur", "oracle_fut"):
            print(f"  {k:10s} {sum(r[k] for r in ok)/len(ok):.4f}")
        gap = sum(r["oracle_fut"] - r["kv_norm"] for r in ok) / len(ok)
        att = sum(r["attn_cur"] - r["kv_norm"] for r in ok) / len(ok)
        print(f"  PAIRED oracle_fut − kv_norm = {gap:+.4f}")
        print(f"  PAIRED attn_cur   − kv_norm = {att:+.4f}")


if __name__ == "__main__":
    main()
