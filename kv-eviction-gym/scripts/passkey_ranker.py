"""THE "algo que ande" experiment: a learned eviction ranker in the passkey arena.

The passkey arena has a +0.43 oracle-over-kv_norm ceiling (FINDINGS 25), unlike
GSM8K's +0.07. This script runs the actual Apple/KVP recipe (offline, per-layer
utility regression, NO online RL) in that arena and, crucially, EVALUATES THE
LEARNED RANKER AS AN EVICTION POLICY end-to-end (accuracy), not just rank
correlation. If the learned ranker beats kv_norm on held-out passkey examples,
we have a positive, publishable result: learned eviction works where the signal
exists, and we built it ourselves.

Pipeline (one process, self-contained, no pre-computed traces):
  1. Generate N passkey examples (reuses eval_passkey.make_passkey_examples).
  2. For each: capture prompt K/V + run full_trace_with_future_attention to get
     the golden per-token future-attention label per layer (oracle_eval).
  3. Train a per-layer MLP ranker: features = raw K/V + rich columns, target =
     log future-attention. Train on [0:n_train], hold out the rest.
  4. Build a learned_evict_fn from the trained rankers (evict the resident slot
     with the LOWEST predicted future attention), and run the paired eviction
     eval on the held-out examples: full / random / kv_norm / learned / oracle.

Reading:
  learned - kv_norm >> 0 on held-out accuracy -> LEARNED EVICTION WORKS in the
    regime that has signal. The whole project's negative result flips to: "online
    RL on GSM8K plateaus because GSM8K lacks the signal; the SAME learned-ranker
    idea, offline, in a retrieval regime, beats the heuristic."

Usage:
  python scripts/passkey_ranker.py --n 160 --n-train 120 --budget 176 \
      --out-dir runs/passkey_ranker
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_eval import full_trace_with_future_attention  # noqa: E402
from eval_passkey import make_passkey_examples            # noqa: E402
from rank_predictability import spearman, overlap_at_k    # noqa: E402

from kv_gym.capture import capture                        # noqa: E402
from kv_gym.features import build_extra_columns, feature_dim  # noqa: E402
from kv_gym.eval_core import (                            # noqa: E402
    _valid_slots, run_online_episode, make_kv_norm_evict_fn,
    make_random_evict_fn,
)
from oracle_eval import make_oracle_evict_fn              # noqa: E402
from kv_gym.vendor.loader import load_model_and_tokenizer  # noqa: E402
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract  # noqa: E402


class LayerRanker(nn.Module):
    def __init__(self, feat_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 64), nn.SiLU(),
                                 nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def token_features(cap_K, cap_V, l, T, H, D, device):
    """Raw K||V (all heads concat) + rich columns for layer l, [T, feat_dim]."""
    K = cap_K[l:l+1].to(device)   # [1, H, T, D]
    V = cap_V[l:l+1].to(device)
    pos = np.arange(T)[None, :]
    rich = build_extra_columns(K, V, T, T, orig_pos=pos)[0]           # [T, 2H+5]
    kv = torch.cat([K[0].permute(1, 0, 2).reshape(T, H * D),
                    V[0].permute(1, 0, 2).reshape(T, H * D)], -1).cpu().numpy()
    return np.concatenate([kv, rich], -1).astype(np.float32)


def make_learned_evict_fn(rankers, mus, sds, L, n_sinks, n_recent, device,
                          feat_builder):
    """Evict the valid slot with the LOWEST predicted future attention, per layer.
    feat_builder(l, orig_positions) -> [len(pos), feat_dim] features for the CURRENT
    resident tokens of layer l (built from the live K/V passed by the episode)."""
    def evict(K, V, tracker, scores=None, n_prompt=0):
        # K, V: [L, H, S, D] live cache
        slots = []
        H = K.shape[1]; D = K.shape[3]
        for l in range(L):
            pos = tracker._pos[l]
            S = len(pos)
            cand = _valid_slots(S, n_sinks, n_recent, n_prompt)
            # build features for all resident slots of layer l from the LIVE cache
            Kl = K[l:l+1].float(); Vl = V[l:l+1].float()
            rich = build_extra_columns(Kl, Vl, S, S,
                                       orig_pos=np.array(pos)[None, :])[0]
            kv = torch.cat([Kl[0].permute(1, 0, 2).reshape(S, H * D),
                            Vl[0].permute(1, 0, 2).reshape(S, H * D)], -1).cpu().numpy()
            feat = np.concatenate([kv, rich], -1).astype(np.float32)
            x = torch.tensor((feat - mus[l]) / sds[l], device=device)
            with torch.no_grad():
                pred = rankers[l](x).cpu().numpy()   # predicted future attention
            vals = pred[cand]
            slots.append(cand[int(np.argmin(vals))])
        return slots
    evict.needs_kv = True
    return evict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=160)
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--budget", type=int, default=176)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--out-dir", default="runs/passkey_ranker")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm, tokenizer, device = load_model_and_tokenizer(
        name="qwen-1.5b", device=device, attn_implementation="eager")
    L = llm.config.num_hidden_layers
    H = getattr(llm.config, "num_key_value_heads", llm.config.num_attention_heads)
    D = llm.config.hidden_size // llm.config.num_attention_heads
    fdim = feature_dim(H, D, rich=True)
    n_sinks, n_recent = 4, 32

    examples = make_passkey_examples(args.n, args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1+2. Build offline dataset: per-layer token features + future-attn labels
    feats = [[] for _ in range(L)]   # each: list over train examples of [T,fdim]
    labs  = [[] for _ in range(L)]
    caps  = []                        # keep test-example capture + trace for eval
    print(f"building dataset: {args.n} passkey examples (eager trace)...", flush=True)
    for idx, ex in enumerate(examples):
        cap = capture(llm, tokenizer, ex, device)
        T = cap.prompt_len
        _, suf, _ = full_trace_with_future_attention(
            llm, tokenizer, cap.input_ids, device, args.max_new_tokens)
        if idx < args.n_train:
            for l in range(L):
                feats[l].append(token_features(cap.K, cap.V, l, T, H, D, device))
                labs[l].append(suf[l][0, :T].astype(np.float32))   # future attn per prompt tok
        else:
            caps.append((ex, cap, suf))
        del suf
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{args.n}", flush=True)

    # ---- 3. Train per-layer rankers
    rankers, mus, sds = [], [], []
    for l in range(L):
        X = torch.tensor(np.concatenate(feats[l]), device=device)
        y = torch.log(torch.tensor(np.concatenate(labs[l]), device=device).clamp(min=1e-6))
        mu = X.mean(0); sd = X.std(0).clamp(min=1e-6)
        Xn = (X - mu) / sd
        net = LayerRanker(fdim).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for ep in range(args.epochs):
            perm = torch.randperm(Xn.shape[0], device=device)
            for i in range(0, len(perm), 8192):
                b = perm[i:i+8192]
                loss = ((net(Xn[b]) - y[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        rankers.append(net.eval())
        mus.append(mu.cpu().numpy()); sds.append(sd.cpu().numpy())
    print("rankers trained.", flush=True)

    # ---- 4. Paired eviction eval on held-out passkey examples
    kv_fn  = make_kv_norm_evict_fn(L, n_sinks, n_recent)
    rng    = np.random.default_rng(args.seed)
    learned_fn = make_learned_evict_fn(rankers, mus, sds, L, n_sinks, n_recent,
                                       device, None)
    res = {k: [] for k in ("full", "random", "kv_norm", "learned", "oracle")}
    for ex, cap, suf in caps:
        gold = ex["gold_answers"][0]
        T = cap.prompt_len
        # full (from a fresh trace is expensive; reuse: run full via budget=inf-ish)
        orc_fn = make_oracle_evict_fn(suf, L, T, n_sinks, n_recent)
        rnd_fn = make_random_evict_fn(L, np.random.default_rng(rng.integers(1 << 30)),
                                      n_sinks, n_recent)
        def run(fn):
            text, _, _, _, _, _ = run_online_episode(
                llm, tokenizer, cap.input_ids, args.budget, args.max_new_tokens,
                device, fn)
            return float(flexible_extract(text, [gold]))
        def run_full():
            text, _, _, _, _, _ = run_online_episode(
                llm, tokenizer, cap.input_ids, 100000, args.max_new_tokens,
                device, kv_fn)  # budget huge -> never evicts
            return float(flexible_extract(text, [gold]))
        res["full"].append(run_full())
        res["random"].append(run(rnd_fn))
        res["kv_norm"].append(run(kv_fn))
        res["learned"].append(run(learned_fn))
        res["oracle"].append(run(orc_fn))

    n = len(caps)
    import statistics as st, math
    summ = {k: sum(v) / n for k, v in res.items()}
    gap = [res["learned"][i] - res["kv_norm"][i] for i in range(n)]
    gse = st.pstdev(gap) / math.sqrt(n) if n > 1 else 0.0
    print(f"\n==== PASSKEY LEARNED RANKER ({n} held-out, budget={args.budget}) ====")
    for k in ("full", "oracle", "learned", "kv_norm", "random"):
        print(f"  {k:9s} {summ[k]:.4f}")
    print(f"  PAIRED learned - kv_norm = {sum(gap)/n:+.4f} +/- {gse:.4f}")
    wins = sum(1 for g in gap if g > 0); loss = sum(1 for g in gap if g < 0)
    print(f"  wins={wins} losses={loss} ties={n-wins-loss}")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"summary": summ, "learned_minus_kvnorm": sum(gap)/n,
                   "se": gse, "wins": wins, "losses": loss, "n": n}, f, indent=1)
    Path(out_dir / "RANKER_EVAL_DONE").touch()


if __name__ == "__main__":
    main()
