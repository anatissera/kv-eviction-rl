"""KVP-lite: is FUTURE UTILITY predictable from K/V content on GSM8K?

Apple's KVP (arXiv 2602.10238) trains lightweight per-head rankers OFFLINE on
pre-computed traces: input = the token's K/V vectors, target = the token's
future utility, loss = ranking. It wins on RULER. Our E5 golden-BC (online
policy, argmin-match metric) failed at 3% match, but that confounds three
things: the online setting, the harsh argmax metric, and the dataset. This
script isolates the DATASET question by running the closest analogue of the
Apple recipe at our scale, entirely offline:

  For each traced example (scripts/trace_gen.py output, ~400 on disk):
    features per PROMPT token per layer: raw K/V (all heads) + the rich
      norm/position columns (build_extra_columns)
    label = SUF[l][0, j] = that token's max future attention over the whole
      generation (the golden-eviction utility)
  Train a per-layer MLP ranker on examples [0:300], test on [300:400].
  Metrics per layer, on held-out examples:
    - Spearman rank correlation (predicted vs true utility ranking)
    - overlap@k: of the k lowest-utility tokens (evictable set, k=T/4),
      what fraction the ranker identifies; compared against the SAME metric
      for the kv_norm ranking (the heuristic's implicit prediction).

Reading:
  Spearman high (>0.5) and overlap >> kv_norm's  -> future utility IS
    learnable from content on GSM8K; our RL failures were the algorithm,
    and an offline-distill pipeline (KVP recipe) should work here.
  Spearman ~ kv_norm's own correlation (or lower)  -> the dataset carries no
    learnable eviction signal beyond norms: the null series is DATASET-driven,
    matching the passkey contrast (scripts/eval_passkey.py).

Usage (on kvp-ab, where traces/ lives):
  python scripts/rank_predictability.py --traces traces --n-train 300 \
      --n-test 100 --out-dir runs/rank_pred
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kv_gym.capture import capture                     # noqa: E402
from kv_gym.features import build_extra_columns        # noqa: E402
from kv_gym.vendor.loader import load_model_and_tokenizer  # noqa: E402
from kv_gym.vendor.gsm8k import load_gsm8k             # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def overlap_at_k(pred_score, true_util, k):
    """Fraction of the k truly-least-useful tokens that the k lowest predicted
    scores capture (higher = better evictable-set identification)."""
    lo_true = set(np.argsort(true_util)[:k].tolist())
    lo_pred = set(np.argsort(pred_score)[:k].tolist())
    return len(lo_true & lo_pred) / max(k, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="traces")
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out-dir", default="runs/rank_pred")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm, tokenizer, device = load_model_and_tokenizer(
        name="qwen-1.5b", device=device, attn_implementation="sdpa")
    L = llm.config.num_hidden_layers
    H = getattr(llm.config, "num_key_value_heads", llm.config.num_attention_heads)
    D = llm.config.hidden_size // llm.config.num_attention_heads
    kv_dim = 2 * H * D                     # raw K||V per token
    n_rich = 2 * H + 5
    feat_dim = kv_dim + n_rich

    n_total = args.n_train + args.n_test
    examples = load_gsm8k(n=n_total, seed=args.seed, split="train")
    tr_dir = Path(args.traces)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build the offline dataset: X[l] (tokens x feat), y[l] (tokens,) ----
    cache_f = out_dir / "dataset.npz"
    if cache_f.exists():
        z = np.load(cache_f, allow_pickle=True)
        Xs = [z[f"X{l:02d}"] for l in range(L)]
        Ys = [z[f"Y{l:02d}"] for l in range(L)]
        EX = z["ex_id"]
        print(f"loaded cached dataset: {Xs[0].shape[0]} tokens")
    else:
        feats = [[] for _ in range(L)]
        labs  = [[] for _ in range(L)]
        exid  = []
        used = 0
        for idx in range(n_total):
            npz = tr_dir / f"ex{idx:05d}.npz"
            if not npz.exists():
                continue
            z = np.load(npz)
            if int(z["skip"]) == 1:
                continue
            T = int(z["T"])
            cap = capture(llm, tokenizer, examples[idx], device)
            if cap.prompt_len != T:
                print(f"[{idx}] T mismatch {cap.prompt_len} vs {T} — skip")
                continue
            pos = np.arange(T)[None, :]
            for l in range(L):
                K = cap.K[l:l+1].to(device)   # [1, H, T, D]
                V = cap.V[l:l+1].to(device)
                rich = build_extra_columns(K, V, T, T, orig_pos=pos)[0]  # [T, n_rich]
                kv = torch.cat([
                    K[0].permute(1, 0, 2).reshape(T, H * D),
                    V[0].permute(1, 0, 2).reshape(T, H * D)], dim=-1).cpu().numpy()
                feats[l].append(np.concatenate([kv, rich], -1).astype(np.float32))
                labs[l].append(z[f"l{l:02d}"][0, :T].astype(np.float32))
            exid.extend([idx] * T)
            used += 1
            if used % 25 == 0:
                print(f"  built {used} examples")
        Xs = [np.concatenate(f) for f in feats]
        Ys = [np.concatenate(y) for y in labs]
        EX = np.array(exid)
        np.savez_compressed(cache_f, ex_id=EX,
                            **{f"X{l:02d}": Xs[l] for l in range(L)},
                            **{f"Y{l:02d}": Ys[l] for l in range(L)})
        print(f"dataset built: {Xs[0].shape[0]} tokens from {used} examples")

    train_mask = EX < args.n_train
    test_ids = sorted(set(EX[~train_mask].tolist()))
    print(f"train tokens={int(train_mask.sum())}  test examples={len(test_ids)}")

    # ---- Train one small MLP per layer (regression on log-utility) ----------
    results = {}
    for l in range(L):
        X = torch.tensor(Xs[l], device=device)
        y = torch.log(torch.tensor(Ys[l], device=device).clamp(min=1e-6))
        mu, sd = X[train_mask].mean(0), X[train_mask].std(0).clamp(min=1e-6)
        Xn = (X - mu) / sd
        net = nn.Sequential(nn.Linear(feat_dim, 128), nn.SiLU(),
                            nn.Linear(128, 64), nn.SiLU(),
                            nn.Linear(64, 1)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        Xtr, ytr = Xn[torch.tensor(train_mask, device=device)], y[torch.tensor(train_mask, device=device)]
        for ep in range(args.epochs):
            perm = torch.randperm(Xtr.shape[0], device=device)
            for i in range(0, len(perm), 8192):
                b = perm[i:i+8192]
                loss = ((net(Xtr[b]).squeeze(-1) - ytr[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()

        # ---- held-out metrics, computed PER EXAMPLE then averaged ----------
        sp_net, sp_kvn, ov_net, ov_kvn = [], [], [], []
        with torch.no_grad():
            pred_all = net(Xn).squeeze(-1).cpu().numpy()
        yl = Ys[l]
        for eid in test_ids:
            m = EX == eid
            util, pred = yl[m], pred_all[m]
            T = util.shape[0]; k = max(T // 4, 8)
            # kv_norm's implicit prediction = kvz column (index kv_dim + 2H+2)
            kvz = Xs[l][m][:, kv_dim + 2 * H + 2]
            sp_net.append(spearman(pred, np.log(np.clip(util, 1e-6, None))))
            sp_kvn.append(spearman(kvz, np.log(np.clip(util, 1e-6, None))))
            ov_net.append(overlap_at_k(pred, util, k))
            ov_kvn.append(overlap_at_k(kvz, util, k))
        results[l] = dict(sp_net=float(np.mean(sp_net)), sp_kvn=float(np.mean(sp_kvn)),
                          ov_net=float(np.mean(ov_net)), ov_kvn=float(np.mean(ov_kvn)))
        print(f"L{l:02d} spearman net={results[l]['sp_net']:+.3f} kvz={results[l]['sp_kvn']:+.3f} "
              f"| overlap@T/4 net={results[l]['ov_net']:.3f} kvz={results[l]['ov_kvn']:.3f}")

    agg = {k: float(np.mean([r[k] for r in results.values()]))
           for k in ("sp_net", "sp_kvn", "ov_net", "ov_kvn")}
    print("\n==== RANK PREDICTABILITY (mean over layers, held-out examples) ====")
    print(f"  learned ranker : spearman={agg['sp_net']:+.3f}  overlap@T/4={agg['ov_net']:.3f}")
    print(f"  kv_norm (kvz)  : spearman={agg['sp_kvn']:+.3f}  overlap@T/4={agg['ov_kvn']:.3f}")
    print(f"  random overlap baseline = 0.25")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"per_layer": results, "aggregate": agg}, f, indent=1)


if __name__ == "__main__":
    main()
