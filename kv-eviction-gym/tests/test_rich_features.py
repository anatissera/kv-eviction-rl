"""Phase 2 E1 — rich-features unit tests.

The load-bearing invariant: the TRAINING observation (`batched_env._obs_episode`)
and the EVAL observation (`features.build_obs`) must be byte-identical for the same
K/V. If they diverge, we reintroduce exactly the train/eval confound Phase 2 exists
to remove. Both call `build_extra_columns`, so this guards the surrounding indexing.

Runs on CPU, no model needed. The PerTokenMLP forward test is skipped if
stable_baselines3 is unavailable (it runs on the VM).
"""
import numpy as np
import torch
import pytest

from kv_gym.features import (
    feature_dim, extra_feature_dim, build_obs, build_extra_columns, POS_SCALE,
)

H, D = 2, 64          # Qwen2.5-1.5B: 2 KV-heads × 64 dim
KVDIM = 2 * H * D     # 256
NEXTRA = 2 * H + 5    # per-head kz(H)+vz(H) + kz_mean + vz_mean + kvz + rec + pos_orig = 9
# column indices within the extra block
IKZ_MEAN, IVZ_MEAN, IKVZ, IREC, IPOS = 2*H, 2*H + 1, 2*H + 2, 2*H + 3, 2*H + 4


def _rand_kv(L, S, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(L, H, S, D, generator=g)
    V = torch.randn(L, H, S, D, generator=g)
    # make norms vary across slots so standardization is non-trivial
    scale = torch.linspace(0.2, 5.0, S).view(1, 1, S, 1)
    return K * scale, V * scale


def _orig_pos(L, S, T=8):
    """Synthetic original positions: prompt [0,T) then generated T,T+1,... per layer."""
    return np.tile(np.arange(S, dtype=np.int64), (L, 1))


def test_feature_dim_math():
    assert feature_dim(H, D, rich=False) == KVDIM
    assert feature_dim(H, D, rich=True) == KVDIM + NEXTRA
    assert extra_feature_dim(False, H) == 0
    assert extra_feature_dim(True, H) == NEXTRA
    # H-dependence
    assert extra_feature_dim(True, 4) == 2*4 + 5


def test_extra_columns_standardized_and_ranges():
    L, S, max_len = 3, 20, 32
    K, V = _rand_kv(L, S)
    op = _orig_pos(L, S)
    cols = build_extra_columns(K, V, cache_size=S, max_len=max_len, orig_pos=op)  # [L,S,2H+4]
    assert cols.shape == (L, S, NEXTRA)
    kz_mean, vz_mean = cols[..., IKZ_MEAN], cols[..., IVZ_MEAN]
    rec, pos = cols[..., IREC], cols[..., IPOS]
    # standardized norms: per-row mean≈0. std uses torch's unbiased (N-1)
    # estimator, so the population std of the z-scores is sqrt((S-1)/S).
    exp_std = np.sqrt((S - 1) / S)
    assert np.allclose(kz_mean.mean(axis=1), 0, atol=1e-4)
    assert np.allclose(kz_mean.std(axis=1), exp_std, atol=1e-2)
    assert np.allclose(vz_mean.mean(axis=1), 0, atol=1e-4)
    # per-head columns also standardized
    for h in range(H):
        assert np.allclose(cols[..., h].mean(axis=1), 0, atol=1e-4)
    # recency: slot 0 least recent (highest rec); pos_orig = orig/POS_SCALE ascending
    assert np.all(rec[:, 0] > rec[:, -1])
    assert np.allclose(pos, op / POS_SCALE)


def test_kz_recovers_kv_norm_ordering():
    """The whole point of E1: kz_mean ranks slots by mean-over-heads ||K||."""
    L, S, max_len = 1, 15, 32
    K, V = _rand_kv(L, S, seed=7)
    kmean = K.norm(dim=-1).mean(dim=1)[0]                 # [S] true mean-over-heads ||K||
    cols = build_extra_columns(K, V, S, max_len, orig_pos=_orig_pos(L, S))
    kz = cols[0, :, IKZ_MEAN]
    assert int(kz.argmin()) == int(kmean.argmin())
    assert np.array_equal(np.argsort(kz), np.argsort(kmean.numpy()))


def test_kvz_recovers_exact_kv_norm():
    """kvz must rank slots by (||K||+||V||) mean-over-heads — kv_norm's EXACT signal,
    so argmin(kvz) == the slot kv_norm evicts. This is what lets the policy represent
    (and warm-start-clone) kv_norm; kz_mean/vz_mean standardized separately cannot."""
    L, S, max_len = 1, 15, 32
    K, V = _rand_kv(L, S, seed=7)
    kvnorm = (K.norm(dim=-1).mean(dim=1) + V.norm(dim=-1).mean(dim=1))[0]   # [S] kv_norm score
    cols = build_extra_columns(K, V, S, max_len, orig_pos=_orig_pos(L, S))
    kvz = cols[0, :, IKVZ]
    assert int(kvz.argmin()) == int(kvnorm.argmin())     # exact kv_norm choice
    assert np.array_equal(np.argsort(kvz), np.argsort(kvnorm.numpy()))


def _obs_episode_like(K_all, V_all, cache_size, max_len, rich, orig):
    """Replicate batched_env._obs_episode's per-layer numpy writes exactly, so we
    can compare the TRAINING construction against eval's build_obs."""
    L = K_all.shape[0]
    fdim = feature_dim(H, D, rich=rich)
    obs = np.zeros((L, max_len, fdim), dtype=np.float32)
    S = cache_size
    ne = extra_feature_dim(rich, H)
    for l in range(L):
        K = K_all[l].float()   # [H, S, D]
        V = V_all[l].float()
        obs[l, :S, :H*D]      = K.transpose(0, 1).reshape(S, H*D).numpy()
        obs[l, :S, H*D:2*H*D] = V.transpose(0, 1).reshape(S, H*D).numpy()
        if rich:
            op = np.asarray(orig[l], dtype=np.int64)[None, :]
            cols = build_extra_columns(K.unsqueeze(0), V.unsqueeze(0), S, max_len, orig_pos=op)
            obs[l, :S, 2*H*D:2*H*D + ne] = cols[0]
    return obs


@pytest.mark.parametrize("rich", [False, True])
def test_train_eval_obs_identical(rich):
    """THE invariant: training (_obs_episode) and eval (build_obs) agree bit-for-bit,
    including the rich orig_pos-derived columns fed from both sides' position tracking."""
    L, S, max_len = 4, 18, 32
    K, V = _rand_kv(L, S, seed=3)
    orig = _orig_pos(L, S)
    train_obs = _obs_episode_like(K, V, S, max_len, rich, orig)
    eval_obs  = build_obs(K, V, S, max_len, rich=rich, orig_pos=(orig if rich else None))
    assert train_obs.shape == eval_obs.shape
    np.testing.assert_allclose(train_obs, eval_obs, rtol=0, atol=0)


def test_padding_is_zero():
    L, S, max_len = 2, 10, 32
    K, V = _rand_kv(L, S)
    obs = build_obs(K, V, S, max_len, rich=True, orig_pos=_orig_pos(L, S))
    # everything beyond the resident cache must be exactly zero
    assert np.all(obs[:, S:, :] == 0.0)
    # resident region has nonzero K/V
    assert np.any(obs[:, :S, :KVDIM] != 0.0)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("stable_baselines3") is None,
    reason="stable_baselines3 not installed (runs on the VM)",
)
@pytest.mark.parametrize("cls_name", ["PerTokenMLP", "PerTokenAttention"])
def test_extra_columns_influence_output(cls_name):
    """The extra columns must actually reach the network (not be dropped/zeroed by
    the LayerNorm bypass), and padded slots must be masked to zero. Covers both the
    per-token MLP (E1) and the cross-token attention extractor (E4)."""
    from gymnasium import spaces
    import kv_gym.policy as P

    L, S, max_len = 1, 12, 32
    K, V = _rand_kv(L, S, seed=11)
    obs = build_obs(K, V, S, max_len, rich=True, orig_pos=_orig_pos(L, S))  # [1, max_len, KVDIM+NEXTRA]
    space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape[1:], dtype=np.float32)
    torch.manual_seed(0)
    net = getattr(P, cls_name)(space, hidden=64, n_extra=NEXTRA)
    net.eval()

    x = torch.tensor(obs)
    with torch.no_grad():
        out = net(x)                                        # [1, max_len]
    assert out.shape == (1, max_len)
    assert torch.isfinite(out).all()
    # padded slots (>= S) must be exactly zero (is_real mask)
    assert torch.allclose(out[0, S:], torch.zeros(max_len - S))
    # perturbing ONLY the extra columns must change the resident outputs → the
    # extra features are wired through (not sliced away or LayerNorm-erased).
    x2 = x.clone()
    x2[0, :S, KVDIM:] += 3.0
    with torch.no_grad():
        out2 = net(x2)
    assert not torch.allclose(out[0, :S], out2[0, :S], atol=1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
