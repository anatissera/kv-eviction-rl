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
    feature_dim, extra_feature_dim, build_obs, build_extra_columns, N_EXTRA_RICH,
)

H, D = 2, 64          # Qwen2.5-1.5B: 2 KV-heads × 64 dim
KVDIM = 2 * H * D     # 256


def _rand_kv(L, S, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(L, H, S, D, generator=g)
    V = torch.randn(L, H, S, D, generator=g)
    # make norms vary across slots so standardization is non-trivial
    scale = torch.linspace(0.2, 5.0, S).view(1, 1, S, 1)
    return K * scale, V * scale


def test_feature_dim_math():
    assert feature_dim(H, D, rich=False) == KVDIM
    assert feature_dim(H, D, rich=True) == KVDIM + N_EXTRA_RICH
    assert extra_feature_dim(False) == 0
    assert extra_feature_dim(True) == N_EXTRA_RICH


def test_extra_columns_standardized_and_ranges():
    L, S, max_len = 3, 20, 32
    K, V = _rand_kv(L, S)
    cols = build_extra_columns(K, V, cache_size=S, max_len=max_len)  # [L, S, 4]
    assert cols.shape == (L, S, N_EXTRA_RICH)
    kz, vz, pos, rec = cols[..., 0], cols[..., 1], cols[..., 2], cols[..., 3]
    # standardized norms: per-row mean≈0. std uses torch's unbiased (N-1)
    # estimator, so the population std of the z-scores is sqrt((S-1)/S).
    exp_std = np.sqrt((S - 1) / S)
    assert np.allclose(kz.mean(axis=1), 0, atol=1e-4)
    assert np.allclose(kz.std(axis=1), exp_std, atol=1e-2)
    assert np.allclose(vz.mean(axis=1), 0, atol=1e-4)
    # position/recency in [0,1)
    assert pos.min() >= 0 and pos.max() < 1.0 + 1e-6
    assert rec.min() >= 0 and rec.max() <= 1.0 + 1e-6
    # slot 0 is oldest (pos=0) and least recent (rec highest); last slot is newest
    assert np.allclose(pos[:, 0], 0.0)
    assert np.all(rec[:, 0] > rec[:, -1])


def test_kz_recovers_kv_norm_ordering():
    """The whole point of E1: kz must rank slots by ||K|| so the policy can
    represent kv_norm's argmin. Lowest-norm slot must have the lowest kz."""
    L, S, max_len = 1, 15, 32
    K, V = _rand_kv(L, S, seed=7)
    kmean = K.norm(dim=-1).mean(dim=1)[0]                 # [S] true per-slot ||K||
    cols = build_extra_columns(K, V, S, max_len)
    kz = cols[0, :, 0]
    # argmin/argmax of kz must match argmin/argmax of the true norm
    assert int(kz.argmin()) == int(kmean.argmin())
    assert int(kz.argmax()) == int(kmean.argmax())
    # full ranking agreement (Spearman == 1)
    assert np.array_equal(np.argsort(kz), np.argsort(kmean.numpy()))


def _obs_episode_like(K_all, V_all, cache_size, max_len, rich):
    """Replicate batched_env._obs_episode's per-layer numpy writes exactly, so we
    can compare the TRAINING construction against eval's build_obs."""
    L = K_all.shape[0]
    fdim = feature_dim(H, D, rich=rich)
    obs = np.zeros((L, max_len, fdim), dtype=np.float32)
    S = cache_size
    for l in range(L):
        K = K_all[l].float()   # [H, S, D]
        V = V_all[l].float()
        obs[l, :S, :H*D]      = K.transpose(0, 1).reshape(S, H*D).numpy()
        obs[l, :S, H*D:2*H*D] = V.transpose(0, 1).reshape(S, H*D).numpy()
        if rich:
            cols = build_extra_columns(K.unsqueeze(0), V.unsqueeze(0), S, max_len)
            obs[l, :S, 2*H*D:2*H*D + N_EXTRA_RICH] = cols[0]
    return obs


@pytest.mark.parametrize("rich", [False, True])
def test_train_eval_obs_identical(rich):
    """THE invariant: training (_obs_episode) and eval (build_obs) agree bit-for-bit."""
    L, S, max_len = 4, 18, 32
    K, V = _rand_kv(L, S, seed=3)
    train_obs = _obs_episode_like(K, V, S, max_len, rich)
    eval_obs  = build_obs(K, V, S, max_len, rich=rich)          # [L, max_len, fdim]
    assert train_obs.shape == eval_obs.shape
    np.testing.assert_allclose(train_obs, eval_obs, rtol=0, atol=0)


def test_padding_is_zero():
    L, S, max_len = 2, 10, 32
    K, V = _rand_kv(L, S)
    obs = build_obs(K, V, S, max_len, rich=True)
    # everything beyond the resident cache must be exactly zero
    assert np.all(obs[:, S:, :] == 0.0)
    # resident region has nonzero K/V
    assert np.any(obs[:, :S, :KVDIM] != 0.0)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("stable_baselines3") is None,
    reason="stable_baselines3 not installed (runs on the VM)",
)
def test_pertokenmlp_extra_columns_influence_output():
    """The extra columns must actually reach the MLP (not be dropped/zeroed), and
    padded slots must be masked to zero."""
    from gymnasium import spaces
    from kv_gym.policy import PerTokenMLP

    L, S, max_len = 1, 12, 32
    K, V = _rand_kv(L, S, seed=11)
    obs = build_obs(K, V, S, max_len, rich=True)           # [1, max_len, KVDIM+4]
    space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape[1:], dtype=np.float32)
    torch.manual_seed(0)
    net = PerTokenMLP(space, hidden=64, n_extra=N_EXTRA_RICH)
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
