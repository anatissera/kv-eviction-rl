"""Unit tests for S4 information-theoretic per-step shaping.

These are dependency-light (no LLM load): they test the KL/entropy math
contract, the shadow-cache deep-copy helper (real code), and that the batched
env exposes the kl_* knobs. The full end-to-end behaviour (real model, real
rollout) is validated by the smoke run in experiments/s4-entropy-shaping/.
"""

import inspect

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Damage math contract (mirrors batched_env.step_wait's inline computation)
# ---------------------------------------------------------------------------

def _kl_full_evict(logits_full, logits_evict):
    """KL(p_full || p_evict) — exact-mode damage."""
    logp_full  = torch.log_softmax(logits_full, dim=-1)
    logp_evict = torch.log_softmax(logits_evict, dim=-1)
    return (logp_full.exp() * (logp_full - logp_evict)).sum().item()


def _entropy(logits):
    """H(p) in nats — proxy-mode damage."""
    logp = torch.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum().item()


def test_kl_zero_when_no_damage():
    """Identical distributions (eviction had no effect) → KL == 0."""
    torch.manual_seed(0)
    logits = torch.randn(151_936)
    assert _kl_full_evict(logits, logits.clone()) == pytest.approx(0.0, abs=1e-5)


def test_kl_positive_when_damaged():
    """A damaging eviction shifts the distribution → KL > 0."""
    torch.manual_seed(0)
    logits_full  = torch.randn(50_000)
    logits_evict = logits_full + torch.randn(50_000)  # perturbed
    assert _kl_full_evict(logits_full, logits_evict) > 0.0


def test_entropy_nonnegative_and_uniform_is_max():
    """H >= 0, and a uniform distribution attains the max entropy log(V)."""
    V = 1000
    uniform = torch.zeros(V)                       # softmax → uniform
    peaked  = torch.zeros(V); peaked[0] = 50.0     # softmax → near one-hot
    assert _entropy(peaked) >= 0.0
    assert _entropy(uniform) == pytest.approx(np.log(V), abs=1e-4)
    assert _entropy(uniform) > _entropy(peaked)


def test_clip_keeps_reward_bounded():
    """clip(damage, 0, kl_clip) bounds the per-step shaping magnitude."""
    kl_clip, kl_weight = 5.0, 0.05
    damage = np.array([-0.1, 0.0, 3.0, 100.0], dtype=np.float32)
    r = -kl_weight * np.clip(damage, 0.0, kl_clip)
    assert r.min() >= -kl_weight * kl_clip - 1e-6
    assert r.max() <= 0.0  # shaping only ever penalises


# ---------------------------------------------------------------------------
# 2. Shadow-cache deep copy (real code under test)
# ---------------------------------------------------------------------------

def test_clone_cache_is_independent_deep_copy():
    from transformers import DynamicCache
    from kv_gym.batched_env import _clone_cache, _get_kv_bat, _set_kv_bat

    cache = DynamicCache()
    K0 = torch.ones(1, 2, 4, 8)
    V0 = torch.ones(1, 2, 4, 8) * 2
    # populate layer 0 via the same accessors the env uses
    if hasattr(cache, "layers"):
        # transformers >= 4.54 uses .layers; build one layer entry
        cache.update(K0, V0, 0)
    else:
        cache.key_cache.append(K0)
        cache.value_cache.append(V0)

    clone = _clone_cache(cache)
    # mutate the original after cloning
    K_orig, _ = _get_kv_bat(cache, 0)
    K_orig.add_(99.0)

    K_clone, V_clone = _get_kv_bat(clone, 0)
    assert torch.allclose(K_clone, torch.ones_like(K_clone)), "clone must not alias original K"
    assert torch.allclose(V_clone, torch.ones_like(V_clone) * 2)


# ---------------------------------------------------------------------------
# 3. Env exposes the kl_* knobs
# ---------------------------------------------------------------------------

def test_batched_env_accepts_kl_kwargs():
    from kv_gym.batched_env import BatchedSharedKVVecEnv
    sig = inspect.signature(BatchedSharedKVVecEnv.__init__)
    for p in ("kl_shaping", "kl_mode", "kl_weight", "kl_clip"):
        assert p in sig.parameters, f"missing kl param: {p}"
    assert sig.parameters["kl_shaping"].default is False
    assert sig.parameters["kl_mode"].default == "exact"
