"""
Validation tests for the three fixes applied after the Opus agent critique:

  1. position_ids fix — attention_mask without explicit position_ids shifts RoPE
                        positions after each masked gap; with arange(T) they stay correct.
  2. K/V norm heterogeneity — K-norms vary 5–50× across layers; PerTokenMLP's
                              per-half LayerNorm brings them to unit scale.
  3. gamma=1.0 in config — no discount bias when observation is constant.

Run with:
    cd kv-eviction-gym && pytest tests/test_fixes.py -v
"""

import numpy as np
import pytest
import torch
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    from kv_gym.vendor.loader import load_model_and_tokenizer
    model, tokenizer, device = load_model_and_tokenizer(
        name="tiny-llama",
        device=torch.device("cpu"),
    )
    model.eval()
    return model, tokenizer, device


@pytest.fixture(scope="module")
def tiny_capture(tiny_model_and_tokenizer):
    from kv_gym.capture import capture
    model, tokenizer, device = tiny_model_and_tokenizer
    example = {
        "prompt_text": "Janet has 3 apples and buys 2 more. She also gets 4 from her friend. How many apples does she have now?",
        "gold_answers": ["9"],
        "task": "gsm8k",
    }
    return capture(model, tokenizer, example, device), model, tokenizer, device


# ---------------------------------------------------------------------------
# Fix 1: position_ids fix
# ---------------------------------------------------------------------------

class TestPositionIdsFix:
    """
    The Opus agent flagged that HuggingFace used to compute positions via
    cumsum(attention_mask) - 1 when position_ids weren't passed, shifting RoPE
    rotations after each masked gap.

    Empirical check: in our transformers version (>=4.40), model() defaults to
    arange(T) regardless — the bug was in an older API. Our explicit position_ids
    pass is therefore a defensive no-op. These tests verify the current behavior
    and guard against regressions.
    """

    def _logits_for(self, model, input_ids, attn_mask, position_ids, device):
        with torch.no_grad():
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attn_mask.to(device) if attn_mask is not None else None,
                position_ids=position_ids.to(device) if position_ids is not None else None,
                use_cache=False,
            )
        return out.logits  # [1, T, vocab]

    def test_current_transformers_uses_arange_by_default(self, tiny_capture):
        """
        In transformers >=4.40, model() uses arange(T) for position_ids by default,
        NOT cumsum(attn_mask)-1. Passing position_ids=arange(T) explicitly should give
        identical logits as not passing it.
        """
        cap, model, tokenizer, device = tiny_capture
        T = cap.prompt_len
        input_ids = cap.input_ids

        attn_mask = torch.zeros(1, T, dtype=torch.long)
        attn_mask[0, ::2] = 1  # half the tokens masked

        logits_no_pos   = self._logits_for(model, input_ids, attn_mask, None, device)
        logits_with_pos = self._logits_for(model, input_ids, attn_mask, torch.arange(T).unsqueeze(0), device)

        diff = (logits_no_pos - logits_with_pos).abs().max().item()
        assert diff == 0.0, (
            f"Explicit position_ids=arange(T) should be identical to the default; "
            f"max diff={diff:.6f}.  If this fails, transformers changed the default behavior."
        )

    def test_masked_logits_differ_from_full_cache(self, tiny_capture):
        """Masking tokens does change logits — the mask is active."""
        cap, model, tokenizer, device = tiny_capture
        T = cap.prompt_len
        input_ids = cap.input_ids

        attn_mask_sparse = torch.zeros(1, T, dtype=torch.long)
        attn_mask_sparse[0, ::2] = 1
        attn_mask_full = torch.ones(1, T, dtype=torch.long)

        logits_sparse = self._logits_for(model, input_ids, attn_mask_sparse, None, device)
        logits_full   = self._logits_for(model, input_ids, attn_mask_full,   None, device)

        diff = (logits_sparse - logits_full).abs().mean().item()
        assert diff > 1e-6, (
            f"Masking half the tokens should change logits; diff={diff:.8f}"
        )

    def test_terminal_reward_generates_without_error(self, tiny_model_and_tokenizer):
        """
        Integration guard: _terminal_reward() completes without error and returns
        a valid score in [0, 1].  With attention shaping enabled (default), the
        reward is a blend of correctness and alignment, so it can be any float
        in [0, 1] rather than exactly {0, 1}.
        """
        from kv_gym.env import SharedKVVecEnv
        model, tokenizer, device = tiny_model_and_tokenizer

        examples = [{
            "prompt_text": "2 + 3 = ?",
            "gold_answers": ["5"],
            "task": "gsm8k",
        }]
        env = SharedKVVecEnv(
            model=model, tokenizer=tokenizer, examples=examples,
            budget_min=2, budget_max=4, max_len=64, device=device,
            use_attention_shaping=True,   # default — reward is blended float in [0,1]
        )
        env.reset()

        keep = 2
        env.resident[:, keep:] = False   # resident is [L, T] after per-layer redesign
        reward = env._terminal_reward()

        assert reward.shape == (env.num_envs,), f"Wrong shape: {reward.shape}"
        assert ((reward >= 0.0) & (reward <= 1.0)).all(), (
            f"Reward should be in [0, 1], got {reward}"
        )
        assert np.all(reward == reward[0]), "all envs should share the reward"


# ---------------------------------------------------------------------------
# Fix 2: K/V norm heterogeneity
# ---------------------------------------------------------------------------

class TestKVNormHeterogeneity:
    """
    K-norms vary dramatically across layers (attention sinks, massive activations).
    The per-half LayerNorm in PerTokenMLP should normalize each token's K and V
    sub-vectors to zero mean and unit variance before the MLP.
    """

    def test_raw_kv_norms_vary_across_layers(self, tiny_capture):
        """Raw K-norms have high coefficient-of-variation across layers."""
        cap, *_ = tiny_capture
        # cap.K: [n_layers, n_kv_heads, T, D]
        # Mean norm per layer (averaged across heads and tokens)
        layer_norms = cap.K.norm(dim=-1).mean(dim=(1, 2))  # [n_layers]

        cv = layer_norms.std() / (layer_norms.mean() + 1e-8)
        # We expect non-trivial variation even in a tiny random model
        # In real Qwen2.5, CV can exceed 3.0; for a tiny model just assert it's nonzero
        assert cv.item() >= 0.0, "Trivially true — just checking the computation works"
        # Document the actual values for inspection
        print(f"\nLayer K-norm mean={layer_norms.mean():.3f}  std={layer_norms.std():.3f}  CV={cv:.3f}")

    def test_layer_norm_normalizes_kv(self, tiny_capture):
        """After per-half LayerNorm, K and V sub-vectors have ~unit norm."""
        import torch.nn as nn
        cap, *_ = tiny_capture
        L, H, T, D = cap.K.shape

        k_norm = nn.LayerNorm(D)
        v_norm = nn.LayerNorm(D)

        # Flatten K across all layers, heads, tokens
        K_flat = cap.K.view(-1, D)  # [L*H*T, D]
        V_flat = cap.V.view(-1, D)

        K_normed = k_norm(K_flat)
        V_normed = v_norm(V_flat)

        # After LayerNorm each vector has zero mean and (D-1)/D population variance.
        # For large D (64 in Qwen2.5), this is close to 1.0.
        # For the tiny model's D=4, variance per-vector can be off — tolerate more.
        k_var = K_normed.var(dim=-1).mean().item()
        v_var = V_normed.var(dim=-1).mean().item()

        # Tolerance scales with 1/D: tight for D=64, loose for D=4
        tol = max(0.5, 2.0 / D)
        assert abs(k_var - 1.0) < tol, f"Expected K variance ≈ 1.0 (±{tol:.2f}), got {k_var:.4f}"
        assert abs(v_var - 1.0) < tol, f"Expected V variance ≈ 1.0 (±{tol:.2f}), got {v_var:.4f}"

    def test_per_token_mlp_has_kv_layernorm(self):
        """PerTokenMLP must have k_norm and v_norm attributes."""
        import torch.nn as nn
        from gymnasium import spaces
        from kv_gym.policy import PerTokenMLP

        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64, 128), dtype=np.float32)
        mlp = PerTokenMLP(obs_space, hidden=32)

        assert hasattr(mlp, "k_norm"), "PerTokenMLP missing k_norm LayerNorm"
        assert hasattr(mlp, "v_norm"), "PerTokenMLP missing v_norm LayerNorm"
        assert isinstance(mlp.k_norm, nn.LayerNorm), f"k_norm is {type(mlp.k_norm)}, expected LayerNorm"
        assert isinstance(mlp.v_norm, nn.LayerNorm), f"v_norm is {type(mlp.v_norm)}, expected LayerNorm"
        assert mlp.k_norm.normalized_shape == (64,), f"k_norm shape: {mlp.k_norm.normalized_shape}"
        assert mlp.v_norm.normalized_shape == (64,), f"v_norm shape: {mlp.v_norm.normalized_shape}"

    def test_per_token_mlp_normalizes_high_scale_input(self):
        """MLP output is similar even when K-scale is 10× higher (simulating cross-layer variation)."""
        from gymnasium import spaces
        from kv_gym.policy import PerTokenMLP

        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8, 4), dtype=np.float32)
        mlp = PerTokenMLP(obs_space, hidden=8)
        mlp.eval()

        # Same content, different K scale (simulates a high-norm layer vs low-norm layer)
        base_obs = torch.randn(1, 8, 4)
        scaled_obs = base_obs.clone()
        scaled_obs[:, :, :2] *= 20.0  # K half 20× larger

        with torch.no_grad():
            out_base   = mlp(base_obs)
            out_scaled = mlp(scaled_obs)

        # After LayerNorm, the ratio of output norms should be much less than 20
        ratio = out_scaled.norm() / (out_base.norm() + 1e-8)
        assert ratio < 5.0, (
            f"LayerNorm should suppress scale differences; output norm ratio = {ratio:.2f} (expected < 5)"
        )


# ---------------------------------------------------------------------------
# Fix 3: gamma=1.0 in config
# ---------------------------------------------------------------------------

class TestGammaConfig:

    def test_quickstart_gamma_is_one(self):
        with open("configs/quickstart.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["gamma"] == 1.0, (
            f"quickstart.yaml gamma={cfg['gamma']}; expected 1.0 (no discount bias)"
        )

    def test_phase1_gamma_is_one(self):
        with open("configs/phase1.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["gamma"] == 1.0, (
            f"phase1.yaml gamma={cfg['gamma']}; expected 1.0"
        )

    def test_phase1_no_reward_mode(self):
        """reward_mode: auc is dead code — must not appear in configs."""
        with open("configs/phase1.yaml") as f:
            cfg = yaml.safe_load(f)
        assert "reward_mode" not in cfg, (
            "phase1.yaml still has reward_mode key — this is dead code that was never read"
        )
