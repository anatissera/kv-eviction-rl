"""
Smoke tests for SharedKVVecEnv.

Uses the tiny-random-LlamaForCausalLM model (CPU, no download needed) so
these run in seconds without a GPU. They verify shapes and invariants, not
reward magnitudes.

Run with:
    cd kv-eviction-gym && pip install -e . && pytest tests/
"""

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    from kv_gym.vendor.loader import load_model_and_tokenizer
    model, tokenizer, device = load_model_and_tokenizer(
        name="tiny-llama",
        device=torch.device("cpu"),
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer, device


@pytest.fixture(scope="module")
def tiny_examples():
    return [
        {
            "prompt_text": "Janet has 3 apples. She buys 2 more. How many?",
            "gold_answers": ["5"],
            "task": "gsm8k",
        },
        {
            "prompt_text": "A train travels 60 miles per hour for 2 hours. How far?",
            "gold_answers": ["120"],
            "task": "gsm8k",
        },
    ]


def test_capture_shapes(tiny_model_and_tokenizer, tiny_examples):
    from kv_gym.capture import capture
    model, tokenizer, device = tiny_model_and_tokenizer
    cap = capture(model, tokenizer, tiny_examples[0], device, max_new_tokens=4)

    cfg = model.config
    L   = cfg.num_hidden_layers
    n_q = cfg.num_attention_heads
    Hkv = getattr(cfg, "num_key_value_heads", n_q)
    D   = cfg.hidden_size // n_q
    T   = cap.prompt_len

    # All captured tensors use KV-head count (Q-heads averaged into groups)
    assert cap.Q.shape == (L, Hkv, T, D), f"Q shape mismatch: {cap.Q.shape}"
    assert cap.K.shape == (L, Hkv, T, D)
    assert cap.V.shape == (L, Hkv, T, D)
    assert cap.attn_score.shape == (L, Hkv, T), f"attn_score shape: {cap.attn_score.shape}"
    assert cap.future_attn.shape == (L, Hkv, T)

    # attn_score should be non-negative (column sums of softmax weights)
    assert (cap.attn_score >= 0).all(), "attn_score has negative values"

    # future_attn should be normalized: each head sums to ~1
    row_sums = cap.future_attn.sum(dim=-1)  # [L, H]
    assert (row_sums > 0).all(), "future_attn has all-zero heads"
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_env_reset_shapes(tiny_model_and_tokenizer, tiny_examples):
    from kv_gym.env import SharedKVVecEnv
    model, tokenizer, device = tiny_model_and_tokenizer

    env = SharedKVVecEnv(
        model=model, tokenizer=tokenizer, examples=tiny_examples,
        budget=4, max_len=64, reward_mode="auc", device=device,
    )
    obs = env.reset()

    n_envs = env.num_envs
    max_len = env.max_len
    feature_dim = 2 * env.head_dim + 5

    assert obs.shape == (n_envs, max_len, feature_dim), f"obs shape: {obs.shape}"
    assert np.isfinite(obs).all(), "obs contains NaN or Inf"


def test_env_action_masks(tiny_model_and_tokenizer, tiny_examples):
    from kv_gym.env import SharedKVVecEnv
    model, tokenizer, device = tiny_model_and_tokenizer

    env = SharedKVVecEnv(
        model=model, tokenizer=tokenizer, examples=tiny_examples,
        budget=4, max_len=64, reward_mode="auc", device=device,
    )
    env.reset()
    masks = env.action_masks()

    T = env.capture.prompt_len
    assert masks.shape == (env.num_envs, env.max_len)
    # Prompt positions should all be valid initially
    assert masks[:, :T].all(), "All prompt positions should be valid after reset"
    # Positions beyond prompt should be masked out
    assert not masks[:, T:].any(), "Positions beyond prompt should be masked"


def test_env_full_episode(tiny_model_and_tokenizer, tiny_examples):
    from kv_gym.env import SharedKVVecEnv
    model, tokenizer, device = tiny_model_and_tokenizer

    budget = 4
    env = SharedKVVecEnv(
        model=model, tokenizer=tokenizer, examples=tiny_examples,
        budget=budget, max_len=64, reward_mode="auc", device=device,
    )
    obs = env.reset()
    T = env.capture.prompt_len

    step_count = 0
    max_steps = T + 10  # should never exceed T - budget steps
    done = False

    while not done and step_count < max_steps:
        masks = env.action_masks()
        # Pick first valid action per env
        actions = np.array([
            np.argmax(masks[i]) for i in range(env.num_envs)
        ])
        obs, rewards, dones, infos = env.step(actions)
        done = bool(dones[0])
        step_count += 1

    assert done, f"Episode did not terminate in {max_steps} steps"

    # Rewards at episode end should all be in (0, 1]
    assert (rewards >= 0).all() and (rewards <= 1.0 + 1e-6).all(), (
        f"Rewards out of range: min={rewards.min():.4f} max={rewards.max():.4f}"
    )
    # Number of steps should equal T - budget (one eviction per step)
    assert step_count == T - budget, (
        f"Expected {T - budget} steps, got {step_count}"
    )


def test_auc_reward_oracle_is_one(tiny_model_and_tokenizer, tiny_examples):
    """Keeping the oracle (top-k by future_attn) tokens gives reward == 1."""
    from kv_gym.capture import capture
    from kv_gym.rewards.auc import future_attention_auc
    model, tokenizer, device = tiny_model_and_tokenizer

    cap = capture(model, tokenizer, tiny_examples[0], device, max_new_tokens=4)
    L, H, T = cap.future_attn.shape
    budget = min(4, T)
    n_envs = L * H

    fa = cap.future_attn.view(n_envs, T)
    _, topk_idx = fa.topk(k=budget, dim=-1)

    resident = torch.zeros(n_envs, T, dtype=torch.bool)
    for i in range(n_envs):
        resident[i, topk_idx[i]] = True

    rewards = future_attention_auc(fa, resident, budget)
    assert torch.allclose(rewards, torch.ones(n_envs), atol=1e-4), (
        f"Oracle should give reward=1, got {rewards}"
    )
