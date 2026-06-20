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

from kv_gym.features import feature_dim


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
    cap = capture(model, tokenizer, tiny_examples[0], device)

    cfg  = model.config
    L    = cfg.num_hidden_layers
    Hkv  = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    D    = cfg.hidden_size // cfg.num_attention_heads
    T    = cap.prompt_len

    assert cap.K.shape == (L, Hkv, T, D), f"K shape: {cap.K.shape}"
    assert cap.V.shape == (L, Hkv, T, D), f"V shape: {cap.V.shape}"
    assert cap.input_ids.shape == (1, T)


def test_env_reset_shapes(tiny_model_and_tokenizer, tiny_examples):
    from kv_gym.env import SharedKVVecEnv
    model, tokenizer, device = tiny_model_and_tokenizer

    env = SharedKVVecEnv(
        model=model, tokenizer=tokenizer, examples=tiny_examples,
        budget=4, max_len=64, device=device,
    )
    obs = env.reset()

    fdim = feature_dim(env.head_dim)
    assert obs.shape == (env.num_envs, env.max_len, fdim), f"obs: {obs.shape}"
    assert np.isfinite(obs).all(), "obs contains NaN or Inf"


def test_env_action_masks(tiny_model_and_tokenizer, tiny_examples):
    from kv_gym.env import SharedKVVecEnv
    model, tokenizer, device = tiny_model_and_tokenizer

    env = SharedKVVecEnv(
        model=model, tokenizer=tokenizer, examples=tiny_examples,
        budget=4, max_len=64, device=device,
    )
    env.reset()
    masks = env.action_masks()
    T = env.capture.prompt_len

    assert masks.shape == (env.num_envs, env.max_len)
    assert masks[:, :T].all(),      "all prompt positions valid after reset"
    assert not masks[:, T:].any(),  "positions beyond prompt masked out"


def test_env_full_episode(tiny_model_and_tokenizer, tiny_examples):
    from kv_gym.env import SharedKVVecEnv
    model, tokenizer, device = tiny_model_and_tokenizer

    budget = 4
    env = SharedKVVecEnv(
        model=model, tokenizer=tokenizer, examples=tiny_examples,
        budget=budget, max_len=64, device=device,
    )
    obs   = env.reset()
    T     = env.capture.prompt_len
    done  = False
    steps = 0

    while not done and steps < T + 10:
        masks   = env.action_masks()
        actions = np.array([np.argmax(masks[i]) for i in range(env.num_envs)])
        obs, rewards, dones, infos = env.step(actions)
        done   = bool(dones[0])
        steps += 1

    assert done, f"episode did not terminate in {T + 10} steps"
    assert steps == T - budget, f"expected {T - budget} steps, got {steps}"

    # terminal reward from correctness: in {0.0, 1.0}
    assert ((rewards == 0.0) | (rewards == 1.0)).all(), (
        f"correctness reward should be 0 or 1, got {rewards}"
    )
    # all envs share the same reward
    assert np.all(rewards == rewards[0]), "all envs should share the reward"
