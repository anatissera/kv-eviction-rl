import numpy as np
import pytest
import torch

from kv_gym.vendor.loader import load_model_and_tokenizer
from kv_gym.eval_core import (
    score_full_cache,
    run_online_episode,
    make_learned_evict_fn,
    make_streaming_evict_fn,
    make_attention_evict_fn,
    make_kv_norm_evict_fn,
    make_random_evict_fn,
)
from kv_gym.capture import capture
from sb3_contrib import MaskablePPO


@pytest.fixture(scope="module")
def tiny_setup():
    model, tokenizer, device = load_model_and_tokenizer(
        name="tiny-llama",
        device=torch.device("cpu"),
    )
    model.config._attn_implementation = "eager"
    model.eval()
    
    example = {
        "prompt_text": "Janet has 3 apples. She buys 2 more. How many?",
        "gold_answers": ["5"],
        "task": "gsm8k",
    }
    cap = capture(model, tokenizer, example, device)
    return model, tokenizer, device, cap


def test_score_full_cache(tiny_setup):
    model, tokenizer, device, cap = tiny_setup
    score, speed, peak_mem = score_full_cache(
        model, tokenizer, cap.input_ids, cap.gold_answer, device, max_new_tokens=5
    )
    assert isinstance(score, float)
    assert isinstance(speed, float)
    assert isinstance(peak_mem, float)
    print(f"score_full_cache: score={score}, speed={speed:.2f} t/s, peak_mem={peak_mem:.2f} MB")


def test_run_online_episodes(tiny_setup):
    model, tokenizer, device, cap = tiny_setup
    L = model.config.num_hidden_layers
    n_sinks = 2
    n_recent = 2
    budget = 10
    max_new_tokens = 5
    
    # 1. Streaming eviction
    evict_fn = make_streaming_evict_fn(n_sinks, L)
    text, corr, evicted, truncated, speed, peak_mem = run_online_episode(
        model, tokenizer, cap.input_ids, budget, max_new_tokens, device, evict_fn
    )
    assert isinstance(text, str)
    assert isinstance(corr, list)
    assert isinstance(evicted, list)
    assert isinstance(truncated, bool)
    assert isinstance(speed, float)
    assert isinstance(peak_mem, float)
    
    # 2. Attention oracle eviction
    evict_fn = make_attention_evict_fn(L, n_sinks, n_recent)
    text, corr, evicted, truncated, speed, peak_mem = run_online_episode(
        model, tokenizer, cap.input_ids, budget, max_new_tokens, device, evict_fn
    )
    assert isinstance(text, str)
    assert isinstance(corr, list)
    assert isinstance(evicted, list)
    assert isinstance(truncated, bool)
    assert isinstance(speed, float)
    assert isinstance(peak_mem, float)
    
    # 3. KV norm eviction
    evict_fn = make_kv_norm_evict_fn(L, n_sinks, n_recent)
    text, corr, evicted, truncated, speed, peak_mem = run_online_episode(
        model, tokenizer, cap.input_ids, budget, max_new_tokens, device, evict_fn
    )
    assert isinstance(text, str)
    assert isinstance(corr, list)
    assert isinstance(evicted, list)
    assert isinstance(truncated, bool)
    assert isinstance(speed, float)
    assert isinstance(peak_mem, float)

    # 4. Random eviction
    rng = np.random.default_rng(42)
    evict_fn = make_random_evict_fn(L, rng, n_sinks, n_recent)
    text, corr, evicted, truncated, speed, peak_mem = run_online_episode(
        model, tokenizer, cap.input_ids, budget, max_new_tokens, device, evict_fn
    )
    assert isinstance(text, str)
    assert isinstance(corr, list)
    assert isinstance(evicted, list)
    assert isinstance(truncated, bool)
    assert isinstance(speed, float)
    assert isinstance(peak_mem, float)
