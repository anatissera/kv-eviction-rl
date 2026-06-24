# RL KV-Cache Eviction — Final Project

Reinforcement learning for adaptive KV-cache eviction in large language models.

## Projects

### `kv-eviction-gym/` — Sequential Gym Environment (main)

A new standalone project that wraps KV-cache eviction as a vectorized
`gymnasium`-style environment and trains it with SB3 `MaskablePPO`.

- **Dataset**: GSM8K math word problems (~100-200 token prompts)
- **Action**: evict 1 token per step (`Discrete(max_len)` + action mask)
- **Agents**: one shared policy across all 56 (layer, head) pairs of Qwen2-1.5B
- **Key efficiency**: one LLM prefill per GSM8K example feeds all 56 environments simultaneously

See `kv-eviction-gym/` for setup and training instructions.

### `ml-learning-to-evict/` — RLOO / PPO Bandit Training (reference)

Adaptation of the [Learning to Evict](https://arxiv.org/abs/...) paper's
Plackett-Luce one-shot ranking approach, extended with a PPO variant with
value critic. Uses RULER and GSM8K data.

See `ml-learning-to-evict/README.md` for the original setup instructions.
