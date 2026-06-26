"""
BatchedSharedKVVecEnv — N episodes decoded in parallel with one batched forward pass.

Architecture
------------
n_envs = N * n_layers.  Each "env" is identified by (episode_idx, layer_idx):
    env_id = episode_b * n_layers + layer_l

All N episodes share the same budget so every episode stays at
cache_size = budget + 1 throughout the eviction phase.  This invariant lets
us maintain a single batched DynamicCache of shape [N, H, S, D] and run ONE
model forward pass per step instead of N sequential batch=1 passes.

Per-layer eviction is applied batch-wise via index-gather (physically removing
the evicted slot per (episode, layer)) — verified equivalent to N independent
_evict_slot calls in test_batched_forward.py TEST 4.

Individual episode resets run sequentially (prefill + free-growth, batch=1)
and reinstate the same shared budget so the replaced cache slice matches
the shape of the running batch.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from kv_gym.env import _get_layer_kv, _evict_slot
from kv_gym.features import feature_dim
from kv_gym.eval_core import valid_action_mask
from kv_gym.rewards.attention_shaping import compute_token_importance
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract
from kv_gym.vendor.prompts import format_gsm8k
from kv_gym.free_growth_cache import FreeGrowthCache
from transformers import DynamicCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DynamicCache helpers (same as env.py but for batched [N, H, S, D] tensors)
# ---------------------------------------------------------------------------

def _n_layers_cache(cache: DynamicCache) -> int:
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache.key_cache)


def _get_kv_bat(cache: DynamicCache, l: int):
    """Return (K, V) [N, H, S, D] for layer l."""
    if hasattr(cache, "layers"):
        return cache.layers[l].keys, cache.layers[l].values
    return cache.key_cache[l], cache.value_cache[l]


def _set_kv_bat(cache: DynamicCache, l: int, K: torch.Tensor, V: torch.Tensor) -> None:
    if hasattr(cache, "layers"):
        cache.layers[l].keys   = K
        cache.layers[l].values = V
    else:
        cache.key_cache[l]   = K
        cache.value_cache[l] = V


def _stack_caches(caches: list[DynamicCache]) -> DynamicCache:
    """Stack N individual [1, H, S, D] caches → one batched [N, H, S, D] cache."""
    n_layers = _n_layers_cache(caches[0])
    batched  = DynamicCache()
    for l in range(n_layers):
        Ks, Vs = zip(*[_get_kv_bat(c, l) for c in caches])
        K_bat = torch.cat(list(Ks), dim=0)
        V_bat = torch.cat(list(Vs), dim=0)
        if hasattr(caches[0], "layers"):
            import copy
            batched.layers.append(copy.copy(caches[0].layers[l]))
            batched.layers[l].keys   = K_bat
            batched.layers[l].values = V_bat
        else:
            batched.key_cache.append(K_bat)
            batched.value_cache.append(V_bat)
    return batched


def _replace_slice(batch_cache: DynamicCache, b: int,
                   ep_cache: DynamicCache, n_layers: int) -> None:
    """Replace episode b's slice in batch_cache with ep_cache (in-place)."""
    for l in range(n_layers):
        K_bat, V_bat = _get_kv_bat(batch_cache, l)
        K_ep,  V_ep  = _get_kv_bat(ep_cache, l)
        K_bat[b:b+1] = K_ep
        V_bat[b:b+1] = V_ep
        _set_kv_bat(batch_cache, l, K_bat, V_bat)


# ---------------------------------------------------------------------------
# Per-episode state
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    past_kv:          DynamicCache | None = None
    next_token:       int   = 0
    true_position:    int   = 0
    cache_size:       int   = 0
    step_count:       int   = 0
    generated:        list  = field(default_factory=list)
    gold_answer:      str   = ""
    prompt_len:       int   = 0
    budget:           int   = 0
    slot_to_pos:      list  = field(default_factory=list)   # [n_layers][S]
    token_importance: object = None  # torch.Tensor | None
    started_at:       float = 0.0
    count:            int   = 0
    prompt_ids:       object = None  # torch.Tensor | None — kept for cache reconstruction


# ---------------------------------------------------------------------------
# Batched VecEnv
# ---------------------------------------------------------------------------

class BatchedSharedKVVecEnv(VecEnv):
    """N episodes decoded in parallel — one batched forward pass per step."""

    metadata    = {}
    render_mode = None
    spec        = None

    def __init__(
        self,
        model,
        tokenizer,
        examples:              list[dict],
        n_parallel:            int   = 4,
        budget_min:            int   = 128,
        budget_max:            int   = 256,
        max_len:               int   = 256,
        max_new_tokens:        int   = 512,
        device:                object = None,
        use_attention_shaping: bool  = False,
        attention_weight:      float = 0.0,
        shaping_mode:          str   = "none",
        n_sinks:               int   = 4,
        n_recent:              int   = 8,
        length_penalty_weight: float = 0.0,
        truncation_penalty:    float = 0.0,
        seed:                  int   = 0,
        free_growth_cache:     FreeGrowthCache | None = None,
    ):
        self.model                 = model
        self.tokenizer             = tokenizer
        self.examples              = examples
        self.N                     = n_parallel
        self.budget_min            = budget_min
        self.budget_max            = budget_max
        self.max_len               = max_len
        self.max_new_tokens        = max_new_tokens
        self.device                = device or next(model.parameters()).device
        self.use_attention_shaping = use_attention_shaping
        self.attention_weight      = attention_weight
        self.shaping_mode          = shaping_mode
        self.n_sinks               = n_sinks
        self.n_recent              = n_recent
        self.length_penalty_weight = length_penalty_weight
        self.truncation_penalty    = truncation_penalty
        self.n_examples            = len(examples)
        self._rng                  = np.random.default_rng(seed)
        self.eos_id                = tokenizer.eos_token_id
        self._cache                = free_growth_cache
        self._example_cursor: int  = 0   # tracks which example we're on for cache keys

        cfg             = model.config
        self.n_layers   = cfg.num_hidden_layers
        self.n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim   = cfg.hidden_size // cfg.num_attention_heads

        fdim      = feature_dim(self.n_kv_heads, self.head_dim)
        obs_space = spaces.Box(low=-np.inf, high=np.inf,
                               shape=(max_len, fdim), dtype=np.float32)
        act_space = spaces.Discrete(max_len)
        super().__init__(self.N * self.n_layers, obs_space, act_space)

        self._example_iter: Iterator[dict] = itertools.cycle(examples)  # kept for compat
        self._pending_actions: np.ndarray | None = None

        # Shared budget for all episodes in the current batch
        self._shared_budget: int = budget_min

        # Per-episode state
        self.episodes: list[EpisodeState] = [EpisodeState() for _ in range(self.N)]

        # Batched KV cache [N, H, S, D] per layer
        self._batch_kv: DynamicCache | None = None

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset(self):
        # Sample one shared budget for all N episodes
        self._shared_budget = int(self._rng.integers(self.budget_min, self.budget_max + 1))
        for b in range(self.N):
            self._reset_episode(b)
        self._batch_kv = _stack_caches([ep.past_kv for ep in self.episodes])
        return self._obs()

    def step_async(self, actions: np.ndarray):
        self._pending_actions = actions

    def step_wait(self):
        # actions: [N * n_layers] → [N, n_layers]
        actions = self._pending_actions.reshape(self.N, self.n_layers)
        slots_t = torch.tensor(actions, device=self.device, dtype=torch.long)  # [N, L]

        S = self.episodes[0].cache_size   # all episodes share same cache_size
        H = self.n_kv_heads
        D = self.head_dim

        # 1. Per-(episode, layer) physical eviction on batched KV
        for l in range(self.n_layers):
            K, V = _get_kv_bat(self._batch_kv, l)   # [N, H, S, D]
            keep = torch.ones(self.N, S, dtype=torch.bool, device=self.device)
            keep[torch.arange(self.N, device=self.device), slots_t[:, l]] = False
            expand = keep.unsqueeze(1).unsqueeze(-1).expand(self.N, H, S, D)
            K_new  = K[expand].reshape(self.N, H, S - 1, D)
            V_new  = V[expand].reshape(self.N, H, S - 1, D)
            _set_kv_bat(self._batch_kv, l, K_new, V_new)

        # Update per-episode slot_to_pos after eviction
        for b in range(self.N):
            ep = self.episodes[b]
            for l in range(self.n_layers):
                slot = int(actions[b, l])
                slot = max(0, min(slot, ep.cache_size - 1))
                ep.slot_to_pos[l].pop(slot)
            ep.cache_size -= 1

        # 2. One batched decode step
        next_toks = torch.tensor(
            [[ep.next_token] for ep in self.episodes], device=self.device
        )  # [N, 1]
        pos_ids = torch.tensor(
            [[ep.true_position] for ep in self.episodes], device=self.device
        )  # [N, 1]
        # All sequences have same length → no padding mask needed
        with torch.no_grad():
            out = self.model(
                input_ids=next_toks,
                past_key_values=self._batch_kv,
                position_ids=pos_ids,
                use_cache=True,
            )
        self._batch_kv = out.past_key_values
        new_tokens = out.logits[:, -1].argmax(dim=-1).cpu().tolist()  # [N]

        # 3. Update per-episode state
        for b, ep in enumerate(self.episodes):
            ep.generated.append(ep.next_token)
            for l in range(self.n_layers):
                ep.slot_to_pos[l].append(ep.true_position)
            ep.true_position += 1
            ep.step_count    += 1
            ep.cache_size    += 1   # net: -1 evict + 1 decode = 0 → budget+1
            ep.next_token     = new_tokens[b]

        # 4. Collect outputs; handle done episodes
        all_obs     = np.zeros((self.N * self.n_layers, self.max_len,
                                2 * self.n_kv_heads * self.head_dim), dtype=np.float32)
        all_rewards = np.zeros(self.N * self.n_layers, dtype=np.float32)
        all_dones   = np.zeros(self.N * self.n_layers, dtype=bool)
        all_infos   = [{} for _ in range(self.N * self.n_layers)]

        for b, ep in enumerate(self.episodes):
            done = (ep.next_token == self.eos_id or ep.step_count >= self.max_new_tokens)
            base = b * self.n_layers

            if done:
                truncated   = (ep.next_token != self.eos_id and
                               ep.step_count >= self.max_new_tokens)
                rewards, correct, align = self._terminal_reward(b, truncated)
                terminal_obs = self._obs_episode(b)

                ep.count += 1
                ep_info = {
                    "terminal_observation": terminal_obs[l] for l in range(self.n_layers)
                }
                for l in range(self.n_layers):
                    all_rewards[base + l] = rewards[l]
                    all_dones  [base + l] = True
                    all_infos  [base + l] = {
                        "terminal_observation": terminal_obs[l],
                        "correct":        correct,
                        "alignment":      align,
                        "episode_seconds": time.perf_counter() - ep.started_at,
                        "context_size":   ep.prompt_len + ep.step_count,
                        "truncated":      truncated,
                    }

                # Reset this episode (batch=1, sequential)
                self._reset_episode(b)
                # Replace slice b in the batched cache
                _replace_slice(self._batch_kv, b, ep.past_kv, self.n_layers)

            # obs for this episode (after possible reset → shows new episode start)
            ep_obs = self._obs_episode(b)
            all_obs[base : base + self.n_layers] = ep_obs

        return all_obs, all_rewards, all_dones, all_infos

    def action_masks(self) -> np.ndarray:
        """[N * n_layers, max_len] — same mask broadcast across layers per episode."""
        masks = np.zeros((self.N * self.n_layers, self.max_len), dtype=bool)
        for b, ep in enumerate(self.episodes):
            row  = valid_action_mask(ep.cache_size, self.max_len, self.n_sinks, self.n_recent)
            base = b * self.n_layers
            masks[base : base + self.n_layers] = row
        return masks

    # ------------------------------------------------------------------
    # VecEnv stubs
    # ------------------------------------------------------------------

    def close(self): pass

    def get_attr(self, attr_name, indices=None):
        indices = list(range(self.num_envs)) if indices is None else list(indices)
        return [getattr(self, attr_name, None)] * len(indices)

    def set_attr(self, attr_name, value, indices=None): pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == "action_masks":
            masks = self.action_masks()
            idx   = list(range(self.num_envs)) if indices is None else list(indices)
            return [masks[i] for i in idx]
        raise NotImplementedError(f"env_method('{method_name}') not supported")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_episode(self, b: int) -> None:
        """Prefill + free-growth for episode b (batch=1, sequential).

        Uses self._shared_budget so the new cache_size matches the running batch.
        Retries if the episode finishes during free-growth (no eviction needed).

        When a FreeGrowthCache is attached, the free-growth phase is replaced by
        a single forward pass (prefill with prompt + cached tokens) on cache hits,
        saving ~300-500 sequential decode steps per episode reset.
        """
        ep = self.episodes[b]
        while True:
            ep.started_at  = time.perf_counter()
            example_idx    = self._example_cursor % self.n_examples
            example        = self.examples[example_idx]
            self._example_cursor += 1

            prompt_txt, _ = format_gsm8k(example)
            ep.gold_answer = example["gold_answers"][0]

            inputs = self.tokenizer(prompt_txt, return_tensors="pt").to(self.device)
            T      = inputs["input_ids"].shape[1]
            if T > self.max_len:
                continue

            ep.budget     = max(self._shared_budget, T)
            ep.prompt_len = T
            ep.step_count = 0
            ep.generated  = []
            ep.prompt_ids = inputs["input_ids"]  # kept for cache reconstruction

            # ── Try cache hit ──────────────────────────────────────────────
            if self._cache is not None:
                cached = self._cache.get(example_idx)
                if cached is not None:
                    n_needed = ep.budget + 1 - T
                    if len(cached) >= n_needed:
                        # Full hit: reconstruct KV in one forward pass
                        fg_tokens = cached[:n_needed].tolist()
                        if self.eos_id in fg_tokens:
                            continue  # would have ended during free-growth — skip
                        self._reconstruct_from_tokens(ep, fg_tokens)
                        break
                    elif len(cached) > 0 and self.eos_id not in cached.tolist():
                        # Partial hit: reconstruct what we have, then extend
                        self._reconstruct_from_tokens(ep, cached.tolist())
                        free_growth_done = self._run_free_growth_sequential(ep, example_idx,
                                                                             cached_so_far=list(cached))
                        if not free_growth_done:
                            break
                        continue

            # ── Cache miss: normal prefill + sequential free-growth ────────
            self.model.eval()
            with torch.no_grad():
                out = self.model(input_ids=inputs["input_ids"], use_cache=True)
            ep.past_kv       = out.past_key_values
            ep.next_token    = int(out.logits[0, -1].argmax().item())
            ep.true_position = T
            ep.cache_size    = T
            ep.slot_to_pos   = [list(range(T)) for _ in range(self.n_layers)]

            if self.use_attention_shaping:
                K_list, V_list = [], []
                for l in range(self.n_layers):
                    K_l, V_l = _get_layer_kv(ep.past_kv, l)
                    K_list.append(K_l); V_list.append(V_l)
                ep.token_importance = compute_token_importance(
                    model=self.model,
                    input_ids=inputs["input_ids"].cpu(),
                    K=torch.stack(K_list), V=torch.stack(V_list),
                    max_new_tokens=self.max_new_tokens, device=self.device,
                )
            else:
                ep.token_importance = None

            free_growth_done = self._run_free_growth_sequential(ep, example_idx,
                                                                 cached_so_far=[])
            if not free_growth_done:
                break

        logger.info("episode=%d  b=%d  prompt_len=%d  budget=%d  cache_size=%d",
                    ep.count, b, ep.prompt_len, ep.budget, ep.cache_size)

    # ------------------------------------------------------------------
    # Free-growth helpers
    # ------------------------------------------------------------------

    def _reconstruct_from_tokens(self, ep: EpisodeState, fg_tokens: list[int]) -> None:
        """Reconstruct the KV cache from cached free-growth tokens in one forward pass.

        Replaces ~len(fg_tokens) sequential decode steps with a single prefill of
        (prompt + fg_tokens), which is equivalent by the causal attention invariant.
        """
        T          = ep.prompt_len
        tok_tensor = torch.tensor([fg_tokens], dtype=torch.long, device=self.device)
        full_input = torch.cat([ep.prompt_ids, tok_tensor], dim=1)  # [1, T+fg_len]
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=full_input, use_cache=True)
        ep.past_kv       = out.past_key_values
        ep.next_token    = int(out.logits[0, -1].argmax().item())
        ep.true_position = T + len(fg_tokens)
        ep.cache_size    = T + len(fg_tokens)
        ep.step_count    = len(fg_tokens)
        ep.generated     = list(fg_tokens)
        ep.slot_to_pos   = [list(range(ep.cache_size)) for _ in range(self.n_layers)]
        ep.token_importance = None  # attention shaping not supported with cache

    def _run_free_growth_sequential(self, ep: EpisodeState, example_idx: int,
                                    cached_so_far: list[int]) -> bool:
        """Sequential decode until cache_size > ep.budget.

        Continues from whatever state ep is currently in (either after normal
        prefill or after partial cache reconstruction).  Saves newly generated
        tokens to cache on success.

        Returns True if the episode ended during free-growth (caller should retry).
        """
        new_tokens: list[int] = []
        while ep.cache_size <= ep.budget:
            pos = torch.tensor([[ep.true_position]], device=self.device)
            with torch.no_grad():
                step_out = self.model(
                    input_ids=torch.tensor([[ep.next_token]], device=self.device),
                    past_key_values=ep.past_kv,
                    position_ids=pos,
                    cache_position=pos.squeeze(0),
                    use_cache=True,
                )
            ep.generated.append(ep.next_token)
            new_tokens.append(ep.next_token)
            ep.past_kv  = step_out.past_key_values
            new_tok     = int(step_out.logits[0, -1].argmax().item())
            for l in range(self.n_layers):
                ep.slot_to_pos[l].append(ep.true_position)
            ep.true_position += 1
            ep.step_count    += 1
            ep.cache_size    += 1
            ep.next_token     = new_tok
            if new_tok == self.eos_id or ep.step_count >= self.max_new_tokens:
                if new_tok != self.eos_id:
                    ep.generated.append(new_tok)
                return True  # ended during free-growth — don't cache this episode

        # Free-growth completed normally: update the cache with all tokens so far.
        if self._cache is not None:
            self._cache.put(example_idx, cached_so_far + new_tokens)

        return False

    def _obs_episode(self, b: int) -> np.ndarray:
        """[n_layers, max_len, feature_dim] observation for episode b from batch KV."""
        ep  = self.episodes[b]
        S   = ep.cache_size
        H   = self.n_kv_heads
        D   = self.head_dim
        obs = np.zeros((self.n_layers, self.max_len, 2 * H * D), dtype=np.float32)
        if self._batch_kv is not None and S > 0:
            for l in range(self.n_layers):
                K_bat, V_bat = _get_kv_bat(self._batch_kv, l)   # [N, H, S, D]
                K = K_bat[b].float().cpu()   # [H, S, D]
                V = V_bat[b].float().cpu()
                K_flat = K.transpose(0, 1).reshape(S, H * D).numpy()
                V_flat = V.transpose(0, 1).reshape(S, H * D).numpy()
                obs[l, :S, :H*D]      = K_flat
                obs[l, :S, H*D:2*H*D] = V_flat
        return obs

    def _obs(self) -> np.ndarray:
        """[N * n_layers, max_len, feature_dim] — all episodes."""
        parts = [self._obs_episode(b) for b in range(self.N)]
        return np.concatenate(parts, axis=0)

    def _terminal_reward(self, b: int, truncated: bool):
        """Terminal reward for episode b → [n_layers] rewards, correct bool, align float."""
        ep   = self.episodes[b]
        text = self.tokenizer.decode(ep.generated, skip_special_tokens=True)
        correct = float(flexible_extract(text, [ep.gold_answer]))

        if ep.token_importance is not None:
            T = ep.prompt_len
            soft_keep = torch.zeros(T, dtype=torch.float32)
            for l in range(self.n_layers):
                for orig_pos in ep.slot_to_pos[l]:
                    if orig_pos < T:
                        soft_keep[orig_pos] += 1.0
            soft_keep /= self.n_layers
            align = (ep.token_importance * soft_keep).sum().item()
        else:
            align = float("nan")

        if self.shaping_mode == "terminal" and ep.token_importance is not None:
            score = (1.0 - self.attention_weight) * correct + self.attention_weight * align
        else:
            score = correct

        if self.length_penalty_weight > 0.0:
            score -= self.length_penalty_weight * (ep.step_count / self.max_new_tokens)
        if self.truncation_penalty > 0.0 and truncated:
            score -= self.truncation_penalty

        return np.full(self.n_layers, score, dtype=np.float32), bool(correct), align
