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
from kv_gym.features import feature_dim, build_extra_columns, extra_feature_dim
from kv_gym.eval_core import valid_action_mask
from kv_gym.rewards.attention_shaping import compute_token_importance
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract
from kv_gym.vendor.prompts import format_gsm8k_chat
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


def _clone_cache(cache: DynamicCache) -> DynamicCache:
    """Deep-copy a DynamicCache (clone K,V tensors per layer).

    Used by the S4 shadow "full" cache: a copy of the episode's KV at the moment
    eviction begins (cache_size = budget+1), grown but NEVER evicted, so we can
    measure the next-token distribution the model WOULD produce with the full
    context at each decode step.
    """
    n_layers = _n_layers_cache(cache)
    new = DynamicCache()
    for l in range(n_layers):
        K, V = _get_kv_bat(cache, l)
        if hasattr(cache, "layers"):
            import copy
            new.layers.append(copy.copy(cache.layers[l]))
            new.layers[l].keys   = K.clone()
            new.layers[l].values = V.clone()
        else:
            new.key_cache.append(K.clone())
            new.value_cache.append(V.clone())
    return new


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
    # S4 information-theoretic shaping: shadow full (never-evicted) cache + KL accumulators
    past_kv_full:     object = None  # DynamicCache | None — full-context shadow cache
    kl_sum:           float = 0.0    # running sum of per-step damage (KL or entropy)
    kl_count:         int   = 0


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
        entropy_reward_weight: float = 0.0,
        seed:                  int   = 0,
        free_growth_cache:     FreeGrowthCache | None = None,
        kl_shaping:            bool  = False,
        kl_mode:               str   = "exact",   # "exact" (KL-to-full) | "proxy" (self-entropy)
        kl_weight:             float = 0.0,
        kl_clip:               float = 5.0,
        per_example_base_budgets: dict[int, int] | None = None,
        eviction_k:               int = 100,
        protect_prompt:           bool = False,
        rich_features:            bool = False,
    ):
        self.model                 = model
        self.tokenizer             = tokenizer
        self.examples              = examples
        self.N                     = n_parallel
        self.n_parallel            = n_parallel  # alias expected by EpisodeMaskablePPO
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
        self.protect_prompt        = protect_prompt
        self.length_penalty_weight = length_penalty_weight
        self.truncation_penalty    = truncation_penalty
        self.entropy_reward_weight = entropy_reward_weight
        self.n_examples            = len(examples)
        self._rng                  = np.random.default_rng(seed)
        self.eos_id                = tokenizer.eos_token_id
        self._cache                = free_growth_cache
        self.kl_shaping            = kl_shaping
        self.kl_mode               = kl_mode
        self.kl_weight             = kl_weight
        self.kl_clip               = kl_clip
        # Per-example budget calibration (solves the cold-start / free-growth skip problem).
        #
        # Root cause of cold-start: _reset_episode skips examples whose EOS appears
        # during free-growth (i.e., where the answer fits within budget).  The training
        # set collapses to hard examples where EOS never fires → 100% truncation.
        #
        # Fix: pre-compute base_budget[i] = T + cached_len for each example i.
        # At runtime: ep.budget = base_budget[i] - eviction_k.
        #   n_needed = ep.budget + 1 - T = cached_len - eviction_k + 1  ≤  cached_len
        #   → full cache hit guaranteed → no EOS in fg_tokens → episode NOT skipped.
        #   → agent makes eviction_k steps and EOS enters range → episode completes.
        #
        # K-curriculum (BudgetCurriculumCallback): _eviction_k starts small (easy,
        # EOS nearby, high completion rate) and grows (harder, more compression needed).
        # This is "lower min and max with time" for the per-example budget regime.
        self._per_example_base_budgets = per_example_base_budgets
        self._eviction_k: int          = eviction_k
        self._example_cursor: int      = 0   # tracks which example we're on for cache keys

        cfg             = model.config
        self.n_layers   = cfg.num_hidden_layers
        self.n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim   = cfg.hidden_size // cfg.num_attention_heads

        self.rich_features = rich_features
        fdim      = feature_dim(self.n_kv_heads, self.head_dim, rich=rich_features)
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

        # Repeat-problem support: when True, reset() re-uses the same examples and
        # budget as the previous reset() instead of sampling fresh ones.
        # Set by EpisodeMaskablePPO before calling env.reset() each rollout.
        self._repeat_episode: bool = False
        # Last example index used per slot — populated by _reset_episode().
        self._current_example_indices: list[int] = [0] * self.N
        # When set, _reset_episode(b) uses this index instead of the cursor.
        self._forced_example_idx: list[int | None] = [None] * self.N

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset(self):
        if self._repeat_episode:
            # Re-use same examples and budget; only re-run the prefill+free-growth.
            # The free-growth cache makes this a single forward pass (fast).
            for b in range(self.N):
                self._forced_example_idx[b] = self._current_example_indices[b]
            for b in range(self.N):
                self._reset_episode(b)
                self._forced_example_idx[b] = None
        else:
            # Normal reset: fresh budget and fresh examples.
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

        # Per-episode next-token entropy (Alex's entropy_reward_weight; computed once,
        # broadcast to layer-envs below). Low entropy → model is confident after this
        # eviction → positive signal. Equivalent to S4 kl_mode='proxy'; left intact so
        # corr_reward_v1 and the kl_shaping arms can coexist (independent flags).
        # Uses float32 for numerical stability; vocab ~150k so log_softmax matters.
        if self.entropy_reward_weight != 0.0:
            with torch.no_grad():
                log_p = torch.nn.functional.log_softmax(
                    out.logits[:, -1].float(), dim=-1
                )  # [N, vocab]
            _entropy_per_ep = -(log_p.exp() * log_p).sum(dim=-1).cpu().numpy()  # [N]
        else:
            _entropy_per_ep = None

        # 2b. S4 information-theoretic per-step shaping.
        #   step_r[b] = -kl_weight * clip(damage[b], 0, kl_clip)
        #     exact : damage = KL(p_full || p_evict) — the causal effect of THIS
        #             step's evictions on the model's next-token distribution,
        #             measured against a never-evicted shadow cache (per episode).
        #     proxy : damage = H(p_evict) — reference-free output entropy (cheap,
        #             biased: a confidently-wrong model scores low).
        #   next_toks / pos_ids still hold the PRE-update values fed to the
        #   evicted forward, so the full forward is fed the identical token+pos.
        step_r = np.zeros(self.N, dtype=np.float32)
        if self.kl_shaping:
            logits_evict = out.logits[:, -1]                        # [N, vocab]
            logp_evict   = torch.log_softmax(logits_evict, dim=-1)  # [N, vocab]
            if self.kl_mode == "proxy":
                p_evict = logp_evict.exp()
                damage  = -(p_evict * logp_evict).sum(dim=-1)       # [N] entropy (nats)
            else:  # exact — one batch=1 forward per episode on the full shadow cache
                damage = torch.zeros(self.N, device=self.device)
                for b, ep in enumerate(self.episodes):
                    with torch.no_grad():
                        full_out = self.model(
                            input_ids=next_toks[b:b+1],
                            past_key_values=ep.past_kv_full,
                            position_ids=pos_ids[b:b+1],
                            use_cache=True,
                        )
                    ep.past_kv_full = full_out.past_key_values  # grow (never evicted)
                    logp_full = torch.log_softmax(full_out.logits[0, -1], dim=-1)
                    damage[b] = (logp_full.exp() * (logp_full - logp_evict[b])).sum()
            damage_np = damage.detach().float().cpu().numpy()
            clipped   = np.clip(damage_np, 0.0, self.kl_clip)
            step_r    = (-self.kl_weight * clipped).astype(np.float32)
            for b, ep in enumerate(self.episodes):
                ep.kl_sum   += float(damage_np[b])
                ep.kl_count += 1

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
                                feature_dim(self.n_kv_heads, self.head_dim,
                                            rich=self.rich_features)), dtype=np.float32)
        all_rewards = np.zeros(self.N * self.n_layers, dtype=np.float32)
        all_dones   = np.zeros(self.N * self.n_layers, dtype=bool)
        all_infos   = [{} for _ in range(self.N * self.n_layers)]

        for b, ep in enumerate(self.episodes):
            done = (ep.next_token == self.eos_id or ep.step_count >= self.max_new_tokens)
            base = b * self.n_layers

            # Per-step KL shaping is broadcast to every layer-slot of this episode
            # (the KL signal is joint over layers — see plan's "layer-shared" note).
            # 0.0 when kl_shaping is off, so the baseline reward is unchanged.
            for l in range(self.n_layers):
                all_rewards[base + l] = step_r[b]

            if done:
                truncated   = (ep.next_token != self.eos_id and
                               ep.step_count >= self.max_new_tokens)
                rewards, correct, align = self._terminal_reward(b, truncated)
                kl_step_mean = ep.kl_sum / max(ep.kl_count, 1)

                # NOTE: we deliberately do NOT compute/store a
                # "terminal_observation" here. EpisodeMaskablePPO ignores the
                # infos returned by env.step() (it derives returns from the
                # terminal reward directly), so a terminal observation would be
                # dead weight. The previous version materialized a full
                # [n_layers, max_len, feature_dim] fp32 array (~35 MB, a
                # GPU→CPU copy) and held n_layers references to it in all_infos
                # every time an episode finished — co-resident with the growing
                # obs lists. Dropping it removes one of two _obs_episode() calls
                # per done step and the dead-weight infos payload.
                ep.count += 1
                for l in range(self.n_layers):
                    all_rewards[base + l] += rewards[l]   # terminal on top of step shaping
                    all_dones  [base + l] = True
                    all_infos  [base + l] = {
                        "correct":        correct,
                        "alignment":      align,
                        "episode_seconds": time.perf_counter() - ep.started_at,
                        "context_size":   ep.prompt_len + ep.step_count,
                        "truncated":      truncated,
                        "kl_step_mean":   kl_step_mean,
                    }

                # Reset this episode (batch=1, sequential)
                self._reset_episode(b)
                # Replace slice b in the batched cache
                _replace_slice(self._batch_kv, b, ep.past_kv, self.n_layers)

            # obs for this episode (after possible reset → shows new episode start)
            ep_obs = self._obs_episode(b)
            all_obs[base : base + self.n_layers] = ep_obs

        # Add entropy reward (dense, every step including terminal).
        # reward = -weight * H(next_token_dist): lower entropy → higher reward.
        # Broadcast episode-level entropy to all n_layers envs for that episode.
        if _entropy_per_ep is not None:
            for b in range(self.N):
                base = b * self.n_layers
                all_rewards[base : base + self.n_layers] -= (
                    self.entropy_reward_weight * _entropy_per_ep[b]
                )

        return all_obs, all_rewards, all_dones, all_infos

    def action_masks(self) -> np.ndarray:
        """[N * n_layers, max_len] — same mask broadcast across layers per episode."""
        masks = np.zeros((self.N * self.n_layers, self.max_len), dtype=bool)
        for b, ep in enumerate(self.episodes):
            n_prompt = ep.prompt_len if self.protect_prompt else 0
            row  = valid_action_mask(ep.cache_size, self.max_len, self.n_sinks,
                                     self.n_recent, n_prompt)
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

        Budget selection (in priority order):
          1. Per-example budget (calibrated): ep.budget = base_budget[i] - self._eviction_k
             where base_budget[i] = T + cached_len.  With this budget, n_needed =
             cached_len - K + 1, which is always < cached_len → guaranteed full cache
             hit → episode is NEVER skipped for free-growth completion.  The agent
             must then make exactly K eviction steps before EOS enters range.
             K is annealed by BudgetCurriculumCallback: small K (easy, EOS nearby)
             at training start → large K (hard, strong compression) at end.
          2. Shared budget (fallback): sampled once per reset() call, used for any
             example not in per_example_base_budgets (probe examples, un-cached).

        When a FreeGrowthCache is attached, the free-growth phase is replaced by
        a single forward pass (prefill with prompt + cached tokens) on cache hits,
        saving ~300-500 sequential decode steps per episode reset.
        """
        ep = self.episodes[b]
        while True:
            ep.started_at = time.perf_counter()
            if self._forced_example_idx[b] is not None:
                # Repeat mode: reuse the same example. Don't advance the cursor.
                example_idx = self._forced_example_idx[b]
            else:
                example_idx = self._example_cursor % self.n_examples
                self._current_example_indices[b] = example_idx
                self._example_cursor += 1
            example = self.examples[example_idx]

            prompt_txt, _ = format_gsm8k_chat(self.tokenizer, example)
            ep.gold_answer = example["gold_answers"][0]

            inputs = self.tokenizer(prompt_txt, return_tensors="pt").to(self.device)
            T      = inputs["input_ids"].shape[1]
            if T > self.max_len:
                # Forced example too long (shouldn't happen — it passed before).
                # Fall back to random sampling so we don't loop forever.
                self._forced_example_idx[b] = None
                continue

            if T > self._shared_budget:
                # Prompt longer than the current shared budget. batch_kv is
                # pre-allocated at _shared_budget+1 slots; an episode with
                # ep.budget=T would grow its KV to T+1 > _shared_budget+1 and
                # crash _replace_slice. Skip and pick a different example.
                # This can trigger mid-rollout when the curriculum anneals
                # budget_min below the prompt length of some examples.
                self._forced_example_idx[b] = None
                continue

            # All episodes in a batch MUST share the same budget so the batched
            # KV cache stays at a uniform seq length (invariant: cache_size = budget+1).
            # Per-example base_budgets are used only to filter which examples are
            # eligible for training (see train.py), NOT to set different cache sizes.
            ep.budget = self._shared_budget
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
                            # Episode ends during free-growth — fall back to random.
                            self._forced_example_idx[b] = None
                            continue
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
            # Episode ended during free-growth — fall back to random on retry.
            self._forced_example_idx[b] = None

        # S4 exact: snapshot the full (pre-eviction) cache as the never-evicted
        # shadow. ep.past_kv is now at cache_size = budget+1; from here the live
        # cache gets evicted while past_kv_full only grows.
        ep.kl_sum   = 0.0
        ep.kl_count = 0
        if self.kl_shaping and self.kl_mode == "exact":
            ep.past_kv_full = _clone_cache(ep.past_kv)
        else:
            ep.past_kv_full = None

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
        """[n_layers, max_len, feature_dim] observation for episode b from batch KV.

        When rich_features is on, appends the same N_EXTRA_RICH scale/position
        columns as eval's build_obs (via the shared build_extra_columns) so the
        training and eval observations are byte-identical."""
        ep   = self.episodes[b]
        S    = ep.cache_size
        H    = self.n_kv_heads
        D    = self.head_dim
        fdim = feature_dim(H, D, rich=self.rich_features)
        obs  = np.zeros((self.n_layers, self.max_len, fdim), dtype=np.float32)
        if self._batch_kv is not None and S > 0:
            for l in range(self.n_layers):
                K_bat, V_bat = _get_kv_bat(self._batch_kv, l)   # [N, H, S, D]
                K = K_bat[b].float().cpu()   # [H, S, D]
                V = V_bat[b].float().cpu()
                K_flat = K.transpose(0, 1).reshape(S, H * D).numpy()
                V_flat = V.transpose(0, 1).reshape(S, H * D).numpy()
                obs[l, :S, :H*D]      = K_flat
                obs[l, :S, H*D:2*H*D] = V_flat
                if self.rich_features:
                    # orig_pos for this layer's resident slots (env tracks slot_to_pos);
                    # same source of truth as eval's PositionTracker → identical features.
                    op   = np.asarray(ep.slot_to_pos[l], dtype=np.int64)[None, :]  # [1, S]
                    cols = build_extra_columns(K.unsqueeze(0), V.unsqueeze(0),
                                               S, self.max_len, orig_pos=op)  # [1, S, 2H+4]
                    ne   = extra_feature_dim(True, H)
                    obs[l, :S, 2*H*D:2*H*D + ne] = cols[0]
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
