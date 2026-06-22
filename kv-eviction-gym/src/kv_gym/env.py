"""
SharedKVVecEnv — online eviction: one decode step per RL step.

Episode structure:
  reset(): prefill the prompt → full prompt cache (T tokens).  Budget ≥ T is
           sampled so no pre-eviction of prompt tokens is needed.
  step():  (1) evict one token per layer if cache > budget,
           (2) run one greedy decode step (new K/V appended to live cache),
           (3) return the next observation from the updated cache.
  done:    EOS token produced OR step_count ≥ max_new_tokens.

The cache starts at T and grows one token per decode step until it reaches
budget, then stays at budget (one evicted per step, one generated per step).
Prompt tokens are included in the evictable set: the agent may evict any
cached position, whether it was in the original prompt or generated.

Generated tokens are accumulated during the episode itself; the terminal reward
is computed in-place from those tokens — no separate model.generate() call.

For attention-alignment shaping, per-token importance comes from a clean
reference generate with output_attentions=True (requires eager attention).
Raises RuntimeError if the backend doesn't support it — use
use_attention_shaping=False to disable.

Architecture
------------
n_envs = n_layers (28 for Qwen2.5-1.5B).  Each env independently evicts from
its own layer of the live DynamicCache.  One shared model forward pass advances
the cache one token per step.

observation_space: Box([max_len, 2*head_dim]) — KV of cached tokens (averaged
                   over KV-heads); zero-padded beyond the current cache size.
action_space:      Discrete(max_len) — cache slot to evict (0-indexed).
                   The action is a no-op when cache_size ≤ budget.
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Iterator

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from kv_gym.features import feature_dim
from kv_gym.eval_core import valid_action_mask
from kv_gym.rewards.attention_shaping import compute_token_importance
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract
from kv_gym.vendor.prompts import format_gsm8k

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DynamicCache helpers — handle both transformers 4.x and 5.x
# ---------------------------------------------------------------------------

def _cache_seq_len(cache, layer: int) -> int:
    """Sequence length stored in one layer of a DynamicCache."""
    if hasattr(cache, "layers"):
        return cache.layers[layer].keys.shape[2]
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer].shape[2]
    raise RuntimeError("Unsupported past_key_values format")


def _get_layer_kv(cache, layer: int):
    """Return (K, V) for one cache layer as [n_kv_heads, seq_len, head_dim] float32 cpu."""
    if hasattr(cache, "layers"):
        K = cache.layers[layer].keys    # [1, H, S, D]
        V = cache.layers[layer].values
    elif hasattr(cache, "key_cache"):
        K = cache.key_cache[layer]
        V = cache.value_cache[layer]
    else:
        raise RuntimeError("Unsupported past_key_values format")
    return K.squeeze(0).float().cpu(), V.squeeze(0).float().cpu()   # [H, S, D]


def _evict_slot(cache, layer: int, slot: int) -> None:
    """Remove one token slot from one layer of a DynamicCache, in place.

    After removal the cache is contiguous: positions after `slot` shift left
    by one.  The slot index is clamped to [0, seq_len-1] for safety.
    """
    def _remove(t: torch.Tensor) -> torch.Tensor:  # t: [1, H, S, D]
        S   = t.shape[2]
        s   = max(0, min(slot, S - 1))
        idx = torch.cat([
            torch.arange(s,     device=t.device),
            torch.arange(s + 1, S, device=t.device),
        ])
        return t.index_select(dim=2, index=idx)

    if hasattr(cache, "layers"):
        cache.layers[layer].keys   = _remove(cache.layers[layer].keys)
        cache.layers[layer].values = _remove(cache.layers[layer].values)
    elif hasattr(cache, "key_cache"):
        cache.key_cache[layer]   = _remove(cache.key_cache[layer])
        cache.value_cache[layer] = _remove(cache.value_cache[layer])
    else:
        raise RuntimeError("Unsupported past_key_values format")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SharedKVVecEnv(VecEnv):
    """Vectorized env — one sub-env per transformer layer, online eviction."""

    metadata    = {}
    render_mode = None
    spec        = None

    def __init__(
        self,
        model,
        tokenizer,
        examples:              list[dict],
        budget_min:            int   = 128,   # total cache capacity lower bound
        budget_max:            int   = 256,   # total cache capacity upper bound
        max_len:               int   = 256,   # observation / action-mask width
        max_new_tokens:        int   = 512,
        device:                torch.device | None = None,
        use_attention_shaping: bool  = True,
        attention_weight:      float = 0.3,
        shaping_mode:          str   = "none",
        n_sinks:               int   = 4,    # protected attention-sink slots (cannot evict)
        n_recent:              int   = 8,    # protected recency-window slots (cannot evict)
        seed:                  int   = 0,
    ):
        self.model                 = model
        self.tokenizer             = tokenizer
        self.examples              = examples
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
        self._rng                  = np.random.default_rng(seed)

        _shaping_needs_imp = {"terminal", "per_step", "per_step_recency"}
        if shaping_mode not in (_shaping_needs_imp | {"none"}):
            raise ValueError(
                f"shaping_mode must be one of {sorted(_shaping_needs_imp | {'none'})}; "
                f"got {shaping_mode!r}"
            )
        if shaping_mode in {"per_step", "per_step_recency"} and not use_attention_shaping:
            raise ValueError(
                f"shaping_mode='{shaping_mode}' needs per-token importance — set "
                "use_attention_shaping=True (it provides the reference-attention signal)."
            )

        cfg            = model.config
        self.n_layers  = cfg.num_hidden_layers
        self.n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim  = cfg.hidden_size // cfg.num_attention_heads

        fdim      = feature_dim(self.n_kv_heads, self.head_dim)  # 2 * n_kv_heads * head_dim
        obs_space = spaces.Box(low=-np.inf, high=np.inf,
                               shape=(max_len, fdim), dtype=np.float32)
        act_space = spaces.Discrete(max_len)
        super().__init__(self.n_layers, obs_space, act_space)

        self._example_iter: Iterator[dict] = itertools.cycle(examples)
        self._pending_actions: np.ndarray | None = None

        # Episode state — initialised in reset()
        self.past_kv             = None          # live DynamicCache on device
        self.next_token:    int  = 0             # pending token (to be fed next step)
        self.true_position: int  = 0             # original seq position of next_token
        self.cache_size:    int  = 0             # tokens in every layer (layers stay in sync)
        self.step_count:    int  = 0
        self.generated:     list[int] = []
        self.gold_answer:   str  = ""
        self.prompt_len:    int  = 0
        self.budget:        int  = budget_min
        self.eos_id:        int | None = tokenizer.eos_token_id

        # Per-layer slot→original-position mapping; used for alignment shaping.
        # slot_to_pos[l][s] = original sequence position of cache slot s in layer l.
        # Updated on eviction (pop) and decode (append).
        self.slot_to_pos: list[list[int]] = []

        # Per-prompt-token importance [T] for alignment shaping, or None.
        self._token_importance: torch.Tensor | None = None

        # Set in reset() when EOS fires during the free-growth phase.
        # step_wait() checks this and returns done immediately without evicting.
        self._free_growth_done: bool = False

        self._episode_count: int = 0
        self._episode_started_at: float = 0.0

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset(self):
        self._episode_started_at = time.perf_counter()
        example    = next(self._example_iter)
        prompt_text, _ = format_gsm8k(example)
        self.gold_answer = example["gold_answers"][0]

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        T      = inputs["input_ids"].shape[1]

        assert T <= self.max_len, (
            f"Prompt length {T} > max_len={self.max_len}. "
            "Increase max_len or truncate prompts before training."
        )

        # Budget = total cache capacity (prompt + generated tokens combined).
        # If T <= budget_max: sample from [max(budget_min, T), budget_max] so the
        # full prompt always fits before eviction starts.
        # If T > budget_max: clamp to budget_max — the cache is already over budget
        # from step 0, so the policy must evict immediately. Still valid signal.
        if T <= self.budget_max:
            budget_lo = max(self.budget_min, T)
            self.budget = int(self._rng.integers(budget_lo, self.budget_max + 1))
        else:
            self.budget = self.budget_max
        self.prompt_len = T

        # --- Prefill ---
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=inputs["input_ids"], use_cache=True)

        self.past_kv       = out.past_key_values
        self.next_token    = int(out.logits[0, -1].argmax().item())
        self.true_position = T
        self.cache_size    = T
        self.step_count    = 0
        self.generated     = []

        # Slot→position maps: after prefill, slot s holds original position s.
        self.slot_to_pos = [list(range(T)) for _ in range(self.n_layers)]

        # Per-token importance for attention-alignment reward shaping.
        # When use_attention_shaping=True: runs model.generate() with
        # output_attentions=True (requires attn_implementation="eager").
        # Raises RuntimeError if the backend doesn't support it.
        # When use_attention_shaping=False: no shaping, pure correctness reward.
        if self.use_attention_shaping:
            K_list, V_list = [], []
            for l in range(self.n_layers):
                K_l, V_l = _get_layer_kv(self.past_kv, l)  # [H, T, D] float32 cpu
                K_list.append(K_l)
                V_list.append(V_l)
            K_all = torch.stack(K_list)  # [L, H, T, D]
            V_all = torch.stack(V_list)
            self._token_importance = compute_token_importance(
                model=self.model,
                input_ids=inputs["input_ids"].cpu(),
                K=K_all,
                V=V_all,
                max_new_tokens=self.max_new_tokens,
                device=self.device,
            )  # [T], sums to 1
        else:
            self._token_importance = None

        # Free-growth phase: decode internally until cache_size > budget.
        # This ensures every call to step_wait() is a genuine eviction step —
        # no no-op steps dilute the policy gradient with zero-reward transitions.
        self._free_growth_done = False
        self._run_free_growth()

        # If the episode ended during free-growth (EOS before any eviction was
        # needed), skip it entirely and reset to the next example.  This prevents
        # SB3 from ever pairing an observation with an action that had no effect
        # on the outcome — the dead episode is simply discarded.
        if self._free_growth_done:
            return self.reset()

        self._episode_count += 1
        logger.info(
            "episode=%d  prompt_len=%d  budget=%d  ratio=%.2f  "
            "free_growth_steps=%d",
            self._episode_count, T, self.budget,
            self.budget / T,
            self.cache_size - T,   # decode steps taken during free-growth
        )

        return self._obs()

    def _run_free_growth(self) -> None:
        """Decode without eviction until cache_size > budget or episode ends.

        After this returns, either:
          - cache_size == budget + 1  → step_wait() will always evict, no no-ops.
          - _free_growth_done == True → EOS/max_steps reached; step_wait() returns
            done immediately and calls reset() for the next episode.

        Why this eliminates credit-assignment dilution
        -----------------------------------------------
        With the old design, the rollout buffer contained up to (budget - T)
        no-op transitions per episode (steps where cache ≤ budget so no eviction
        fired).  With gamma=1 and terminal-only reward, those steps received the
        same advantage as genuine eviction steps, diluting the gradient by a factor
        of max_new_tokens / n_eviction_steps.  Moving free-growth here means the
        rollout buffer only ever contains real eviction decisions.
        """
        while self.cache_size <= self.budget:
            pos = torch.tensor([[self.true_position]], device=self.device)
            with torch.no_grad():
                step_out = self.model(
                    input_ids=torch.tensor([[self.next_token]], device=self.device),
                    past_key_values=self.past_kv,
                    position_ids=pos,
                    cache_position=pos.squeeze(0),
                    use_cache=True,
                )
            self.generated.append(self.next_token)
            self.past_kv      = step_out.past_key_values
            new_next_token    = int(step_out.logits[0, -1].argmax().item())
            for l in range(self.n_layers):
                self.slot_to_pos[l].append(self.true_position)
            self.true_position += 1
            self.step_count    += 1
            self.cache_size    += 1
            self.next_token    = new_next_token
            if new_next_token == self.eos_id or self.step_count >= self.max_new_tokens:
                self._free_growth_done = True
                # Same truncation fix as step_wait: include the last predicted token
                # if episode ended by step limit rather than EOS.
                if new_next_token != self.eos_id:
                    self.generated.append(new_next_token)
                break

    def step_async(self, actions: np.ndarray):
        self._pending_actions = actions

    def step_wait(self):
        actions = self._pending_actions

        # --- 1. Per-layer eviction — always fires (free-growth handled in reset) ---
        #
        # Why cache_size is always > budget here:
        #   reset() runs _run_free_growth() which decodes until cache_size > budget.
        #   Each step_wait evicts 1 token then decodes 1 token: net change = 0,
        #   so cache_size stays at budget+1 for the lifetime of the episode.
        # Per-step shaping (only in "per_step" mode): the terminal alignment reward
        # is additive over the prompt tokens still cached, so its *loss* is additive
        # over the tokens evicted.  We therefore pay that loss exactly where and when
        # it happens — layer-env l receives −attention_weight·importance[p] at the step
        # it evicts original prompt position p.  Summed over an episode this equals the
        # terminal alignment (up to a per-episode constant that the PPO advantage
        # baseline absorbs), but it is localized in time (real per-eviction credit
        # instead of one terminal scalar shared by every step) and per layer (env l's
        # reward reflects layer l's own decision, not the mean over all layers — far
        # richer gradient for the shared policy).  Evicting a generated token
        # (orig_pos ≥ prompt_len) costs 0: importance is defined over prompt positions.
        step_shaping = np.zeros(self.num_envs, dtype=np.float32)
        mode       = self.shaping_mode
        have_imp   = self._token_importance is not None
        do_attn    = (mode == "per_step"         and have_imp)   # original (biased) form
        do_recency = (mode == "per_step_recency" and have_imp)   # fixed form
        T = self.prompt_len
        imp_max = float(self._token_importance.max()) if do_recency else 1.0
        if imp_max <= 0:
            imp_max = 1.0
        for l, slot in enumerate(int(a) for a in actions):
            slot = max(0, min(slot, self.cache_size - 1))
            evicted_pos = self.slot_to_pos[l][slot]
            if do_attn:
                # −w·importance[p], prompt-only (generated → 0 → free; this is the mode
                # that caused the recency collapse — kept selectable for comparison).
                if evicted_pos < T:
                    step_shaping[l] = -self.attention_weight * float(
                        self._token_importance[evicted_pos]
                    )
            elif do_recency:
                # −w·keep_value, keep_value = max(normalized attention, recency) ∈ [0,1].
                # Recent tokens (prompt OR generated) become costly to evict → removes the
                # "generated = free" bug. No new hyperparameter (both terms normalized).
                attn_v    = (float(self._token_importance[evicted_pos]) / imp_max
                             if evicted_pos < T else 0.0)
                recency_v = (evicted_pos + 1) / (self.true_position + 1)
                step_shaping[l] = -self.attention_weight * max(attn_v, recency_v)
            _evict_slot(self.past_kv, l, slot)
            self.slot_to_pos[l].pop(slot)
        self.cache_size -= 1

        # --- 2. One greedy decode step ---
        # position_ids = true_position (original sequence coordinate) for correct RoPE.
        # cache_position = true_position for the causal mask.
        #
        # Why true_position is correct for cache_position (not cache_size):
        #   The causal mask at row R allows attending to positions 0..R-1.
        #   All tokens currently in the cache have original positions < true_position,
        #   so they all pass the causal check.  Passing cache_size instead would use
        #   a mask row of ~budget, which would INCORRECTLY mask out any cached token
        #   whose original position > cache_size (common after many evictions).
        pos = torch.tensor([[self.true_position]], device=self.device)
        with torch.no_grad():
            step_out = self.model(
                input_ids=torch.tensor([[self.next_token]], device=self.device),
                past_key_values=self.past_kv,
                position_ids=pos,
                cache_position=pos.squeeze(0),
                use_cache=True,
            )

        self.generated.append(self.next_token)
        self.past_kv       = step_out.past_key_values
        new_next_token     = int(step_out.logits[0, -1].argmax().item())

        for l in range(self.n_layers):
            self.slot_to_pos[l].append(self.true_position)

        self.true_position += 1
        self.step_count    += 1
        self.cache_size    += 1   # net: -1 (evict) + 1 (decode) → stays at budget+1

        # --- 3. Termination check ---
        truncated = (new_next_token != self.eos_id and self.step_count >= self.max_new_tokens)
        done = (new_next_token == self.eos_id or self.step_count >= self.max_new_tokens)
        self.next_token = new_next_token

        if done:
            # When done by EOS, new_next_token is a special token — don't include it.
            # When done by truncation, new_next_token is real content that was predicted
            # but never fed as input: include it so the decoded answer is complete.
            if new_next_token != self.eos_id:
                self.generated.append(new_next_token)
            term_rewards, correct, align = self._terminal_reward()
            episode_seconds = time.perf_counter() - self._episode_started_at
            # Terminal step also evicted: add this step's shaping on top of the
            # terminal (correctness) reward.
            rewards      = step_shaping + term_rewards
            dones        = np.ones(self.num_envs, dtype=bool)
            terminal_obs = self._obs()
            infos        = [{"terminal_observation": terminal_obs[i],
                             "correct": correct, "alignment": align,
                             "episode_seconds": episode_seconds,
                             "truncated": truncated}
                            for i in range(self.num_envs)]
            new_obs = self.reset()
        else:
            # Non-terminal steps carry only the per-step shaping (zeros unless
            # shaping_mode == "per_step").
            rewards = step_shaping
            dones   = np.zeros(self.num_envs, dtype=bool)
            infos   = [{} for _ in range(self.num_envs)]
            new_obs = self._obs()

        return new_obs, rewards, dones, infos

    def action_masks(self) -> np.ndarray:
        """[n_envs, max_len] bool — evictable cache slots, with the first `n_sinks`
        (attention sinks) and last `n_recent` (recency window) protected.

        The recency window is the key fix for the collapse where the policy learned
        to evict its own most-recent generated tokens. All layer-envs share the same
        cache_size (lockstep), so one row is broadcast to all envs.
        """
        if self.past_kv is None:
            return np.zeros((self.num_envs, self.max_len), dtype=bool)
        row = valid_action_mask(self.cache_size, self.max_len, self.n_sinks, self.n_recent)
        return np.broadcast_to(row, (self.num_envs, self.max_len)).copy()

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

    def _obs(self) -> np.ndarray:
        """Build [n_envs, max_len, 2*n_kv_heads*head_dim] observation from the live cache.

        K and V for each layer are concatenated across KV-heads:
            [K_head0 | K_head1 | V_head0 | V_head1]  per token slot.
        Slots beyond the current cache size are zero-padded to max_len.
        """
        S    = self.cache_size
        H    = self.n_kv_heads
        D    = self.head_dim
        obs  = np.zeros((self.n_layers, self.max_len, 2 * H * D), dtype=np.float32)

        if self.past_kv is not None and S > 0:
            for l in range(self.n_layers):
                K, V = _get_layer_kv(self.past_kv, l)   # [H, S, D] float32 cpu
                # [H, S, D] → [S, H*D] by interleaving heads into the feature axis
                K_flat = K.transpose(0, 1).reshape(S, H * D).numpy()
                V_flat = V.transpose(0, 1).reshape(S, H * D).numpy()
                obs[l, :S, :H*D]       = K_flat
                obs[l, :S, H*D:2*H*D]  = V_flat

        return obs

    def _terminal_reward(self) -> tuple[np.ndarray, bool, float]:
        """Compute and broadcast the episode reward to all layer-envs.

        Reward:
            correctness  — 1.0 if generated answer matches gold, 0.0 otherwise.
            alignment    — fraction of attention mass captured by prompt tokens
                           still present in any layer's cache, averaged over layers.

            score = (1 - attention_weight) * correctness + attention_weight * alignment
                    [when use_attention_shaping=True and importance is available]
            score = correctness   [otherwise]

        All layer-envs receive the same scalar reward (cooperative setting).

        Returns:
            (rewards [n_envs], correct bool, alignment float)
        """
        text        = self.tokenizer.decode(self.generated, skip_special_tokens=True)
        correctness = float(flexible_extract(text, [self.gold_answer]))

        # Alignment is always computed when importance is available — used as the
        # terminal reward term in "terminal" mode, and as a diagnostic (info dict)
        # in every mode.
        if self._token_importance is not None:
            T = self.prompt_len
            soft_keep = torch.zeros(T, dtype=torch.float32)
            for l in range(self.n_layers):
                for orig_pos in self.slot_to_pos[l]:
                    if orig_pos < T:
                        soft_keep[orig_pos] += 1.0
            soft_keep /= self.n_layers
            align = (self._token_importance * soft_keep).sum().item()
        else:
            align = float("nan")

        # Terminal reward = the sparse global objective (correctness), shared by all
        # layer-envs.  Alignment is added here ONLY in "terminal" mode; in "per_step"
        # mode it was already paid out incrementally during step_wait (one
        # −w·importance[evicted] per eviction), so adding it again would double-count.
        if self.shaping_mode == "terminal" and self._token_importance is not None:
            score = ((1.0 - self.attention_weight) * correctness
                     + self.attention_weight * align)
        elif self.shaping_mode in {"per_step", "per_step_recency"}:
            # alignment already paid out incrementally in step_wait → terminal is
            # correctness only (scaled so the magnitudes stay comparable to terminal mode).
            score = (1.0 - self.attention_weight) * correctness
        else:  # "none"
            score = correctness

        return np.full(self.num_envs, score, dtype=np.float32), bool(correctness), align
