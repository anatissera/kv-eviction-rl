"""
SharedKVVecEnv — SB3-compatible VecEnv for sequential KV-cache eviction.

One sub-environment per (layer, kv_head) pair.  All envs share a single
LLM prefill per episode reset and step in lockstep (same prompt, same
budget, same episode length), so they always terminate together.

observation_space: Box([max_len, 3])   — [||K||, ||V||, position] per token
action_space:      Discrete(max_len)
n_envs:            n_layers × n_kv_heads  (56 for Qwen2.5-1.5B: 28×2)

Terminal reward:
    Aggregate per-head resident masks (top-budget by mean keep-score),
    re-run generate with that attention mask, score vs gold answer.
    All 56 envs receive the same reward (cooperative multi-agent setting).

SB3 VecEnv contract:
  - step_wait() auto-resets on done; terminal obs in infos["terminal_observation"].
  - action_masks() returns [n_envs, max_len] bool for MaskablePPO.
  - env_method("action_masks") also works (older sb3-contrib).
"""

from __future__ import annotations

import itertools
from typing import Iterator

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from kv_gym.capture import capture, AllHeadCapture
from kv_gym.features import build_obs, feature_dim
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract


class SharedKVVecEnv(VecEnv):
    """Vectorized env — one sub-env per (layer, kv_head) pair."""

    metadata   = {}
    render_mode = None
    spec       = None

    def __init__(
        self,
        model,
        tokenizer,
        examples:    list[dict],
        budget:      int = 32,
        max_len:     int = 256,
        max_new_tokens: int = 64,
        device:      torch.device | None = None,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.examples       = examples
        self.budget         = budget
        self.max_len        = max_len
        self.max_new_tokens = max_new_tokens
        self.device         = device or next(model.parameters()).device

        cfg = model.config
        self.n_layers  = cfg.num_hidden_layers
        self.n_heads   = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim  = cfg.hidden_size // cfg.num_attention_heads
        n_envs = self.n_layers * self.n_heads

        self.head_indices = [
            (l, h) for l in range(self.n_layers) for h in range(self.n_heads)
        ]

        fdim = feature_dim(self.head_dim)
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(max_len, fdim), dtype=np.float32,
        )
        act_space = spaces.Discrete(max_len)
        super().__init__(n_envs, obs_space, act_space)

        self._example_iter: Iterator[dict] = itertools.cycle(examples)
        self._pending_actions: np.ndarray | None = None

        # Episode state (set in reset)
        self.capture:  AllHeadCapture | None = None
        self.resident: torch.Tensor   | None = None  # [L, H, T] bool

        # Per-env K/V views for observation building [n_envs, T, D]
        self._K_flat: torch.Tensor | None = None
        self._V_flat: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset(self):
        example = next(self._example_iter)
        cap     = capture(self.model, self.tokenizer, example, self.device)
        self.capture = cap

        T = cap.prompt_len
        L, H = self.n_layers, self.n_heads

        self.resident = torch.ones(L, H, T, dtype=torch.bool)
        self._K_flat  = cap.K.view(L * H, T, self.head_dim)   # [n_envs, T, D]
        self._V_flat  = cap.V.view(L * H, T, self.head_dim)

        return self._obs()

    def step_async(self, actions: np.ndarray):
        self._pending_actions = actions

    def step_wait(self):
        actions = self._pending_actions
        T = self.capture.prompt_len

        # Evict one token per environment
        for env_idx, (l, h) in enumerate(self.head_indices):
            token_idx = int(actions[env_idx])
            if token_idx < T and self.resident[l, h, token_idx]:
                self.resident[l, h, token_idx] = False

        n_resident = self.resident.sum(dim=-1)   # [L, H]
        done = bool((n_resident <= self.budget).all())

        if done:
            rewards     = self._terminal_reward()
            dones       = np.ones(self.num_envs, dtype=bool)
            terminal_obs = self._obs()
            infos       = [{"terminal_observation": terminal_obs[i]}
                           for i in range(self.num_envs)]
            new_obs     = self.reset()
        else:
            rewards = np.zeros(self.num_envs, dtype=np.float32)
            dones   = np.zeros(self.num_envs, dtype=bool)
            infos   = [{} for _ in range(self.num_envs)]
            new_obs = self._obs()

        return new_obs, rewards, dones, infos

    def action_masks(self) -> np.ndarray:
        """Return [n_envs, max_len] bool — True where action is valid."""
        T = self.capture.prompt_len if self.capture else 0
        masks = np.zeros((self.num_envs, self.max_len), dtype=bool)
        if T > 0:
            masks[:, :T] = self.resident.view(self.num_envs, T).cpu().numpy()
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
            idx = list(range(self.num_envs)) if indices is None else list(indices)
            return [masks[i] for i in idx]
        raise NotImplementedError(f"env_method('{method_name}') not supported")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None): pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        """Build [n_envs, max_len, FEATURE_DIM] observation."""
        return build_obs(
            K=self._K_flat,
            V=self._V_flat,
            prompt_len=self.capture.prompt_len,
            max_len=self.max_len,
        )

    def _terminal_reward(self) -> np.ndarray:
        """Generate from the evicted cache and score correctness.

        Aggregation: each token's keep-score = fraction of heads that kept it.
        We keep the top-budget tokens by this score and re-run generate with
        an attention mask that zeros out the rest.  All envs share the reward.
        """
        T = self.capture.prompt_len

        # Aggregate per-head decisions into one global token ranking
        token_scores = self.resident.float().mean(dim=(0, 1))  # [T]
        n_keep = min(self.budget, T)
        _, topk = token_scores.topk(n_keep)
        attn_mask = torch.zeros(1, T, dtype=torch.long, device=self.device)
        attn_mask[0, topk] = 1

        with torch.no_grad():
            out = self.model.generate(
                input_ids=self.capture.input_ids.to(self.device),
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        text  = self.tokenizer.decode(out[0, T:], skip_special_tokens=True)
        score = flexible_extract(text, [self.capture.gold_answer])
        return np.full(self.num_envs, score, dtype=np.float32)
