"""
SharedKVVecEnv — SB3-compatible VecEnv for sequential KV-cache eviction.

One sub-environment per LAYER.  Each env decides independently which tokens to
evict for its layer; the K/V features for that layer are the average across all
KV-heads (e.g. 2 for Qwen2.5-1.5B).  This gives n_envs = n_layers = 28 for
Qwen2.5-1.5B rather than n_layers × n_kv_heads = 56, fixing multi-agent credit
misassignment: a layer's reward now reflects that layer's eviction decision, not
a confused mix of per-head votes.

observation_space: Box([max_len, 2*head_dim])  — [K_avg[t], V_avg[t]] per token;
                   evicted positions are zeroed so the obs changes each step.
action_space:      Discrete(max_len)
n_envs:            n_layers  (28 for Qwen2.5-1.5B)

Terminal reward:
    Aggregate per-layer resident masks (top-budget by mean keep-score across
    layers), re-run generate with that attention mask, score vs gold answer.
    All envs receive the same reward (cooperative setting).

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
from kv_gym.rewards.attention_shaping import attention_alignment, compute_token_importance
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
        examples:              list[dict],
        budget_min:            int = 32,
        budget_max:            int = 256,
        max_len:               int = 256,
        max_new_tokens:        int = 512,
        device:                torch.device | None = None,
        use_attention_shaping: bool = True,
        attention_weight:      float = 0.3,
        seed:                  int = 0,
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
        self._rng                  = np.random.default_rng(seed)

        # budget for the current episode (sampled in reset())
        self.budget: int = budget_min

        cfg = model.config
        self.n_layers  = cfg.num_hidden_layers
        self.n_heads   = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim  = cfg.hidden_size // cfg.num_attention_heads
        n_envs = self.n_layers   # one sub-env per layer, not per (layer, head)

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

        # Per-layer K/V views for observation: average over KV-heads [n_layers, T, D]
        self._K_flat: torch.Tensor | None = None
        self._V_flat: torch.Tensor | None = None

        # Per-token attention importance from the clean reference run [T], or None
        self._token_importance: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset(self):
        example = next(self._example_iter)
        cap     = capture(self.model, self.tokenizer, example, self.device)
        self.capture = cap

        T = cap.prompt_len
        L = self.n_layers

        assert T <= self.max_len, (
            f"Prompt length {T} exceeds max_len={self.max_len}. "
            f"Increase max_len or truncate prompts before training."
        )

        # Sample a fresh budget each episode so the policy learns a total
        # ranking rather than a budget-specific selection.
        self.budget = int(self._rng.integers(self.budget_min,
                                             min(self.budget_max, T - 1) + 1))

        # Per-layer eviction: resident[l, t] = True if token t is still in layer l's cache.
        # K/V features are averaged over KV-heads so each layer-env sees one [T, D] view.
        self.resident = torch.ones(L, T, dtype=torch.bool)          # [L, T]
        self._K_flat  = cap.K.mean(dim=1)                           # [L, T, D]
        self._V_flat  = cap.V.mean(dim=1)

        # Clean reference run: one full-context generate to get per-token importance.
        # Falls back to KV-norm proxy if output_attentions returns empty (SDPA/flash).
        # Stored on CPU; used at terminal to blend correctness with attention alignment.
        if self.use_attention_shaping:
            self._token_importance = compute_token_importance(
                self.model, cap.input_ids,
                K=cap.K, V=cap.V,
                max_new_tokens=self.max_new_tokens,
                device=self.device,
            )
        else:
            self._token_importance = None

        return self._obs()

    def step_async(self, actions: np.ndarray):
        self._pending_actions = actions

    def step_wait(self):
        actions = self._pending_actions
        T = self.capture.prompt_len

        # Evict one token per layer-environment
        for l, token_idx in enumerate(int(a) for a in actions):
            if token_idx < T and self.resident[l, token_idx]:
                self.resident[l, token_idx] = False

        n_resident = self.resident.sum(dim=-1)   # [L]
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
            masks[:, :T] = self.resident.cpu().numpy()  # [L, T]
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

    def seed(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        """Build [n_envs, max_len, FEATURE_DIM] observation.

        Evicted positions are zeroed so the observation changes as tokens are
        removed, giving the value function a non-constant signal across steps.
        """
        T = self.capture.prompt_len
        return build_obs(
            K=self._K_flat,          # [L, T, D]
            V=self._V_flat,
            prompt_len=T,
            max_len=self.max_len,
            resident_mask=self.resident,  # [L, T]
        )

    def _terminal_reward(self) -> np.ndarray:
        """Score the eviction decision and return reward to all envs.

        Aggregation: each token's keep-score = fraction of heads that kept it.
        We keep the top-budget tokens by this score.

        Reward:
            correctness  — re-run generate with eviction mask; 1 if answer
                           matches gold, 0 otherwise.  (Sparse signal.)
            alignment    — fraction of the clean-run attention mass that falls
                           on the kept tokens.  Available whenever attention
                           shaping is enabled.  (Dense proxy signal.)

            score = (1 - attention_weight) * correctness
                  + attention_weight * alignment   [if shaping enabled]
            score = correctness                    [if shaping disabled or failed]

        All 56 envs receive the same scalar reward (cooperative setting).
        """
        T = self.capture.prompt_len

        # Aggregate per-layer decisions into one global token ranking
        token_scores = self.resident.float().mean(dim=0)  # mean over layers → [T]
        n_keep = min(self.budget, T)
        _, topk = token_scores.topk(n_keep)

        attn_mask    = torch.zeros(1, T, dtype=torch.long, device=self.device)
        attn_mask[0, topk] = 1
        position_ids = torch.arange(T, device=self.device).unsqueeze(0)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=self.capture.input_ids.to(self.device),
                attention_mask=attn_mask,
                position_ids=position_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        text        = self.tokenizer.decode(out[0, T:], skip_special_tokens=True)
        correctness = flexible_extract(text, [self.capture.gold_answer])

        if self._token_importance is not None:
            align = attention_alignment(self._token_importance, topk.cpu())
            score = (1.0 - self.attention_weight) * correctness + self.attention_weight * align
        else:
            score = float(correctness)

        return np.full(self.num_envs, score, dtype=np.float32)
