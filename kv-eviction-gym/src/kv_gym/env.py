"""
SharedKVVecEnv — SB3-compatible VecEnv for sequential KV-cache eviction.

All n_layers * n_heads environments share one LLM call per episode reset.
They all step in lockstep (same seq_len, same budget, same episode length).

observation_space: Box([max_len, feature_dim])
action_space:      Discrete(max_len)
n_envs:            n_layers * n_kv_heads  (56 for Qwen2.5-1.5B: 28×2)

SB3 VecEnv contract:
  - step_wait() auto-resets all envs on done and returns fresh obs.
    Terminal obs is saved in infos[i]["terminal_observation"].
  - action_masks() returns [n_envs, max_len] bool for MaskablePPO.
  - env_method("action_masks") also works (older sb3-contrib versions).
"""

from __future__ import annotations

import itertools
from typing import Iterator

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

from kv_gym.capture import capture, AllHeadCapture
from kv_gym.attention import recompute_all_heads
from kv_gym.features import build_obs
from kv_gym.rewards.auc import future_attention_auc


class SharedKVVecEnv(VecEnv):
    """Vectorized env — one sub-env per (layer, head) pair.

    Args:
        model:       HuggingFace CausalLM loaded with attn_implementation="eager".
        tokenizer:   Matching tokenizer.
        examples:    List of GSM8K example dicts (cycled indefinitely).
        budget:      Number of KV tokens to keep per head at episode end.
        max_len:     Fixed observation width (pad prompt to this length).
        reward_mode: "auc" (Phase 1, no LLM during steps) or "correctness"
                     (Phase 2, LLM call at episode end).
        device:      Device for torch tensors.
    """

    metadata = {}
    render_mode = None
    spec = None

    def __init__(
        self,
        model,
        tokenizer,
        examples: list[dict],
        budget:   int = 32,
        max_len:  int = 256,
        reward_mode: str = "auc",
        device: torch.device | None = None,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.examples  = examples
        self.budget    = budget
        self.max_len   = max_len
        self.reward_mode = reward_mode
        self.device    = device or next(model.parameters()).device

        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        # Use KV-heads as the per-environment head count: one env per (layer, kv_head).
        # For GQA models, multiple Q-heads share one KV cache — eviction decisions
        # are made at the KV-head level, not the Q-head level.
        self.n_heads  = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        n_envs = self.n_layers * self.n_heads

        self.head_indices = [
            (l, h) for l in range(self.n_layers) for h in range(self.n_heads)
        ]

        feature_dim = 2 * self.head_dim + 5
        obs_space  = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(max_len, feature_dim), dtype=np.float32,
        )
        act_space  = spaces.Discrete(max_len)

        super().__init__(n_envs, obs_space, act_space)

        self._example_iter: Iterator[dict] = itertools.cycle(examples)
        self._pending_actions: np.ndarray | None = None

        # Episode state (initialised in reset)
        self.capture: AllHeadCapture | None = None
        self.resident: torch.Tensor | None = None    # [L, H, T]  bool
        self.attn_score: torch.Tensor | None = None  # [L, H, T]

        self._K_flat: torch.Tensor | None = None
        self._V_flat: torch.Tensor | None = None
        self._fa_flat: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset(self):
        example = next(self._example_iter)
        cap = capture(self.model, self.tokenizer, example, self.device)
        self.capture = cap

        T = cap.prompt_len
        L, H = self.n_layers, self.n_heads

        self.resident   = torch.ones(L, H, T, dtype=torch.bool)
        self.attn_score = cap.attn_score.clone()

        self._K_flat  = cap.K.view(L * H, T, self.head_dim)
        self._V_flat  = cap.V.view(L * H, T, self.head_dim)
        self._fa_flat = cap.future_attn.view(L * H, T)

        return self._obs()

    def step_async(self, actions: np.ndarray):
        self._pending_actions = actions

    def step_wait(self):
        actions = self._pending_actions
        L, H = self.n_layers, self.n_heads
        T = self.capture.prompt_len

        # Evict one token per environment
        for env_idx, (l, h) in enumerate(self.head_indices):
            token_idx = int(actions[env_idx])
            if token_idx < T and self.resident[l, h, token_idx]:
                self.resident[l, h, token_idx] = False

        # Recompute attention scores for all heads
        self.attn_score = recompute_all_heads(
            self.capture.Q, self.capture.K, self.resident
        )

        n_resident = self.resident.sum(dim=-1)  # [L, H]
        done = bool((n_resident <= self.budget).all())

        if done:
            rewards = self._terminal_reward()
            dones   = np.ones(self.num_envs, dtype=bool)

            # Save terminal obs before resetting (SB3 reads this from infos)
            terminal_obs = self._obs()
            infos = [{"terminal_observation": terminal_obs[i]}
                     for i in range(self.num_envs)]

            # Auto-reset: SB3 VecEnv contract requires this
            new_obs = self.reset()
        else:
            rewards = np.zeros(self.num_envs, dtype=np.float32)
            dones   = np.zeros(self.num_envs, dtype=bool)
            infos   = [{} for _ in range(self.num_envs)]
            new_obs = self._obs()

        return new_obs, rewards, dones, infos

    def action_masks(self) -> np.ndarray:
        """Return [n_envs, max_len] bool — True where action is valid."""
        L, H = self.n_layers, self.n_heads
        T = self.capture.prompt_len if self.capture else 0

        masks = np.zeros((self.num_envs, self.max_len), dtype=bool)
        if T > 0:
            res_flat = self.resident.view(L * H, T).cpu().numpy()
            masks[:, :T] = res_flat
        return masks

    # ------------------------------------------------------------------
    # Stubs required by VecEnv ABC
    # ------------------------------------------------------------------

    def close(self): pass

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        val = getattr(self, attr_name, None)
        return [val] * len(indices)

    def set_attr(self, attr_name, value, indices=None): pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        # sb3-contrib calls env_method("action_masks") on older versions
        if method_name == "action_masks":
            masks = self.action_masks()
            n = self.num_envs if indices is None else len(indices)
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
        """Build [n_envs, max_len, feature_dim] observation."""
        L, H = self.n_layers, self.n_heads
        T = self.capture.prompt_len

        layer_frac = torch.tensor(
            [l / max(L - 1, 1) for l, h in self.head_indices], dtype=torch.float32
        )
        head_frac = torch.tensor(
            [h / max(H - 1, 1) for l, h in self.head_indices], dtype=torch.float32
        )

        n_envs = self.num_envs
        K_pad = torch.zeros(n_envs, self.max_len, self.head_dim)
        V_pad = torch.zeros(n_envs, self.max_len, self.head_dim)
        A_pad = torch.zeros(n_envs, self.max_len)
        R_pad = torch.zeros(n_envs, self.max_len, dtype=torch.bool)

        K_pad[:, :T, :] = self._K_flat
        V_pad[:, :T, :] = self._V_flat
        A_pad[:, :T]    = self.attn_score.view(n_envs, T)
        R_pad[:, :T]    = self.resident.view(n_envs, T)

        return build_obs(K_pad, V_pad, A_pad, R_pad, layer_frac, head_frac,
                         T, self.max_len)

    def _terminal_reward(self) -> np.ndarray:
        if self.reward_mode == "auc":
            L, H = self.n_layers, self.n_heads
            T = self.capture.prompt_len
            fa   = self._fa_flat
            res  = self.resident.view(L * H, T)
            rew  = future_attention_auc(fa, res, self.budget)
            return rew.cpu().numpy().astype(np.float32)
        raise NotImplementedError("correctness reward not wired yet")
