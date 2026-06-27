"""Memory-efficient rollout buffer for the KV-eviction PPO policy.

Observations in the KV gym are large — shape [n_steps, n_envs, max_len,
2*n_kv_heads*head_dim] — making the buffer the dominant RAM consumer.
Two optimisations here:

1. fp16 storage: observations are stored as float16 instead of float32,
   halving the steady-state buffer footprint.  Minibatch tensors are cast
   back to float32 by PerTokenMLP.forward() before the first network op.

2. Faster swap_and_flatten: SB3's default does arr.swapaxes(0,1).reshape(...)
   on a non-contiguous view, which forces NumPy to allocate a full copy and
   hold BOTH arrays simultaneously (2× peak).  Our override makes the array
   contiguous first, then reshapes as a view — so the peak is still ~2× but
   the two copies are float16 (half the size) rather than one float32 original
   plus one float32 copy.
"""

import numpy as np
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer


class FP16ObsMaskableRolloutBuffer(MaskableRolloutBuffer):
    """MaskableRolloutBuffer that stores observations in float16.

    Halves steady-state RAM and halves the transient 2× swap_and_flatten peak.
    Cast back to float32 happens in PerTokenMLP.forward() — transparent to the
    rest of the training loop.
    """

    def reset(self) -> None:
        # Do NOT call super().reset(): it allocates observations as fp32 first
        # (using observation_space.dtype), which for large buffers causes OOM
        # before we could ever downcast to fp16. Replicate the parent chain here
        # but allocate observations directly as fp16.
        # np.empty (not np.zeros) avoids the eager memset that would bring all
        # 12 GB of pages into physical RAM before the copy loop. Every position
        # is written by collect_rollouts before the buffer is read, so zeros are
        # not needed. np.empty uses mmap(MAP_ANONYMOUS) → OS demand-paging → 0
        # physical pages until first write.
        self.observations = np.empty(
            (self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float16
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        # MaskableRolloutBuffer also needs action_masks.
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")
        self.mask_dims = mask_dims
        self.action_masks = np.ones(
            (self.buffer_size, self.n_envs, mask_dims), dtype=np.float32
        )
        # BaseBuffer.reset()
        self.pos = 0
        self.full = False

    # add() deliberately NOT overridden: NumPy auto-casts float32 → float16 on
    # assignment when self.observations.dtype == float16.

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """Reshape [n_steps, n_envs, *shape] → [n_steps*n_envs, *shape].

        Merges the first two dimensions with a plain reshape (zero-copy view on
        C-contiguous arrays).  The standard SB3 implementation transposes first
        which forces a full copy — for the observation array (~17 GB fp16) that
        copy alone OOMs.  Sample ordering within the merged axis changes vs the
        transpose approach, but PPO training shuffles mini-batch indices with a
        random permutation so ordering is irrelevant.
        """
        if arr.ndim < 3:
            arr = arr[..., None]
        return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
