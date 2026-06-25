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
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer


class FP16ObsMaskableRolloutBuffer(MaskableRolloutBuffer):
    """MaskableRolloutBuffer that stores observations in float16.

    Halves steady-state RAM and halves the transient 2× swap_and_flatten peak.
    Cast back to float32 happens in PerTokenMLP.forward() — transparent to the
    rest of the training loop.
    """

    def reset(self) -> None:
        super().reset()
        # Parent allocated observations as float32; downcast to float16.
        # The transient double-allocation here is fine — reset is not on the
        # critical RAM path (the model graph is already freed after the update).
        self.observations = self.observations.astype(np.float16)

    # add() deliberately NOT overridden: NumPy auto-casts float32 → float16 on
    # assignment when self.observations.dtype == float16.

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """Reshape [n_steps, n_envs, *shape] → [n_steps*n_envs, *shape].

        Makes the transposed view contiguous before reshape so that reshape
        returns a cheap view instead of a second full copy.  Peak is still ~2×
        but both copies share the dtype of arr (float16 for observations,
        int64/float32 for other small tensors).
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        contiguous = np.ascontiguousarray(arr.swapaxes(0, 1))
        return contiguous.reshape(shape[0] * shape[1], *shape[2:])
