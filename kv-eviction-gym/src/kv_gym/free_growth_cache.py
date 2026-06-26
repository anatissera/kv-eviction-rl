"""Disk cache for free-growth token sequences.

During episode reset, the env runs a "free-growth" phase: it decodes the LLM
greedily (no eviction) from the end of the prompt until the KV cache reaches
budget+1 tokens.  This phase is expensive (~300-500 sequential decode steps)
but entirely deterministic — the LLM weights are frozen throughout RL training,
so the same (model, prompt) pair always produces the same token sequence.

This cache stores those generated tokens on disk.  On a cache hit, we skip
the sequential decode and instead run ONE forward pass (prefill with prompt +
cached tokens), which is orders of magnitude faster for long sequences.

Storage
-------
  cache_dir / model_name / {example_idx:06d}.npy  →  int32 array of tokens

Layout
------
Tokens are the ones fed into the model during free-growth in order:
  generated[0] is the token predicted after the prompt (fed at position T),
  generated[1] is the token fed at position T+1, and so on.
After free-growth completes normally (no EOS), len(tokens) == budget+1-T.
We cache as many tokens as we generated, so partial caches grow over time
as the example is seen at different (increasing) budgets.

Sharing across runs
-------------------
Multiple training runs with the same model share the same cache directory.
The model_name subdirectory isolates caches from different LLMs.
Writes are atomic (temp file + rename) so concurrent writes are safe —
the worst case is two processes writing identical content simultaneously.

Invalidation
------------
The only way a cached entry becomes stale is if the LLM weights change.
Using a different model → different model_name → different subdirectory.
There is NO automatic content-hash check; rely on model_name to separate
caches for different models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FreeGrowthCache:
    """Read/write cache for per-example free-growth token sequences."""

    def __init__(self, cache_dir: str | Path, model_name: str):
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        self._dir  = Path(cache_dir).expanduser() / safe_name
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def _path(self, example_idx: int) -> Path:
        return self._dir / f"{example_idx:06d}.npy"

    def get(self, example_idx: int) -> np.ndarray | None:
        """Return cached tokens (int32 array) or None if not cached."""
        p = self._path(example_idx)
        if p.exists():
            try:
                return np.load(p)
            except Exception:
                return None   # corrupt file — treat as miss
        return None

    def put(self, example_idx: int, tokens: list[int] | np.ndarray) -> None:
        """Save tokens atomically.  Safe for concurrent writes (last write wins)."""
        p   = self._path(example_idx)
        tmp = p.with_suffix(".tmp")
        np.save(tmp, np.array(tokens, dtype=np.int32))
        tmp.replace(p)   # atomic on POSIX

    def stats(self) -> tuple[int, int]:
        """Return (n_cached, total_tokens) for logging."""
        files = list(self._dir.glob("*.npy"))
        total = 0
        for f in files:
            try:
                total += len(np.load(f))
            except Exception:
                pass
        return len(files), total
