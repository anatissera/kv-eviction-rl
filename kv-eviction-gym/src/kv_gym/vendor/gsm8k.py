# Copied from internal-signals-context-compression/src/data/gsm8k.py
# Source: /Users/alexanderbodner/Documents/Udesa/5to/tesis/internal-signals-context-compression
"""
GSM8K loader.

GSM8K (Grade School Math 8K) is a benchmark of 8,500 grade-school math word
problems. Each problem requires multi-step arithmetic reasoning and has a
single numeric answer.

The raw dataset stores the answer as a chain-of-thought string ending in
'#### N', e.g. "She baked 3 * 4 = 12 cookies. #### 12". We keep only the
final number — the chain-of-thought is not used for evaluation.

Every example returned by :func:`load_gsm8k` follows the project-wide schema:

    {
        "prompt_text":  str,        # the math problem, plain text
        "gold_answers": list[str],  # always length 1 for GSM8K, e.g. ["42"]
        "task":         "gsm8k",
    }
"""

import warnings


TASK_NAME = "gsm8k"


def load_gsm8k(
    n: int = 100,
    seed: int = 0,
    streaming: bool = False,
    split: str = "test",
    min_answer_words: int = 0,
) -> list[dict]:
    """Load ``n`` examples from a GSM8K split.

    Args:
        n:         Number of examples to return.
        seed:      Random seed for the shuffle (ignored when ``streaming=True``).
        streaming: If False (default), download the full split, shuffle with
                   ``seed``, then take the first ``n`` examples. Fully
                   reproducible — use this for experiments.

                   If True, stream examples from HuggingFace Hub and take the
                   first ``n`` without downloading the full split. No shuffle.
                   Use this for local development when you just need a few
                   examples to look at quickly.
        split:     Which split to load ("train" or "test"). Use "train" for
                   training to avoid overlap with held-out evaluation data.
                   GSM8K has 7,473 train and 1,319 test examples.
        min_answer_words: Phase-A regime filter — keep only examples whose GOLD
                   reasoning (answer text before "####") has at least this many
                   words, a free proxy for CoT/generation length. The oracle-gap
                   analysis (FINDINGS §11-§13) showed eviction damage — ALL the
                   learnable headroom — concentrates in long generations
                   (gen<150 → zero damage; gen 400-600 → full−kv_norm=+0.11,
                   oracle recovers 100%). Applied AFTER the seeded shuffle →
                   deterministic and prefix-stable for any n.

    Returns:
        List of example dicts following the project-wide schema (see the
        module docstring).
    """
    from datasets import load_dataset

    if streaming:
        ds = load_dataset("openai/gsm8k", "main", split=split, streaming=True)
        return [_row_to_example(row) for row in ds.take(n)]

    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.shuffle(seed=seed)

    if min_answer_words > 0:
        keep = [i for i, a in enumerate(ds["answer"])
                if len(a.split("####")[0].split()) >= min_answer_words]
        ds = ds.select(keep)

    if n > len(ds):
        warnings.warn(
            f"load_gsm8k: requested n={n} but dataset only has {len(ds)} examples"
            + (f" after min_answer_words={min_answer_words}" if min_answer_words else "")
            + f". Returning all {len(ds)}."
        )
        n = len(ds)

    return [_row_to_example(row) for row in ds.select(range(n))]


def _row_to_example(row: dict) -> dict:
    raw_answer = row["answer"]
    numeric_answer = raw_answer.split("####")[-1].strip().replace(",", "")
    return {
        "prompt_text":  row["question"],
        "gold_answers": [numeric_answer],
        "task":         TASK_NAME,
    }
