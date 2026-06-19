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
) -> list[dict]:
    """Load ``n`` examples from the GSM8K test split.

    Args:
        n:         Number of examples to return.
        seed:      Random seed for the shuffle (ignored when ``streaming=True``,
                   see below).
        streaming: If False (default), download the full test split, shuffle
                   with ``seed``, then take the first ``n`` examples. Fully
                   reproducible — use this for experiments.

                   If True, stream examples from HuggingFace Hub and take the
                   first ``n`` without downloading the full split. No shuffle
                   (``IterableDataset`` shuffling needs a buffer the size of
                   the dataset to match the non-streaming order, which defeats
                   the purpose). Use this for local development when you just
                   need a few examples to look at quickly.

    Returns:
        List of example dicts following the project-wide schema (see the
        module docstring).
    """
    from datasets import load_dataset

    if streaming:
        ds = load_dataset("gsm8k", "main", split="test", streaming=True)
        return [_row_to_example(row) for row in ds.take(n)]

    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed)

    if n > len(ds):
        warnings.warn(
            f"load_gsm8k: requested n={n} but dataset only has {len(ds)} examples. "
            f"Returning all {len(ds)}."
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
