# Copied from internal-signals-context-compression/src/eval/metrics/answer_extraction_gsm8k.py
# Source: /Users/alexanderbodner/Documents/Udesa/5to/tesis/internal-signals-context-compression
"""
GSM8K answer extraction and scoring.

These two functions reproduce the two filter chains that lm-evaluation-harness
uses for GSM8K, with a small extension to handle modern model formatting
(boxed answers).

Source:
    https://github.com/EleutherAI/lm-evaluation-harness
    File:    lm_eval/tasks/gsm8k/gsm8k.yaml
    Patterns and filter idea: copied verbatim, then extended for \\boxed{}.
"""

import re


_STRICT_RE   = re.compile(r"####\s*(\-?[0-9\.\,]+)")
_BOXED_RE    = re.compile(r"\\boxed\s*\{\s*(\-?[0-9\.\,]+)\s*\}")
_FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def _to_number(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def _matches_any_gold(candidate: str | None, gold_answers: list[str]) -> float:
    if candidate is None:
        return 0.0
    cand_num = _to_number(candidate)
    if cand_num is None:
        return 0.0
    for gold in gold_answers:
        gold_num = _to_number(gold)
        if gold_num is not None and gold_num == cand_num:
            return 1.0
    return 0.0


def strict_match(prediction: str, gold_answers: list[str]) -> float:
    """Number after ``####`` or inside ``\\boxed{}``."""
    m = _STRICT_RE.search(prediction)
    if m:
        return _matches_any_gold(m.group(1), gold_answers)
    m = _BOXED_RE.search(prediction)
    candidate = m.group(1) if m else None
    return _matches_any_gold(candidate, gold_answers)


def flexible_extract(prediction: str, gold_answers: list[str]) -> float:
    """lm-eval-harness ``flexible-extract`` filter — last number anywhere."""
    matches = _FLEXIBLE_RE.findall(prediction)
    if not matches:
        return 0.0
    last = matches[-1]
    candidate = (last[0] or last[1]) if isinstance(last, tuple) else last
    return _matches_any_gold(candidate, gold_answers)


def score(prediction: str, gold_answers: list[str]) -> dict[str, float]:
    """Return both lm-eval-harness GSM8K scores for one prediction."""
    return {
        "strict_match":     strict_match(prediction, gold_answers),
        "flexible_extract": flexible_extract(prediction, gold_answers),
    }
