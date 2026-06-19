# Copied from internal-signals-context-compression/src/eval/prompts.py
# Source: /Users/alexanderbodner/Documents/Udesa/5to/tesis/internal-signals-context-compression
# Only the GSM8K-relevant parts are used in this project.
"""
Prompt templates and generation caps.

We only use the GSM8K parts here. The LongBench templates are kept for
completeness but are not used in the gym environment.
"""


# ---------------------------------------------------------------------------
# GSM8K prompt
# ---------------------------------------------------------------------------

GSM8K_INSTRUCTION: str = (
    "Solve the following grade-school math problem step by step. "
    "After your reasoning, put the final numeric answer on a new line "
    "after \"####\".\n\n"
    "Problem: {question}"
)

# Most GSM8K solutions are short, but verbose models like Qwen need more
# headroom for detailed chain-of-thought.
GSM8K_MAX_NEW_TOKENS: int = 524


def format_gsm8k(example: dict) -> tuple[str, int]:
    """Return ``(prompt_text, max_new_tokens)`` for one GSM8K example."""
    prompt = GSM8K_INSTRUCTION.format(question=example["prompt_text"])
    return prompt, GSM8K_MAX_NEW_TOKENS
