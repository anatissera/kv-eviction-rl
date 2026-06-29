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


def format_gsm8k_chat(tokenizer, example: dict) -> tuple[str, int]:
    """GSM8K prompt wrapped in the model's chat template.

    Instruct models (e.g. Qwen2.5-Instruct) were fine-tuned to answer inside a
    ``<|im_start|>assistant ... <|im_end|>`` frame and emit their end-of-turn
    token — which IS ``tokenizer.eos_token_id`` — to stop.  Feeding the *raw*
    instruction text skips that frame, so the model never emits EOS and
    generates until the token cap.  That broke episode termination: nothing
    finished, every episode truncated, correctness was always 0, and the
    free-growth EOS skip never even fired.  Wrapping with the chat template
    restores natural stopping (gen_len ~200-300 on GSM8K, EOS emitted).

    Returns the templated *string* (not token ids) so existing
    ``tokenizer(prompt, ...)`` call sites stay unchanged — they just receive a
    prompt that now carries the chat control tokens.  Qwen adds no extra BOS on
    encode, so re-tokenizing this string is equivalent to
    ``apply_chat_template(tokenize=True)``.
    """
    raw, max_new = format_gsm8k(example)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return text, max_new
