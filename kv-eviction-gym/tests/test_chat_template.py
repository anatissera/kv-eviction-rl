"""Contract test for the chat-template prompt wrapper (format_gsm8k_chat).

Dependency-light: no LLM, no torch — uses a fake tokenizer that records the
arguments passed to apply_chat_template.  The real behaviour (Instruct model
emits EOS and stops) is validated on-GPU; this just locks the call contract so
the raw-prompt regression (model never emits EOS → 100% truncation) can't
silently come back.
"""

from kv_gym.vendor.prompts import format_gsm8k, format_gsm8k_chat


class _FakeTokenizer:
    """Records the apply_chat_template call and returns a marker string."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        self.calls.append(
            dict(messages=messages, tokenize=tokenize,
                 add_generation_prompt=add_generation_prompt)
        )
        # Mimic a real chat template wrapping the user content in im_* tokens.
        user = messages[-1]["content"]
        gen = "<|im_start|>assistant\n" if add_generation_prompt else ""
        return f"<|im_start|>user\n{user}<|im_end|>\n{gen}"


_EXAMPLE = {"prompt_text": "A robot has 3 apples and eats 1. How many remain?",
            "gold_answers": ["2"]}


def test_wraps_raw_prompt_as_single_user_message():
    tok = _FakeTokenizer()
    text, max_new = format_gsm8k_chat(tok, _EXAMPLE)

    assert len(tok.calls) == 1
    call = tok.calls[0]
    # exactly one user message carrying the raw GSM8K instruction
    assert [m["role"] for m in call["messages"]] == ["user"]
    raw, raw_max = format_gsm8k(_EXAMPLE)
    assert call["messages"][0]["content"] == raw
    # must return the string (not token ids) and open the assistant turn
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True
    assert max_new == raw_max


def test_output_contains_assistant_frame():
    """The whole point: the prompt must open the assistant turn so the model
    enters the frame where it learned to emit <|im_end|> (EOS) and stop."""
    tok = _FakeTokenizer()
    text, _ = format_gsm8k_chat(tok, _EXAMPLE)
    assert "<|im_start|>assistant" in text
    assert _EXAMPLE["prompt_text"] in text
