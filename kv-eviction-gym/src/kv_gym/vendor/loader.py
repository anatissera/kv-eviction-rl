# Copied from internal-signals-context-compression/src/models/loader.py
# Source: /Users/alexanderbodner/Documents/Udesa/5to/tesis/internal-signals-context-compression
"""
Load instruction-tuned Llama or Qwen models from HuggingFace Hub.

The eviction research targets small instruct models so the experimental
loop is fast. The two defaults are:

    - Qwen 2.5 1.5B Instruct — non-gated, runs on a Mac with MPS or CPU
    - Llama 3.2 1B  Instruct — gated (needs HF token), used on CUDA servers

:func:`load_model_and_tokenizer` takes a canonical short name (e.g.
``"qwen-1.5b"``) and returns ``(model, tokenizer, device)``. It does
*not* call ``.eval()`` or set ``output_attentions`` — those are concerns
of the inference loop, not the loader.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Canonical short names → HuggingFace model IDs.
# Add a row here to make a new model available everywhere in the project.
KNOWN_MODELS: dict[str, str] = {
    # 1B-class (small models we can run on a single A6000 for every step,
    # including attention-output-based concentration analysis).
    "qwen-1.5b":  "Qwen/Qwen2.5-1.5B-Instruct",
    "llama-1b":   "meta-llama/Llama-3.2-1B-Instruct",

    # 3B-class — a mid-size point that still fits on a 16-24 GB Mac in fp16
    # (~6 GB), between the 1.5B and 7B peers.
    "qwen-3b":    "Qwen/Qwen2.5-3B-Instruct",

    # 7B/8B-class (large peers). Llama 3.2 doesn't ship an 8B, so Llama
    # 3.1's 8B is the closest peer of Qwen 2.5 7B.
    "qwen-7b":    "Qwen/Qwen2.5-7B-Instruct",
    "llama-8b":   "meta-llama/Llama-3.1-8B-Instruct",

    # Tiny CPU-only synthetic model used by the test suite. Lives here so
    # tests go through the same loading path as real runs.
    "tiny-llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
}


def pick_device() -> torch.device:
    """Return the best available device for inference.

    Preference order: CUDA > MPS (Apple Silicon) > CPU. We don't try to be
    clever about multi-GPU here — the eviction work is single-GPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(
    name: str | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    attn_implementation: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """Load a known model + its tokenizer and move the model to ``device``.

    Args:
        name:   Canonical short name from :data:`KNOWN_MODELS`. If None,
                picks a default based on the device.
        device: torch device. If None, auto-detected.
        dtype:  Model dtype. If None: bfloat16 on CUDA (when supported),
                else float32 (MPS has spotty float16 support for some ops).
                bf16 — not fp16 — on CUDA is deliberate: these models are
                trained in bf16, and **eager** attention (forced for
                attention-based scoring) computes QK^T in the model dtype.
                In fp16 that overflows 65504 on long contexts → inf → NaN
                softmax → NaN cascades through every deeper layer. bf16 has
                fp32's exponent range, so it stays finite. (The no-eviction
                baseline used SDPA and was unaffected; the eviction path uses
                eager and is not.)
        attn_implementation:
                Attention backend. Default ``None`` lets HuggingFace pick
                (currently ``"sdpa"`` on most setups). Pass ``"eager"`` if
                you need ``output_attentions=True`` — SDPA and FlashAttention
                do not expose attention weights, and silently return None.
                Required for attention-based scoring (see :mod:`src.scoring`).

    Returns:
        (model, tokenizer, device)

    Raises:
        ValueError: if ``name`` is not in :data:`KNOWN_MODELS`.
    """
    device = device if device is not None else pick_device()
    # Default: Llama on CUDA (we run on a server with a Llama-gated token),
    # Qwen everywhere else (non-gated, easy to run locally).
    if name is None:
        name = "llama-1b" if device.type == "cuda" else "qwen-1.5b"

    if name not in KNOWN_MODELS:
        raise ValueError(
            f"Unknown model name: '{name}'. "
            f"Known: {list(KNOWN_MODELS.keys())}"
        )

    if dtype is None:
        if device.type == "cuda":
            # bf16 (not fp16): eager attention computes QK^T in the model dtype,
            # which overflows fp16 on long contexts → NaN. bf16 matches fp32's
            # exponent range. Fall back to fp16 only if the GPU lacks bf16.
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

    model_id = KNOWN_MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model = model.to(device)

    # Some tokenizers ship without a pad token. We set it to EOS so that
    # the generation code can pad freely without hitting an error.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


if __name__ == "__main__":
    print(f"device:    {pick_device()}")
    print(f"available: {list(KNOWN_MODELS.keys())}")
