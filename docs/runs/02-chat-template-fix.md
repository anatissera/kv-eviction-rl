# 02 · The root cause that invalidated every initial run: the chat template (+ budget fix)

**Report section:** §4.4 "The chat-template problem".
**Code fix:** `format_gsm8k_chat` in `kv-eviction-gym/src/kv_gym/vendor/prompts.py`
(commit `7c6d2d8` on main, 2026-06-29). Contract test: `tests/test_chat_template.py`.

## Symptom

Every run, regardless of the reward, died against the same wall:

| run | reward | result |
|---|---|---|
| `corr_reward_v1` | correctness + entropy | 100% truncation, retention 0 |
| `s4_kl_v1` | correctness + KL-to-full | 100% truncation, retention 0-5%, identical pattern |

100% truncation, 0% correctness, `explained_variance=NaN`: PPO with no gradient.
Before finding the real cause, mitigations that did not attack the problem were tried
(easy-example sets, per-example budgets, prompt protection); the handoff of that era
(now in the git history) attributed the problem to the "free-growth skip" that discarded
easy examples. That was a symptom, not the cause.

## Root cause

The pipeline tokenized the **raw text** of the problem statement, without the model's
**chat template**. Qwen2.5-1.5B-**Instruct** only emits its end-of-turn token `<|im_end|>`
(which IS `tokenizer.eos_token_id`, id 151645) when it answers inside the
`<|im_start|>assistant ... <|im_end|>` frame it was fine-tuned on. With a raw prompt the
model never enters that frame, **never emits EOS**, and generates until `max_new_tokens`
on every episode. That explains everything at once: 100% truncation, 0 correctness,
EV=NaN, and why the free-growth skip never fired (EOS never appeared).

## Decisive evidence

Same model, same GSM8K examples, only the prompt format changes:

| format | finish (EOS < 600 tok) | gen_len |
|---|---|---|
| raw (old pipeline) | **0 / 300** | 600 for all (min = p50 = max) |
| chat template | **5 / 5** | 206-302, `<|im_end|>` always emitted |

## The fix

```python
def format_gsm8k_chat(tokenizer, example):
    raw, max_new = format_gsm8k(example)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw}],
        tokenize=False, add_generation_prompt=True,
    )
    return text, max_new
```

Applied at every tokenization point: `capture.py` (eval + probe), `batched_env.py` and
`env.py` (training). Answer extraction needs no changes: `flexible_extract`
(lm-eval-harness's flexible filter) takes the last number in the text and already handles
the Instruct formats (`\boxed{}`, `**$X**`) without requiring `####`.

Implications:
1. **All previous runs are invalid** (they trained on the raw format and learned nothing;
   they sat at 0% correctness).
2. The easy-example and per-example-budget machinery becomes unnecessary: with EOS
   working, almost every example finishes on its own at gen_len 200-400.
3. The previous free-growth cache became invalid (raw-format tokens).

## The coupled fix: budget floor and the `_replace_slice` crash

With the chat template active, `chat_full_v1` (control) crashed at 309k steps:
`_replace_slice tensor size mismatch (195 vs 218)`. Cause: the budget curriculum went
down to 194 while the batched environment requires a uniform cache size (`budget+1` slots
shared by the N episodes); resetting an episode with a 217-token prompt makes
`ep.budget = max(shared_budget, T) = 217`, breaking the invariant and crashing.

Chat-format prompt lengths over the 1000 training examples:

| min | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| 80 | 116 | 147 | 158 | 186 | **232** |

With a curriculum floor of 120, 40.6% of the examples have a prompt longer than the whole
budget. **Fix: fixed budget = 256** (larger than max(T)=232), no curriculum, `max_len=288`.
Compression still happens where it should: over the generated reasoning chain. This is
also the realistic regime (a fixed memory budget, independent of output length). This
architectural restriction (budget >= prompt in the batched training environment) is what
later explains the regime insight ([05](05-wide-eval-regime.md)).

## What the report says

§4.4 tells the story of the bug (including the trap of the earlier "cold start"
diagnosis) and §4.8 uses the batched environment's budget restriction to explain the
low-difficulty regime. The prompt-percentile table is behind the budget=256 decision used
in every scaled experiment.

## Status

Current: the fix is in the code and is a validity condition for everything that follows.
Any result dated before 2026-06-29 is invalid due to this bug.
