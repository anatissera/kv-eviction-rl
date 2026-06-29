# The chat-template fix — why every run truncated 100% (and how it's solved)

**Status:** fixed on `main` (2026-06-29). This document supersedes the
*cold-start / free-growth-skip* root-cause analysis in `HANDOFF.md` §Phase 2:
that was a **symptom**, not the cause. The real bug is below.

---

## TL;DR

The pipeline tokenized the **raw instruction text**, never wrapping it in the
model's **chat template**. Qwen2.5-1.5B-**Instruct** only emits its end-of-turn
token — which *is* `tokenizer.eos_token_id` (`<|im_end|>`, id 151645) — when it
is answering inside the `<|im_start|>assistant … <|im_end|>` frame it was
fine-tuned on. Fed a raw prompt, the model never enters that frame, **never
emits EOS, and generates until the token cap on every example**. That single
fact produced every pathology we chased for weeks:

- 100 % truncation (no episode ever "finishes")
- correctness = 0 % (episodes are cut off mid-stream)
- `explained_variance = NaN` / no learning
- the *free-growth EOS skip* (`if eos_id in fg_tokens`) **never even fired**,
  because EOS never appeared anywhere

The fix is one helper, `format_gsm8k_chat(tokenizer, example)`, applied at every
tokenization site. It is already on `main`.

---

## The symptom we kept seeing

Every run — regardless of reward design — died at the same wall:

| Run | Reward | Result |
|---|---|---|
| `corr_reward_v1` (Alex) | correctness + entropy | 100 % trunc, retention 0, `evict_generated_frac` ≈ 0.90 |
| `s4_kl_v1` (ours) | correctness + KL-to-full | 100 % trunc, retention 0–5 %, identical pattern |

Because *every* reward failed identically, the problem was clearly **structural**,
not in the reward. `HANDOFF.md` attributed it to the free-growth skip discarding
easy examples. That turned out to be downstream of the actual cause.

## The actual root cause

`format_gsm8k()` (in `src/kv_gym/vendor/prompts.py`) returns a **raw string**:

```
Solve the following grade-school math problem step by step. After your
reasoning, put the final numeric answer on a new line after "####".

Problem: <question>
```

Every call site then did `tokenizer(prompt_text, …)` directly. No chat template.

Qwen2.5-**Instruct** is a *chat* model. Its supervised fine-tuning data always
looks like:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<the request><|im_end|>
<|im_start|>assistant
<the answer><|im_end|>      ← the model learned to emit <|im_end|> here
```

`<|im_end|>` (id 151645) is the assistant's end-of-turn token, and it is exactly
`tokenizer.eos_token_id`. With a **raw** prompt the model never sees
`<|im_start|>assistant`, so it is not in the state where it learned to stop. It
writes the answer (often correctly) and then **keeps going** — restating,
re-deriving, drifting — until generation hits `max_new_tokens`.

### Why this explains *everything* at once

- **100 % truncation.** An episode is "done" only when `next_token == eos_id`
  (`batched_env.py`). EOS never fires ⇒ every episode is truncated at the cap.
- **correctness = 0 %.** Episodes are cut off; the eviction phase never reaches a
  clean finish, so the correctness reward never lands.
- **`correct_full ≈ 0.59` in the probe is NOT a contradiction.** The full-cache
  scorer reads the answer out of the (capped) text via `flexible_extract`
  (= last number anywhere). The model *does* produce the right number around
  token ~200; it just doesn't *stop*. So full-cache scoring looked fine while
  every RL episode truncated.
- **The free-growth skip never fired.** `if self.eos_id in fg_tokens: continue`
  can only skip an example whose EOS appears during free-growth. EOS never
  appeared, so the skip was a no-op. The "skip discards easy examples" story in
  `HANDOFF.md` describes a mechanism that never triggered.

## The evidence (decisive)

A direct on-GPU diagnostic on `ppo-kvp`, same model, same GSM8K examples, only
the prompt format changed:

| Prompt format (what the pipeline fed) | finished (EOS < 600 tok) | gen_len |
|---|---|---|
| **raw** (current pipeline) | **0 / 300** | 600 for *all* (min = p50 = max = 600) |
| **chat template** | **5 / 5** | 206 – 302, `<|im_end|>` emitted every time |

Token-level confirmation:
- `tokenizer.eos_token_id = 151645 = <|im_end|>` (the right token *was*
  configured; the model simply never produced it from a raw prompt).
- `model.generation_config.eos_token_id = [151645, 151643]`.
- Raw-prompt tail: `… #### 26 ` — answer present, **no `<|im_end|>`**, runs to cap.
- Chat tail: `… **$250**<|im_end|>` — stops cleanly at gen_len ~270.

Reproduce: `scripts/find_easy_examples.py` over the raw prompt reports
`finished: 0/300` (all gen_len = 600). The same examples through the chat
template finish at 206–302.

## The fix

New helper in `src/kv_gym/vendor/prompts.py`:

```python
def format_gsm8k_chat(tokenizer, example: dict) -> tuple[str, int]:
    raw, max_new = format_gsm8k(example)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw}],
        tokenize=False,
        add_generation_prompt=True,   # opens the <|im_start|>assistant frame
    )
    return text, max_new
```

It returns the **templated string**, so existing `tokenizer(prompt, …)` calls
stay unchanged — they just receive a prompt that now carries the chat control
tokens. (Qwen adds no extra BOS on encode, so re-tokenizing the templated
string is equivalent to `apply_chat_template(tokenize=True)`.)

Applied at **every** tokenization chokepoint:

| File | Role |
|---|---|
| `src/kv_gym/capture.py` | prefill capture — covers `scripts/eval.py` **and** `probe.py` |
| `src/kv_gym/batched_env.py` | training env (`n_parallel > 1`) |
| `src/kv_gym/env.py` | sequential env (`n_parallel = 1`) |
| `scripts/find_easy_examples.py` | gen-length measurement |
| `scripts/compute_per_example_budgets.py` | budget calibration (must match training format) |
| `scripts/measure_gen_lengths.py` | analysis |

Contract locked by `tests/test_chat_template.py` (no torch needed — a fake
tokenizer records the `apply_chat_template` call).

### Answer extraction needs no change

The chat model answers in its own style (`\boxed{5}`, `**$250**`) and usually
**drops the `####` marker**. That is fine: `flexible_extract` (the lm-eval-harness
"flexible" filter) takes the **last number anywhere** in the text, which already
handles `\boxed{}` / `**$X**` / bare numbers. No `####` is required.

## Implications

1. **All prior runs are invalid** (they trained on the raw format and learned
   nothing). Nothing of value is lost — they were all stuck at 0 % correctness.
2. **`find_easy` / per-example budgets are now unnecessary.** With EOS working,
   essentially every example finishes at gen_len ~200–400 on its own. The
   per-example-budget machinery existed only to dodge the cold-start bug.
3. **Budget scheme: fixed / uniform.** The batched env (`n_parallel = 2`) needs a
   uniform cache length across the N episodes (single `[N,H,S,D]` tensor), so the
   shared budget is fixed per reset. This is also the *realistic* setup: a real
   KV-cache budget is a fixed memory size, independent of (unknown) output
   length. A per-example budget keyed on `gen_len` would use future information.
   Variable budget *could* be parallelized via padding + attention masking, but
   it is neither needed (the bug it worked around is gone) nor more defensible.
4. **Free-growth cache is stale.** Cached free-growth tokens were generated from
   the raw format. **Clear `~/.kv_eviction_cache` before the next run.**

## Operational checklist before the next run

```bash
# on the VM
rm -rf ~/.kv_eviction_cache/*          # stale raw-format tokens
# then launch the chat-template smoke (fixed budget, no find_easy):
#   configs/chat_smoke_v1.yaml   (budget=200, n_examples=1000, protect_prompt=true)
```

**H0 to confirm** in `probe_curve.csv`: `truncation_rate < 1.0` **and**
`correct_learned > 0`. The smoke doubles as budget calibration — if
`truncation_rate` stays high, adjust `budget_min/max`.
