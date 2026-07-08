# Third-party notices

This repository contains code under **two different licences**. Which one applies
depends on the directory.

| Path | Owner | Licence |
|---|---|---|
| `kv-eviction-gym/` | us | [MIT](LICENSE) |
| `report/`, `docs/`, `README.md` | us | [MIT](LICENSE) |
| `ml-learning-to-evict/` | **Apple Inc.** | **Apple Sample Code License** ([`ml-learning-to-evict/LICENSE`](ml-learning-to-evict/LICENSE)) |

## Apple's code

`ml-learning-to-evict/` is a copy of
[`apple/ml-learning-to-evict`](https://github.com/apple/ml-learning-to-evict)
(KVP), the work this project starts from. It stays under the **Apple Sample Code
License** and **we cannot and do not relicense it**. Apple's files keep their
original copyright headers.

About ten files in that directory are ours, added to reproduce KVP with
Qwen2-1.5B. They carry an "Added for this project" header and are listed in the
PROVENANCE block of
[`ml-learning-to-evict/README.md`](ml-learning-to-evict/README.md). Those files
are MIT, like the rest of our code.

If you want to reuse anything from `ml-learning-to-evict/`, read Apple's licence
first: it is more restrictive than MIT, and in particular it is not an
open-source licence in the OSI sense.

## Our code

`kv-eviction-gym/` is **independent code of ours**: it contains no imports from,
and no copies of, Apple's code. The only two connections are deliberate and
documented:

- KVP's general idea as inspiration (comment in `src/kv_gym/policy.py`).
- A reimplementation of the KVP recipe in `scripts/passkey_ranker.py`, written
  from the paper and used as a reference point in the report.

The utilities under `src/kv_gym/vendor/` come from an earlier repository of
ours, not from Apple.

## Dependencies

| Component | Used for | Licence |
|---|---|---|
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) / [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) | MaskablePPO, the RL trainer | MIT |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | environment API | MIT |
| [transformers](https://github.com/huggingface/transformers) | the frozen LLM and its KV cache | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | tensors and autograd | BSD-3-Clause |
| NumPy, pandas, matplotlib | numerics and figures | BSD-3-Clause / BSD / PSF-like |

## Models and data

Qwen2.5-1.5B-Instruct is used frozen, under its own licence, and is not
redistributed here. GSM8K, HotpotQA and RULER are public datasets under their
respective terms; the `passkey` set is generated synthetically by our own code.
