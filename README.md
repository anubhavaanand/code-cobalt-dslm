# Cobalt

**A domain-specific language model for Python, verified by tests rather than trusted.**

Cobalt is an experiment in a single question:

> If a task is narrow enough to be machine-verifiable, does a small model plus
> free retries beat an expensive model on cost and latency at comparable quality?

The answer is probably yes, and the reason is arithmetic rather than opinion. If a
single attempt succeeds with probability `p`, then `n` free retries succeed with
probability `1 − (1−p)^n`:

| per-try `p` | success in 100 tries |
|---|---|
| 0.20 | 99.99999998% |
| 0.10 | 99.997% |
| **0.05** | **99.4%** |
| 0.01 | 63.4% |

A model with 5% per-attempt accuracy is functionally perfect given free retries.
That reframes the requirement from *"be as good as the frontier"* to *"have a
non-trivial chance of being right"* — a far more achievable target.

## How it works

```
frontier model  →  decomposes a task into verifiable subtasks
                →  writes test cases (the oracle)
                →  Cobalt drafts code for one subtask
                →  pytest / mypy / ruff verify
                →  retry until pass or budget exhausted
```

The model never plans. It executes. **It is a component, not a product.**

## Status

**Phase 0 — pipeline bring-up.** The pipeline runs and is verified end to end;
nothing has been trained at scale yet.

## Machine policy

**Training does not happen on the laptop.** The dev machine is a 4-thread i3 with
no GPU — a training run there takes hours and makes it unusable. All training runs
on **Kaggle** (free 2×T4, 30 h/week).

| Task | Where |
|---|---|
| Editing, reading logs, docs | laptop |
| Tokenising <10M tokens | laptop |
| Any training run | Kaggle |
| Full-scale data prep | Kaggle |
| Inference on a finished model | laptop (~20–40 tok/s) |

## Quick start (Kaggle)

```python
# Kaggle notebook, accelerator: GPU T4 x2
!git clone <this-repo> cobalt && cd cobalt
%pip install -q -r requirements.txt
!python scripts/prepare_data.py --preset kaggle
!python -m dslm.train --preset kaggle
!python -m dslm.sample --ckpt checkpoints/latest.pt --prompt "Once upon a time"
```

The smoke model will not be good. It exists to prove the pipeline connects.

## Layout

```
dslm/
  config.py      model + training presets (smoke, kaggle, base)
  model.py       the transformer: RMSNorm, RoPE, SwiGLU, weight tying
  data.py        memory-mapped uint16 token streams
  tokenizer.py   byte-level BPE (no OOV, so code always round-trips)
  train.py       pretraining loop, with checkpoint/resume
  sample.py      generation from a checkpoint
scripts/
  prepare_data.py         download -> split -> tokenize -> shard
  make_kaggle_notebook.py regenerates the notebook
notebooks/
  Cobalt_Phase0.ipynb     15 cells, ready to run on Kaggle
```

## Verified working

| Check | Result |
|---|---|
| `smoke` preset | 13.77M params |
| `base` preset | 122.70M params |
| forward + backward | finite loss and gradients |
| tokenizer | trains, round-trips, no OOV |
| data prep | 4.27M tokens, 4.20 bytes/token (TinyStories) |
| train loop | loss 9.11 -> 4.11 on Kaggle T4, checkpoint written |
| sampling | produces grammatical text |

## What this is not

- Not a general chatbot. No pretense of broad capability.
- Not a frontier competitor. That is a compute problem, not an engineering one.
- Not multi-language. Capacity divided across languages is capacity lost.
- Not human-facing. The interface is structured specs and test cases.

## The honest scope fence

Cobalt works when correctness is machine-checkable. It breaks when:

- the tests encode the wrong behaviour (the #1 failure mode)
- the task needs cross-file or global reasoning
- the task needs a novel algorithm
- the edit chain is long and dependent

**A repo with no tests is outside its competence, no matter how good the model
is.** That is a scope statement, not a limitation to hide.

Two constraints from the literature that shape the design:

- **Single-shot only.** Multi-turn tool use collapses below 1B params (TinyLLM).
  No conversation, no memory — the harness holds all state.
- **The decomposer must be a frontier model.** Decomposition prompts do not
  generalise across domains (Least-to-Most).

## Prior work

The test-input selection idea is structurally close to **CodeT** (ICLR 2023,
arXiv:2207.10397), which generates programs and tests jointly and selects by dual
execution agreement. The difference: CodeT selects among generated programs;
Cobalt proposes selecting among the model's own function repertoire. That
difference is the contribution and the risk — CodeT's assumption that *incorrect
solutions rarely agree* is unproven for *incorrect functions*.

## Rights

**Proprietary. All rights reserved.** See [LICENSE](LICENSE).

The code is public so you can read it. It is not open source: no permission is
granted to copy, modify, redistribute, or use it commercially, or to train a
competing model. Publication for visibility is not a grant of rights.