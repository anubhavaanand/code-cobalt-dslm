"""Config sanity.

The bug this guards against: a preset whose `total_steps` is far too low, so a
run finishes "successfully" on a badly undertrained model. That happened once —
4,000 steps was 33M tokens for a 35M-param model, roughly 30x short — and it
would have produced garbage that looked like an architecture problem.

`tokens_per_param` is the invariant. Chinchilla-optimal is ~20; below ~5 is
undertrained; above ~100 is a waste of compute for a model this size.
"""

import pytest

from dslm.config import PRESETS

# Rough parameter counts, so we can compute tokens/param without building a model
# (which would require torch).
APPROX_PARAMS = {
    "smoke": 13.8e6,
    "kaggle": 35.5e6,
    "base": 122.7e6,
}


def tokens_per_step(model_cfg, train_cfg) -> int:
    return train_cfg.batch_size * train_cfg.grad_accum * model_cfg.block_size


# `smoke` is a pipeline check, not a trainable model — it is deliberately
# undertrained so it finishes in seconds. The token-budget invariant applies to
# the presets you would actually train on.
PIPELINE_ONLY = {"smoke"}


@pytest.mark.parametrize("name", [n for n in PRESETS if n not in PIPELINE_ONLY])
def test_preset_is_trainable(name):
    """Every real preset must have a sane token budget for its size."""
    model_cfg, train_cfg = PRESETS[name]

    total_tokens = tokens_per_step(model_cfg, train_cfg) * train_cfg.total_steps
    tpp = total_tokens / APPROX_PARAMS[name]

    assert 5 <= tpp <= 100, (
        f"{name}: {total_tokens/1e6:.1f}M tokens for ~{APPROX_PARAMS[name]/1e6:.0f}M "
        f"params is {tpp:.1f} tokens/param. Undertrained below ~5, wasteful above ~100."
    )


@pytest.mark.parametrize("name", list(PRESETS))
def test_preset_dims_are_consistent(name):
    model_cfg, _ = PRESETS[name]

    assert model_cfg.n_embd % model_cfg.n_head == 0, (
        f"{name}: n_embd ({model_cfg.n_embd}) must divide by n_head ({model_cfg.n_head})"
    )
    assert model_cfg.head_dim >= 32, f"{name}: head_dim {model_cfg.head_dim} is too small"
    assert model_cfg.vocab_size > 0
    assert model_cfg.block_size >= 256, f"{name}: block_size {model_cfg.block_size} too small"


@pytest.mark.parametrize("name", list(PRESETS))
def test_preset_schedule_is_sane(name):
    _, train_cfg = PRESETS[name]

    assert 0 < train_cfg.warmup_steps < train_cfg.total_steps, (
        f"{name}: warmup ({train_cfg.warmup_steps}) must be inside the run "
        f"({train_cfg.total_steps} steps)"
    )
    assert train_cfg.min_lr < train_cfg.lr, f"{name}: min_lr must be below lr"
    assert train_cfg.min_lr > 0, f"{name}: min_lr must be positive"
    assert train_cfg.grad_accum >= 1
    assert train_cfg.batch_size >= 1

def test_smoke_is_short():
    """The smoke preset exists to prove the pipeline connects, nothing more."""
    model_cfg, train_cfg = PRESETS["smoke"]
    total = tokens_per_step(model_cfg, train_cfg) * train_cfg.total_steps
    assert total < 20e6, (
        f"smoke should be a few minutes, got {total/1e6:.1f}M tokens"
    )
