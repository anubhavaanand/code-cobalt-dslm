"""Model tests. Requires torch."""

import pytest

torch = pytest.importorskip("torch")

from dslm.config import PRESETS, ModelConfig
from dslm.model import GPT


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return GPT(ModelConfig())


def test_param_count(model):
    """The default config is the smoke preset: ~13.8M params."""
    n = model.num_params()
    assert 12e6 < n < 16e6, f"expected ~13.8M, got {n/1e6:.2f}M"


def test_weight_tying(model):
    """Input embedding and output projection must share storage."""
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()


def test_forward_shape(model):
    cfg = model.cfg
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, _ = model(x)
    assert logits.shape == (2, 32, cfg.vocab_size)


def test_loss_is_finite(model):
    cfg = model.cfg
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    _, loss = model(x, y)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_backward_produces_finite_grads(model):
    cfg = model.cfg
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    _, loss = model(x, y)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients were produced"
    for g in grads:
        assert torch.isfinite(g).all(), "non-finite gradient"


def test_loss_decreases_on_a_fixed_batch(model):
    """One optimizer step on a single repeated batch must reduce the loss.

    A model that cannot overfit a single batch is broken; this catches
    misconfigured attention masks, bad init, and label/logit misalignment.
    """
    cfg = model.cfg
    torch.manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    _, before = model(x, y)
    for _ in range(20):
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
    _, after = model(x, y)

    assert after.item() < before.item(), (
        f"loss did not fall on a fixed batch: {before.item():.4f} -> {after.item():.4f}"
    )


def test_generate_shape(model):
    cfg = model.cfg
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 13)
    assert out[:, :8].equal(idx), "generate must preserve the prompt"


def test_generate_appends_exactly_max_new_tokens(model):
    """Generation appends the requested count to the prompt.

    The returned sequence can exceed block_size: the model only ever *reads*
    the last block_size tokens (see the slice in generate), but the full history
    is returned so callers can decode it. This is the same behaviour as nanoGPT.
    """
    cfg = model.cfg
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape[1] == idx.shape[1] + 10
    assert out[:, : idx.shape[1]].equal(idx), "prompt must be preserved"


def test_every_preset_builds():
    """All presets must instantiate without error."""
    for name, (model_cfg, _) in PRESETS.items():
        m = GPT(model_cfg)
        assert m.num_params() > 0, f"{name} built with zero params"