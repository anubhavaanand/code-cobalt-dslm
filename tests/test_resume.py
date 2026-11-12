"""Checkpoint/resume tests.

Kaggle kills sessions at ~9 hours and the base preset needs ~30. If resume does
not work, every session throws away the whole run. That makes this the highest
consequence untested path in the codebase.
"""

import pytest

torch = pytest.importorskip("torch")

from dslm.config import PRESETS, ModelConfig, TrainConfig
from dslm.model import GPT
from dslm.train import load_checkpoint, save_checkpoint


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(0)
    model_cfg, train_cfg = PRESETS["smoke"]
    train_cfg.ckpt_dir = str(tmp_path)

    model = GPT(model_cfg)
    opt = model.configure_optimizers(
        train_cfg.weight_decay, train_cfg.lr, device_type="cpu"
    )
    return model, opt, model_cfg, train_cfg


def test_save_then_load_restores_step(setup):
    model, opt, model_cfg, train_cfg = setup
    save_checkpoint(model, opt, model_cfg, train_cfg, step=1234, val_loss=4.5)

    fresh = GPT(model_cfg)
    fresh_opt = fresh.configure_optimizers(
        train_cfg.weight_decay, train_cfg.lr, device_type="cpu"
    )

    step, val = load_checkpoint(train_cfg.ckpt_dir + "/ckpt_001234.pt", fresh, fresh_opt, "cpu")

    assert step == 1234
    assert val == 4.5


def test_resume_restores_weights(setup):
    model, opt, model_cfg, train_cfg = setup

    # Perturb the weights so a successful load is detectable.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    save_checkpoint(model, opt, model_cfg, train_cfg, step=10, val_loss=1.0)

    fresh = GPT(model_cfg)
    fresh_opt = fresh.configure_optimizers(
        train_cfg.weight_decay, train_cfg.lr, device_type="cpu"
    )
    load_checkpoint(train_cfg.ckpt_dir + "/latest.pt", fresh, fresh_opt, "cpu")

    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.allclose(a, b), "weights were not restored"


def test_resume_restores_optimizer_state(setup):
    model, opt, model_cfg, train_cfg = setup

    # Take a step so the optimizer has state worth saving.
    x = torch.randint(0, model_cfg.vocab_size, (2, 16))
    y = torch.randint(0, model_cfg.vocab_size, (2, 16))
    _, loss = model(x, y)
    loss.backward()
    opt.step()

    save_checkpoint(model, opt, model_cfg, train_cfg, step=1, val_loss=None)

    fresh = GPT(model_cfg)
    fresh_opt = fresh.configure_optimizers(
        train_cfg.weight_decay, train_cfg.lr, device_type="cpu"
    )
    load_checkpoint(train_cfg.ckpt_dir + "/latest.pt", fresh, fresh_opt, "cpu")

    assert len(fresh_opt.state) > 0, "optimizer state was not restored"
    for group in fresh_opt.param_groups:
        assert "lr" in group


def test_resume_without_val_loss_is_safe(setup):
    """Mid-run checkpoints pass val_loss=None; loading must not crash."""
    model, opt, model_cfg, train_cfg = setup
    save_checkpoint(model, opt, model_cfg, train_cfg, step=5, val_loss=None)

    fresh = GPT(model_cfg)
    fresh_opt = fresh.configure_optimizers(
        train_cfg.weight_decay, train_cfg.lr, device_type="cpu"
    )
    step, val = load_checkpoint(train_cfg.ckpt_dir + "/ckpt_000005.pt", fresh, fresh_opt, "cpu")

    assert step == 5
    assert val is None