"""Configuration.

Three presets:

  smoke   ~14M params  — pipeline check, seconds of GPU time
  base   ~122M params  — the real model. Kaggle 2xT4, ~30h for 2B tokens.
  kaggle  ~35M params  — ablation workhorse. ~2h per run on a T4.

`base` is deliberately close to SmolLM-135M so we can compare against a
published model and know whether our pipeline is broken.

Tokens per step = batch_size * grad_accum * block_size:

  smoke   16 * 1 *  512 =  8,192   ->   600 steps =    4.9M tokens
  base     8 * 4 * 2048 = 65,536   -> 30,000 steps = 1.97B tokens
  kaggle   8 * 4 * 1024 = 32,768   ->  9,000 steps =  295M tokens

Throughput measured on a Kaggle T4 (fp16, 13.77M params, block 512):
~85,000 tok/s. On the dev laptop (4-thread i3, fp32): ~1,000 tok/s.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 512
    vocab_size: int = 8192
    mlp_ratio: float = 8.0 / 3.0
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data"
    token_file: str = "train.bin"
    val_token_file: str = "val.bin"

    # optimisation
    batch_size: int = 32          # micro-batch (sequences per step)
    grad_accum: int = 1           # effective batch = batch_size * grad_accum
    total_steps: int = 2000
    warmup_steps: int = 100
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # runtime
    device: str = "auto"          # auto | cpu | cuda
    dtype: str = "auto"           # auto | float32 | bfloat16 | float16
    compile: bool = False
    seed: int = 1337

    # logging / checkpointing
    log_every: int = 20
    eval_every: int = 200
    eval_iters: int = 20
    ckpt_dir: str = "checkpoints"
    save_every: int = 500


PRESETS: dict[str, tuple[ModelConfig, TrainConfig]] = {
    # ~14M params. Seconds of GPU time; minutes on a laptop CPU.
    "smoke": (
        ModelConfig(n_layer=6, n_head=6, n_embd=384, block_size=512, vocab_size=8192),
        TrainConfig(
            batch_size=16,
            total_steps=600,
            warmup_steps=50,
            lr=1e-3,
            min_lr=1e-4,
            log_every=10,
            eval_every=100,
            save_every=200,
        ),
    ),
    # ~122M params, ~2B tokens. The real model: ~30h on a T4, so this needs
    # checkpoint/resume across several Kaggle sessions.
    "base": (
        ModelConfig(n_layer=12, n_head=12, n_embd=768, block_size=2048, vocab_size=49152),
        TrainConfig(
            batch_size=8,
            grad_accum=4,
            total_steps=30000,
            warmup_steps=1500,
            lr=6e-4,
            min_lr=6e-5,
            log_every=500,
            eval_every=2500,
            save_every=2500,
        ),
    ),
    # ~35M params, 1k context. 295M tokens = ~8 tokens/param, ~2h per run.
    # Five ablations fit inside a week of free quota.
    "kaggle": (
        ModelConfig(n_layer=6, n_head=8, n_embd=512, block_size=1024, vocab_size=32768),
        TrainConfig(
            batch_size=8,
            grad_accum=4,
            total_steps=9000,
            warmup_steps=200,
            lr=6e-4,
            min_lr=6e-5,
            log_every=100,
            eval_every=1000,
            save_every=1000,
        ),
    ),
}