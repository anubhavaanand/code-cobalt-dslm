"""Pretraining loop.

    python -m dslm.train --preset smoke
    python -m dslm.train --preset kaggle --resume checkpoints/ckpt_002000.pt

Reads uint16 token files produced by scripts/prepare_data.py, trains a GPT, and
writes checkpoints plus a sample of generated text so you can eyeball progress.

Everything here is deliberately explicit. No config objects from a framework, no
if-then-else monsters. If you want to change something, change it here.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from dslm.config import PRESETS, ModelConfig, TrainConfig
from dslm.data import TokenStream
from dslm.model import GPT


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    if device == "cuda":
        # bf16 is what we want on Ampere+; fp16 with a GradScaler elsewhere.
        major = torch.cuda.get_device_capability()[0]
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)

    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: GPT, stream: TokenStream, cfg: TrainConfig, ctx) -> float:
    """Average loss over eval_iters batches. Lower is better."""
    model.eval()
    losses = torch.zeros(cfg.eval_iters)
    for i in range(cfg.eval_iters):
        x, y = stream.get_batch(cfg.batch_size)
        with ctx:
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def resolve_amp_ctx(device: str, dtype: torch.dtype):
    """Return (ctx, scaler). scaler is only used for fp16 on CUDA."""
    if device == "cpu" or dtype == torch.float32:
        return torch.autocast(device_type=device, enabled=False), None

    ctx = torch.autocast(device_type=device, dtype=dtype)
    scaler = torch.amp.GradScaler(device, enabled=(dtype == torch.float16))
    return ctx, scaler


def save_checkpoint(model, optimizer, model_cfg, train_cfg, step: int, val_loss):
    out = Path(train_cfg.ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": vars(model_cfg).copy(),
            "step": step,
            "val_loss": val_loss,
        },
        out / f"ckpt_{step:06d}.pt",
    )
    # `latest.pt` is a stable name so --resume does not need to know the step.
    torch.save(
        {
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": vars(model_cfg).copy(),
            "step": step,
            "val_loss": val_loss,
        },
        out / "latest.pt",
    )


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    getattr(model, "_orig_mod", model).load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0), ckpt.get("val_loss")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="DSLM pretraining")
    parser.add_argument("--preset", choices=list(PRESETS), default="smoke")
    parser.add_argument("--data-dir", default=None, help="override data_dir")
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--val-token-file", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="override model vocab to match the trained tokenizer")
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--out", default=None, help="checkpoint directory")
    parser.add_argument("--resume", default=None,
                        help="checkpoint path, or 'latest' for <ckpt_dir>/latest.pt")
    args = parser.parse_args()

    model_cfg, train_cfg = PRESETS[args.preset]

    # CLI overrides
    if args.vocab_size:
        model_cfg.vocab_size = args.vocab_size
    if args.data_dir:
        train_cfg.data_dir = args.data_dir
    if args.token_file:
        train_cfg.token_file = args.token_file
    if args.val_token_file:
        train_cfg.val_token_file = args.val_token_file
    if args.total_steps:
        train_cfg.total_steps = args.total_steps
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.grad_accum:
        train_cfg.grad_accum = args.grad_accum
    if args.lr:
        train_cfg.lr = args.lr
    if args.device:
        train_cfg.device = args.device
    if args.dtype:
        train_cfg.dtype = args.dtype
    if args.compile:
        train_cfg.compile = True
    if args.seed:
        train_cfg.seed = args.seed
    if args.log_every is not None:
        train_cfg.log_every = args.log_every
    if args.eval_every is not None:
        train_cfg.eval_every = args.eval_every
    if args.save_every is not None:
        train_cfg.save_every = args.save_every
    if args.out:
        train_cfg.ckpt_dir = args.out

    # -- setup ------------------------------------------------------------- #
    device = resolve_device(train_cfg.device)
    dtype = resolve_dtype(train_cfg.dtype, device)
    torch.manual_seed(train_cfg.seed)
    generator = torch.Generator(device="cpu").manual_seed(train_cfg.seed)

    data_dir = Path(train_cfg.data_dir)
    train_stream = TokenStream(data_dir / train_cfg.token_file,
                               model_cfg.block_size, device=device)
    val_stream = TokenStream(data_dir / train_cfg.val_token_file,
                             model_cfg.block_size, device=device)

    print(f"device={device}  dtype={dtype}")
    print(f"train tokens={train_stream.n_tokens:,}  val tokens={val_stream.n_tokens:,}")
    print(f"block_size={model_cfg.block_size}  batch={train_cfg.batch_size} "
          f"x grad_accum={train_cfg.grad_accum}")

    model = GPT(model_cfg).to(device)
    print(f"params={model.num_params()/1e6:.2f}M  "
          f"(non-embedding {model.num_params(non_embedding=True)/1e6:.2f}M)")

    optimizer = model.configure_optimizers(
        train_cfg.weight_decay, train_cfg.lr, device_type=device
    )

    if train_cfg.compile:
        print("compiling model (this takes a minute)...")
        model = torch.compile(model)

    ctx, scaler = resolve_amp_ctx(device, dtype)

    # -- resume ------------------------------------------------------------ #
    start_step = 0
    best_val = float("inf")
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        if str(args.resume) == "latest":
            resume_path = Path(train_cfg.ckpt_dir) / "latest.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"cannot resume: {resume_path} not found")
        start_step, best_val = load_checkpoint(resume_path, model, optimizer, device)
        if best_val is None:
            best_val = float("inf")
        print(f"resumed from {resume_path} at step {start_step} "
              f"(best val {best_val:.4f})")

    # -- training ---------------------------------------------------------- #
    t0 = time.time()
    model.train()

    for step in range(start_step, train_cfg.total_steps):
        lr = lr_at(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Gradient accumulation: `micro_steps` forward/backward passes per
        # optimizer step.
        for _ in range(train_cfg.grad_accum):
            x, y = train_stream.get_batch(train_cfg.batch_size, generator=generator)
            with ctx:
                _, loss = model(x, y)
                loss = loss / train_cfg.grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if step % train_cfg.log_every == 0:
            dt = time.time() - t0
            tok_per_sec = (
                train_cfg.log_every * train_cfg.grad_accum
                * train_cfg.batch_size * model_cfg.block_size / max(dt, 1e-6)
            )
            print(
                f"step {step:5d}/{train_cfg.total_steps}  "
                f"loss {loss.item() * train_cfg.grad_accum:.4f}  "
                f"lr {lr:.2e}  {tok_per_sec:,.0f} tok/s  {dt:.1f}s",
                flush=True,
            )
            t0 = time.time()

        if train_cfg.eval_every > 0 and step > 0 and step % train_cfg.eval_every == 0:
            val_loss = evaluate(model, val_stream, train_cfg, ctx)
            improved = val_loss < best_val
            best_val = min(best_val, val_loss)
            print(f"  >> val loss {val_loss:.4f}  (best {best_val:.4f})"
                  f"{'  *' if improved else ''}", flush=True)

        if train_cfg.save_every > 0 and step > 0 and step % train_cfg.save_every == 0:
            save_checkpoint(model, optimizer, model_cfg, train_cfg, step, None)
            print(f"  >> saved checkpoint at step {step}", flush=True)

    # -- final ------------------------------------------------------------- #
    val_loss = evaluate(model, val_stream, train_cfg, ctx)
    print(f"\nfinal val loss: {val_loss:.4f} nats/token", flush=True)
    save_checkpoint(model, optimizer, model_cfg, train_cfg,
                    train_cfg.total_steps, val_loss)
    print(f"saved to {train_cfg.ckpt_dir}/", flush=True)


if __name__ == "__main__":
    main()