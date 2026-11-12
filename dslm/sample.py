"""Generate text from a checkpoint.

    python -m dslm.sample --ckpt checkpoints/ckpt_000600.pt --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dslm.config import ModelConfig
from dslm.model import GPT
from dslm.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a DSLM checkpoint")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if args.device == "auto" else args.device
    )

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])

    tok = Tokenizer.from_file(args.tokenizer)
    if tok.vocab_size != cfg.vocab_size:
        raise ValueError(
            f"tokenizer vocab ({tok.vocab_size}) != model vocab ({cfg.vocab_size}); "
            "retrain the tokenizer with the matching size."
        )

    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = tok.encode(args.prompt) or [tok.eos_token_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tok.eos_token_id,
    )

    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()