"""Data loading.

Tokens live in flat uint16 binary files (one .bin per split), which is how
nanoGPT does it and for a good reason: memory-mapped random access with no
per-batch tokenisation cost and no dataloader worker processes.

Two uint16 tokens per 32-bit word means a 2B-token corpus is ~4 GB on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenStream:
    """Random contiguous windows out of a memory-mapped token file."""

    def __init__(self, path: str | Path, block_size: int, device: str = "cpu"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/prepare_data.py first."
            )
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.device = device

        if len(self.tokens) < block_size + 2:
            raise ValueError(
                f"{path} has only {len(self.tokens)} tokens; need at least "
                f"{block_size + 2}. Prepare more data."
            )

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size - 1)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def get_batch(self, batch_size: int, generator: torch.Generator | None = None):
        """Return (inputs, targets) of shape (batch_size, block_size)."""
        ix = torch.randint(0, len(self), (batch_size,), generator=generator)
        x = np.stack([self.tokens[i : i + self.block_size] for i in ix])
        y = np.stack([self.tokens[i + 1 : i + 1 + self.block_size] for i in ix])
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        return x.to(self.device), y.to(self.device)


def bytes_per_token(path: str | Path) -> float:
    """Diagnostic: compression rate of the tokeniser on this corpus.

    Values around 4.0+ are healthy for BPE on English text. Code tends to sit
    lower (3.2-3.8) because identifiers and punctuation fragment.
    """
    path = Path(path)
    n_tokens = len(np.memmap(path, dtype=np.uint16, mode="r"))
    n_bytes = path.stat().st_size
    return n_bytes / max(1, n_tokens)