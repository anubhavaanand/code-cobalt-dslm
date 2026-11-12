"""Byte-level BPE tokenizer.

We train our own rather than borrowing GPT-2's, because:
  - vocab size is a real architecture decision (it sets the embedding cost)
  - it is one of the parts of the pipeline worth understanding firsthand
  - for a code-specialised model we may later want indent-aware tokens

Byte-level BPE guarantees no OOV: every byte is representable, so any input
round-trips. That property is non-negotiable for code.

Note the aliasing: `tokenizers.Tokenizer` is imported as `HFTokenizer` so our
own `Tokenizer` wrapper can keep the short name.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, processors, trainers

SPECIAL_TOKENS = ["<|endoftext|>"]


def train_tokenizer(
    input_files: list[str | Path],
    save_path: str | Path,
    vocab_size: int = 8192,
    min_frequency: int = 2,
) -> HFTokenizer:
    tok = HFTokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train([str(p) for p in input_files], trainer)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(save_path))
    return tok


class Tokenizer:
    """Thin wrapper so callers do not depend on `tokenizers` directly."""

    def __init__(self, tok: HFTokenizer):
        self._tok = tok

    @classmethod
    def from_file(cls, path: str | Path) -> "Tokenizer":
        return cls(HFTokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def eos_token_id(self) -> int:
        return self._tok.token_to_id("<|endoftext|>")

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def encode_file(self, path: str | Path) -> list[int]:
        """Encode a text file, appending EOS so documents are separable."""
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        ids = self.encode(text)
        return ids + [self.eos_token_id]