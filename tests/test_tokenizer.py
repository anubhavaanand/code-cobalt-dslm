"""Tokenizer tests.

The bug this guards against: our `Tokenizer` wrapper class shadowed the
`tokenizers.Tokenizer` import, so `train_tokenizer` tried to call `.train()` on
the wrapper and crashed with a confusing AttributeError. A round-trip test catches
that immediately; a smoke test does not.
"""

import numpy as np
import pytest

from dslm.tokenizer import Tokenizer, train_tokenizer

SAMPLES = [
    "Once upon a time",
    "def add(a, b):\n    return a + b",
    "def add(a, b):\n\treturn a + b",  # tabs must survive
    "x = [i**2 for i in range(10)]",
    "def f(x):\n    'docstring'\n    return x",
    "héllo wörld — em dash and accents",
]


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    """Train a small tokenizer on synthetic text."""
    d = tmp_path_factory.mktemp("tok")
    text = d / "corpus.txt"
    words = ("the quick brown fox jumps over lazy dog once upon a time there was "
             "a little girl who lived in village near forest and every day she "
             "would walk to school with her friends").split()
    lines = [" ".join(np.random.choice(words, size=12)) for _ in range(2000)]
    text.write_text("\n".join(lines))

    tok = train_tokenizer([text], d / "tok.json", vocab_size=512)
    return tok, Tokenizer.from_file(d / "tok.json")


def test_train_returns_hf_tokenizer_not_wrapper(tokenizer):
    """Regression: the wrapper must not shadow the underlying tokenizer."""
    hf_tok, wrapped = tokenizer
    assert hasattr(hf_tok, "train"), "train_tokenizer returned the wrong type"
    assert not isinstance(hf_tok, wrapped.__class__)


def test_vocab_size(tokenizer):
    _, wrapped = tokenizer
    assert wrapped.vocab_size > 256


def test_eos_exists(tokenizer):
    _, wrapped = tokenizer
    assert wrapped.eos_token_id is not None
    assert wrapped.eos_token_id >= 0


@pytest.mark.parametrize("text", SAMPLES)
def test_roundtrip(tokenizer, text):
    """Byte-level BPE must round-trip exactly, including tabs and accents."""
    _, wrapped = tokenizer
    ids = wrapped.encode(text)
    assert wrapped.decode(ids) == text


def test_no_unrepresentable_bytes(tokenizer):
    """Every byte value must be encodable — that is the point of byte-level BPE."""
    _, wrapped = tokenizer
    for code in range(256):
        raw = bytes([code])
        # decode with errors='replace' so a broken tokenizer fails loudly
        ids = wrapped.encode(raw.decode("latin-1"))
        assert len(ids) > 0, f"byte {code} produced no tokens"


def test_encode_file_appends_eos(tokenizer, tmp_path):
    _, wrapped = tokenizer
    p = tmp_path / "doc.txt"
    p.write_text("hello world")
    ids = wrapped.encode_file(p)
    assert ids[-1] == wrapped.eos_token_id


def test_tokens_fit_in_uint16(tokenizer):
    """Token IDs must fit in uint16 — data.py memory-maps them as such."""
    _, wrapped = tokenizer
    assert wrapped.vocab_size <= 65535, (
        f"vocab {wrapped.vocab_size} exceeds uint16 range; data.py would corrupt"
    )