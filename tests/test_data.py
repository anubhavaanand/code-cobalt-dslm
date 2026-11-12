"""Data tests.

The bug this guards against: `bytes_per_token` divided the *binary* file size by
the token count. Since tokens are stored as uint16, that is always exactly 2.0 —
a plausible-looking number that silently tells you nothing.

The fixture tokenises real prose rather than inventing a token count, so the
ratio it produces is genuine and clearly distinguishable from the 2.0 the bug
would give.
"""

import numpy as np
import pytest

from dslm.data import TokenStream, bytes_per_token
from dslm.tokenizer import Tokenizer, train_tokenizer

WORDS = (
    "the quick brown fox jumps over a lazy dog while little girl walked to "
    "school with her friends and told stories about dragons and knights in "
    "far away lands beyond the forest near the river where boats sail"
).split()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")

    lines = [" ".join(WORDS[i % len(WORDS):][:8]) for i in range(2000)]
    text = "\n".join(lines) + "\n"

    text_path = d / "train.txt"
    text_path.write_text(text)

    # Tokenise for real. A synthetic token count would make the ratio
    # meaningless and could accidentally land near 2.0.
    tok_path = d / "tok.json"
    train_tokenizer([text_path], tok_path, vocab_size=1024)
    tok = Tokenizer.from_file(tok_path)
    ids = tok.encode(text)

    token_path = d / "train.bin"
    np.array(ids, dtype=np.uint16).tofile(token_path)

    return text_path, token_path, len(ids)


def test_bytes_per_token_is_not_the_binary_size(corpus):
    """Regression: must measure against source text, not the uint16 file.

    The uint16 file is 2 bytes/token by construction. If this returns ~2.0, the
    diagnostic is measuring the wrong thing.
    """
    text_path, token_path, _ = corpus
    bpt = bytes_per_token(text_path, token_path)

    assert abs(bpt - 2.0) > 0.2, (
        f"bytes/token = {bpt:.2f}, which is the uint16 binary size. "
        f"It must be measured against the source text."
    )
    assert bpt > 3.0, (
        f"bytes/token = {bpt:.2f} is implausibly low for English prose "
        f"(expect 3.5-5.0)."
    )


def test_bytes_per_token_matches_hand_computation(corpus):
    text_path, token_path, n_tokens = corpus
    expected = text_path.stat().st_size / n_tokens
    assert bytes_per_token(text_path, token_path) == pytest.approx(expected)


def test_token_stream_len(corpus):
    _, token_path, n_tokens = corpus
    stream = TokenStream(token_path, block_size=64)
    assert len(stream) == n_tokens - 64 - 1
    assert stream.n_tokens == n_tokens


def test_token_stream_batch_shapes(corpus):
    _, token_path, _ = corpus
    stream = TokenStream(token_path, block_size=64)
    x, y = stream.get_batch(4)

    assert x.shape == (4, 64)
    assert y.shape == (4, 64)
    assert x.dtype == y.dtype


def test_token_stream_targets_are_shifted_by_one(corpus):
    """y must be x shifted by one position — the definition of next-token prediction.

    x = t[i : i+B], y = t[i+1 : i+B+1], so y[j] == x[j+1].
    """
    _, token_path, _ = corpus
    stream = TokenStream(token_path, block_size=64)
    x, y = stream.get_batch(1)

    try:
        import torch

        assert torch.equal(x[0, 1:], y[0, :-1])
    except ImportError:
        assert np.array_equal(x[0, 1:], y[0, :-1])


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenStream(tmp_path / "nope.bin", block_size=64)