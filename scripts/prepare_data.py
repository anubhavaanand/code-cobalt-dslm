"""Download a corpus, write text shards, then tokenise into uint16 .bin files.

Phase 0 uses TinyStories: small, clean, and specifically designed so that tiny
models can learn coherent language. That makes it the right substrate for a
smoke test — if the pipeline is broken, the output will be obviously garbage.

Phase 1 will swap in The Stack / Stack-Edu (Python code). The rest of the
pipeline does not change.

    python scripts/prepare_data.py --preset smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow `python scripts/prepare_data.py` as well as `python -m scripts.prepare_data`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dslm.config import PRESETS
from dslm.tokenizer import Tokenizer, train_tokenizer

TINYSTORIES_REPO = "roneneldan/TinyStories"


def download_tinystories(out_dir: Path, max_docs: int) -> Path:
    """Fetch TinyStories and write one text file, one story per line.

    Reads parquet row-groups directly instead of going through `datasets`.
    Streaming the whole dataset holds too much in memory on a small machine
    and its teardown is unreliable; this path is flat and predictable.
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        sys.exit(
            "This script needs `huggingface_hub` and `pyarrow`.\n"
            "    pip install huggingface_hub pyarrow\n"
            "or point --text-file at your own corpus."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / "tinystories.txt"

    if text_path.exists():
        print(f"reusing {text_path}")
        return text_path

    print(f"listing {TINYSTORIES_REPO} ...")
    files = sorted(
        f
        for f in HfApi().list_repo_files(TINYSTORIES_REPO, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".parquet")
    )
    if not files:
        sys.exit(f"no parquet shards found in {TINYSTORIES_REPO}")

    print(f"downloading {len(files)} parquet shard(s)...")
    n = 0
    with text_path.open("w", encoding="utf-8") as f:
        for shard in files:
            local = hf_hub_download(TINYSTORIES_REPO, shard, repo_type="dataset")
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=2000):
                cols = batch.schema.names
                # `story` in older revisions, `text` in newer ones.
                col = "story" if "story" in cols else "text"
                for value in batch.column(col).to_pylist():
                    if not value:
                        continue
                    story = value.strip().replace("\n", " ")
                    if not story:
                        continue
                    f.write(story + "\n")
                    n += 1
                    if n >= max_docs:
                        break
                if n >= max_docs:
                    break
            print(f"  {n:,} stories so far...")
            if n >= max_docs:
                break

    print(f"wrote {n:,} stories to {text_path}")
    return text_path


def split_text(
    text_path: Path, out_dir: Path, val_frac: float = 0.01
) -> tuple[Path, Path]:
    """Carve out a validation split."""
    lines = text_path.read_text(encoding="utf-8", errors="replace").splitlines()
    n_val = max(1, int(len(lines) * val_frac))
    val_lines, train_lines = lines[:n_val], lines[n_val:]

    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    train_path.write_text("\n".join(train_lines), encoding="utf-8")
    val_path.write_text("\n".join(val_lines), encoding="utf-8")
    return train_path, val_path


def encode_shard(tok: Tokenizer, text_path: Path, out_path: Path) -> int:
    """Tokenise a text file into a flat uint16 binary."""
    ids: list[int] = []
    for line in text_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        ids.extend(tok.encode(line))
        ids.append(tok.eos_token_id)

    arr = np.array(ids, dtype=np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out_path)
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Phase 0 data")
    parser.add_argument("--preset", choices=list(PRESETS), default="smoke")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=60_000,
        help="stories to pull (60k is plenty for a smoke test)",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--text-file", default=None, help="skip download; use this text file instead"
    )
    args = parser.parse_args()

    model_cfg, train_cfg = PRESETS[args.preset]
    data_dir = Path(args.data_dir or train_cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. get text -------------------------------------------------------- #
    if args.text_file:
        text_path = Path(args.text_file)
    else:
        text_path = download_tinystories(data_dir, args.max_docs)

    train_txt, val_txt = split_text(text_path, data_dir)

    # -- 2. train tokenizer ------------------------------------------------- #
    tok_path = data_dir / "tokenizer.json"
    if tok_path.exists():
        print(f"reusing {tok_path}")
        tok = Tokenizer.from_file(tok_path)
    else:
        print(f"training BPE tokenizer (vocab={model_cfg.vocab_size})...")
        train_tokenizer([train_txt], tok_path, vocab_size=model_cfg.vocab_size)
        tok = Tokenizer.from_file(tok_path)

    if tok.vocab_size != model_cfg.vocab_size:
        print(
            f"  note: tokenizer has {tok.vocab_size} tokens, "
            f"preset wants {model_cfg.vocab_size}"
        )

    # -- 3. encode ---------------------------------------------------------- #
    print("encoding train split...")
    n_train = encode_shard(tok, train_txt, data_dir / train_cfg.token_file)
    print(f"  {n_train:,} tokens")

    print("encoding val split...")
    n_val = encode_shard(tok, val_txt, data_dir / train_cfg.val_token_file)
    print(f"  {n_val:,} tokens")

    # -- 4. report ---------------------------------------------------------- #
    from dslm.data import bytes_per_token

    bpt = bytes_per_token(train_txt, data_dir / train_cfg.token_file)
    print(f"\ndone. bytes/token = {bpt:.2f}  (healthy is 3.5-4.5 for text)")
    print(f"data in {data_dir}/")

    if n_train < model_cfg.block_size * 100:
        print(
            f"\nWARNING: only {n_train:,} train tokens. That is very little - "
            "the model will memorise rather than generalise. Fine for a smoke "
            "test, not for anything else."
        )


if __name__ == "__main__":
    main()