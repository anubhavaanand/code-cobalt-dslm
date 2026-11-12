"""Cobalt — a domain-specific language model for Python.

The point of this package is to be *readable*. Every design decision in the
project's VISION.md should be traceable to a line of code in here.

`GPT` is imported lazily so that `import dslm` works without PyTorch installed —
useful for reading the config or running the tokenizer on a machine that only
has `tokenizers` and `numpy`.
"""

from dslm.config import PRESETS, ModelConfig, TrainConfig

__all__ = ["ModelConfig", "TrainConfig", "PRESETS", "GPT"]


def __getattr__(name: str):
    # PEP 562: only pull in torch when someone actually asks for the model.
    if name == "GPT":
        from dslm.model import GPT

        return GPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")