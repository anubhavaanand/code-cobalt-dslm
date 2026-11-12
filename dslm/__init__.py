"""DSLM — a small, domain-specialised language model for Python.

The point of this package is to be *readable*. Every design decision in
`VISION.md` should be traceable to a line of code in here.
"""

from dslm.config import ModelConfig, TrainConfig, PRESETS
from dslm.model import GPT

__all__ = ["ModelConfig", "TrainConfig", "PRESETS", "GPT"]