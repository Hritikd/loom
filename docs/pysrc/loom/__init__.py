"""loom — a GPT from scratch in NumPy, every gradient derived by hand."""

from .checkpoint import load, save
from .model import (
    GPTConfig,
    backward,
    forward,
    generate,
    generate_stream,
    init_params,
    loss_and_grads,
    param_count,
)
from .optim import AdamW, clip_grad_norm, cosine_lr
from .tokenizer import CharTokenizer

__version__ = "0.1.0"

__all__ = [
    "GPTConfig",
    "CharTokenizer",
    "AdamW",
    "forward",
    "backward",
    "loss_and_grads",
    "init_params",
    "param_count",
    "generate",
    "generate_stream",
    "clip_grad_norm",
    "cosine_lr",
    "save",
    "load",
    "__version__",
]
