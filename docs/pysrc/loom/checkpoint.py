"""Checkpoints: one .npz holds the weights, the config, and the vocabulary.

A checkpoint is fully self-contained — load it and you can generate, no other
files needed. Weights can be stored float16 (`half=True`) to keep shipped
demo models small; they are always loaded back as float32.
"""

from __future__ import annotations

import json

import numpy as np

from .model import GPTConfig
from .tokenizer import CharTokenizer


def save(path, params: dict, cfg: GPTConfig, tokenizer: CharTokenizer, half: bool = False):
    arrays = {f"p.{k}": (v.astype(np.float16) if half else v) for k, v in params.items()}
    meta = json.dumps({"config": cfg.to_dict(), "vocab": tokenizer.chars})
    np.savez_compressed(path, meta=np.frombuffer(meta.encode(), dtype=np.uint8), **arrays)


def load(path) -> tuple[dict, GPTConfig, CharTokenizer]:
    with np.load(path) as z:
        meta = json.loads(bytes(z["meta"]).decode())
        params = {
            k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("p.")
        }
    cfg = GPTConfig(**meta["config"])
    tok = CharTokenizer(meta["vocab"])
    return params, cfg, tok
