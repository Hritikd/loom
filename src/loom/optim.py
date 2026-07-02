"""AdamW and the learning-rate schedule, from the papers, in NumPy."""

from __future__ import annotations

import math

import numpy as np


class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Decay applies only to weight matrices (ndim >= 2, excluding embeddings) —
    the same policy GPT-2 uses: biases, layernorm gains, and embeddings are
    left undecayed.
    """

    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self._decay = {k for k, a in params.items() if a.ndim >= 2 and not k.endswith("_emb")}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        bc1 = 1.0 - self.b1**self.t
        bc2 = 1.0 - self.b2**self.t
        for k, g in grads.items():
            m = self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            v = self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * g * g
            update = (m / bc1) / (np.sqrt(v / bc2) + self.eps)
            if k in self._decay:
                update = update + self.weight_decay * params[k]
            params[k] -= lr * update


def clip_grad_norm(grads: dict[str, np.ndarray], max_norm: float = 1.0) -> float:
    """Scale all gradients so their global L2 norm is at most max_norm."""
    total = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for g in grads.values():
            g *= scale
    return total


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup: int = 100,
              min_ratio: float = 0.1) -> float:
    """Linear warmup, then cosine decay to min_ratio * base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
