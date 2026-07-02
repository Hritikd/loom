"""Reproducible verification report: gradient precision + throughput.

    python benchmarks/run.py

Prints (1) the worst relative error between every hand-derived gradient and
its float64 central-difference numerical gradient, and (2) training/inference
throughput on this machine. No seeds are left implicit; rerunning reproduces
the same gradient numbers exactly.
"""

from __future__ import annotations

import time

import numpy as np

from loom.model import GPTConfig, backward, forward, generate, init_params, loss_and_grads
from loom.optim import AdamW, clip_grad_norm


def gradient_report():
    cfg = GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=8)
    params = init_params(cfg, seed=1, dtype=np.float64)
    rng = np.random.default_rng(42)
    ids = rng.integers(0, 11, size=(2, 6))
    targets = rng.integers(0, 11, size=(2, 6))

    _, _, cache, _ = forward(params, cfg, ids, targets)
    grads = backward(params, cfg, cache)

    def loss_fn():
        _, loss, _, _ = forward(params, cfg, ids, targets)
        return float(loss)

    check_rng = np.random.default_rng(7)
    worst_abs = 0.0
    worst_rel = 0.0  # over gradients large enough for relative error to mean anything
    checked = 0
    for _key, p in params.items():
        for fi in check_rng.choice(p.size, size=min(8, p.size), replace=False):
            i = np.unravel_index(fi, p.shape)
            orig = p[i]
            eps = 1e-5
            p[i] = orig + eps
            hi = loss_fn()
            p[i] = orig - eps
            lo = loss_fn()
            p[i] = orig
            num = (hi - lo) / (2 * eps)
            ana = grads[_key][i]
            worst_abs = max(worst_abs, abs(num - ana))
            if max(abs(num), abs(ana)) >= 1e-3:
                worst_rel = max(worst_rel, abs(num - ana) / max(abs(num), abs(ana)))
            checked += 1
    print(f"gradient check : {checked} coordinates across all {len(params)} tensors (float64)")
    print(f"                 worst absolute error vs numerical: {worst_abs:.2e}")
    print(f"                 worst relative error (|grad| >= 1e-3): {worst_rel:.2e}")


def throughput_report():
    cfg = GPTConfig(vocab_size=65, block_size=128, n_layer=3, n_head=6, n_embd=96)
    params = init_params(cfg, seed=0)
    opt = AdamW(params, lr=1e-3)
    rng = np.random.default_rng(0)

    # training throughput
    x = rng.integers(0, 65, size=(32, 128))
    y = rng.integers(0, 65, size=(32, 128))
    loss_and_grads(params, cfg, x, y)  # warmup
    t0 = time.time()
    steps = 10
    for _ in range(steps):
        _, grads = loss_and_grads(params, cfg, x, y)
        clip_grad_norm(grads, 1.0)
        opt.step(params, grads)
    dt = time.time() - t0
    print(f"training       : {steps * x.size / dt:,.0f} tokens/sec "
          f"(batch 32 x 128, {sum(a.size for a in params.values()):,} params)")

    # generation throughput (KV cache)
    t0 = time.time()
    n = 120
    generate(params, cfg, [1], n, temperature=1.0, seed=0)
    dt = time.time() - t0
    print(f"generation     : {n / dt:,.0f} tokens/sec (KV-cache incremental decode)")


if __name__ == "__main__":
    gradient_report()
    throughput_report()
