"""Every hand-derived backward pass, checked against numerical gradients.

All checks run in float64 with central differences (eps=1e-5), so a correct
derivative shows relative error around 1e-8 or better. Any sign error, missing
term, or transposed matrix in a backward pass fails these loudly.
"""

import numpy as np
import pytest

from loom.layers import (
    attention_backward,
    attention_forward,
    causal_mask,
    cross_entropy_backward,
    cross_entropy_forward,
    gelu_backward,
    gelu_forward,
    layernorm_backward,
    layernorm_forward,
    linear_backward,
    linear_forward,
)
from loom.model import GPTConfig, backward, forward, init_params

RNG = np.random.default_rng(42)
TOL = 1e-6


def numerical_grad(f, x, eps=1e-5):
    """Central-difference gradient of scalar-valued f at x."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        orig = x[i]
        x[i] = orig + eps
        hi = f()
        x[i] = orig - eps
        lo = f()
        x[i] = orig
        g[i] = (hi - lo) / (2 * eps)
        it.iternext()
    return g


def rel_err(a, b):
    denom = max(np.abs(a).max(), np.abs(b).max(), 1e-12)
    return np.abs(a - b).max() / denom


def test_linear_grads():
    x = RNG.normal(size=(2, 5, 4))
    w = RNG.normal(size=(4, 7))
    b = RNG.normal(size=7)
    dout = RNG.normal(size=(2, 5, 7))

    def loss():
        return float((linear_forward(x, w, b)[0] * dout).sum())

    _, cache = linear_forward(x, w, b)
    dx, dw, db = linear_backward(dout, cache)
    assert rel_err(dx, numerical_grad(loss, x)) < TOL
    assert rel_err(dw, numerical_grad(loss, w)) < TOL
    assert rel_err(db, numerical_grad(loss, b)) < TOL


def test_layernorm_grads():
    x = RNG.normal(size=(3, 4, 8))
    g = RNG.normal(size=8)
    b = RNG.normal(size=8)
    dout = RNG.normal(size=(3, 4, 8))

    def loss():
        return float((layernorm_forward(x, g, b)[0] * dout).sum())

    _, cache = layernorm_forward(x, g, b)
    dx, dg, db = layernorm_backward(dout, cache)
    assert rel_err(dx, numerical_grad(loss, x)) < TOL
    assert rel_err(dg, numerical_grad(loss, g)) < TOL
    assert rel_err(db, numerical_grad(loss, b)) < TOL


def test_gelu_grad():
    x = RNG.normal(size=(4, 9)) * 2.0
    dout = RNG.normal(size=(4, 9))

    def loss():
        return float((gelu_forward(x)[0] * dout).sum())

    _, cache = gelu_forward(x)
    dx = gelu_backward(dout, cache)
    assert rel_err(dx, numerical_grad(loss, x)) < TOL


def test_attention_grads():
    b, h, t, dh = 2, 2, 5, 4
    q = RNG.normal(size=(b, h, t, dh))
    k = RNG.normal(size=(b, h, t, dh))
    v = RNG.normal(size=(b, h, t, dh))
    mask = causal_mask(t, dtype=np.float64)
    dout = RNG.normal(size=(b, h, t, dh))

    def loss():
        return float((attention_forward(q, k, v, mask)[0] * dout).sum())

    _, cache = attention_forward(q, k, v, mask)
    dq, dk, dv = attention_backward(dout, cache)
    assert rel_err(dq, numerical_grad(loss, q)) < TOL
    assert rel_err(dk, numerical_grad(loss, k)) < TOL
    assert rel_err(dv, numerical_grad(loss, v)) < TOL


def test_cross_entropy_grad():
    logits = RNG.normal(size=(2, 4, 9))
    targets = RNG.integers(0, 9, size=(2, 4))

    def loss():
        return float(cross_entropy_forward(logits, targets)[0])

    _, cache = cross_entropy_forward(logits, targets)
    d = cross_entropy_backward(cache)
    assert rel_err(d, numerical_grad(loss, logits)) < TOL


@pytest.mark.parametrize("key_sample", [None])
def test_full_model_grads(key_sample):
    """End-to-end: numerical check of d(loss)/d(param) through the whole GPT."""
    cfg = GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=8)
    params = init_params(cfg, seed=1, dtype=np.float64)
    ids = RNG.integers(0, 11, size=(2, 6))
    targets = RNG.integers(0, 11, size=(2, 6))

    _, loss, cache, _ = forward(params, cfg, ids, targets)
    grads = backward(params, cfg, cache)

    def loss_fn():
        _, x, _, _ = forward(params, cfg, ids, targets)
        return float(x)

    check_rng = np.random.default_rng(7)
    worst = 0.0
    for key, p in params.items():
        analytic = grads[key]
        # sample a handful of coordinates per tensor; full numerical grads
        # over ~30k parameters would be slow for no extra signal.
        flat_idx = check_rng.choice(p.size, size=min(4, p.size), replace=False)
        for fi in flat_idx:
            i = np.unravel_index(fi, p.shape)
            orig = p[i]
            eps = 1e-5
            p[i] = orig + eps
            hi = loss_fn()
            p[i] = orig - eps
            lo = loss_fn()
            p[i] = orig
            num = (hi - lo) / (2 * eps)
            scale = max(abs(num), abs(analytic[i]), 1e-8)
            worst = max(worst, abs(num - analytic[i]) / scale)
    assert worst < 1e-5, f"worst relative gradient error {worst:.2e}"
