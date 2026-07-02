"""The layers of a transformer, as forward/backward function pairs.

Every function here is pure: `*_forward` returns (output, cache) and the
matching `*_backward` consumes (upstream gradient, cache) and returns the
gradients of every input. No autograd, no framework — each backward pass is
the hand-derived derivative of its forward pass, and tests/test_gradients.py
checks every one of them against central-difference numerical gradients.

Shape conventions: batch B, sequence T, model width D, heads H, head width
Dh = D // H, vocabulary V.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# linear
# ---------------------------------------------------------------------------


def linear_forward(x, w, b):
    """y = x @ w + b.  x: (..., Din), w: (Din, Dout), b: (Dout,)."""
    return x @ w + b, (x, w)


def linear_backward(dout, cache):
    x, w = cache
    din, dout_dim = w.shape
    dx = dout @ w.T
    dw = x.reshape(-1, din).T @ dout.reshape(-1, dout_dim)
    db = dout.reshape(-1, dout_dim).sum(axis=0)
    return dx, dw, db


# ---------------------------------------------------------------------------
# layernorm
# ---------------------------------------------------------------------------


def layernorm_forward(x, g, b, eps: float = 1e-5):
    """Normalize the last axis to zero mean / unit variance, then scale+shift."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)


def layernorm_backward(dout, cache):
    # Derivation: with xhat = (x - mu) * rstd, the Jacobian of layernorm
    # couples every element of the row through mu and var. Collapsing the
    # chain rule gives the standard three-term form below.
    xhat, rstd, g = cache
    n = xhat.shape[-1]
    dxhat = dout * g
    dg = (dout * xhat).reshape(-1, n).sum(axis=0)
    db = dout.reshape(-1, n).sum(axis=0)
    dx = rstd * (
        dxhat
        - dxhat.mean(axis=-1, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)
    )
    return dx, dg, db


# ---------------------------------------------------------------------------
# GELU (tanh approximation, as used by GPT-2)
# ---------------------------------------------------------------------------

_GELU_C = math.sqrt(2.0 / math.pi)
_GELU_A = 0.044715


def gelu_forward(x):
    u = _GELU_C * (x + _GELU_A * x**3)
    t = np.tanh(u)
    return 0.5 * x * (1.0 + t), (x, t)


def gelu_backward(dout, cache):
    x, t = cache
    du_dx = _GELU_C * (1.0 + 3.0 * _GELU_A * x**2)
    dx = dout * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du_dx)
    return dx


# ---------------------------------------------------------------------------
# softmax + scaled dot-product attention
# ---------------------------------------------------------------------------


def softmax(x, axis: int = -1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention_forward(q, k, v, mask):
    """Scaled dot-product attention over heads.

    q, k, v: (B, H, T, Dh).  mask: additive, broadcastable to (T, T) scores —
    0 where attention is allowed, a large negative number where it is not.
    Returns (output (B, H, T, Dh), cache); the cache keeps the post-softmax
    attention weights, which is also what the visualizer renders.
    """
    dh = q.shape[-1]
    scale = 1.0 / math.sqrt(dh)
    s = q @ np.swapaxes(k, -1, -2) * scale
    s = s + mask
    p = softmax(s, axis=-1)
    o = p @ v
    return o, (q, k, v, p, scale)


def attention_backward(dout, cache):
    # Softmax backward: for p = softmax(s) along the last axis,
    # ds = p * (dp - sum(dp * p)). Masked positions have p == 0, so their
    # gradient is exactly 0 and the mask needs no special handling here.
    q, k, v, p, scale = cache
    dp = dout @ np.swapaxes(v, -1, -2)
    dv = np.swapaxes(p, -1, -2) @ dout
    ds = p * (dp - (dp * p).sum(axis=-1, keepdims=True))
    dq = ds @ k * scale
    dk = np.swapaxes(ds, -1, -2) @ q * scale
    return dq, dk, dv


def causal_mask(t: int, dtype=np.float32):
    """(T, T) additive mask: position i may attend to positions <= i."""
    return np.triu(np.full((t, t), -1e9, dtype=dtype), k=1)


# ---------------------------------------------------------------------------
# cross-entropy from logits
# ---------------------------------------------------------------------------


def cross_entropy_forward(logits, targets):
    """Mean cross-entropy over all positions. logits: (B, T, V), targets: (B, T)."""
    v = logits.shape[-1]
    z = logits - logits.max(axis=-1, keepdims=True)
    logsum = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    logp = z - logsum
    flat = logp.reshape(-1, v)
    idx = targets.reshape(-1)
    loss = -flat[np.arange(idx.size), idx].mean()
    return loss, (np.exp(logp), targets)


def cross_entropy_backward(cache):
    """d loss / d logits — the famous (softmax - onehot) / N."""
    probs, targets = cache
    v = probs.shape[-1]
    n = targets.size
    d = probs.copy().reshape(-1, v)
    d[np.arange(n), targets.reshape(-1)] -= 1.0
    d /= n
    return d.reshape(probs.shape)
