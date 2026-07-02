"""A GPT — decoder-only transformer — in pure NumPy.

Parameters live in a flat dict of arrays (key -> ndarray), which keeps the
optimizer, checkpointing, and gradient checking trivial. `forward` returns
logits plus a cache of intermediates; `backward` walks the cache in reverse
and returns a gradient dict with exactly the same keys as the parameters.

Architecture (GPT-2 lineage): learned token + position embeddings, pre-norm
blocks of [LayerNorm -> causal multi-head self-attention -> residual,
LayerNorm -> GELU MLP -> residual], final LayerNorm, linear head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .layers import (
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
    softmax,
)


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 3
    n_head: int = 6
    n_embd: int = 96

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

    def to_dict(self) -> dict:
        return asdict(self)


def init_params(cfg: GPTConfig, seed: int = 0, dtype=np.float32) -> dict[str, np.ndarray]:
    """GPT-2-style init: N(0, 0.02), residual projections scaled by 1/sqrt(2L)."""
    rng = np.random.default_rng(seed)
    std = 0.02
    resid_std = std / np.sqrt(2 * cfg.n_layer)
    d, v = cfg.n_embd, cfg.vocab_size

    def n(shape, s=std):
        return rng.normal(0.0, s, shape).astype(dtype)

    p: dict[str, np.ndarray] = {
        "tok_emb": n((v, d)),
        "pos_emb": n((cfg.block_size, d)),
        "ln_f.g": np.ones(d, dtype=dtype),
        "ln_f.b": np.zeros(d, dtype=dtype),
        "head.w": n((d, v)),
    }
    for i in range(cfg.n_layer):
        p[f"h{i}.ln1.g"] = np.ones(d, dtype=dtype)
        p[f"h{i}.ln1.b"] = np.zeros(d, dtype=dtype)
        p[f"h{i}.attn.w_qkv"] = n((d, 3 * d))
        p[f"h{i}.attn.b_qkv"] = np.zeros(3 * d, dtype=dtype)
        p[f"h{i}.attn.w_proj"] = n((d, d), resid_std)
        p[f"h{i}.attn.b_proj"] = np.zeros(d, dtype=dtype)
        p[f"h{i}.ln2.g"] = np.ones(d, dtype=dtype)
        p[f"h{i}.ln2.b"] = np.zeros(d, dtype=dtype)
        p[f"h{i}.mlp.w_fc"] = n((d, 4 * d))
        p[f"h{i}.mlp.b_fc"] = np.zeros(4 * d, dtype=dtype)
        p[f"h{i}.mlp.w_proj"] = n((4 * d, d), resid_std)
        p[f"h{i}.mlp.b_proj"] = np.zeros(d, dtype=dtype)
    return p


def param_count(params: dict[str, np.ndarray]) -> int:
    return sum(a.size for a in params.values())


def _split_heads(x, n_head):
    b, t, d = x.shape
    return x.reshape(b, t, n_head, d // n_head).transpose(0, 2, 1, 3)


def _merge_heads(x):
    b, h, t, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, t, h * dh)


def forward(params, cfg: GPTConfig, ids, targets=None, return_attn: bool = False):
    """Run the model.

    ids: (B, T) int array, T <= cfg.block_size.
    Returns (logits, loss, cache, attn):
      loss is None unless targets given; attn is None unless return_attn —
      otherwise a list of (B, H, T, T) post-softmax attention maps per layer.
    """
    b, t = ids.shape
    if t > cfg.block_size:
        raise ValueError(f"sequence length {t} exceeds block_size {cfg.block_size}")
    dtype = params["tok_emb"].dtype

    x = params["tok_emb"][ids] + params["pos_emb"][:t]
    mask = causal_mask(t, dtype=dtype)
    blocks = []
    attns = [] if return_attn else None

    for i in range(cfg.n_layer):
        pre = f"h{i}"
        a, ln1_c = layernorm_forward(x, params[f"{pre}.ln1.g"], params[f"{pre}.ln1.b"])
        qkv, qkv_c = linear_forward(a, params[f"{pre}.attn.w_qkv"], params[f"{pre}.attn.b_qkv"])
        q, k, v = np.split(qkv, 3, axis=-1)
        q, k, v = (_split_heads(z, cfg.n_head) for z in (q, k, v))
        o, att_c = attention_forward(q, k, v, mask)
        if return_attn:
            attns.append(att_c[3])  # post-softmax weights
        merged = _merge_heads(o)
        proj, proj_c = linear_forward(
            merged, params[f"{pre}.attn.w_proj"], params[f"{pre}.attn.b_proj"]
        )
        x = x + proj

        m, ln2_c = layernorm_forward(x, params[f"{pre}.ln2.g"], params[f"{pre}.ln2.b"])
        fc, fc_c = linear_forward(m, params[f"{pre}.mlp.w_fc"], params[f"{pre}.mlp.b_fc"])
        g, g_c = gelu_forward(fc)
        mlp, mlp_c = linear_forward(g, params[f"{pre}.mlp.w_proj"], params[f"{pre}.mlp.b_proj"])
        x = x + mlp

        blocks.append((ln1_c, qkv_c, att_c, proj_c, ln2_c, fc_c, g_c, mlp_c))

    xf, lnf_c = layernorm_forward(x, params["ln_f.g"], params["ln_f.b"])
    logits = xf @ params["head.w"]

    loss, ce_c = (None, None)
    if targets is not None:
        loss, ce_c = cross_entropy_forward(logits, targets)

    cache = (ids, xf, lnf_c, blocks, ce_c)
    return logits, loss, cache, attns


def backward(params, cfg: GPTConfig, cache, dlogits=None):
    """Gradient of the loss (or of `dlogits`) w.r.t. every parameter."""
    ids, xf, lnf_c, blocks, ce_c = cache
    if dlogits is None:
        if ce_c is None:
            raise ValueError("no targets were given to forward(); pass dlogits explicitly")
        dlogits = cross_entropy_backward(ce_c)

    grads: dict[str, np.ndarray] = {}
    d = params["head.w"].shape[0]

    grads["head.w"] = xf.reshape(-1, d).T @ dlogits.reshape(-1, dlogits.shape[-1])
    dxf = dlogits @ params["head.w"].T
    dx, grads["ln_f.g"], grads["ln_f.b"] = layernorm_backward(dxf, lnf_c)

    for i in reversed(range(cfg.n_layer)):
        pre = f"h{i}"
        ln1_c, qkv_c, att_c, proj_c, ln2_c, fc_c, g_c, mlp_c = blocks[i]

        # MLP branch: x_out = x_mid + mlp(ln2(x_mid))
        dg_out, grads[f"{pre}.mlp.w_proj"], grads[f"{pre}.mlp.b_proj"] = linear_backward(
            dx, mlp_c
        )
        dfc = gelu_backward(dg_out, g_c)
        dm, grads[f"{pre}.mlp.w_fc"], grads[f"{pre}.mlp.b_fc"] = linear_backward(dfc, fc_c)
        dres, grads[f"{pre}.ln2.g"], grads[f"{pre}.ln2.b"] = layernorm_backward(dm, ln2_c)
        dx = dx + dres

        # Attention branch: x_mid = x_in + proj(attn(ln1(x_in)))
        dmerged, grads[f"{pre}.attn.w_proj"], grads[f"{pre}.attn.b_proj"] = linear_backward(
            dx, proj_c
        )
        do = _split_heads(dmerged, cfg.n_head)
        dq, dk, dv = attention_backward(do, att_c)
        dqkv = np.concatenate([_merge_heads(z) for z in (dq, dk, dv)], axis=-1)
        da, grads[f"{pre}.attn.w_qkv"], grads[f"{pre}.attn.b_qkv"] = linear_backward(
            dqkv, qkv_c
        )
        dres, grads[f"{pre}.ln1.g"], grads[f"{pre}.ln1.b"] = layernorm_backward(da, ln1_c)
        dx = dx + dres

    t = ids.shape[1]
    grads["tok_emb"] = np.zeros_like(params["tok_emb"])
    np.add.at(grads["tok_emb"], ids, dx)
    grads["pos_emb"] = np.zeros_like(params["pos_emb"])
    grads["pos_emb"][:t] = dx.sum(axis=0)
    return grads


def loss_and_grads(params, cfg: GPTConfig, ids, targets):
    logits, loss, cache, _ = forward(params, cfg, ids, targets)
    return loss, backward(params, cfg, cache)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def _sample_from_logits(logits, temperature, top_k, rng):
    if temperature <= 0.0:  # greedy
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k is not None and top_k < logits.size:
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    probs = softmax(logits.astype(np.float64))
    return int(rng.choice(probs.size, p=probs / probs.sum()))


def generate_stream(
    params,
    cfg: GPTConfig,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    use_cache: bool = True,
):
    """Yield new token ids one at a time.

    While the sequence fits in block_size, generation runs incrementally with
    a per-layer KV cache (each step feeds ONE token and attends to cached
    keys/values). Once the context outgrows block_size, it falls back to
    re-encoding the last block_size tokens per step — correct, just slower.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token")
    rng = np.random.default_rng(seed)
    ids = list(prompt_ids)

    kv: list[tuple[np.ndarray, np.ndarray]] | None = None
    next_pos = 0
    if use_cache and len(ids) <= cfg.block_size:
        logits_row, kv = _prefill(params, cfg, np.array([ids]))
        next_pos = len(ids)
    else:
        window = np.array([ids[-cfg.block_size :]])
        logits, _, _, _ = forward(params, cfg, window)
        logits_row = logits[0, -1]

    for _ in range(max_new_tokens):
        tok = _sample_from_logits(logits_row, temperature, top_k, rng)
        ids.append(tok)
        yield tok
        if kv is not None and next_pos < cfg.block_size:
            logits_row, kv = _decode_step(params, cfg, tok, next_pos, kv)
            next_pos += 1
        else:
            kv = None  # context is full: windowed recompute from here on
            window = np.array([ids[-cfg.block_size :]])
            logits, _, _, _ = forward(params, cfg, window)
            logits_row = logits[0, -1]


def generate(params, cfg, prompt_ids, max_new_tokens, **kw) -> list[int]:
    return list(generate_stream(params, cfg, prompt_ids, max_new_tokens, **kw))


def _attn_block(params, cfg, x, i, k_all, v_all, mask):
    """One block given precomputed full k/v (used by both cache paths)."""
    pre = f"h{i}"
    a, _ = layernorm_forward(x, params[f"{pre}.ln1.g"], params[f"{pre}.ln1.b"])
    qkv, _ = linear_forward(a, params[f"{pre}.attn.w_qkv"], params[f"{pre}.attn.b_qkv"])
    q, k, v = np.split(qkv, 3, axis=-1)
    q = _split_heads(q, cfg.n_head)
    k_new = _split_heads(k, cfg.n_head)
    v_new = _split_heads(v, cfg.n_head)
    k_all = k_new if k_all is None else np.concatenate([k_all, k_new], axis=2)
    v_all = v_new if v_all is None else np.concatenate([v_all, v_new], axis=2)
    o, _ = attention_forward(q, k_all, v_all, mask)
    proj, _ = linear_forward(
        _merge_heads(o), params[f"{pre}.attn.w_proj"], params[f"{pre}.attn.b_proj"]
    )
    x = x + proj
    m, _ = layernorm_forward(x, params[f"{pre}.ln2.g"], params[f"{pre}.ln2.b"])
    fc, _ = linear_forward(m, params[f"{pre}.mlp.w_fc"], params[f"{pre}.mlp.b_fc"])
    g, _ = gelu_forward(fc)
    mlp, _ = linear_forward(g, params[f"{pre}.mlp.w_proj"], params[f"{pre}.mlp.b_proj"])
    return x + mlp, k_all, v_all


def _prefill(params, cfg: GPTConfig, ids):
    """Encode the whole prompt once, returning last-position logits + KV cache."""
    t = ids.shape[1]
    dtype = params["tok_emb"].dtype
    x = params["tok_emb"][ids] + params["pos_emb"][:t]
    mask = causal_mask(t, dtype=dtype)
    kv = []
    for i in range(cfg.n_layer):
        x, k_all, v_all = _attn_block(params, cfg, x, i, None, None, mask)
        kv.append((k_all, v_all))
    xf, _ = layernorm_forward(x, params["ln_f.g"], params["ln_f.b"])
    return (xf @ params["head.w"])[0, -1], kv


def _decode_step(params, cfg: GPTConfig, tok: int, pos: int, kv):
    """Feed ONE token at position `pos`, attending to all cached keys/values."""
    x = params["tok_emb"][np.array([[tok]])] + params["pos_emb"][pos]
    new_kv = []
    for i in range(cfg.n_layer):
        k_all, v_all = kv[i]
        # A single query attends to every cached position: no mask needed.
        x, k_all, v_all = _attn_block(params, cfg, x, i, k_all, v_all, 0.0)
        new_kv.append((k_all, v_all))
    xf, _ = layernorm_forward(x, params["ln_f.g"], params["ln_f.b"])
    return (xf @ params["head.w"])[0, -1], new_kv
