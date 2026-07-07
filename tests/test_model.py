"""Model behavior: causality, KV-cache equivalence, sampling, training."""

import numpy as np

from loom.model import (
    GPTConfig,
    forward,
    generate,
    init_params,
    loss_and_grads,
    param_count,
)
from loom.optim import AdamW, clip_grad_norm

CFG = GPTConfig(vocab_size=13, block_size=16, n_layer=2, n_head=2, n_embd=16)


def test_shapes_and_param_count():
    params = init_params(CFG, seed=0)
    ids = np.zeros((3, 10), dtype=np.int64)
    logits, loss, _, attns = forward(params, CFG, ids, return_attn=True)
    assert logits.shape == (3, 10, 13)
    assert loss is None
    assert len(attns) == CFG.n_layer and attns[0].shape == (3, 2, 10, 10)
    d, v, blk, ff = CFG.n_embd, CFG.vocab_size, CFG.block_size, 4 * CFG.n_embd
    per_block = 2 * (2 * d) + (d * 3 * d + 3 * d) + (d * d + d) + (d * ff + ff) + (ff * d + d)
    expected = v * d + blk * d + CFG.n_layer * per_block + 2 * d + d * v
    assert param_count(params) == expected


def test_causality():
    """Changing a future token must not change logits at earlier positions."""
    params = init_params(CFG, seed=0)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 13, size=(1, 12))
    logits_a, _, _, _ = forward(params, CFG, ids)
    ids2 = ids.copy()
    ids2[0, -1] = (ids2[0, -1] + 1) % 13
    logits_b, _, _, _ = forward(params, CFG, ids2)
    np.testing.assert_allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-6)
    assert not np.allclose(logits_a[0, -1], logits_b[0, -1])


def test_kv_cache_matches_full_forward():
    """Greedy generation with the KV cache == greedy without it."""
    params = init_params(CFG, seed=3)
    prompt = [1, 2, 3]
    with_cache = generate(params, CFG, prompt, 10, temperature=0.0, use_cache=True)
    without = generate(params, CFG, prompt, 10, temperature=0.0, use_cache=False)
    assert with_cache == without


def test_generation_past_block_size():
    """Generation keeps going correctly once context exceeds block_size."""
    params = init_params(CFG, seed=3)
    out = generate(params, CFG, [1, 2], 30, temperature=0.0)  # 32 > block 16
    assert len(out) == 30
    assert all(0 <= t < 13 for t in out)


def test_sampling_determinism_and_top_k():
    params = init_params(CFG, seed=4)
    a = generate(params, CFG, [5], 15, temperature=1.0, top_k=3, seed=11)
    b = generate(params, CFG, [5], 15, temperature=1.0, top_k=3, seed=11)
    c = generate(params, CFG, [5], 15, temperature=1.0, top_k=3, seed=12)
    assert a == b
    assert a != c  # overwhelmingly likely over 15 draws from top-3


def test_top_k_restricts_support():
    params = init_params(CFG, seed=5)
    # top_k=1 must equal greedy regardless of temperature
    greedy = generate(params, CFG, [2, 7], 12, temperature=0.0)
    topk1 = generate(params, CFG, [2, 7], 12, temperature=5.0, top_k=1, seed=0)
    assert greedy == topk1


def test_top_p_tiny_collapses_to_greedy():
    """A vanishing nucleus keeps only the single most-probable token."""
    params = init_params(CFG, seed=5)
    greedy = generate(params, CFG, [2, 7], 12, temperature=0.0)
    nucleus = generate(params, CFG, [2, 7], 12, temperature=5.0, top_p=1e-9, seed=0)
    assert greedy == nucleus


def test_top_p_determinism_and_support():
    params = init_params(CFG, seed=4)
    a = generate(params, CFG, [5], 15, temperature=1.0, top_p=0.9, seed=11)
    b = generate(params, CFG, [5], 15, temperature=1.0, top_p=0.9, seed=11)
    c = generate(params, CFG, [5], 15, temperature=1.0, top_p=0.9, seed=12)
    assert a == b  # same seed -> identical stream
    assert a != c  # a narrowed-but-nontrivial nucleus still samples


def test_loss_decreases_when_training():
    """60 AdamW steps on a repeating pattern must cut the loss sharply."""
    cfg = GPTConfig(vocab_size=5, block_size=8, n_layer=1, n_head=2, n_embd=8)
    params = init_params(cfg, seed=0)
    opt = AdamW(params, lr=1e-2)
    rng = np.random.default_rng(0)
    pattern = np.array([0, 1, 2, 3, 4] * 40, dtype=np.int64)

    def batch():
        ix = rng.integers(0, len(pattern) - cfg.block_size - 1, size=8)
        x = np.stack([pattern[i : i + cfg.block_size] for i in ix])
        y = np.stack([pattern[i + 1 : i + cfg.block_size + 1] for i in ix])
        return x, y

    x, y = batch()
    first, _ = loss_and_grads(params, cfg, x, y)
    for _ in range(60):
        x, y = batch()
        loss, grads = loss_and_grads(params, cfg, x, y)
        clip_grad_norm(grads, 1.0)
        opt.step(params, grads)
    assert loss < first * 0.5, f"loss {first:.3f} -> {loss:.3f}: did not learn"


def test_attention_rows_sum_to_one():
    params = init_params(CFG, seed=6)
    ids = np.random.default_rng(0).integers(0, 13, size=(2, 9))
    _, _, _, attns = forward(params, CFG, ids, return_attn=True)
    for a in attns:
        np.testing.assert_allclose(a.sum(axis=-1), 1.0, atol=1e-5)
        # causal: strictly-upper triangle is zero
        assert np.abs(np.triu(a, k=1)).max() < 1e-9
