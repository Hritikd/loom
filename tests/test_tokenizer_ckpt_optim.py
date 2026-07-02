"""Tokenizer round-trips, checkpoint round-trips, optimizer behavior."""

import numpy as np
import pytest

from loom.checkpoint import load, save
from loom.model import GPTConfig, forward, init_params
from loom.optim import AdamW, clip_grad_norm, cosine_lr
from loom.tokenizer import CharTokenizer


def test_tokenizer_round_trip():
    text = "To be, or not to be — that is the question!\n"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_tokenizer_strict_and_replace():
    tok = CharTokenizer.from_text("abc ")
    with pytest.raises(ValueError):
        tok.encode("abz")
    assert tok.decode(tok.encode("abz", errors="replace")) == "ab "


def test_tokenizer_json_round_trip():
    tok = CharTokenizer.from_text("hello world")
    tok2 = CharTokenizer.from_json(tok.to_json())
    assert tok2.chars == tok.chars
    assert tok2.encode("hello") == tok.encode("hello")


def test_checkpoint_round_trip(tmp_path):
    cfg = GPTConfig(vocab_size=7, block_size=8, n_layer=1, n_head=2, n_embd=8)
    params = init_params(cfg, seed=0)
    tok = CharTokenizer([chr(97 + i) for i in range(7)])
    path = tmp_path / "ckpt.npz"
    save(path, params, cfg, tok)
    p2, cfg2, tok2 = load(path)
    assert cfg2 == cfg and tok2.chars == tok.chars
    ids = np.zeros((1, 5), dtype=np.int64)
    a, _, _, _ = forward(params, cfg, ids)
    b, _, _, _ = forward(p2, cfg2, ids)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_checkpoint_half_precision(tmp_path):
    cfg = GPTConfig(vocab_size=7, block_size=8, n_layer=1, n_head=2, n_embd=8)
    params = init_params(cfg, seed=0)
    tok = CharTokenizer(list("abcdefg"))
    path = tmp_path / "half.npz"
    save(path, params, cfg, tok, half=True)
    p2, _, _ = load(path)
    assert p2["tok_emb"].dtype == np.float32
    assert np.abs(p2["tok_emb"] - params["tok_emb"]).max() < 1e-3  # fp16 quantization


def test_adamw_minimizes_quadratic():
    params = {"w": np.array([5.0, -3.0])}
    opt = AdamW(params, lr=0.1, weight_decay=0.0)
    for _ in range(200):
        opt.step(params, {"w": 2.0 * params["w"]})  # d/dw of w^2
    assert np.abs(params["w"]).max() < 1e-2


def test_adamw_weight_decay_policy():
    params = {"w_mat": np.ones((2, 2)), "bias": np.ones(2), "tok_emb": np.ones((2, 2))}
    opt = AdamW(params, lr=0.1, weight_decay=0.5)
    zero = {k: np.zeros_like(v) for k, v in params.items()}
    opt.step(params, zero)
    assert params["w_mat"].max() < 1.0  # matrices decay
    assert params["bias"].min() == 1.0  # biases don't
    assert params["tok_emb"].min() == 1.0  # embeddings don't


def test_clip_grad_norm():
    grads = {"a": np.array([3.0, 4.0])}  # norm 5
    total = clip_grad_norm(grads, max_norm=1.0)
    assert abs(total - 5.0) < 1e-9
    assert abs(np.linalg.norm(grads["a"]) - 1.0) < 1e-9


def test_cosine_schedule_shape():
    base = 1e-3
    assert cosine_lr(0, 1000, base, warmup=100) < base * 0.02  # warmup starts low
    assert abs(cosine_lr(100, 1000, base, warmup=100) - base) < base * 0.01  # peak
    assert cosine_lr(999, 1000, base, warmup=100) < base * 0.11  # decayed to ~min
