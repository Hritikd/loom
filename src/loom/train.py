"""Train a character-level GPT on a text file. CPU, NumPy, no framework.

    loom-train --corpus data/shakespeare.txt --out ckpt.npz --steps 3000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .checkpoint import save
from .model import GPTConfig, forward, init_params, loss_and_grads, param_count
from .optim import AdamW, clip_grad_norm, cosine_lr
from .tokenizer import CharTokenizer


def get_batch(data: np.ndarray, block_size: int, batch_size: int, rng) -> tuple:
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def estimate_loss(params, cfg, data, batch_size, rng, iters: int = 20) -> float:
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, cfg.block_size, batch_size, rng)
        _, loss, _, _ = forward(params, cfg, x, y)
        losses.append(loss)
    return float(np.mean(losses))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="path to a plain-text training file")
    ap.add_argument("--out", default="ckpt.npz")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--half", action="store_true", help="save weights as float16")
    args = ap.parse_args(argv)

    text = open(args.corpus, encoding="utf-8").read()
    tok = CharTokenizer.from_text(text)
    data = np.array(tok.encode(text), dtype=np.int32)
    n_val = max(1, len(data) // 10)
    train_data, val_data = data[:-n_val], data[-n_val:]

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    params = init_params(cfg, seed=args.seed)
    opt = AdamW(params, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1)

    print(f"corpus: {len(data):,} chars, vocab {tok.vocab_size} | params: {param_count(params):,}")

    best_val = float("inf")
    tokens_seen = 0
    t0 = time.time()
    for step in range(args.steps):
        x, y = get_batch(train_data, cfg.block_size, args.batch_size, rng)
        loss, grads = loss_and_grads(params, cfg, x, y)
        clip_grad_norm(grads, 1.0)
        opt.step(params, grads, lr=cosine_lr(step, args.steps, args.lr))
        tokens_seen += x.size

        if step % args.eval_every == 0 or step == args.steps - 1:
            val = estimate_loss(params, cfg, val_data, args.batch_size, eval_rng)
            tps = tokens_seen / (time.time() - t0)
            print(f"step {step:5d} | train {loss:.4f} | val {val:.4f} | {tps:,.0f} tok/s")
            if val < best_val:
                best_val = val
                save(args.out, params, cfg, tok, half=args.half)

    print(f"done in {time.time() - t0:.1f}s | best val {best_val:.4f} | saved {args.out}")


if __name__ == "__main__":
    main()
