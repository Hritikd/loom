# loom

**A GPT, from scratch, in pure NumPy — every gradient derived by hand.**

No PyTorch. No JAX. No autograd. ~800 lines of readable NumPy that train a real transformer,
where every backward pass is a derivative worked out on paper and verified against numerical
differentiation to **~1e-8**.

**[▶ Live demo](https://hritikd.github.io/loom/)** — a model trained by this code writes
Shakespeare in your browser, and you can inspect what every attention head was looking at
while it wrote.

[![ci](https://github.com/Hritikd/loom/actions/workflows/ci.yml/badge.svg)](https://github.com/Hritikd/loom/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![deps](https://img.shields.io/badge/runtime%20deps-numpy%20only-success)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Why this exists

Frameworks make transformers easy to *run* and hard to *see*. `loss.backward()` is one line;
what it actually does — the softmax Jacobian inside attention, the three coupled terms of the
layernorm gradient, the scatter-add into the embedding table — stays invisible. loom is the
version where nothing is invisible: if it isn't written down in this repo, it doesn't happen.

It's the final piece of a from-scratch LLM stack:

| Stage | Project | What it builds from scratch |
|---|---|---|
| Text → tokens | **[mosaic](https://github.com/Hritikd/mosaic)** | byte-pair-encoding tokenizer |
| Gradients | **[nabla](https://github.com/Hritikd/nabla)** | reverse-mode autograd engine |
| The model | **loom** (this repo) | the GPT itself — forward, backward, training, KV-cache inference |

nabla derives gradients *automatically* for scalar graphs; loom deliberately does the opposite —
every gradient of the full tensor model is derived **by hand**, because doing it once by hand is
how you actually learn what autograd automates.

## What a transformer actually is

The entire forward pass, from [`model.py`](src/loom/model.py) — this is not pseudocode, it's the code:

```python
x = params["tok_emb"][ids] + params["pos_emb"][:t]          # tokens + positions
for i in range(cfg.n_layer):
    a = layernorm(x)                                        # pre-norm
    q, k, v = split_heads(a @ w_qkv + b_qkv)                # project to heads
    x = x + proj(merge_heads(softmax(q @ k.T / sqrt(dh) + causal_mask) @ v))
    x = x + mlp_proj(gelu(layernorm(x) @ w_fc + b_fc))      # position-wise MLP
logits = layernorm(x) @ w_head
```

That's it. Everything else in the field — KV caches, flash attention, LoRA, quantization — is
an optimization of some line above.

## The backward pass is the point

Each layer in [`layers.py`](src/loom/layers.py) is a `*_forward` / `*_backward` pair. The
backward passes are the derivatives you'd work out on paper:

| Layer | The gradient you have to derive |
|---|---|
| softmax (inside attention) | `ds = p * (dp − Σ(dp·p))` — the Jacobian collapsed to two terms |
| layernorm | three coupled terms through the mean and variance |
| attention | chain rule through `P@V` and `Q@Kᵀ`, with the causal mask handled *by* the softmax gradient |
| cross-entropy | the famous `(softmax − onehot) / N` |
| embedding | scatter-add: `np.add.at(d_emb, ids, dx)` |

And every one of them is **checked**. `tests/test_gradients.py` compares each analytic gradient
against float64 central differences, layer by layer and then end-to-end through the whole model:

```
gradient check : 232 coordinates across all 29 tensors (float64)
                 worst absolute error vs numerical: 4.52e-09
                 worst relative error (|grad| >= 1e-3): 5.93e-08
```

A sign error, a missing term, or a transposed matrix anywhere in the chain fails these tests
loudly. Reproduce with `python benchmarks/run.py`.

## Quickstart

```bash
pip install git+https://github.com/Hritikd/loom
```

Train a ~360K-parameter character-level GPT on Shakespeare, on your laptop's CPU:

```bash
loom-train --corpus data/shakespeare.txt --out ckpt.npz --steps 3000
loom-sample --ckpt ckpt.npz --prompt "ROMEO:" --tokens 400
```

Or from Python:

```python
from loom import GPTConfig, init_params, loss_and_grads, generate, AdamW

cfg = GPTConfig(vocab_size=65, block_size=128, n_layer=3, n_head=6, n_embd=96)
params = init_params(cfg)
opt = AdamW(params, lr=1e-3)

loss, grads = loss_and_grads(params, cfg, x, y)   # forward + hand-derived backward
opt.step(params, grads)

tokens = generate(params, cfg, prompt_ids, 200, temperature=0.8, top_k=40)
```

## Measured results

Training run behind the [live demo](https://hritikd.github.io/loom/) — 3 layers, 6 heads,
d_model 96 (360,096 params), 500KB of Shakespeare, single CPU (Apple M-series), NumPy only:

```
corpus: 500,000 chars, vocab 63 | params: 360,096
step     0 | train 4.1853 | val 4.1728
step   500 | train 2.1558 | val 2.2667
step  1000 | train 1.7833 | val 2.0275
step  2000 | train 1.5410 | val 1.8518
step  2999 | train 1.4138 | val 1.7970
done in 659.8s (~11 min) | ~22,000 tokens/sec | best val 1.7970
```

A sample from the trained model (temperature 0.8, top-k 40):

```
ROMEO:
What,
The fear your come; when I say your sleep!

KING RICHARD III:
Flatter. O Norfolk, before I alone.

KING RICHARD III:
The cause were to my tends to such a to others.

GLOUCESTER:
A all of Rome.
```

Shakespeare-*shaped*, not Shakespeare — dialogue structure, speaker names, line lengths, and
mostly-real words, learned from scratch in minutes on a CPU. Scale is the only thing missing,
and scale is the one thing this repo intentionally doesn't chase.

Generation runs incrementally with a **per-layer KV cache** — each step feeds one token and
attends to cached keys/values (~2,700 tokens/sec on CPU), falling back to windowed
re-encoding once the context outgrows `block_size`. `tests/test_model.py` proves cached and
uncached generation produce identical tokens.

## The browser demo

[hritikd.github.io/loom](https://hritikd.github.io/loom/) runs **this exact code** — not a port —
via [Pyodide](https://pyodide.org): the page fetches `src/loom/*.py` and the float16 checkpoint,
and generates locally in your browser. No server, no API. The attention panel re-runs the
forward pass with `return_attn=True` and renders the post-softmax weights of any head, so you
can see induction-style behavior (heads that lock onto the previous occurrence of a pattern)
emerge even at this scale.

## Read the source in this order

| Order | File | Lines | What you'll learn |
|---|---|---|---|
| 1 | [`tokenizer.py`](src/loom/tokenizer.py) | 64 | chars ↔ ids (for real BPE, see mosaic) |
| 2 | [`layers.py`](src/loom/layers.py) | 161 | each layer + its hand-derived backward |
| 3 | [`model.py`](src/loom/model.py) | 317 | the GPT: forward, backward, KV-cache generation |
| 4 | [`optim.py`](src/loom/optim.py) | 65 | AdamW + cosine schedule, from the papers |
| 5 | [`train.py`](src/loom/train.py) | 93 | the whole training loop, no magic |

## Honest limitations

- **It's tiny and character-level.** The goal is a model you can fully read and verify, not one
  that competes with anything. Expect Shakespeare-shaped text, not poetry.
- **No dropout.** Dropout adds RNG state threading through every backward pass for little
  benefit at this scale; regularization comes from weight decay and early stopping.
- **Pure NumPy is slow.** Orders of magnitude slower than PyTorch on a GPU. CPU-only, O(T²)
  attention, no fusion, no flash attention — clarity was chosen over speed at every fork.
- **float32 training, float16 checkpoints.** No mixed-precision machinery.
- Past `block_size`, generation re-encodes a sliding window per step (correct, but the KV cache
  no longer applies).

## License

MIT
