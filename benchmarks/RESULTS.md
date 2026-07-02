# Measured results

All numbers below are the output of `python benchmarks/run.py` and `loom-train`
on a single Apple M-series CPU, NumPy only. Rerun them yourself — the gradient
check is seeded and reproduces exactly.

## Gradient correctness (the number that matters)

```
gradient check : 232 coordinates across all 29 tensors (float64)
                 worst absolute error vs numerical: 4.52e-09
                 worst relative error (|grad| >= 1e-3): 5.93e-08
```

Every hand-derived backward pass, checked end-to-end through the full model
against float64 central differences. This is machine-precision agreement —
a sign error or transposed matrix anywhere would show up around 1e-1.

## Throughput

```
training       : 22,348 tokens/sec (batch 32 x 128, 360,480 params)
generation     : 2,738 tokens/sec  (KV-cache incremental decode)
```

## The demo model's training run

`loom-train --corpus data/shakespeare.txt --steps 3000` (3L / 6H / d96,
360,096 params, 500KB Shakespeare, vocab 63):

```
step     0 | train 4.1853 | val 4.1728
step   500 | train 2.1558 | val 2.2667
step  1000 | train 1.7833 | val 2.0275
step  2000 | train 1.5410 | val 1.8518
step  2999 | train 1.4138 | val 1.7970
done in 659.8s | best val 1.7970
```

4.17 → 1.80 validation loss in ~11 minutes on a laptop CPU with no framework.
For calibration: uniform over 63 chars would be ln(63) ≈ 4.14, so the model
starts at "knows nothing" and ends producing the Shakespeare-shaped text you
can generate in the [live demo](https://hritikd.github.io/loom/).
