"""Generate text from a trained checkpoint.

    loom-sample --ckpt ckpt.npz --prompt "ROMEO:" --tokens 400
"""

from __future__ import annotations

import argparse
import sys

from .checkpoint import load
from .model import generate_stream


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    params, cfg, tok = load(args.ckpt)
    prompt_ids = tok.encode(args.prompt, errors="replace")
    sys.stdout.write(args.prompt)
    for t in generate_stream(
        params, cfg, prompt_ids, args.tokens,
        temperature=args.temperature, top_k=args.top_k, seed=args.seed,
    ):
        sys.stdout.write(tok.decode([t]))
        sys.stdout.flush()
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
