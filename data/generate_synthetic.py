# data/generate_synthetic.py
"""
Synthetic dataset generator for Chimera experiments.

This script produces a small synthetic dataset suitable for
development and unit testing of the Chimera pipeline. It writes
a compressed NumPy .npz file containing:
  - keys:   (T, d) array of key/query embeddings
  - values: (T, d_v) array of value vectors
  - labels: (T,) binary labels (optional)
  - meta:   dictionary with generation parameters

Usage:
    python data/generate_synthetic.py --out data/synthetic.npz --T 500 --d 16 --d_v 8 --seed 0
"""
import argparse
import os
from typing import Tuple

import numpy as np


def generate_sequence(T: int, d: int, d_v: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic keys, values and labels.

    Args:
        T: sequence length (number of tokens)
        d: dimensionality of keys/queries
        d_v: dimensionality of values
        seed: RNG seed for reproducibility

    Returns:
        keys: (T, d) float32 array
        values: (T, d_v) float32 array
        labels: (T,) int8 array with binary labels (0/1)
    """
    rng = np.random.RandomState(seed)
    # Random Gaussian features
    keys = rng.normal(loc=0.0, scale=1.0, size=(T, d)).astype(np.float32)
    # Add small temporal correlation
    for t in range(1, T):
        keys[t] += 0.1 * keys[t - 1]

    # Values are correlated with keys via a random linear map + noise
    W = rng.normal(scale=0.5, size=(d, d_v)).astype(np.float32)
    values = keys.dot(W) + 0.1 * rng.normal(size=(T, d_v)).astype(np.float32)

    # Binary labels: a simple synthetic anomaly rule
    # label = 1 if projection on a random vector exceeds threshold
    proj = keys.dot(rng.normal(size=(d,)).astype(np.float32))
    threshold = np.percentile(proj, 95)
    labels = (proj > threshold).astype(np.int8)

    return keys, values, labels


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for Chimera.")
    parser.add_argument("--out", type=str, default="data/synthetic.npz", help="Output .npz filename.")
    parser.add_argument("--T", type=int, default=1000, help="Sequence length (number of tokens).")
    parser.add_argument("--d", type=int, default=16, help="Key/query dimensionality.")
    parser.add_argument("--d_v", type=int, default=8, help="Value dimensionality.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    keys, values, labels = generate_sequence(args.T, args.d, args.d_v, seed=args.seed)
    meta = {"T": args.T, "d": args.d, "d_v": args.d_v, "seed": args.seed}

    np.savez_compressed(args.out, keys=keys, values=values, labels=labels, meta=meta)
    print(f"Synthetic dataset saved to {args.out}")
    print(f"keys.shape={keys.shape}, values.shape={values.shape}, labels.shape={labels.shape}")


if __name__ == "__main__":
    main()
