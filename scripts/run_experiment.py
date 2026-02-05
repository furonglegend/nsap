# scripts/run_experiment.py
"""
End-to-end experiment runner for Chimera.

This script wires together:
  - synthetic data generation,
  - feature extraction,
  - incremental attention accumulation,
  - key selection,
  - symbolic fusion,
  - control-plane table generation,
  - and evaluation.

The goal is to provide a reproducible reference pipeline that mirrors
the methodology described in the Chimera paper, while remaining
hardware-agnostic and easy to extend.
"""
import argparse
import numpy as np

from data.generate_synthetic import generate_dataset
from chimerapy.features import extract_features
from chimerapy.incremental import incremental_attention
from chimerapy.key_selection import two_level_key_select
from chimerapy.fusion import SymbolicEngine
from chimerapy.control_plane import recluster_and_generate_tables, TableInstaller
from chimerapy.eval import classification_metrics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Chimera experiment")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--feature-dim", type=int, default=16, help="Feature dimension")
    parser.add_argument("--n-keys", type=int, default=32, help="Number of attention keys")
    parser.add_argument("--n-centroids", type=int, default=16, help="Number of mapping centroids")
    parser.add_argument("--eta", type=float, default=0.1, help="EMA step size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def main():
    """Main experiment entry point."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Generate synthetic data
    X, y = generate_dataset(
        n_samples=args.n_samples,
        feature_dim=args.feature_dim,
        rng=rng,
    )

    # Feature extraction
    features = np.array([extract_features(x) for x in X])

    # Incremental attention accumulation
    attn_outputs = []
    for f in features:
        out = incremental_attention(f, eta=args.eta)
        attn_outputs.append(out)
    attn_outputs = np.array(attn_outputs)

    # Two-level key selection
    selected_keys = two_level_key_select(
        attn_outputs,
        n_local=args.n_keys // 2,
        n_static=args.n_keys // 2,
    )

    # Symbolic engine setup
    sym = SymbolicEngine()

    # Example hard rule: threshold on first feature
    sym.add_hard_rule(lambda z: z[0] > 1.5)

    # Example soft rule: distance from origin
    sym.add_soft_rule(lambda z: float(np.linalg.norm(z)), weight=1.0)

    # Compute fused scores and predictions
    fused_scores = []
    for f, nn_score in zip(selected_keys, attn_outputs[:, 0]):
        score = sym.fused_score(f, s_nn=float(nn_score))
        fused_scores.append(score)

    fused_scores = np.array(fused_scores)
    y_pred = (fused_scores > 0.5).astype(int)

    # Evaluation
    metrics = classification_metrics(y, y_pred)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Control-plane reclustering and table installation
    centroids, quantized = recluster_and_generate_tables(
        features=selected_keys,
        n_centroids=args.n_centroids,
    )
    installer = TableInstaller()
    install_time = installer.install_table(quantized, atomic=True)
    print(f"Simulated table install time: {install_time:.6f} seconds")


if __name__ == "__main__":
    main()
