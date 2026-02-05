# chimerapy/eval.py
"""
Evaluation utilities for Chimera experiments.

This module provides:
  - standard classification metrics (accuracy, precision, recall, f1);
  - resource usage estimators for SRAM/TCAM based on table sizes and
    accumulator shapes;
  - robustness test helpers that measure score degradation under noise.
"""
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute basic classification metrics.

    Args:
        y_true: (N,) ground-truth labels (0/1)
        y_pred: (N,) predicted labels (0/1) or scores thresholded externally

    Returns:
        dict with keys: accuracy, precision, recall, f1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": float(p), "recall": float(r), "f1": float(f)}


def estimate_resource_usage(
    S_shape: Tuple[int, int],
    Z_shape: Tuple[int],
    quant_bits: int,
    n_tcam_entries: int,
    tcam_entry_bits: int = 32,
) -> Dict[str, float]:
    """
    Roughly estimate SRAM and TCAM bit usage for Chimera's mapping.

    Args:
        S_shape: (m, d_v) shape of accumulator S (number of scalars)
        Z_shape: (m,) shape of accumulator Z
        quant_bits: bits used per stored scalar (fixed-point)
        n_tcam_entries: number of TCAM entries used for static index
        tcam_entry_bits: bit width per TCAM entry (implementation-specific)

    Returns:
        dict with approximate bit counts: sram_bits, tcam_bits, total_bits
    """
    m, d_v = int(S_shape[0]), int(S_shape[1])
    n_sram_scalars = m * d_v + m  # S and Z
    sram_bits = n_sram_scalars * int(quant_bits)
    tcam_bits = n_tcam_entries * int(tcam_entry_bits)
    total_bits = sram_bits + tcam_bits
    return {"sram_bits": float(sram_bits), "tcam_bits": float(tcam_bits), "total_bits": float(total_bits)}


def robustness_test_noise(
    compute_score_fn,
    features: np.ndarray,
    noise_std_list: Tuple[float, ...] = (0.01, 0.05, 0.1),
) -> Dict[float, float]:
    """
    Measure degradation of a scoring function when additive Gaussian noise
    is applied to features.

    Args:
        compute_score_fn: callable(feature_array) -> scalar score (or array)
        features: (N, d) array of baseline features
        noise_std_list: list of noise standard deviations to evaluate

    Returns:
        dict mapping noise_std -> mean absolute score change
    """
    baseline_scores = np.array([compute_score_fn(x) for x in features])
    results = {}
    for sigma in noise_std_list:
        noisy = features + np.random.normal(scale=sigma, size=features.shape)
        noisy_scores = np.array([compute_score_fn(x) for x in noisy])
        mean_abs_change = float(np.mean(np.abs(noisy_scores - baseline_scores)))
        results[float(sigma)] = mean_abs_change
    return results
