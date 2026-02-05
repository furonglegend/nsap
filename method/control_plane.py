# chimerapy/control_plane.py
"""
Control-plane utilities for Chimera.

This module implements:
  - EMA-based occupancy statistics update (fast dataplane estimator).
  - Re-clustering and table generation (slow control-plane path).
  - A TableInstaller simulator that models atomic installs and tracks
    the install duration for use in two-timescale stability analysis.
"""
from typing import Optional, Tuple

import time
import math

import numpy as np
from sklearn.cluster import KMeans


def ema_update(C: np.ndarray, u: np.ndarray, eta: float) -> np.ndarray:
    """
    Exponential moving average update for occupancy statistics.

    Args:
        C: (n_centroids,) current EMA state
        u: (n_centroids,) instantaneous 0/1 observation vector
        eta: EMA step size in (0,1)

    Returns:
        updated C vector
    """
    C = np.asarray(C, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    assert 0.0 < eta < 1.0, "eta must be in (0,1)"
    return (1.0 - eta) * C + eta * u


def recluster_and_generate_tables(
    features: np.ndarray,
    n_centroids: int,
    quant_bits: int = 8,
    random_state: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run k-means clustering over feature vectors and produce quantized table
    entries suitable for exporting to the dataplane.

    Args:
        features: (N, d) array of feature vectors used to build mapping tables
        n_centroids: desired number of centroids / table entries
        quant_bits: number of bits for symmetric integer quantization
        random_state: RNG seed

    Returns:
        centroids: (n_centroids, d) float centroids
        quantized: (n_centroids, d) int quantized centroids
    """
    features = np.asarray(features, dtype=np.float64)
    if features.shape[0] < n_centroids:
        # If too few points, pad by sampling with replacement
        idx = np.random.choice(features.shape[0], size=n_centroids - features.shape[0], replace=True)
        features = np.vstack([features, features[idx]])

    kmeans = KMeans(n_clusters=n_centroids, random_state=random_state, n_init=10)
    kmeans.fit(features)
    centroids = kmeans.cluster_centers_

    # Symmetric quantization to signed integers
    qmax = (2 ** (quant_bits - 1)) - 1
    max_abs = float(np.max(np.abs(centroids))) if np.max(np.abs(centroids)) != 0 else 1.0
    scale = max_abs / qmax
    quantized = np.round(centroids / scale).astype(int)
    return centroids, quantized


class TableInstaller:
    """
    Simulated table installer that performs batched installs and tracks
    atomic install duration. Use this class to model installation time
    and measure the impact on two-timescale stability.
    """

    def __init__(self, per_entry_install_ms: float = 0.001):
        """
        Args:
            per_entry_install_ms: expected time in milliseconds required to
                                  install a single table entry (empirical).
        """
        self.per_entry_install_ms = float(per_entry_install_ms)
        self.last_install_time = 0.0

    def estimate_install_duration(self, n_entries: int) -> float:
        """
        Estimate the duration (seconds) to atomically install n_entries.
        """
        ms = n_entries * self.per_entry_install_ms
        return max(0.0, ms / 1000.0)

    def install_table(self, entries: np.ndarray, atomic: bool = True) -> float:
        """
        Simulate installation of table entries. Returns actual elapsed seconds.

        Args:
            entries: arbitrary object representing entries (shape interpretable)
            atomic: if True, simulate an atomic batch install (single duration);
                    otherwise simulate per-entry incremental installs.

        Returns:
            elapsed_seconds: float
        """
        try:
            n_entries = int(entries.shape[0])
        except Exception:
            n_entries = int(len(entries))
        dur = self.estimate_install_duration(n_entries)
        # In a real implementation we would call P4Runtime here; we simulate delay.
        time.sleep(dur)
        self.last_install_time = dur
        return dur


def mapping_similarity(old_table: np.ndarray, new_table: np.ndarray) -> float:
    """
    Compute a simple similarity metric between two mapping tables.

    The metric returns a scalar in [0,1], where 1 means identical mapping,
    and 0 means maximally different. The function computes per-row L2
    differences normalized by row magnitude and returns 1 - mean(relative diff).

    Args:
        old_table: (N, d)
        new_table: (N, d)

    Returns:
        similarity: float in [0,1]
    """
    old = np.asarray(old_table, dtype=np.float64)
    new = np.asarray(new_table, dtype=np.float64)
    if old.shape != new.shape:
        # If shapes differ, compute similarity on the overlapping prefix
        n = min(old.shape[0], new.shape[0])
        old = old[:n]
        new = new[:n]
    row_norm = np.linalg.norm(old, axis=1) + 1e-12
    rel_diff = np.linalg.norm(old - new, axis=1) / row_norm
    mean_rel = float(np.mean(rel_diff))
    sim = max(0.0, 1.0 - mean_rel)
    return sim
