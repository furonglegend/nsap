# chimerapy/fusion.py
"""
Cascade fusion and symbolic scoring utilities for Chimera.

This module implements:
  - a cascade fusion operator that enforces hard-symbolic vetoes,
    and otherwise returns a differentiable blend of neural and
    soft-symbolic scores;
  - helper utilities to compute soft-symbolic scores from rule
    distances or feature-based penalties;
  - a small SymbolicEngine class that simulates TCAM-based hard
    rule hits and provides rule-distance lookups for soft scoring.
"""
from typing import Iterable, List, Optional

import numpy as np


def sigmoid(x: float) -> float:
    """Numerically-stable sigmoid for scalars."""
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


def cascade_fusion(
    s_nn: float,
    s_sym: float,
    hard_hit: bool,
    lambda_hard: bool = True,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Cascade fusion operator.

    Args:
        s_nn: neural score (scalar). Can be any real number; typical range is
              model-dependent (e.g., logits or normalized score).
        s_sym: soft-symbolic score (scalar). Higher means stronger symbolic evidence.
        hard_hit: whether a hard symbolic rule matched (boolean).
        lambda_hard: if True then a hard hit forces output to the hard value (1.0).
                     If False, hard hits are ignored and soft blending is used.
        alpha: scaling coefficient applied to s_nn in soft blend.
        beta: scaling coefficient applied to s_sym in soft blend.

    Returns:
        fused_score: scalar in (0,1) resulting from the cascade fusion logic.
    """
    if hard_hit and lambda_hard:
        # Hard veto / forced accept depending on rule semantics.
        return 1.0
    x = alpha * s_nn + beta * s_sym
    return float(sigmoid(x))


def compute_soft_symbolic_score(rule_distances: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Compute a soft symbolic score from rule distances.

    The function converts distances-to-satisfaction into a single continuous
    score. Smaller distances indicate better rule satisfaction; we negate the
    distance-weighted sum to obtain a larger-is-better score.

    Args:
        rule_distances: array of non-negative distances for each ground rule (M,).
        weights: optional weights for rules (M,). If None, uniform weights are used.

    Returns:
        score: scalar soft-symbolic score (higher = more satisfied).
    """
    rd = np.asarray(rule_distances, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(rd, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    # Defensive clipping: distances must be non-negative
    rd = np.clip(rd, 0.0, None)
    # Negative weighted sum gives higher score for better satisfaction
    score = -float(np.dot(w, rd))
    return score


class SymbolicEngine:
    """
    Lightweight symbolic engine that simulates hard rule matching (TCAM)
    and computes soft rule distances.

    The engine stores:
      - hard_rules: a list of (predicate_fn) that return bool for a packet/feature
      - soft_rules: a list of (distance_fn) that return non-negative distance

    This is a lightweight, easily replaceable component intended for
    algorithmic evaluations and unit tests.
    """

    def __init__(self):
        self.hard_rules = []  # List[callable(feature) -> bool]
        self.soft_rules = []  # List[callable(feature) -> float]
        self.soft_weights = []  # List[float]

    def add_hard_rule(self, predicate_fn):
        """Add a hard rule predicate that returns True on match."""
        self.hard_rules.append(predicate_fn)

    def add_soft_rule(self, distance_fn, weight: float = 1.0):
        """Add a soft rule represented by a distance function and weight."""
        self.soft_rules.append(distance_fn)
        self.soft_weights.append(float(weight))

    def hard_hit(self, feature) -> bool:
        """Return True if any hard rule matches the given feature."""
        for p in self.hard_rules:
            try:
                if p(feature):
                    return True
            except Exception:
                # Rules should not raise; treat exceptions as non-match
                continue
        return False

    def soft_scores(self, feature) -> (np.ndarray, np.ndarray):
        """
        Compute soft rule distances and return (distances, weights).

        Returns:
            distances: np.ndarray (M,)
            weights: np.ndarray (M,)
        """
        if len(self.soft_rules) == 0:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        dists = np.array([f(feature) for f in self.soft_rules], dtype=np.float64)
        w = np.array(self.soft_weights, dtype=np.float64)
        return dists, w

    def fused_score(self, feature, s_nn: float, lambda_hard: bool = True, alpha: float = 1.0, beta: float = 1.0) -> float:
        """
        Compute fused score for a given feature and neural score.

        This convenience function calls hard_hit, soft_scores, computes the
        soft symbolic score, and applies cascade_fusion.

        Args:
            feature: opaque feature object consumed by rules
            s_nn: neural score scalar
            lambda_hard: whether hard hits force output
            alpha, beta: fusion coefficients

        Returns:
            fused scalar score in (0,1)
        """
        hh = self.hard_hit(feature)
        dists, weights = self.soft_scores(feature)
        s_sym = compute_soft_symbolic_score(dists, weights) if dists.size > 0 else 0.0
        return cascade_fusion(s_nn, s_sym, hh, lambda_hard=lambda_hard, alpha=alpha, beta=beta)
