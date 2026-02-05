# chimerapy/incremental.py
"""
Incremental aggregation utilities for Chimera.

Implements the S_t and Z_t updates used in the linearized attention
formulation and provides a function to recover the approximate output
o(q) = (phi(q)^T (Phi(K)^T V)) / (phi(q)^T (Phi(K)^T 1)).
"""
from typing import Tuple

import numpy as np

# import quantization helper from features module
try:
    # package import when chimerapy is installed as a package
    from chimerapy.features import quantize_array
except Exception:
    # fallback to relative import if run as a script in package layout
    from .features import quantize_array  # type: ignore


def incremental_update(
    S: np.ndarray,
    Z: np.ndarray,
    phi_k: np.ndarray,
    v: np.ndarray,
    quantize_bits: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform an incremental accumulator update:
      S <- S + phi_k * v^T
      Z <- Z + phi_k

    Args:
        S: accumulator matrix of shape (m, d_v)
        Z: accumulator vector of shape (m,)
        phi_k: feature vector of shape (m,)
        v: value vector of shape (d_v,)
        quantize_bits: optional bitwidth for fixed-point simulation

    Returns:
        (S_new, Z_new): updated accumulators
    """
    # outer product: (m, d_v)
    incr = np.outer(phi_k, v)
    S_new = S + incr
    Z_new = Z + phi_k
    if quantize_bits is not None:
        S_new = quantize_array(S_new, quantize_bits)
        Z_new = quantize_array(Z_new, quantize_bits)
    return S_new, Z_new


def compute_output(phi_q: np.ndarray, S: np.ndarray, Z: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Compute the linearized attention output for a single query:
      o(q) ≈ (phi(q)^T S) / (phi(q)^T Z)

    Args:
        phi_q: feature vector for query, shape (m,)
        S: accumulator matrix, shape (m, d_v)
        Z: accumulator vector, shape (m,)
        eps: small numerical guard for denominator

    Returns:
        out: output vector of shape (d_v,)
    """
    numerator = phi_q.dot(S)  # (d_v,)
    denom = float(phi_q.dot(Z))  # scalar
    if abs(denom) < eps:
        denom = eps
    return numerator / denom
