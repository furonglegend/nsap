# tests/test_features.py
"""
Unit tests for feature extraction utilities.

These tests validate:
  - output dimensionality,
  - numerical stability,
  - deterministic behavior under fixed input.

The tests are intentionally lightweight so they can be executed
quickly during continuous integration.
"""
import numpy as np

from chimerapy.features import extract_features


def test_feature_shape():
    """Feature extractor should return a 1D vector with fixed length."""
    x = np.random.randn(16)
    f = extract_features(x)
    assert isinstance(f, np.ndarray)
    assert f.ndim == 1
    assert f.shape[0] > 0


def test_feature_determinism():
    """Feature extraction should be deterministic for identical input."""
    x = np.random.randn(16)
    f1 = extract_features(x)
    f2 = extract_features(x)
    assert np.allclose(f1, f2)


def test_feature_finite_values():
    """Extracted features should not contain NaN or Inf."""
    x = np.random.randn(16)
    f = extract_features(x)
    assert np.all(np.isfinite(f))


def test_feature_zero_input():
    """Zero input should not cause numerical issues."""
    x = np.zeros(16)
    f = extract_features(x)
    assert np.all(np.isfinite(f))
    assert np.linalg.norm(f) >= 0.0
