# chimerapy/__init__.py
"""
Chimera Python package exports.

Expose main helper functions and classes for ease of import.
"""
from .features import init_random_features, rff_positive_features, quantize_array
from .incremental import incremental_update, compute_output
from .key_selection import TwoLayerKeySelector

__all__ = [
    "init_random_features",
    "rff_positive_features",
    "quantize_array",
    "incremental_update",
    "compute_output",
    "TwoLayerKeySelector",
]
