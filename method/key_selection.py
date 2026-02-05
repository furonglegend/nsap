# chimerapy/key_selection.py
"""
Two-layer key selection module.

Provides a simple software-model of the hybrid local-window + static-TCAM
selection used by Chimera. The 'TCAM' is simulated with a top-k similarity
lookup implemented via dot products.
"""
from collections import deque
from typing import Optional

import numpy as np


class TwoLayerKeySelector:
    """
    Two-layer key selector that maintains a local circular buffer of recent
    keys and a static candidate keybank that acts like a TCAM-backed index.

    Methods:
        push_local(key): append a new key to the local window
        tcam_match(q, topk): return the top-k static keys by similarity
        get_selected(q, topk): return the union of local-window keys and TCAM hits
    """

    def __init__(self, local_window_size: int, static_keys: np.ndarray, normalize: bool = True):
        """
        Args:
            local_window_size: maximum number of recent keys to keep
            static_keys: (N_static, d) array of static candidate keys
            normalize: whether to L2-normalize vectors before similarity
        """
        self.local_window_size = int(local_window_size)
        self.local_buf = deque(maxlen=self.local_window_size)
        self.static_keys = np.asarray(static_keys)
        self.normalize = bool(normalize)
        # Pre-normalize static keys if requested
        if self.normalize and self.static_keys.size > 0:
            norms = np.linalg.norm(self.static_keys, axis=1, keepdims=True) + 1e-12
            self.static_keys = self.static_keys / norms

    def push_local(self, key: np.ndarray) -> None:
        """
        Push a new key vector into the local circular buffer.

        Args:
            key: (d,) array
        """
        self.local_buf.append(np.asarray(key))

    def tcam_match(self, q: np.ndarray, topk: int = 4) -> np.ndarray:
        """
        Simulate a TCAM-like match by returning the top-k static keys
        that maximize dot-product similarity with query q.

        Args:
            q: (d,) query vector
            topk: number of matches to return

        Returns:
            matches: (k, d) array of matching static keys
        """
        if self.static_keys.size == 0:
            return np.zeros((0, q.shape[0]), dtype=q.dtype)
        qv = np.asarray(q)
        if self.normalize:
            qv = qv / (np.linalg.norm(qv) + 1e-12)
        sims = self.static_keys.dot(qv)
        idx = np.argsort(-sims)[:topk]
        return self.static_keys[idx]

    def get_selected(self, q: np.ndarray, topk: int = 4, deduplicate: bool = True) -> np.ndarray:
        """
        Return the union of local-window keys and TCAM top-k matches.

        Args:
            q: (d,) query vector
            topk: number of static matches to include
            deduplicate: if True, attempt to deduplicate near-duplicate keys

        Returns:
            selected: (N_sel, d) array of selected keys
        """
        local_arr = np.array(self.local_buf) if len(self.local_buf) > 0 else np.zeros((0, q.shape[0]))
        tcam_arr = self.tcam_match(q, topk=topk)
        if local_arr.shape[0] == 0:
            return tcam_arr
        if tcam_arr.shape[0] == 0:
            return local_arr
        concat = np.vstack([local_arr, tcam_arr])
        if not deduplicate:
            return concat
        # Simple deduplication by rounding and unique rows
        rounded = np.round(concat, decimals=6)
        _, idx = np.unique(rounded.view([('', rounded.dtype)] * rounded.shape[1]), return_index=True)
        selected = concat[np.sort(idx)]
        return selected
