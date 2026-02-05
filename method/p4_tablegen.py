# chimerapy/p4_tablegen.py
"""
Helpers to export mapping tables to CSV and to create simple
P4Runtime-style table add command strings for offline conversion.

This module does not depend on an actual P4Runtime client library;
instead it creates textual artifacts that are commonly used by control
plane installers (CSV exports, JSON blobs, or simple `table_add` strings).
"""
from typing import List, Optional, Tuple

import csv
import json
import os

import numpy as np
import pandas as pd


def export_table_csv(quantized_centroids: np.ndarray, filename: str) -> None:
    """
    Export quantized centroids to a CSV file. Each row corresponds to a
    single table entry whose columns are the quantized feature components.

    Args:
        quantized_centroids: (N, d) integer array
        filename: output CSV filename
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    df = pd.DataFrame(quantized_centroids)
    df.to_csv(filename, index=False)
    return None


def export_table_json(centroids: np.ndarray, filename: str, meta: Optional[dict] = None) -> None:
    """
    Export floating-point centroids and optional metadata to a JSON file.

    Args:
        centroids: (N, d) float array
        filename: output JSON filename
        meta: optional metadata dictionary
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    payload = {"centroids": centroids.tolist()}
    if meta is not None:
        payload["meta"] = meta
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return None


def generate_p4_table_add_commands(
    quantized_centroids: np.ndarray,
    table_name: str,
    key_fields: Optional[List[str]] = None,
    action_name: str = "set_value",
    action_params: Optional[List[str]] = None,
) -> List[str]:
    """
    Produce a list of textual P4 table_add commands in a simple structured
    form. Control-plane scripts commonly translate such strings into
    P4Runtime API calls.

    Args:
        quantized_centroids: (N, d) integer array representing table keys/values
        table_name: name of the P4 table
        key_fields: optional list of key field names (length d). If None,
                    generic keys k0, k1, ... are used.
        action_name: P4 action to invoke for each entry
        action_params: optional list of action parameter names; if provided,
                       must have length <= d and will be bound to centroid values.

    Returns:
        commands: list of strings that represent table entries
    """
    q = np.asarray(quantized_centroids, dtype=int)
    n, d = q.shape
    key_fields = key_fields or [f"k{i}" for i in range(d)]
    cmd_list = []
    for i in range(n):
        key_pairs = ",".join([f"{k}={int(v)}" for k, v in zip(key_fields, q[i])])
        if action_params:
            params = ",".join([f"{p}={int(v)}" for p, v in zip(action_params, q[i])])
        else:
            params = ",".join([f"val{j}={int(v)}" for j, v in enumerate(q[i])])
        cmd = f"table_add {table_name} {action_name} {key_pairs} => {params}"
        cmd_list.append(cmd)
    return cmd_list
