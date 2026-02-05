
# Overview

This repository provides a reference implementation for a dataplane-oriented
learning and reasoning pipeline designed for programmable network devices.
The code focuses on expressing learning-style computations and rule-based
constraints in forms that are compatible with match-action abstractions.

The implementation separates fast per-packet processing from slower
configuration updates, enabling online inference under strict resource and
latency constraints.

---

# Repository Structure

```

data/
generate_synthetic.py        Synthetic dataset generation

chimerapy/
features.py                  Feature extraction utilities
incremental.py               Incremental statistics and accumulation
key_selection.py             Hierarchical key selection logic
fusion.py                    Rule fusion and constraint enforcement
control_plane.py             Control-plane update logic
p4_tablegen.py               Table export and resource estimation
eval.py                      Evaluation utilities

scripts/
run_experiment.py            End-to-end experiment runner

tests/
test_features.py             Unit tests for feature extraction

```

---

# Installation

Python 3.9 or later is recommended.

Install dependencies using:

```bash
pip install -r requirements.txt
````

No specialized hardware, P4 compiler, or vendor-specific SDK is required to
run the experiments. All dataplane behavior is simulated at the algorithmic
level.

---

# Running Experiments

To execute the default experimental pipeline:

```
python scripts/run_experiment.py
```

The script performs the following steps:

* generates or loads input data,
* extracts features and updates online statistics,
* applies table-based inference logic,
* simulates control-plane table updates,
* reports classification and detection metrics.

Key parameters such as update rates, feature dimensions, and table sizes can
be adjusted via command-line arguments.

---

# Design Principles

The codebase is structured around three core design principles:

* **Incremental Processing**
  All statistics are updated online using constant-time operations to match
  dataplane execution constraints.

* **Timescale Separation**
  Fast per-packet updates are decoupled from slower configuration changes,
  improving stability and predictability.

* **Table-Oriented Representation**
  Computation and constraints are expressed as lookup and aggregation
  operations compatible with match-action pipelines.

The implementation favors clarity and explicitness over aggressive
optimization to facilitate analysis and extension.

---

# Evaluation and Testing

Unit tests can be executed using:

```
pytest tests/
```

Evaluation utilities report standard metrics and estimate memory usage based
on table dimensions and quantization settings.

---

# Extending the Code

The repository is intended for research and experimentation. Possible
extensions include:

* replacing synthetic inputs with real traffic traces,
* modifying feature mappings and update rules,
* experimenting with alternative rule sets,
* exporting generated tables to external runtimes.

