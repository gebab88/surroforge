# Quickstart

This walkthrough takes you from a raw data table to a **validated, reproducible,
exportable** surrogate model. It is self-contained: copy each block in order and
you will have a working model in a few minutes.

## 0. Install

```bash
pip install 'surroforge[smt]'   # Kriging/RBF need the SMT backend
```

## 1. Get some data

A SurroForge dataset is a tabular file (CSV, and Parquet/HDF5 with the `formats`
extra) where each row is one simulation and columns are the inputs and outputs.
For this walkthrough, create a small `samples.csv`:

```python title="make_samples.py"
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
length = rng.uniform(1.0, 5.0, 60)
width = rng.uniform(0.1, 0.5, 60)
load = rng.uniform(100.0, 1000.0, 60)
# A stand-in "solver" output (max stress on a beam).
max_stress = (load * length) / (width**2) + rng.normal(0, 5.0, 60)

pd.DataFrame(
    {"length": length, "width": width, "load": load, "max_stress": max_stress}
).to_csv("samples.csv", index=False)
```

```bash
python make_samples.py
```

## 2. Create a project

A project is an on-disk workspace that holds your data, configs, runs, and reports.

```bash
surroforge init beam-study
cd beam-study
```

This creates:

```
beam-study/
├── surroforge.yaml   # project metadata + registered datasets
├── data/             # input data
├── configs/          # YAML training configurations
├── runs/             # one directory per training run
└── reports/          # generated reports
```

## 3. Register the dataset

Registering validates the schema, records a content hash, and copies the file into
`data/` so the project is self-contained.

```bash
surroforge data add ../samples.csv --name beam
surroforge data list
```

## 4. Describe the run in YAML

The configuration is the complete specification of a training run — one file, fully
versionable. Save this as `configs/kriging.yaml`:

```yaml title="configs/kriging.yaml"
experiment: beam-kriging
data:
  source: beam            # a registered dataset name, or a file path
  inputs: [length, width, load]
  outputs: [max_stress]
  split:
    test_fraction: 0.2
    seed: 42
model:
  method: kriging         # kriging | rbf | polynomial | mlp
  corr: squar_exp         # method-specific hyperparameters pass straight through
validation:
  cross_validation: 5     # k-fold; or {folds: 5, shuffle: true}
  metrics: [rmse, r2, mae]
```

See the [configuration reference](reference/configuration.md) for every field.

## 5. Train

```bash
surroforge train --config configs/kriging.yaml
```

You will see a run id (`run-001`) and a metrics summary (train / test / CV mean /
CV std for each metric). The run directory under `runs/run-001/` holds the saved
model, a config snapshot, metrics, and a complete lineage record.

## 6. Inspect and validate

```bash
surroforge list                 # all runs in the project
surroforge validate run-001     # re-print this run's metrics
surroforge info run-001         # full lineage: hashes, seed, versions
```

## 7. Predict on new inputs

```bash
# new_points.csv needs the same input columns: length, width, load
surroforge predict run-001 --input ../new_points.csv --output preds.csv
```

## 8. Export the model

```bash
surroforge export run-001 --format pickle           # always available
surroforge export run-001 --format onnx             # polynomial only (needs onnx extra)
```

A method that does not support a format fails with a clear message — export support
is declared per method, not guessed.

## 9. Report

```bash
surroforge report run-001       # Markdown report under reports/
```

## 10. Reproduce

Because the run recorded its config, data hash, seed, and versions, you can verify
it reproduces:

```bash
surroforge reproduce run-001
```

A deterministic method reports a **bit-identical** reproduction; an iterative one
(Kriging/RBF) reports reproduction **within its declared tolerance**. See
[Reproduction](scenarios/reproduction.md).

---

That is the **greenfield** scenario end to end. From here:

- Compare methods head to head → [Method comparison](scenarios/method-comparison.md)
- Drive the whole thing from Python → [Python API](reference/python-api.md)
