# SurroForge

**Surrogate models for FEM/CFD simulation data — scriptable, versionable, reproducible.**

SurroForge is a command-line tool (with a matching Python API) for building,
validating, and managing surrogate models from tabular simulation data. It takes
you from a solver's output table to a validated, exportable surrogate model — with
the data, configuration, and model cleanly versioned and reproducible months or
years later.

This repository contains the **Core** (Apache 2.0) edition: the open-source kernel
described in the [PRD](PRD.md), Phase 1 (MVP).

## Install

```bash
pip install surroforge            # core
pip install 'surroforge[smt]'     # + Kriging / RBF backends
pip install 'surroforge[onnx]'    # + ONNX export
pip install 'surroforge[formats]' # + Parquet / HDF5 input
pip install 'surroforge[all]'     # everything
```

Requires Python 3.10+.

## Quickstart

```bash
# 1. Create a project
surroforge init my-study
cd my-study

# 2. Register a dataset (schema validated, content hashed)
surroforge data add ../samples.csv --name beam

# 3. Describe a training run in YAML (configs/kriging.yaml below), then train
surroforge train --config configs/kriging.yaml

# 4. Inspect, compare, reproduce
surroforge list
surroforge validate run-001
surroforge compare --runs run-001,run-002
surroforge reproduce run-001

# 5. Use the model
surroforge predict run-001 --input ../new_points.csv --output preds.csv
surroforge export run-001 --format onnx
surroforge report run-001
```

### Example configuration

A single YAML file is the complete specification of a training run (PRD 4.3):

```yaml
experiment: beam-kriging
data:
  source: beam            # a registered dataset name, or a file path
  inputs: [length, width, load]
  outputs: [max_stress]
  split:
    test_fraction: 0.2
    seed: 42
model:
  method: kriging         # kriging | rbf | polynomial
  corr: squar_exp
validation:
  cross_validation: 5     # k-fold; or {folds: 5, shuffle: true}
  metrics: [rmse, r2, mae]
```

## Python API

The API mirrors the CLI one-to-one (PRD 4.10):

```python
from surroforge import Project, Trainer

project = Project.open("my-study")
trainer = Trainer.from_config("configs/kriging.yaml")
run = trainer.fit(project.dataset("beam"))
print(run.metrics)
```

## Methods (Core / MVP)

| Method | `model.method` | Backend |
|---|---|---|
| Kriging | `kriging` | SMT |
| Radial Basis Functions | `rbf` | SMT |
| Polynomial regression | `polynomial` | scikit-learn |

Cross-validation is k-fold; standard metrics are RMSE, R², and MAE.

## Reproducibility

Every run records its full configuration, a SHA-256 hash of the data, the random
seed, and the exact Python/library versions in `runs/<id>/lineage.json`. A
prediction fingerprint lets `surroforge reproduce` verify that a re-run produced a
bit-identical model.

## Extending SurroForge

New methods, exporters, and reporters are registered through the
`surroforge.trainers`, `surroforge.exporters`, and `surroforge.reporters`
entry-point groups (PRD 4.11). The built-in methods use exactly this mechanism.

## License

Apache License 2.0. See [LICENSE](LICENSE).
