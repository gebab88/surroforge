# Python API reference

The Python API mirrors the CLI with the same vocabulary, and is a first-class
interface suitable for embedding in larger pipelines. The two entry points are
re-exported at the top level:

```python
from surroforge import Project, Trainer
```

## End-to-end

```python
from surroforge import Project, Trainer

project = Project.open("beam-study")
trainer = Trainer.from_config("configs/kriging.yaml")
run = trainer.fit(project.dataset("beam"))   # or trainer.fit(project)
print(run.run_id, run.metrics["test"])
```

## `Project`

The on-disk workspace holding data, configs, runs, and reports.

**Construction**

| Method | Description |
|---|---|
| `Project.init(root, name, description=None)` | Create a new project (raises `ProjectError` if one already exists). |
| `Project.open(root)` | Open an existing project (raises `ProjectError` if none at `root`). |
| `Project.discover(start=None)` | Walk up from `start` (default: cwd) to the enclosing project. |

**Datasets**

| Method | Description |
|---|---|
| `add_dataset(file, name=None)` | Register, validate, hash, and copy a dataset into `data/`; returns a `Dataset`. |
| `datasets()` | List registered `Dataset` objects. |
| `dataset(name)` | Look up a registered dataset by name (raises `DataError` if unknown). |

**Runs**

| Method | Description |
|---|---|
| `runs()` | All runs, ordered by run number. |
| `run(run_id)` | Load one run (raises `RunNotFoundError` if missing). |

**Properties:** `root`, `name`, `metadata`, `data_dir`, `configs_dir`, `runs_dir`,
`reports_dir`.

## `Trainer`

A training configuration bound to its method, ready to fit.

| Method | Description |
|---|---|
| `Trainer.from_config(path)` | Build from a YAML config file. |
| `Trainer.from_dict(data)` | Build from an already-parsed mapping. |
| `fit(target, *, export_onnx=False)` | Train and persist a run. `target` is a `Project` (data taken from the config) or a `Dataset` from `project.dataset(name)` (overrides `data.source`). ONNX export is opt-in (`export_onnx=True`) and needs a Professional license (unlicensed, the export is skipped and the run completes); otherwise produce it later via `export_model`. Returns a `Run`. |

## `Run`

Returned by `Trainer.fit` and `Project.run`. Read-only accessors over the run
directory and its `lineage.json`:

| Member | Description |
|---|---|
| `run_id` | The run's id (e.g. `run-001`). |
| `method` | The method used. |
| `metrics` | Dict of train / test / cross_validation metrics and shapes. |
| `config` | The parsed `ExperimentConfig` snapshot. |
| `created_at`, `seed`, `config_hash`, `data_hash` | Lineage fields. |
| `holdout_fingerprint` | Out-of-sample predictions recorded for the reproducibility verdict. |
| `parent_run` | The id this run reproduces, if any. |
| `lineage` | The full lineage mapping (raw data stays readable; the gated surfaces are `reproduce` and the aggregated views). |
| `load_model()` | Unpickle the fitted trainer (emits a one-time `ProvenanceWarning` for non-local artifacts). |
| `has_onnx()` | Whether an ONNX artifact exists for the run (written by `fit(export_onnx=True)` or a later `export`). |

## `surroforge.services`

High-level verbs shared by the CLI and API:

| Function | Description |
|---|---|
| `reproduce(project, run_id) -> ReproduceResult` | Re-run from the config snapshot and compare fingerprints. Professional (licensed). |
| `assert_reproducible(result, *, tolerance=None)` | Raise `ReproducibilityError` unless verifiably reproduced (uses the run's declared tolerance by default). |
| `compare_runs(project, run_ids, *, metric="rmse") -> dict` | Rank runs by a metric with per-run train/test/CV, best-per-metric, a config/hyperparameter diff, a lineage summary, and warnings (`runs`/`best`/`config_diff`/`lineage_summary`/`warnings`). Without a Professional license the result omits `lineage`/`lineage_summary`. |
| `predict_with_run(run, frame) -> DataFrame` | Predict for an in-memory dataframe. |
| `predict_from_file(run, path) -> DataFrame` | Predict for a data file. |
| `export_model(run, fmt, output=None) -> Path` | Export to `pickle`/`onnx`/a plugin format — Professional (licensed); clear `ExportError` for unsupported pairs. |
| `generate_report(project, run, fmt="markdown", output=None) -> (Path, str)` | Render and write a report. |

`ReproduceResult` carries `verdict`, `bit_identical`, `max_abs_diff`, `tolerance`,
`data_hash_match`, and the `parent`/`reproduction` runs.

## Errors

All domain errors subclass `surroforge.errors.SurroForgeError`, so you can catch the
base or a specific type:

`ProjectError`, `ConfigError`, `DataError`, `TrainerError`, `PluginError`,
`ExportError`, `RunNotFoundError`, `ReproducibilityError`, `ConformanceError`.

## Extending SurroForge

New methods, exporters, and reporters register through the `surroforge.trainers`,
`surroforge.exporters`, and `surroforge.reporters` entry-point groups — the built-ins
use exactly this mechanism. A trainer subclasses `surroforge.trainers.base.BaseTrainer`
and can self-verify against the shipped conformance suite:

```python
from surroforge.conformance import check_trainer
from my_package import MyTrainer

check_trainer(MyTrainer).raise_for_status()   # raises ConformanceError on a violation
```

Plugins may declare an `api_version` (checked at resolution) and their
`export_formats` / `reproduction_tolerance` so Core's export dispatch and
reproducibility verdict handle them without Core changes.
