# CLI reference

Every command mirrors a Python API operation one-to-one. Domain errors are printed
as a clean `Error: …` line (no traceback) and exit with status 1. Run
`surroforge --help` or `surroforge <command> --help` for the live signature.

## Global options

Pass these before the command:

| Option | Description |
|---|---|
| `-v`, `--verbose` | Show detailed per-step progress. |
| `-q`, `--quiet` | Suppress non-error output. |
| `--config-dir PATH` | Project directory (default: discovered by walking up from the cwd). |

## Project & data

### `surroforge init <name>`
Create a new project directory (`data/`, `configs/`, `runs/`, `reports/`, and
`surroforge.yaml`).

### `surroforge data add <file> [-n NAME]`
Register a dataset: validate the schema, record a content hash, and copy the file
into `data/`. `--name/-n` defaults to the file stem. Accepts CSV (and Parquet/HDF5
with the `formats` extra).

### `surroforge data list`
List registered datasets with row/column counts and paths.

## Train & inspect

### `surroforge train -c <config.yaml> [--onnx]`
Train a surrogate from a YAML [configuration](configuration.md). Prints the new run
id and a train/test/CV metrics summary. `--config/-c` is required. The model is
always saved in the run store; ONNX export is **opt-in** with `--onnx` (it pulls in
and runs `skl2onnx`, the dominant cost of a cheap method) — otherwise produce it
later on demand with `surroforge export <run> --format onnx`. Model export is a
Professional feature: without a license, `--onnx` logs `ONNX export skipped: …`
and the run still completes.

### `surroforge list`
List all runs in the project (id, method, created-at, test RMSE).

### `surroforge validate <run_id>`
Re-print a run's validation metrics.

### `surroforge info <run_id>`
Show a run's full lineage — experiment, method, created-at, data source, config and
data hashes, seed, prediction hash, Python/SurroForge versions, parent run, whether
an ONNX artifact exists, and dependencies — plus the metrics summary. The provenance
block (hashes, seed, versions, dependencies) needs a Professional license; identity
fields and metrics stay free.

## Use the model

### `surroforge predict <run_id> -i <input> [-o <output>]`
Predict with a trained model. `--input/-i` is a data file with the run's input
columns. With `--output/-o` (`.csv`/`.parquet`) predictions are written there;
otherwise they print to stdout.

### `surroforge export <run_id> [-f FORMAT] [-o OUTPUT]`
Export a run's model — a **Professional** feature (licensed). `--format/-f` is
`onnx` (default), `pickle`, or a plugin format such as `pt`; ONNX only for methods
that declare it (e.g. polynomial, mlp). An unsupported method/format pair fails
with a clear message. The run store's own `model.pkl` is written at train time
regardless — predict never needs an export.

### `surroforge report <run_id> [-f FORMAT] [-o OUTPUT]`
Generate a report (default `markdown`) under `reports/`.

## Reproduce

### `surroforge reproduce <run_id>`
Re-run a stored run from its config snapshot and verify reproducibility. A
Professional feature (the recording happens on every run). Succeeds on
`reproduced` / `reproduced_within_tolerance`; exits 1 on `diverged` / `data_changed`.
See [Reproduction](../scenarios/reproduction.md).

## Misc

### `surroforge version`
Print the installed SurroForge version.
