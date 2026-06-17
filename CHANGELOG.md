# Changelog

All notable changes to SurroForge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Python API

- **`Run.export(fmt, output=None)`** — an object-level binding of
  `surroforge.services.export_model`, so the returned run exports itself the way it already
  loads itself: `run = trainer.fit(project); run.export("onnx", "model.onnx")`. It mirrors the
  `surroforge export` CLI, carries the declaration-driven format gate (an unsupported
  method/format pair raises `ExportError`), and reaches plugin formats such as `pt`. The NACA
  examples now export through `run.export(...)` instead of calling the service function directly.

### Added — Plugins

- **`surroforge-torch` plugin** (`plugins/surroforge-torch/`, separate distribution): a
  **PyTorch** neural-network surrogate registered as the `torch-mlp` method through the
  `surroforge.trainers` entry point only — **never** Core's `_BUILTIN`, so Core stays
  PyTorch-free. It is the first heavyweight-framework method and the first real exercise of
  the plugin seam. You switch between scikit-learn and PyTorch by changing one config line
  (`model.method: mlp` ↔ `torch-mlp`) — `torch-mlp` mirrors `mlp`'s hyperparameter surface.
  Deterministic by construction (own integer `random_state`, seeded weight init, full-batch
  training; honest float32 `reproduction_tolerance = 1e-4`), exports to **ONNX** (torch's
  built-in exporter, standardisation baked into the graph) **and to a native PyTorch `.pt`
  checkpoint**, passes the shipped `check_trainer` conformance suite, and works as a
  `reduced-field` inner (POD-NN). PyTorch is BSD-3 → NFR8 holds; install with
  `pip install ./plugins/surroforge-torch`. See `examples/08_torch_mlp.py` for the one-line
  `mlp` ↔ `torch-mlp` switch.
- **`pt` export format** — the plugin registers a native PyTorch checkpoint exporter on Core's
  `surroforge.exporters` entry-point seam (so Core gains the `pt` format only while the plugin
  is installed). `torch-mlp.to_pt()` writes a `state_dict` + architecture checkpoint that loads
  under the safe `torch.load(..., weights_only=True)` path; `surroforge export --format pt` (or
  `export_model(run, "pt")`) produces it, and a non-torch method refuses `pt` with a clear error.

### Added — Examples

- **`examples/09_naca0012_openfoam_surrogate.py`** — a NACA 0012 study over **angle of attack and
  freestream speed** end to end: ingest a 2-D OpenFOAM DoE (`read_openfoam_sweep`), then train
  **two** surrogates — a `reduced-field` POD-NN of the pressure field (`p = Cp(AoA)·q(U)`, so speed
  scales it via the dynamic pressure) and a scalar `(AoA, speed) → [C_L, C_D]` model (both
  coefficients carry the stall, with a Reynolds stall drift). The stall makes the neural surrogate
  beat a quadratic on both; `torch-mlp` joins when the plugin is present. Self-contained/CI-safe (it
  fabricates the sweep in genuine OpenFOAM format, no solver). Exports one artifact per format —
  the best coefficient model to **ONNX**, the `torch-mlp` model to a native **`.pt`** checkpoint,
  the field model to pickle — degrading gracefully when a backend is absent. Pass `--out DIR` to
  keep them (scratch stays in a temp dir, so the default run leaves nothing behind and re-runs never
  collide).
- **`examples/naca0012_openfoam/`** — the real-CFD companion (not CI-tested): `run_sweep.py` drives
  `simpleFoam` over a 2-D `angle × speed` DoE on the validated `airFoil2D` mesh (rotating the
  freestream + `forceCoeffs`, `--angles`/`--speeds`), and `ingest_and_train.py` builds the same
  surrogates — `(AoA, speed) → field` and `(AoA, speed) → [Cl, Cd]` — from the genuine fields and
  the real `Cl`/`Cd`. Convergence is checked, not assumed: the sweep pins `SIMPLE/residualControl`
  (`--residual`) so `simpleFoam` stops at convergence and reports any case that hit the iteration
  cap, and ingestion flags any case whose `Cl` is still swinging over its last writes. Verified
  against OpenFOAM v2512.

## [0.2.0] — 2026-06-15

First public release. SurroForge — the open-source **Core** (Apache 2.0) — takes you from a
solver's output table to a validated, exportable surrogate model, with the data, configuration,
and model cleanly versioned and reproducible. This release contains the Phase-1 MVP, **Field ROM
v1** (reduced-order field models), Phase-2 **solver-native ingestion** (OpenFOAM), **and** a first
**neural-network method** (`mlp`).

### Added — Solver-native ingestion (Phase 2)

- **OpenFOAM field ingestion** (`surroforge.data.read_openfoam_sweep`,
  `read_openfoam_field`, `read_openfoam_dimensioned_scalar`, `latest_time_field`): read an
  OpenFOAM case sweep's ASCII `internalField` (scalar and vector, `nonuniform`/`uniform`)
  straight into the wide `(n_samples × n_dof)` snapshot matrix the `reduced-field` method
  trains on — **mesh-free** (no coordinates/connectivity) and dependency-free (pure-Python
  parsing on existing Core deps). See `examples/05_openfoam_field_rom.py`.
- **`surroforge data add-openfoam <case-root>`**: a CLI command that ingests an OpenFOAM case
  sweep (`--field`, `--samples` DoE table, `--name`, `--output`, `--time`) and registers it as a
  dataset in one step — a thin wrapper over `read_openfoam_sweep`, same lineage/hash as `data add`.
- **Binary/mesh field reading** (`surroforge.data.read_mesh_field`) via `meshio`, behind the
  `surroforge[rom]` extra (now `meshio>=5.3`, MIT-licensed): reads a named point/cell field from
  any `meshio`-supported format, with a lazy import + `pip install 'surroforge[rom]'` hint.

### Added — Projects, data & training

- `surroforge init` to create a project; `surroforge data add` to register a dataset with
  schema validation and a SHA-256 content hash; `surroforge list` to enumerate datasets and runs.
- A single YAML file as the complete specification of a training run (`data`, `model`,
  `validation`). `surroforge train --config …` trains from it.
- k-fold cross-validation with the standard metrics **RMSE, R², MAE**.
- Four built-in methods: **polynomial** (scikit-learn), **kriging** and **rbf** (SMT backend), and
  **mlp** — a neural network (multi-layer perceptron via scikit-learn's `MLPRegressor`, input
  `StandardScaler`, `lbfgs` by default). It adds **no new dependency**, fits multi-output targets
  natively, declares an honest reproduction tolerance for its iterative optimiser, and **exports to
  ONNX** via `skl2onnx` — the second ONNX-capable method after `polynomial`. As an ordinary
  `BaseTrainer` it also serves as a `reduced-field` inner surrogate (POD-NN: `model.inner.method:
  mlp`). A heavyweight framework (PyTorch/DeepONet/FNO) stays available on the external plugin seam.
  See `examples/06_mlp_surrogate.py` (scalar) and `examples/07_pod_nn_field.py` (POD-NN field).

### Added — Compare, predict, export & report

- `surroforge compare` to rank runs and identify the best; `surroforge predict` for out-of-sample
  inputs; `surroforge export` to **ONNX** or **pickle**; `surroforge report` for a Markdown summary.

### Added — Reproducibility & lineage

- Every run records its full configuration, a SHA-256 data hash, the random seed, and exact
  Python/library versions in `runs/<id>/lineage.json`.
- `surroforge reproduce` re-trains from config and returns a trustworthy verdict via an
  out-of-sample prediction fingerprint (it never unpickles the model — no code-execution risk).
- A one-time provenance warning when loading a non-local model artifact (`predict`/`export`).

### Added — Field ROM v1 (reduced-order field models)

- A new **`reduced-field`** method: POD reduction (energy- or rank-truncated SVD) of a field —
  "output columns that move together" — with a per-mode inner surrogate. **Mesh-free**: no
  coordinates or connectivity are read. The inner method is any registered trainer
  (kriging/rbf/polynomial/mlp all work unchanged). Configured via a `model.reduction` block
  (`basis: pod`, `energy`/`rank`/`rank: adaptive`) and `model.inner`.
- **Field-primary reproducibility fingerprint**: sign/order canonicalization with degeneracy
  handling; the scalar (non-field) path stays byte-identical; probe-coverage recorded in lineage.
- **Field validation & report**: `energy_l2` and `truncation_error` as registered metrics; a
  two-source error split (truncation floor vs. surrogate error — "add modes or add data"); and
  honest-basis cross-validation (a fresh basis per fold).
- **Prediction**: modal coefficients by default, full-field reconstruction on `--field`, with
  lazy content-addressed basis storage (the run dir stays smaller than the raw data).
- **Introspection**: `surroforge rom inspect` (energy spectrum, retained rank, probe coverage).
- **Conformance**: a `coeffs→field→coeffs` reconstruction round-trip check and a no-mesh
  (DOF-permutation equivariance) check in the shipped conformance suite.
- The **`surroforge[rom]`** packaging extra (lazy imports; v1 adds no new heavy dependency).

### Added — Python API & extensibility

- A Python API that mirrors the CLI one-to-one (`Project`, `Trainer`, …).
- Plugin registration through the `surroforge.trainers`, `surroforge.exporters`, and
  `surroforge.reporters` entry-point groups (the built-ins use the same mechanism).
- A shippable, framework-agnostic conformance suite (`check_trainer`) so a plugin can
  self-certify against the `BaseTrainer` contract.
- Methods declare their reproducibility tolerance and export capabilities; a Core↔commercial
  plugin seam is reserved (shape-only, a documented no-op gate in Core).

### Added — CLI ergonomics, quality & distribution

- Consistent global CLI options; errors-as-a-feature (clear messages and actionable install
  hints, not tracebacks); graceful degradation when an optional backend is absent.
- Enforced architectural and quality gates (ruff, mypy, import-linter) and a cross-OS/Python CI
  matrix (Ubuntu/macOS/Windows × Python 3.10–3.12), plus reproducibility and license-hygiene
  (no GPL/LGPL in Core) jobs.
- A MkDocs Material documentation site under `docs/`.
- Optional extras: `smt`, `onnx`, `formats`, `rom`, `all` (and `dev`, `docs`).

[Unreleased]: https://github.com/gebab88/surroforge/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/gebab88/surroforge/releases/tag/v0.2.0
</content>
