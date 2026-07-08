# Changelog

All notable changes to SurroForge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — Editionen (Breaking)

- **`mlp`, `random-forest`, `gradient-boosting` sind Professional-Methoden** und leben jetzt
  im `surroforge_professional`-Plugin (Entry-Points statt Core-Builtins). Auflösung ohne
  Lizenz schlägt mit einem klaren `PluginError` fehl; `SURROFORGE_PRO_LICENSE` (oder
  `~/.surroforge/professional.lic`) schaltet frei. Core behält Kriging, RBF, Polynom und
  `reduced-field` (mit Nicht-NN-Inner).
- **`torch-mlp` + `pt`-Exporter** sind in `surroforge_professional` aufgegangen (optionales
  Extra: `pip install 'surroforge_professional[torch]'`); das separate `surroforge-torch`-Paket
  entfällt.
- **`autoencoder-field` ist eine Enterprise-Methode** im neuen `surroforge_enterprise`-Paket;
  eine Professional-Lizenz schaltet sie nicht frei (Stufen: professional < enterprise). Das
  separate `surroforge-autoencoder`-Paket entfällt.
- **Plugin-Seam (additiv, API v1 unverändert):** Plugins können `required_tier` deklarieren;
  Editionen registrieren Authorizer über die neue Entry-Point-Gruppe `surroforge.authorizers`.
- **Hinweis:** bestehende lokale Run-Stores mit `mlp`/`torch-mlp`-Modellen (Pickle-Modulpfade)
  müssen neu trainiert werden; die Modulpfade haben sich durch den Umzug geändert.

### Added — Python API

- **`Run.export(fmt, output=None)`** — an object-level binding of
  `surroforge.services.export_model`, so the returned run exports itself the way it already
  loads itself: `run = trainer.fit(project); run.export("onnx", "model.onnx")`. It mirrors the
  `surroforge export` CLI, carries the declaration-driven format gate (an unsupported
  method/format pair raises `ExportError`), and reaches plugin formats such as `pt`. The NACA
  examples now export through `run.export(...)` instead of calling the service function directly.
- **Extended run comparisons** — `surroforge compare` now prints a ranking table
  (train/test/CV mean±std with deltas to the best), an all-test-metrics table with the
  best run per metric marked, a configuration/hyperparameter **diff** (only the fields
  that differ between runs, derived from each run's canonical config), and a **lineage**
  block flagging whether the runs share the same data. Heterogeneous comparisons (different
  metric sets, datasets, experiments, or columns) emit warnings rather than failing. The
  structured result is available via `surroforge.comparison.build_comparison(runs, metric=...)`
  / `services.compare_runs(...)`. Run selection is unchanged (`--runs a,b,c`).

### Added — Methods

- **`gradient-boosting` method** — a tree-ensemble surrogate backed by scikit-learn's
  `HistGradientBoostingRegressor`, the boosting counterpart to `mlp`. It is a Core builtin
  (scikit-learn is already Core → **no new dependency**), needs no input scaling, supports
  single- and multi-output targets (the latter via `MultiOutputRegressor`, one booster per
  target), runs with `early_stopping=False` for a deterministic small-DOE fit, is persisted
  via **pickle**, passes the `check_trainer` conformance suite, and works as a `reduced-field`
  inner. Select it with `model.method: gradient-boosting`; hyperparameters: `learning_rate`,
  `max_iter`, `max_leaf_nodes`, `max_depth`, `min_samples_leaf`, `l2_regularization`,
  `random_state`. (ONNX export is not offered: the current `skl2onnx`/`onnx` stack cannot
  convert tree-ensemble regressors.)
- **`random-forest` method** — a bagged tree-ensemble surrogate backed by scikit-learn's
  `RandomForestRegressor`, completing the Phase 2 extended methods (alongside `mlp`,
  `kriging`, and `gradient-boosting`). It is a Core builtin (scikit-learn is already Core →
  **no new dependency**), needs no input scaling, is **natively multi-output** (no
  `MultiOutputRegressor` wrapper), exports to **pickle and ONNX** (skl2onnx — the
  RandomForest converter is unaffected by the `HistGradientBoosting` converter issue that
  kept `gradient-boosting` pickle-only), passes the `check_trainer` conformance suite, and works
  as a `reduced-field` inner. A fixed integer `random_state` makes it reproducible. Select it
  with `model.method: random-forest`; hyperparameters: `n_estimators`, `max_depth`,
  `max_leaf_nodes`, `min_samples_leaf`, `max_features`, `random_state`.

### Added — Reports

- **Interactive HTML visualizations** — the `surroforge-professional` `html` reporter now
  embeds four interactive charts: a **parity** plot (predicted vs. actual), a **residual**
  plot, **per-fold cross-validation** values, and a **train/test/CV-mean** metric
  comparison. Rendered with **uPlot** (MIT, ~51 KB) vendored and inlined — the report stays
  a single self-contained file with **no network/external assets** and remains XSS-safe
  (chart data is numeric and `<`-escaped in its JSON block). Multi-output models get a
  dropdown to switch the parity/residual plots between output columns; hovering shows exact
  values, drag-zoom and double-click-reset are available. Charts degrade gracefully: runs
  trained before this release (no holdout actuals) and field/ROM runs simply omit the
  parity/residual plots while keeping the CV and comparison charts. To support the parity
  plot, Core now persists out-of-sample **holdout actuals** alongside the prediction
  fingerprint, exposed as **`Run.holdout_actual`** — additive lineage that feeds no hash
  and does not affect `reproduce()`.
- **Interactive HTML comparison report** — `surroforge compare --runs a,b,c --format html`
  now renders the extended comparison as a **single self-contained HTML file**
  (`reports/comparison.html` by default; `--output/-o` overrides, `--format terminal`
  stays the unchanged default). The page carries a best-run callout (with per-metric
  winners when they disagree), a prominent **warnings banner** for apples-to-oranges
  comparisons, the ranking table, **one interactive uPlot chart per metric** (layered
  train/test/CV-mean bars per run with **±std whiskers** and the per-metric best marked),
  the full train/test/CV table, the configuration **diff**, and the lineage block. Same
  guarantees as the single-run report: uPlot vendored inline, no network/external assets,
  Jinja autoescape + `<`-escaped JSON (XSS-safe), license-gated. Core gained the seam this
  plugs into: a `BaseComparisonReporter` contract on the new
  `surroforge.comparison_reporters` entry-point group and
  `services.generate_comparison_report(...)` — third parties can register their own
  comparison formats the same way.

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
- **`surroforge-professional` plugin** (`plugins/surroforge-professional/`, separate distribution):
  the first **Professional-edition** plugin and the first real use of the Core↔commercial seam — it
  registers an `html` reporter on Core's `surroforge.reporters` entry point only, so Core still ships
  Markdown alone and gains `html` only while the plugin is installed. Rendering is gated by a **local
  Professional license** (`SURROFORGE_PRO_LICENSE`, or `~/.surroforge/professional.lic`); a missing
  licence raises a `PluginError` with an actionable message before any run data is read, and there is
  no online activation in v1. The reporter renders a **single self-contained HTML file** — embedded
  CSS, **no JavaScript, no external/network assets**, and every dynamic value autoescaped (XSS-safe).
  Scalar reports carry an Executive Summary, a Quality Snapshot of metric cards, Data And Model,
  Validation Details, and a Reproducibility Appendix (run id, experiment, method, data/config hashes,
  prediction fingerprint, train/test/CV metrics, hyperparameters, dependencies, the full config YAML,
  and the `surroforge reproduce <run-id>` line). A **Field-ROM Details** section (the two-source
  truncation-vs-surrogate error split) appears only when complete `energy_l2`/`truncation_error`
  metrics and lineage are present. Produced via `surroforge report <run> --format html` or
  `generate_report(project, run, fmt="html")` (written to `reports/<run-id>.html`); install with
  `pip install ./plugins/surroforge-professional`.
- **`surroforge-autoencoder` plugin** (`plugins/surroforge-autoencoder/`, separate distribution):
  a **PyTorch autoencoder** field surrogate registered as the `autoencoder-field` method through the
  `surroforge.trainers` entry point only — **never** Core's `_BUILTIN`, so Core stays PyTorch-free.
  It is the **nonlinear counterpart to `reduced-field`**: instead of a linear POD basis it trains an
  encoder/decoder MLP to learn a nonlinear latent representation of the (standardised) field
  snapshots, then fits an ordinary SurroForge **inner trainer** (default `mlp`, e.g. `torch-mlp`) to
  map inputs such as `(aoa, speed)` to latent coordinates; prediction runs input → inner model →
  latent → decoder → field. Hyperparameters cover `latent_dim`, `encoder_hidden_layer_sizes`
  (the `decoder_hidden_layer_sizes` default to the reverse), `activation`, `alpha` (weight decay),
  `max_iter`, `tol`, `learning_rate_init`, `random_state`, and the nested `inner` spec; field
  mean/std standardisation is baked into the model. Deterministic by construction (seeded weight
  init that restores the global RNG state; honest float32 `reproduction_tolerance = 1e-4`) and
  exports to pickle. Apache-2.0 (PyTorch is BSD-3 → NFR8 holds); install with
  `pip install ./plugins/surroforge-autoencoder`.
- **DoE generator (`surroforge doe`)** — the `surroforge-professional` plugin adds a
  design-of-experiments generator on a new generic Core command seam (entry-point group
  `surroforge.commands`, the first plugin-contributed CLI command). `surroforge doe run
  spec.yaml` reads a Pydantic-validated YAML spec (numeric continuous/discrete and
  categorical factors), samples it by **full-factorial**, **Latin Hypercube**, **Sobol**,
  or **random** (LHS/Sobol via the optional `scipy` extra `surroforge-professional[doe]`,
  imported lazily), and writes `samples.csv` (a `case` column plus one column per factor)
  plus a reproducibility `doe.manifest.json`. For **numeric factors only**
  (continuous/discrete), the CSV is directly consumable by
  `surroforge data add-openfoam --samples`; specs with **categorical** factors produce
  string columns that ingest rejects — encode them before passing to that command.
  License-gated like the HTML reporter; deterministic per `seed`.

### Added — Examples

- **`examples/09_naca0012_openfoam/`** — a real-CFD NACA 0012 study (not CI-tested): `run_sweep.py` drives
  `simpleFoam` over a 2-D `angle × speed` DoE on the validated `airFoil2D` mesh (rotating the
  freestream + `forceCoeffs`, `--angles`/`--speeds`), and `ingest_and_train.py` builds the same
  surrogates — `(AoA, speed) → field` (pressure **and** velocity) and `(AoA, speed) → [Cl, Cd]` —
  from the genuine fields and the real `Cl`/`Cd`. Convergence is checked, not assumed: the sweep
  pins `SIMPLE/residualControl` (`--residual`) so `simpleFoam` stops at convergence and reports
  any case that hit the iteration cap, and ingestion flags any case whose `Cl` is still swinging
  over its last writes. The DoE is embarrassingly parallel, so `run_sweep.py --jobs N` solves N
  independent cases concurrently (near-linear speed-up to the core count — a 100-case sweep ran
  in ~1 min at `--jobs 12`), and `--vtk` additionally writes each case's converged field to
  `case/VTK/` (`foamToVTK`) for ParaView, inline with the sweep. After training,
  `ingest_and_train.py` now evaluates the best field surrogates on every original `samples.csv`
  case and writes a single VTKHDF time series `surrogate_vtk/surrogate_fields.vtkhdf`, using the
  original VTU meshes as the geometry source and writing the surrogate pressure as cell field `p`
  and the surrogate velocity as vector cell field `U`. Verified against OpenFOAM v2512.

### Changed

- **`surroforge-professional[doe]` now requires `scipy>=1.15`** (was `>=1.10`): the qmc
  engines (LHS/Sobol) are seeded via SciPy's SPEC-7 `rng=` parameter instead of the legacy
  `seed=` alias slated for removal. Same streams for the same integer seeds — existing
  designs reproduce unchanged.
- **NACA real-CFD example slimmed** — the VTK/VTKHDF I/O of
  `examples/09_naca0012_openfoam/ingest_and_train.py` now lives in a reusable, SurroForge-free
  sibling module `surrogate_vtk.py`, the dead per-case `.vtu`/`.pvd` export path was
  removed, and the directory's README absorbed the full verified reference (case setup,
  DoE, exact commands, fit numbers from the real 110-case run). Behaviour is unchanged —
  re-verified end to end against the on-disk sweep.

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
