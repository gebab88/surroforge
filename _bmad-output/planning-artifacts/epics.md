---
stepsCompleted: ["step-01-validate-prerequisites", "step-02-design-epics", "step-03-create-stories", "step-04-final-validation"]
status: 'complete'
completedAt: '2026-06-04'
inputDocuments:
  - PRD.md
  - _bmad-output/planning-artifacts/architecture.md
  - _bmad-output/project-context.md
  - SurroForge-Tech-Spikes.md
  - README.md
  - "SurroForge-UX-Briefing.md (requirements captured as UX-DR1–8; source file since removed from repo root)"
scope: "Phase 1 (MVP) — SurroForge Core (PRD section 9.1)"
note: "Phase 1 is already implemented (working CLI, 121 passing tests, incl. a corrected out-of-sample reproducibility fingerprint and CI boundary gates). These epics/stories are a structured, traceable backlog and acceptance baseline, reconciled against the completed architecture.md."
---

# SurroForge - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for **SurroForge**, decomposing the **Phase 1 (MVP / Core)** requirements from the PRD, the UX Design Briefing, the formal Architecture (`architecture.md`), and the AI-agent rules (`project-context.md`) into implementable stories. Because Phase 1 is already implemented, most stories are **brownfield**: they document and pin existing, tested behaviour as an acceptance baseline, and a minority capture the architecture's remaining/corrective work.

Out of scope here (Phase 2/3 per PRD 9.2/9.3): advanced methods (GP, RF, Gradient Boosting, NN), HTML/PDF reports, multi-output models, Active Learning, Multi-Fidelity, DoE generation, and the licensing/monetisation mechanics.

## Requirements Inventory

### Functional Requirements

FR1: Initialize a new project directory via `surroforge init <name>`, creating the standard layout (`data/`, `configs/`, `runs/`, `reports/`, `surroforge.yaml`). [PRD 4.2, 7.1]
FR2: Register a dataset via `surroforge data add <file>` — validate the schema, compute a content hash, copy it into `data/`, and record it in project metadata. [PRD 4.2, 4.4, 7.3]
FR3: List registered datasets via `surroforge data list`. [PRD 4.2]
FR4: Train a surrogate model from a YAML configuration via `surroforge train --config <yaml>`. [PRD 4.2, 4.3]
FR5: Validate a run and display its metrics via `surroforge validate <run-id>`. [PRD 4.2]
FR6: Compare multiple training runs side by side via `surroforge compare --runs <ids>`. [PRD 4.1, 4.2]
FR7: Predict with a trained model via `surroforge predict <run-id> --input <file>`. [PRD 4.2]
FR8: Export a run's model via `surroforge export <run-id> --format <fmt>` (ONNX, pickle). [PRD 4.2, 4.4]
FR9: Generate a Markdown report for a run via `surroforge report <run-id>`. [PRD 4.2, 4.9]
FR10: List all runs in a project via `surroforge list`. [PRD 4.2]
FR11: Reproduce a run identically via `surroforge reproduce <run-id>`. [PRD 4.2, 4.5]
FR12: Show detailed run information via `surroforge info <run-id>`. [PRD 4.2]
FR13: Treat a single YAML file as the complete and only specification of a run, validated with Pydantic (required fields, optional fields with documented defaults). [PRD 4.3]
FR14: Accept tabular inputs in CSV, Parquet, and HDF5 with named input/output columns. [PRD 4.4]
FR15: Persist every run with a unique id, full config snapshot, SHA-256 data hash, random seed, Python/library versions, train/test/CV metrics, and the model artifact. [PRD 4.5, 7.2]
FR16: Provide three Core methods — Kriging (SMT), RBF (SMT), and polynomial regression (scikit-learn). [PRD 4.6]
FR17: Provide k-fold cross-validation with configurable k and standard metrics RMSE, R², MAE. [PRD 4.6]
FR18: Emit model outputs — pickle and ONNX (where supported), metadata as JSON, config snapshot as YAML. [PRD 4.4]
FR19: Provide a full Python API parallel to the CLI (`Project`, `Trainer`) using the same concepts and vocabulary. [PRD 4.10]
FR20: Expose plugin interfaces via entry points — `surroforge.trainers`, `surroforge.exporters`, `surroforge.reporters`. [PRD 4.11]
FR21: Support global CLI options `--verbose/-v`, `--quiet/-q`, `--config-dir`, `--help`. [PRD 4.2]

### NonFunctional Requirements

NFR1: Performance — train a Kriging model on 200 points / 5 inputs in under 5 s; hash datasets up to 100 MB in under 5 s; k=5 CV slows training by at most ~6×. [PRD 5.1]
NFR2: Reproducibility — every run is fully reproducible at identical library versions; config, data hash, seed, and versions are captured automatically with no opt-out. [PRD 5.2]
NFR3: Platform support — Linux, macOS, Windows; Python 3.10+; distributed via PyPI (`pip install surroforge`). [PRD 5.3]
NFR4: Stability & versioning — Semantic Versioning from 1.0; CLI/API contracts backward-compatible within a major version; deprecation notices one minor ahead. [PRD 5.4]
NFR5: Security & privacy — no outbound network in Core (except an opt-in update check); no telemetry without opt-in. [PRD 5.5]
NFR6: Language — CLI/API output and documentation primarily English. [PRD 5.6]
NFR7: Adoption metrics — time-to-first-trained-model under 30 min from install; reproduction success > 95 % (measured in CI); docs sustain > 80 % of first-time users unaided. [PRD 10.2]
NFR8: Licensing of dependencies — Core depends only on permissively licensed packages (Apache/MIT/BSD); no GPL/LGPL. [PRD 6.3]
NFR9: Error UX — helpful, tested error messages with concrete fix suggestions; no tracebacks for domain errors; precise, sober tone; no emojis or marketing in output. [UX 5, 7]

### Additional Requirements

(From the formal Architecture `architecture.md` and the AI-agent rules `project-context.md`. Status: ✅ implemented & tested · ⏳ remaining Phase-1 / Tier-A work, not yet built · ⏸ Phase-2 reservation, shape-only.)

- ✅ **AR1 — Tech stack (PRD 6.1):** Python 3.10+, Typer, YAML+Pydantic, SMT + scikit-learn, Pandas/PyArrow/h5py, SHA-256/JSON/Pickle, skl2onnx/onnxruntime, Jinja2, pytest, ruff + mypy; hatchling build, `src/` layout.
- ✅ **AR2 — Module structure (PRD 6.2):** `cli / core / data / trainers / validators / exporters / reporters / persistence / plugins / api` plus `engine.py`, `services.py`, `errors.py`.
- ✅ **AR3 — Verified dependency direction:** core/engine/services never import the adapters; engine has no back-edge to services; `errors` is a leaf; `plugins/registry.py` is the sole dynamic resolver. Enforced by `.importlinter` contracts + `tests/test_architecture.py`.
- ✅ **AR4 — ONNX export differentiated (Spike 1):** ONNX for scikit-learn methods; SMT Kriging/RBF via pickle, with a clear `ExportError` for unsupported method/format pairs.
- ✅ **AR5 — Reproducibility, corrected (Spike 3 + decision):** fingerprint over an **out-of-sample** held-out sample (not training inputs); `reproduce()` emits a verdict against a single named tolerance; legacy runs report `unverifiable`. Lineage capture automatic / no-opt-out.
- ✅ **AR6 — Data hashing (Spike 4):** canonical in-memory representation (values + column names + dtypes), order-sensitive; CSV acknowledged lossy.
- ✅ **AR7 — Walking skeleton (Spike 5):** end-to-end `init → data add → train → report` path exists; run-id sequential `run-001`; atomic persistence (temp + `os.replace`).
- ✅ **AR8 — CI boundary/quality gates:** GitHub Actions scaffold (OS×Python matrix, ruff, mypy, import-linter, pinned-threading reproducibility, GPL/LGPL license scan); import-linter + AST guard already gate the suite.
- ✅ **AR9 — Edition-agnostic registry (Spike 2):** built-ins register via the same entry points third-party/commercial plugins use; Core runs standalone with graceful degradation (lazy optional-backend imports).
- ⏳ **AR10 — Plugin conformance suite:** a shippable `src/surroforge/conformance/` parametrised contract suite so third-party/commercial plugins can self-verify (Tier-A; not built).
- ⏳ **AR11 — Trainer reproducibility-characteristics declaration** (deterministic vs. tolerance) and **queried exporter-capability** (replacing the binary `supports_onnx`) on `BaseTrainer` (Tier-A; not built).
- **AR12 — ✅ Pickle provenance warning** on loading non-locally-produced artifacts (Story 4.4); **⏳ MkDocs docs site** (PRD §13, Story 6.5).
- ⏸ **AR13 — Core↔commercial seam, shape-only:** `api_version` check at registration, name-collision/precedence policy, post-resolution no-op `gate.check` hook; edition catalog + composed `--help` with upgrade hints. Frozen before the first external plugin binds (Phase-2 precondition).

### UX Design Requirements

(From the UX Design Briefing. SurroForge is CLI-only in the MVP — no GUI — so these are interaction/ergonomics requirements rather than visual-component work.)

UX-DR1: CLI commands grouped by activity (`init`, `train`, `validate`, `compare`, `export`, `report`, `reproduce`, …), with ergonomics comparable to `git`/`dvc`. [UX 5]
UX-DR2: YAML configuration is the single source of truth for each run and is Git-versionable. [UX 5]
UX-DR3: The Python API is a first-class interface, equal to the CLI, suitable for embedding in larger pipelines. [UX 5]
UX-DR4: Output tone is precise, sober, and professional — no emojis, no marketing language. [UX 5]
UX-DR5: Error messages are a feature — curated, tested, and accompanied by concrete fix suggestions. [UX 7]
UX-DR6: Sensible, documented defaults that work for ~80 % of standard cases and are always overridable. [UX 7]
UX-DR7: Standards over convenience — no guessing of column names, no hidden assumptions. [UX 7]
UX-DR8: The three test scenarios must work end-to-end — greenfield (data → validated model quickly), method comparison (one command, unified metrics), and reproduction (config + data only). [UX 6]

### FR Coverage Map

- FR1 (init) → Epic 1 — create project layout
- FR2 (data add) → Epic 1 — register + hash dataset
- FR3 (data list) → Epic 1 — list datasets
- FR4 (train) → Epic 2 — train from YAML
- FR5 (validate) → Epic 2 — show run metrics
- FR6 (compare) → Epic 3 — compare runs, pick best
- FR7 (predict) → Epic 3 — predict on new inputs
- FR8 (export) → Epic 3 — export ONNX/pickle
- FR9 (report) → Epic 3 — Markdown report
- FR10 (list) → Epic 4 — list runs
- FR11 (reproduce) → Epic 4 — reproduce with verdict
- FR12 (info) → Epic 4 — run details
- FR13 (YAML config) → Epic 2 — single-file Pydantic-validated spec
- FR14 (formats) → Epic 1 — CSV/Parquet/HDF5 ingest
- FR15 (run persistence) → Epic 2 — lineage + artifacts
- FR16 (methods) → Epic 2 — Kriging/RBF/polynomial
- FR17 (CV/metrics) → Epic 2 — k-fold + RMSE/R²/MAE
- FR18 (artifacts) → Epic 3 — pickle/ONNX/JSON/YAML outputs
- FR19 (Python API) → Epic 5 — `Project`/`Trainer` parity
- FR20 (plugins) → Epic 5 — entry-point trainers/exporters/reporters
- FR21 (global options) → Epic 6 — `--verbose/-v`, `--quiet/-q`, `--config-dir`, `--help`

## Epic List

### Epic 1: Project & Dataset Foundation
A CAE engineer can create a project and register validated, content-hashed datasets, ready to train on.
**FRs covered:** FR1, FR2, FR3, FR14
**Supports:** AR1, AR2, AR6, AR7, UX-DR7

### Epic 2: Train & Validate Surrogates
From a single YAML config, train a Core method (Kriging, RBF, polynomial), get cross-validated metrics, and a fully persisted, reproducible run.
**FRs covered:** FR4, FR5, FR13, FR15, FR16, FR17
**Supports:** NFR1, AR9, UX-DR2, UX-DR6

### Epic 3: Compare, Predict, Export & Report
Compare runs to pick the best model, predict on new inputs, export to ONNX/pickle, and generate a Markdown report.
**FRs covered:** FR6, FR7, FR8, FR9, FR18
**Supports:** AR4

### Epic 4: Reproducibility & Run Lineage
List and inspect every run, and reproduce any past run with a trustworthy verdict — the headline guarantee.
**FRs covered:** FR10, FR11, FR12
**Supports:** NFR2, NFR7, AR5, AR12, UX-DR8

### Epic 5: Python API & Extensibility
Use SurroForge as a first-class Python library and extend it with new trainers/exporters/reporters via plugins.
**FRs covered:** FR19, FR20
**Supports:** NFR4, AR3, AR9, AR10, AR11, AR13

### Epic 6: CLI Ergonomics, Quality Gates & Distribution
A polished, trustworthy CLI (consistent global options, helpful errors) backed by enforced quality gates and clean packaging.
**FRs covered:** FR21
**Supports:** NFR3, NFR6, NFR8, NFR9, UX-DR1, UX-DR4, UX-DR5, AR8, AR12

> **Story status legend:** ✅ implemented & tested (story is the acceptance baseline) · ⏳ remaining Phase-1 / Tier-A work · ⏸ Phase-2 reservation (shape-only).

## Epic 1: Project & Dataset Foundation

A CAE engineer can create a project and register validated, content-hashed datasets, ready to train on.

### Story 1.1: Initialize a project

As a methods engineer,
I want to create a new SurroForge project with one command,
So that I have a standard, versionable layout for data, configs, runs, and reports.

**Acceptance Criteria:**

**Given** an empty target directory
**When** I run `surroforge init my-study`
**Then** the project layout is created (`data/`, `configs/`, `runs/`, `reports/`) with a `surroforge.yaml` recording project name, schema version, and creation date
**And** re-running `init` on an existing project fails with a clear `ProjectError`, not an overwrite.

**Status:** ✅ implemented (FR1, AR2, AR7)

### Story 1.2: Register a dataset with validation and hashing

As a methods engineer,
I want to register a tabular dataset and have it validated and content-hashed,
So that every later run is pinned to exactly the data it used.

**Acceptance Criteria:**

**Given** a CSV/Parquet/HDF5 file with named columns
**When** I run `surroforge data add <file> --name beam`
**Then** the schema is validated (no column guessing), a SHA-256 is computed over the canonical in-memory representation (values + column names + dtypes), the file is copied into `data/`, and the dataset is recorded in project metadata
**And** the same data re-encoded losslessly (e.g. Parquet) yields the same hash; CSV is acknowledged lossy
**And** an unreadable or schema-invalid file raises a `DataError` with a concrete fix suggestion.

**Status:** ✅ implemented (FR2, FR14, AR6, UX-DR7)

### Story 1.3: List registered datasets

As a methods engineer,
I want to list the datasets registered in my project,
So that I can see what is available to train on.

**Acceptance Criteria:**

**Given** a project with one or more registered datasets
**When** I run `surroforge data list`
**Then** each dataset is shown with its name, hash, and schema summary
**And** an empty project prints a clear "no datasets registered" message, not an error.

**Status:** ✅ implemented (FR3)

## Epic 2: Train & Validate Surrogates

From a single YAML config, train a Core method, get cross-validated metrics, and a fully persisted, reproducible run.

### Story 2.1: Define a run in a single YAML config

As a methods engineer,
I want one YAML file to fully specify a training run,
So that my experiments are explicit, Git-versionable, and reproducible.

**Acceptance Criteria:**

**Given** a YAML file with `experiment`, `data.source/inputs/outputs`, `model.method`, and `validation.cross_validation`
**When** it is loaded
**Then** it is validated with Pydantic; required fields are enforced and optional fields take documented defaults (`split.method=random`, `split.test_fraction=0.2`, `split.seed=42`, `validation.metrics=[rmse,r2,mae]`)
**And** an unknown or malformed field raises a `ConfigError` naming the offending field.

**Status:** ✅ implemented (FR13, UX-DR2, UX-DR6)

### Story 2.2: Train a surrogate from config

As a methods engineer,
I want to train a Core surrogate method from my config,
So that I get a fitted model without writing bespoke code.

**Acceptance Criteria:**

**Given** a registered dataset and a valid config naming `kriging`, `rbf`, or `polynomial`
**When** I run `surroforge train --config configs/kriging.yaml`
**Then** the method is resolved through the plugin registry, a holdout split + final model are fit, and a run is produced
**And** requesting a method whose backend extra is not installed (e.g. `kriging` without `[smt]`) yields an actionable install hint, not an `ImportError`.

**Status:** ✅ implemented (FR4, FR16, AR9)

### Story 2.3: Cross-validate and compute standard metrics

As a methods engineer,
I want k-fold cross-validation with standard metrics,
So that I can judge a model's quality, not just its training fit.

**Acceptance Criteria:**

**Given** a training run with `validation.cross_validation: k`
**When** the run executes
**Then** k-fold CV (seeded from `split.seed`) is performed and RMSE, R², and MAE are reported for train, test, and CV (mean ± std)
**And** an invalid k (e.g. larger than the sample count) raises a `DataError` with guidance.

**Status:** ✅ implemented (FR17)

### Story 2.4: Persist every run with complete lineage

As a methods engineer,
I want each run recorded with everything needed to trust and re-create it,
So that results remain auditable months later.

**Acceptance Criteria:**

**Given** a completed training run
**When** it is persisted
**Then** a `runs/run-NNN/` directory is written atomically (temp + replace) containing the config snapshot, `model.pkl` (+ `model.onnx` where supported), `metrics.json`, and `lineage.json` with run_id, created_at, config_hash, data_hash, seed, Python/library versions, and the reproducibility fingerprint
**And** lineage capture is automatic with no opt-out (NFR2).

**Status:** ✅ implemented (FR15, NFR2, AR7)

### Story 2.5: Show a run's validation metrics

As a methods engineer,
I want to view a run's metrics on demand,
So that I can assess a model without re-training.

**Acceptance Criteria:**

**Given** a persisted run id
**When** I run `surroforge validate run-001`
**Then** train/test/CV metrics are displayed in a readable table
**And** an unknown run id raises `RunNotFoundError` with the available ids.

**Status:** ✅ implemented (FR5)

## Epic 3: Compare, Predict, Export & Report

Compare runs to pick the best model, predict on new inputs, export, and report.

### Story 3.1: Compare runs and identify the best

As a methods engineer,
I want to compare several runs on unified metrics in one command,
So that I can choose the best method with confidence.

**Acceptance Criteria:**

**Given** two or more run ids on the same dataset
**When** I run `surroforge compare --runs run-001,run-002`
**Then** their test and CV metrics are tabulated and the best run by the selected metric is identified (lower-is-better for RMSE/MAE, higher for R²)
**And** an empty or single-id request is handled with a clear message.

**Status:** ✅ implemented (FR6)

### Story 3.2: Predict on new inputs

As a methods engineer,
I want to run a trained model on new input points,
So that I can use the surrogate in downstream work.

**Acceptance Criteria:**

**Given** a trained run and an input file containing the run's input columns
**When** I run `surroforge predict run-001 --input new.csv --output preds.csv`
**Then** predictions are appended under the run's output column names and written out
**And** an input missing required columns raises a `DataError` listing the missing columns.

**Status:** ✅ implemented (FR7)

### Story 3.3: Export a model to ONNX or pickle

As a methods engineer,
I want to export a model in an interoperable format,
So that it can be consumed by other tools.

**Acceptance Criteria:**

**Given** a trained run
**When** I run `surroforge export run-001 --format onnx`
**Then** scikit-learn-based methods (polynomial) export to ONNX, and pickle export is always available
**And** requesting ONNX for an SMT method (kriging/rbf) raises a clear `ExportError` stating ONNX is unsupported for that method (Spike 1).

**Status:** ✅ implemented (FR8, FR18, AR4)

### Story 3.4: Generate a Markdown report

As a methods engineer,
I want a Markdown report for a run,
So that I can share results and keep a record.

**Acceptance Criteria:**

**Given** a persisted run
**When** I run `surroforge report run-001`
**Then** a Markdown report is written under `reports/` with data overview, model spec, validation metrics, and config details
**And** the report renders deterministically for the same run.

**Status:** ✅ implemented (FR9)

## Epic 4: Reproducibility & Run Lineage

List and inspect every run, and reproduce any past run with a trustworthy verdict.

### Story 4.1: List all runs in a project

As a methods engineer,
I want to list every run with its key facts,
So that I can navigate my experiment history.

**Acceptance Criteria:**

**Given** a project with runs
**When** I run `surroforge list`
**Then** runs are listed by id with experiment, method, and a headline metric
**And** ids sort naturally (`run-001` … `run-010`).

**Status:** ✅ implemented (FR10)

### Story 4.2: Inspect a run's details

As a methods engineer,
I want full detail on a single run,
So that I can audit exactly how it was produced.

**Acceptance Criteria:**

**Given** a run id
**When** I run `surroforge info run-001`
**Then** its lineage (config_hash, data_hash, seed, versions, fingerprint) and metrics are shown
**And** an unknown id raises `RunNotFoundError`.

**Status:** ✅ implemented (FR12)

### Story 4.3: Reproduce a run with a trustworthy verdict

As a methods engineer,
I want to reproduce a past run and be told whether it truly matched,
So that I can trust the reproducibility guarantee — the product's core promise.

**Acceptance Criteria:**

**Given** a stored run and its unchanged data
**When** I run `surroforge reproduce run-001`
**Then** the run is re-executed from its config snapshot and a verdict is emitted from an **out-of-sample** fingerprint against a single named tolerance: `reproduced` (bit-identical), `reproduced_within_tolerance`, `diverged`, `data_changed`, or `unverifiable` (legacy run without a fingerprint)
**And** `diverged`, `data_changed`, and `unverifiable` exit non-zero with a clear message; `reproduced`/`within_tolerance` exit zero
**And** the reproduction scenario (config + data only) works end-to-end (UX-DR8).

**Status:** ✅ implemented (FR11, NFR2, NFR7, AR5)

### Story 4.4: Warn when loading a non-local model artifact

As a methods engineer,
I want a warning when I load a pickle a model I did not produce locally,
So that I am aware of the arbitrary-code-execution risk of shared artifacts.

**Acceptance Criteria:**

**Given** a `model.pkl` whose recorded provenance differs from the current machine/project
**When** it is loaded (predict/reproduce/export)
**Then** a clear, one-time warning is emitted naming the risk, and loading proceeds
**And** locally-produced artifacts load silently.

**Status:** ✅ implemented (AR12 provenance warning). A one-time `UserWarning` fires at the unpickle choke point (`Run.load_model`), covering predict + export. `reproduce` re-trains from the config snapshot and never unpickles the model, so it carries no arbitrary-code-execution risk and stays silent by design. Provenance keys on a privacy-preserving `machine_id` (hash of host+user) recorded in lineage; legacy runs without it are treated as foreign.

## Epic 5: Python API & Extensibility

Use SurroForge as a first-class library and extend it with plugins.

### Story 5.1: Drive the full workflow from the Python API

As a research engineer,
I want a Python API equal to the CLI,
So that I can embed SurroForge in larger pipelines.

**Acceptance Criteria:**

**Given** an installed project
**When** I use `Project.open(...)`, `Trainer.from_config(...)`, and `trainer.fit(...)`
**Then** I can run the same train/validate/compare/predict/export/report/reproduce operations as the CLI, with the same vocabulary and results
**And** behaviour stays identical because both call the shared service layer (no logic duplicated in the adapters).

**Status:** ✅ implemented (FR19, UX-DR3)

### Story 5.2: Register a custom method via entry points

As a plugin author,
I want to add a trainer/exporter/reporter through Python entry points,
So that I can extend SurroForge without modifying Core.

**Acceptance Criteria:**

**Given** a package declaring an entry point under `surroforge.trainers` (or `.exporters`/`.reporters`)
**When** it is installed
**Then** the new method is discoverable and usable by name exactly as a built-in (which use the identical mechanism)
**And** Core runs standalone when the plugin is absent (graceful degradation).

**Status:** ✅ implemented (FR20, AR9)

### Story 5.3: Self-verify a plugin against a shipped conformance suite

As a plugin author,
I want an importable conformance suite I can run against my trainer,
So that I can confirm it honours the Core contract (including reproducibility) before publishing.

**Acceptance Criteria:**

**Given** a `src/surroforge/conformance/` parametrised suite shipped with Core
**When** I parametrise it with my trainer class
**Then** it asserts the `BaseTrainer` contract (signatures, picklability, 2-D `predict` shape), determinism, fingerprint participation, and `SurroForgeError`-only failures
**And** Core's own `tests/` run the same suite against the three built-ins.

**Status:** ⏳ not yet built (AR10)

### Story 5.4: Declare a method's reproducibility and export capabilities

As a plugin author,
I want to declare how my method behaves on reproducibility and which exporters it supports,
So that Core's verdict logic and exporter dispatch handle my method without Core changes.

**Acceptance Criteria:**

**Given** a trainer subclass
**When** it declares reproducibility characteristics (deterministic vs. tolerance) and a queried exporter capability (replacing the binary `supports_onnx`)
**Then** `reproduce()` picks the correct comparison and `export` gives a clear error for unsupported method/format pairs — driven by the declaration, not a hard-coded list.

**Status:** ⏳ not yet built (AR11)

### Story 5.5: Reserve the Core↔commercial plugin seam

As a maintainer,
I want the commercial seam contract reserved before any external plugin binds,
So that Phase 2 (Professional/Enterprise) is additive, not a breaking change.

**Acceptance Criteria:**

**Given** the plugin registry
**When** the seam is reserved
**Then** a plugin `api_version` is checked at registration (clean error on mismatch), a deterministic name-collision/precedence policy is defined, and a no-op `gate.check(plugin_metadata)` call site exists after resolution
**And** no edition-named logic enters Core; the gate is a documented Phase-1 no-op.

**Status:** ⏸ Phase-2 reservation (AR13) — freeze before the first external plugin binds.

## Epic 6: CLI Ergonomics, Quality Gates & Distribution

A polished, trustworthy CLI backed by enforced quality gates and clean packaging.

### Story 6.1: Consistent global CLI options

As a CLI user,
I want uniform global options and activity-grouped commands,
So that the tool feels predictable, like `git`/`dvc`.

**Acceptance Criteria:**

**Given** any command
**When** I pass `--verbose/-v`, `--quiet/-q`, `--config-dir`, or `--help`
**Then** they behave consistently across commands (verbose adds progress, quiet suppresses non-error output)
**And** commands are grouped by activity with discoverable help.

**Status:** ✅ implemented (FR21, UX-DR1)

### Story 6.2: Errors as a feature

As a CLI user,
I want failures to be clear and actionable,
So that I can fix problems without reading a traceback.

**Acceptance Criteria:**

**Given** a domain failure (bad config, missing data, unknown run, etc.)
**When** it occurs
**Then** it is rendered as a clean `Error: …` line with a concrete fix suggestion and a non-zero exit; the full traceback appears only under `--verbose`
**And** output tone is precise and sober — no emojis, no marketing.

**Status:** ✅ implemented (NFR9, UX-DR4, UX-DR5)

### Story 6.3: Graceful degradation without optional backends

As a CLI user,
I want missing optional backends to produce guidance, not crashes,
So that a minimal install stays usable.

**Acceptance Criteria:**

**Given** Core installed without an optional extra
**When** I request a feature needing it (e.g. `kriging` without `[smt]`, ONNX without `[onnx]`, Parquet without `[formats]`)
**Then** I get an actionable install hint (`pip install 'surroforge[...]'`), not an `ImportError`
**And** listing available methods still works without the backend installed.

**Status:** ✅ implemented (AR9)

### Story 6.4: Enforced architectural and quality gates

As a maintainer,
I want the architecture's invariants enforced automatically,
So that the codebase stays correct and trustworthy as it evolves.

**Acceptance Criteria:**

**Given** the codebase
**When** the gates run
**Then** import-linter contracts (dependency direction) and `tests/test_architecture.py` (no Core→commercial import) pass in the suite, and the CI workflow runs ruff, mypy, import-linter, an OS×Python test matrix, a pinned-threading reproducibility job, and a GPL/LGPL license scan
**And** the locally-runnable gates (import-linter, AST guard, license scan) are active today; the GitHub workflow activates once the project is a pushed git repo.

**Status:** ✅ partially (import-linter contracts, AST guard, license scan, CI scaffold) · ⏳ full CI activation pending a git remote (AR8, AR3, NFR3, NFR8)

### Story 6.5: Documentation site

As a first-time user,
I want tutorials and an API reference,
So that I can reach a trained model unaided in under 30 minutes.

**Acceptance Criteria:**

**Given** the repository
**When** docs are built
**Then** a MkDocs Material site provides install, quickstart, the three scenarios, config reference, and Python API reference
**And** the docs sustain a first-time user through the greenfield scenario without external help (NFR7).

**Status:** ⏳ not yet built (AR12, NFR7)
