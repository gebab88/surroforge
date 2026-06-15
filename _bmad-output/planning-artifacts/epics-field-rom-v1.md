---
stepsCompleted: ["step-01-validate-prerequisites", "step-02-design-epics", "step-03-create-stories"]
status: 'draft'
createdAt: '2026-06-05'
inputDocuments:
  - _bmad-output/planning-artifacts/prd-field-rom-v1.md
  - _bmad-output/brainstorming/brainstorming-session-2026-06-04-231637.md
  - _bmad-output/planning-artifacts/architecture.md
  - PRD.md
scope: "Field ROM v1 — the thin field-surrogate slice beyond Phase 1 (PRD-ROM-v1). Greenfield/forward-looking, NOT implemented."
note: "Unlike epics.md (Phase 1, brownfield, complete), every story here is planned (📋) and not built. Idea-IDs (#NN) trace to the 2026-06-04 brainstorm; ADR-ROM-001 is the fingerprint design. De-risk-first: the fingerprint spike (Story R2.1) must land before the v1 reproduce() contract freezes."
---

# SurroForge — Field ROM v1 Epic Breakdown

## Overview

This document decomposes the **Field ROM v1** feature PRD ([`prd-field-rom-v1.md`](prd-field-rom-v1.md)) into implementable epics and stories, in the same format as the Phase-1 [`epics.md`](epics.md). Field ROM adds **field-valued outputs** (displacement/stress/temperature over a mesh) reduced onto a small modal basis (v1: POD), with a surrogate mapping parameters → modal coefficients and on-demand full-field reconstruction.

The organizing principle is the brainstorm's **spine theorem**: **#36 (basis hidden inside the trainer) + a field-primary fingerprint (ADR-ROM-001)** make ROM ship as *one new trainer + a fingerprint evolution* — downstream (export/compare/conformance/registry) sees a normal `(n, n_dof)` trainer, so the remaining ~100 ideas become optional enhancements. v1 is **Core** (Apache) + a `surroforge[rom]` extra; commercial ROM surfaces (basis hub, embedded twin, inverse design, Kalman predict) layer on the same seam in Phase 2.

Out of scope here (PRD-ROM §1.5 / §11): solver-native & mesh readers, learned/greedy/Krylov bases, multi-output coefficient vector, living HTML report, the cross-solver "physics" verdict (#73, v1 is bitwise-equivalent on the field), and the coordinate-channel research track (#63).

## Requirements Inventory

### Functional Requirements (Field ROM)

FR-R1: Ingest a field as a snapshot matrix `(n_samples × n_dof)` from CSV/Parquet/HDF5, **mesh-free**; declare the field via an ordered column glob/prefix in `data.outputs`. [PRD-ROM 4.2; #19, #53]
FR-R2: Configure reduction via a `model.reduction` block (`basis: pod`; `energy` or `rank` target; `rank: adaptive` resolved at fit and written to the config snapshot) plus a `model.inner` method block. [PRD-ROM 4.3; #29, #34]
FR-R3: Train via a `ReducedFieldTrainer(BaseTrainer)` that builds a POD basis (energy/rank-determined), projects snapshots to coefficients, and fits **one inner surrogate per mode**, behind the unchanged `BaseTrainer` `predict()` contract. [PRD-ROM 4.4; #36, #11, #2/#7]
FR-R4: Reproduce a field run with a **field-primary verdict** — reconstruct at pinned, canonicalized probes; field equivalence is ground truth; canonicalized coefficients + a bound (diagnostic, non-gating) basis hash are the fast/audit lens; tolerance is folded into the hashed spec. [PRD-ROM 4.5; ADR-ROM-001; #16, #64, #43]
FR-R5: Report a **two-source error split** (truncation floor vs. surrogate error) and field metrics (`energy_l2`, `truncation_error`) registered as ordinary metrics; cross-validation re-derives the basis **inside each fold**. [PRD-ROM 4.6; #25, #42, #41, #40]
FR-R6: `predict` returns compact **coefficients by default**, full field only on `--field`; coefficients + basis stored lazily and reconstructed on demand. [PRD-ROM 4.7; #35, #22, #21]
FR-R7: `surroforge rom inspect <run-id>` shows the energy spectrum, resolved rank, per-mode energy, and probe coverage. [PRD-ROM 4.8; #30]
FR-R8: The shipped conformance suite gains a `coeffs→field→coeffs` reconstruction round-trip check **and** a no-mesh (no coordinates/connectivity) check; `surroforge[rom]` packaging extra with lazy imports. [PRD-ROM 4.9, 4.10; #57, #53, #31]

### NonFunctional Requirements (deltas to PRD §5)

NFR-R1: Reproducibility — ROM runs reproducible via the field verdict; the **scalar path stays byte-identical** (identity-basis equivalence), tested as a regression guard. [PRD-ROM 5; inherits NFR2]
NFR-R2: SVD determinism under pinned BLAS/OpenMP threads; sign/order-canonicalization + the field verdict absorb basis gauge ambiguity. [PRD-ROM 5, 8]
NFR-R3: Artifact size — the run dir is **smaller than the raw snapshots** (lazy coefficients + content-addressed basis). [PRD-ROM 4.7; #47, #32]
(Inherits NFR1 performance, NFR3 platform, NFR8 licensing — v1 adds **no new heavy dependency**: SVD from scipy.)

### Additional Requirements (architecture, Field ROM)

(Status: 📋 planned — none built. ADR-ROM-001 is the fingerprint decision record in the brainstorm doc.)

- 📋 **AR-R1 — Basis-inside-trainer (#36):** ROM ships as a single `surroforge.trainers` entry point; downstream sees a normal `(n, n_dof)` trainer ⇒ **zero new downstream touchpoints** (export/compare/conformance/ONNX-path/registry unchanged). A closed-source reducer can ship as an ordinary entry point.
- 📋 **AR-R2 — Field-primary fingerprint (ADR-ROM-001):** observable-field equivalence is the *definition* (First-Principles flip); coefficient-space is the fast/audit lens; sign/order canonicalization (#16) is the non-negotiable foundation; the basis hash is **diagnostic, not a gate** (pre-mortem); tolerance is **folded into the hashed spec**; degenerate/repeated modes fall back to subspace/principal-angle or the gauge-invariant field verdict; the cross-solver "physics" verdict (#73) + per-mode bands (#72) are **deferred** (v1 = bitwise-equivalent on the field).
- 📋 **AR-R3 — Honest-basis CV (#40):** the basis is re-derived inside each CV fold via the existing per-fold fresh-factory contract; `validators/cross_validation.py` is unchanged. Surfaces basis variance and prevents test-snapshot leakage into the basis.
- 📋 **AR-R4 — Metric-as-key (#41):** field metrics are ordinary `AVAILABLE_METRICS` callables, so they flow through CV, best-run selection, and the report with no new plumbing.
- 📋 **AR-R5 — Reconstruction conformance + no-mesh invariant (#57, #53):** the shipped suite proves a basis is a faithful projector (round-trip within `DEFAULT_TOLERANCE`) and enforces that the reducer touches no coordinates/connectivity — reusing the existing `record()`/tolerance plumbing.
- 📋 **AR-R6 — Lineage & artifact deltas:** `lineage.json` gains the resolved `reduction` spec (part of the hashed spec), a diagnostic `basis_hash`, a field-primary `holdout_fingerprint`, a field-energy-relative `reproduction_tolerance`, and `probe_coverage_energy`; the run dir stores a content-addressed `basis.npz` (deduplicated across runs) + lazy coefficients. [#34, #32, #47]
- 📋 **AR-R7 — Edition & packaging:** Field ROM v1 is **Core** (Apache) behind the `surroforge[rom]` extra, mirroring `[smt]`/`[onnx]`; the reproduce() change must be Core. (Confirmation: OQ-1.)

### FR Coverage Map

- FR-R1 (snapshot ingest, mesh-free) → Epic R1
- FR-R2 (`model.reduction`, rank-as-resolved) → Epic R1
- FR-R3 (`ReducedFieldTrainer`, per-mode) → Epic R1
- FR-R4 (field-primary reproduce verdict) → Epic R2
- FR-R5 (two-source error, field metrics, honest-basis CV) → Epic R3
- FR-R6 (coefficients-by-default, `--field`, lazy storage) → Epic R4
- FR-R7 (`rom inspect`) → Epic R4
- FR-R8 (reconstruction + no-mesh conformance, `[rom]` extra) → Epic R4

## Epic List

### Epic R1: Reduction Engine & Trainer
Ingest a field snapshot matrix, configure reduction declaratively, and train a `ReducedFieldTrainer` (POD + per-mode inner surrogate) that passes the existing `BaseTrainer` conformance unmodified.
**FRs covered:** FR-R1, FR-R2, FR-R3
**Supports:** AR-R1, AR-R7, NFR-R3, UX-DR2, UX-DR6, UX-DR7

### Epic R2: Field Reproducibility (Fingerprint) — *critical path*
Evolve `reproduce()` to a field-primary verdict (ADR-ROM-001) without disturbing the scalar path. Highest leverage and highest risk: its spike (R2.1) front-runs and must land before the v1 contract freezes.
**FRs covered:** FR-R4
**Supports:** AR-R2, NFR-R1, NFR-R2, NFR2, NFR7, UX-DR8

### Epic R3: Field Validation & Report
Report a two-source error split and field metrics, with cross-validation re-deriving the basis per fold (honest-basis CV).
**FRs covered:** FR-R5
**Supports:** AR-R3, AR-R4

### Epic R4: Ingestion, Query, Conformance & Packaging
Coefficients-by-default prediction with `--field` reconstruction and lazy storage, `rom inspect`, the reconstruction + no-mesh conformance probe, and the `surroforge[rom]` extra.
**FRs covered:** FR-R6, FR-R7, FR-R8
**Supports:** AR-R5, AR-R6, AR-R7, NFR-R3, NFR8, UX-DR1

> **Story status legend (this doc):** 📋 planned (not built; forward-looking) · 🔬 spike (de-risk before contract) · ⏸ deferred to Phase 2. (The Phase-1 ✅/⏳ markers do not apply — nothing here is implemented yet.)

> **Dependencies & sequencing.** R1 is foundational (a trainer that emits coefficients/fields). **R2.1 (fingerprint spike) is the de-risk gate** and runs in parallel with R1; the rest of R2 (the verdict contract) must be settled **before** any v1 run is persisted with a field fingerprint, because that fingerprint is the reproducibility contract. R3 and R4 follow once R1+R2 stabilize.

## Epic R1: Reduction Engine & Trainer

Ingest a field snapshot matrix, configure reduction declaratively, and train a `ReducedFieldTrainer` behind the unchanged trainer contract.

### Story R1.1: Ingest a field as a snapshot matrix (mesh-free)

As a methods engineer,
I want to register a field as a wide `(n_samples × n_dof)` table and name its columns as one output field,
So that I can build a field surrogate without exporting a mesh.

**Acceptance Criteria:**

**Given** a CSV/Parquet/HDF5 file whose rows are DOE points and whose output columns are an ordered field group (e.g. `temp_0001 … temp_NNNN`)
**When** I register it and reference it with `data.outputs: [temp_*]`
**Then** the glob expands deterministically (stable order) into the `n_dof` field vector, hashed and recorded like any dataset
**And** no coordinates or connectivity are read or required — the field is "columns that move together"
**And** a ragged/misaligned field group raises a `DataError` naming the offending columns.

**Status:** 📋 planned (FR-R1, AR-R1, UX-DR7; #19, #53)

### Story R1.2: Configure reduction in a `model.reduction` block

As a methods engineer,
I want to declare the reduction (POD, energy or rank target) in YAML beside the inner method,
So that the reduced run is a single versionable source of truth.

**Acceptance Criteria:**

**Given** a config with `model.method: reduced-field`, a `model.reduction` block (`basis: pod`, and `energy: 0.999` **or** `rank: 12`), and a `model.inner` method block
**When** the config is loaded
**Then** Pydantic validates it; exactly one of `energy`/`rank` is required; the inner block is an ordinary method spec resolved through `get_trainer`
**And** with `rank: adaptive` the engine resolves the rank at fit time and writes the **resolved** rank into the config snapshot (the snapshot, not the input YAML, is canonical)
**And** an invalid combination (both `energy` and `rank`, or neither) raises a `ConfigError`.

**Status:** 📋 planned (FR-R2, AR-R6; #29, #34)

### Story R1.3: ReducedFieldTrainer — POD basis + per-mode inner surrogate

As a methods engineer,
I want training to reduce the field and fit one inner surrogate per mode,
So that I reuse the entire existing trainer stack on a low-dimensional target.

**Acceptance Criteria:**

**Given** a registered field dataset and a valid `reduced-field` config
**When** I run `surroforge train --config <yaml>`
**Then** a POD/SVD basis `V (n_dof × r)` is built (energy- or rank-determined, sign/order-canonicalized), snapshots are projected to coefficients `(n × r)`, and `r` independent inner surrogates are fit (one per mode)
**And** the trainer exposes the unchanged `BaseTrainer` contract: `predict(x)` returns a 2-D array and downstream (export/compare/registry) treats it as an ordinary `(n, n_dof)` trainer
**And** `export_formats = {pickle}` for v1 (field ONNX deferred), and `reproduction_tolerance` is field-energy-relative.

**Status:** ✅ delivered (FR-R3, AR-R1; #36, #11, #2/#7) — `src/surroforge/trainers/reduced_field.py` (POD energy/rank + sign-canonicalized basis + per-mode inner surrogate), a **pure plugin**: registered via the `surroforge.trainers` entry point + `_BUILTIN` fallback, **zero engine/config/services changes**. Passes the existing conformance suite (added to `tests/test_conformance.py` BUILTINS) and runs end-to-end through the unchanged engine (`tests/test_reduced_field.py`, incl. a field e2e: R²>0.95 on a rank-3 field). Honest-basis CV (R3.3) falls out for free via the per-fold fresh-factory. Partially covers R1.1 (field = output column-group; glob expansion still planned) and R1.2 (reduction/inner parsed+validated as hyperparameters; the `rank: adaptive` snapshot write-back still planned).

### Story R1.4: Built-in methods work as inner surrogates and pass conformance unmodified

As a plugin author / maintainer,
I want the three Core methods to serve as inner surrogates with no special-casing,
So that ROM is genuinely "one trainer on top of the existing stack".

**Acceptance Criteria:**

**Given** a `reduced-field` config whose `model.inner.method` is `kriging`, `rbf`, or `polynomial`
**When** the run trains
**Then** each inner method fits per-mode coefficients with no ROM-specific changes to the inner trainers
**And** the existing `check_trainer()` conformance suite passes for each built-in used as an inner surrogate (the BaseTrainer contract is untouched).

**Status:** 📋 planned (FR-R3, AR-R1; supports DoD)

## Epic R2: Field Reproducibility (Fingerprint) — critical path

Evolve `reproduce()` to a field-primary verdict per ADR-ROM-001, with the scalar path byte-identical by construction.

### Story R2.1: Fingerprint spike — prove the field-primary verdict (de-risk)

As the architect,
I want a prototype that proves the field-primary verdict and the coefficient fast-path agree under a pinned basis,
So that the v1 reproduce() contract is settled before any field run is persisted against it.

**Acceptance Criteria:**

**Given** a prototype field run with a pinned basis
**When** I compute both the reconstructed-field verdict (at canonicalized probes) and the canonicalized-coefficient fast-path
**Then** they agree (the fast-path is a valid accelerator under the bound basis), and a deliberately sign-flipped/re-ordered equivalent basis still reproduces
**And** a symmetric part (repeated singular values → degenerate subspace) is shown to false-diverge under naive per-mode canonicalization, and the subspace/principal-angle (or field-verdict) fallback fixes it
**And** the output is an updated ADR + a concrete `reproduce()` contract proposal feeding R2.2–R2.4.

**Status:** ✅ spike delivered (FR-R4, AR-R2, NFR-R2; #16, #64, #43) — runnable evidence `spikes/field_fingerprint_spike.py` (all 3 claims PASS), contract proposal `_bmad-output/implementation-artifacts/spec-field-rom-fingerprint.md`. Findings: field verdict is the definition; canonical coeffs are a valid fast/audit lens but false-diverge on degeneracy; **basis hash demoted to diagnostic, not a gate** (adjusts ADR-ROM-001).

### Story R2.2: Sign/order canonicalization with degeneracy handling

As a methods engineer,
I want the modal basis canonicalized before any hashing or comparison,
So that SVD gauge ambiguity never causes a false `diverged`.

**Acceptance Criteria:**

**Given** a POD basis (possibly with near-equal singular values)
**When** it is canonicalized
**Then** mode sign (max-abs component) and order (energy) are fixed deterministically, and near-equal singular-value clusters are detected and compared as a subspace (principal angles), not per-mode
**And** the same physics re-imported with a re-gauged basis canonicalizes identically (no false diverge)
**And** canonicalization runs under pinned BLAS/OpenMP threads in CI.

**Status:** 📋 planned (FR-R4, AR-R2, NFR-R2; #16)

### Story R2.3: Field-primary verdict in `reproduce()`

As a methods engineer / auditor,
I want `reproduce()` to judge a field run by the reconstructed field (the thing I depend on),
So that "reproduced" means the engineer-observable field matches — not an incidental coefficient/byte detail.

**Acceptance Criteria:**

**Given** a persisted field run
**When** I run `surroforge reproduce <run-id>`
**Then** the verdict reconstructs the out-of-sample holdout field at pinned, canonicalized probes and compares within a field-energy-relative tolerance that is folded into the hashed spec
**And** canonicalized coefficients + a bound `basis_hash` are recorded as a fast/audit lens (which mode moved), with the basis hash **diagnostic, not a gate** (a re-gauged basis with an identical field reads "reproduced, basis re-gauged")
**And** widening the tolerance is recorded as a *different* run (spec re-hash), not a looser compare.

**Status:** 📋 planned (FR-R4, AR-R2, NFR2, NFR7; #64, #43, #72-deferred)

### Story R2.4: Scalar path stays byte-identical (regression guard)

As a maintainer,
I want the generalized fingerprint to leave scalar runs bit-for-bit unchanged,
So that adding ROM never regresses the existing reproducibility guarantee.

**Acceptance Criteria:**

**Given** the pre-ROM scalar fingerprint baseline
**When** the field-primary fingerprint code path runs on a scalar run (identity basis ⇒ field *is* the prediction)
**Then** the produced fingerprint and verdict are byte-identical to the pre-ROM behaviour
**And** a CI regression test pins this equivalence (in the spirit of the existing profiling/boundary gates).

**Status:** 📋 planned (FR-R4, NFR-R1, NFR2)

### Story R2.5: Record probe coverage; prefer a global energy-norm

As an auditor,
I want the run to record how much field energy the probes cover,
So that "reproduced at probes" can't hide divergence in unprobed regions.

**Acceptance Criteria:**

**Given** a field run whose verdict uses pinned probes
**When** it is persisted
**Then** `probe_coverage_energy` is recorded in lineage, and a global energy-weighted field norm is the definition with probes as the fast path
**And** an out-of-probe energy fraction above a threshold emits a warning, not a silent pass.

**Status:** 📋 planned (FR-R4, AR-R2, AR-R6; #49, #84-deferred)

## Epic R3: Field Validation & Report

Report a two-source error split and field metrics, with the basis re-derived inside each CV fold.

### Story R3.1: Field metrics as registered callables

As a methods engineer,
I want field-appropriate metrics available like any other metric,
So that they flow through CV, comparison, and the report with no special handling.

**Acceptance Criteria:**

**Given** a config with `validation.metrics: [energy_l2, truncation_error, rmse]`
**When** the run executes
**Then** `energy_l2` and `truncation_error` are resolved from `AVAILABLE_METRICS` and computed for train/test/CV like RMSE today
**And** an unknown metric name raises the same clear error as the scalar path (no new plumbing).

**Status:** 📋 planned (FR-R5, AR-R4; #41)

### Story R3.2: Two-source error split in the report

As a methods engineer,
I want the report to separate the projection floor from surrogate-added error,
So that I know whether to add modes or add data.

**Acceptance Criteria:**

**Given** a completed field run
**When** I generate its Markdown report
**Then** the report renders both the coefficient-space and reconstructed-field errors in one table, separating truncation error (the basis floor) from surrogate error
**And** a "great R² from a tiny basis doing nothing" case is visibly exposed (low surrogate error, high truncation floor).

**Status:** 📋 planned (FR-R5; #25, #42)

### Story R3.3: Honest-basis cross-validation

As a methods engineer,
I want each CV fold to re-derive the basis from only its training snapshots,
So that my CV score isn't inflated by test snapshots leaking into the basis.

**Acceptance Criteria:**

**Given** a `reduced-field` run with `validation.cross_validation: k`
**When** CV runs
**Then** the basis is re-derived inside each fold via the existing per-fold fresh-factory contract (no change to `cross_validation.py`)
**And** the report surfaces basis variance across folds as a first-class number, and a global-basis (leaky) CV is demonstrably more optimistic in tests.

**Status:** 📋 planned (FR-R5, AR-R3; #40)

## Epic R4: Ingestion, Query, Conformance & Packaging

Coefficients-by-default prediction with `--field` reconstruction and lazy storage, `rom inspect`, conformance, and the `[rom]` extra.

### Story R4.1: Predict coefficients by default; reconstruct on `--field`

As a methods engineer,
I want predictions to be compact coefficients unless I ask for the full field,
So that querying a million-node surrogate stays KB-sized.

**Acceptance Criteria:**

**Given** a trained field run
**When** I run `surroforge predict <run-id> --input new.csv`
**Then** compact modal coefficients are returned by default; `--field` reconstructs and writes the full field (Parquet/HDF5/CSV column group in v1)
**And** the run dir stores a content-addressed `basis.npz` (deduplicated across runs by hash) + lazy coefficients, making the run dir smaller than the raw snapshots.

**Status:** 📋 planned (FR-R6, AR-R6, NFR-R3; #35, #22, #21, #47, #32)

### Story R4.2: `surroforge rom inspect`

As a methods engineer / buyer persona,
I want to see the energy spectrum, chosen rank, and probe coverage,
So that the reduction is auditable.

**Acceptance Criteria:**

**Given** a persisted field run
**When** I run `surroforge rom inspect <run-id>`
**Then** it prints the energy spectrum, the resolved rank, per-mode energy contribution, and probe coverage, in the precise/sober house tone (no emojis/marketing)
**And** running it on a scalar (non-reduced) run prints a clear "this run is not a reduced-field run" message, not an error.

**Status:** 📋 planned (FR-R7, UX-DR4; #30)

### Story R4.3: Reconstruction round-trip + no-mesh conformance probe

As a plugin author,
I want the shipped conformance suite to verify my reducer is a faithful, mesh-free projector,
So that a third-party/commercial ROM can self-certify before claiming conformance.

**Acceptance Criteria:**

**Given** a reduced-field trainer under `check_trainer()`
**When** the conformance suite runs
**Then** a `coeffs → field → coeffs` round-trip is asserted within `DEFAULT_TOLERANCE` (faithful projector, no sign/normalization bug)
**And** a no-mesh check asserts the reducer reads no coordinates/connectivity (mesh-free is a tested invariant)
**And** these reuse the existing `record()`/tolerance plumbing and appear in the `ConformanceReport`.

**Status:** 📋 planned (FR-R8, AR-R5; #57, #53)

### Story R4.4: `surroforge[rom]` packaging extra

As a user,
I want ROM to install as an optional extra without bloating the base install,
So that the Core stays lean and permissively licensed.

**Acceptance Criteria:**

**Given** a base `pip install surroforge`
**When** I `pip install 'surroforge[rom]'`
**Then** ROM becomes available with lazy imports (mirroring `[smt]`/`[onnx]`), adding **no new heavy dependency** in v1 (SVD from scipy) and no GPL/LGPL dependency (NFR8)
**And** requesting `reduced-field` without the extra installed yields an actionable install hint, not an `ImportError`.

**Status:** 📋 planned (FR-R8, AR-R7, NFR8; #31)

## Epic R5: Solver-Native & Mesh Readers — *Phase 2*

> **Phase 2 — opens after Field ROM v1 (R1–R4) shipped.** v1 deliberately deferred
> *solver-native & mesh readers* (Overview / PRD-ROM §1.5, §11); this epic begins that track on
> the reserved `surroforge[rom]` seam. It extends the FR-R1 ingestion contract (a field is
> "columns that move together", mesh-free) to solver output, and rides the spine theorem
> (AR-R1): readers only emit an ordinary wide `(n_samples × n_dof)` DataFrame, so the engine,
> registry, `reduced-field` trainer, fingerprint, and reproduce path stay **unchanged**.

### Story R5.1: OpenFOAM field ingestion (solver-native, mesh-free)

As a methods engineer with OpenFOAM CFD results,
I want a first-class library reader that turns an OpenFOAM case sweep into the wide snapshot
matrix SurroForge already trains on,
So that I can build a field surrogate straight from solver output — without SurroForge ever
reading a mesh.

**Acceptance Criteria:** (full BDD in the story file)

**Given** an OpenFOAM case sweep (one case dir per DOE point, ASCII `volScalarField`/
`volVectorField` `internalField`)
**When** I ingest field `p` (or a vector field) via the new `surroforge.data` reader
**Then** I get a deterministic-order wide DataFrame (`p_0000…`, vectors `_x/_y/_z`) that
registers and trains through the unchanged `reduced-field` pipeline with `data.outputs: ['p_*']`
**And** only `internalField` values are read (no coordinates/connectivity); ASCII parsing is
pure-Python with **no new dependency** (binary/`meshio` readers deferred to `[rom]`)
**And** the working prototype in `examples/05_openfoam_field_rom.py` is promoted into the
package and the example refactored to import it.

**Status:** 📋 planned — Phase 2 (FR-R1 extension, AR-R1, AR-R5; story
`r5-1-openfoam-field-ingestion.md`). Prototype already runs in `examples/05_openfoam_field_rom.py`.

### Story R5.2 (deferred): `surroforge data add-openfoam` CLI + binary/meshio readers

⏸ Deferred follow-on: a CLI ingestion command and `meshio`-backed binary/mesh readers behind the
`surroforge[rom]` extra (lazy import + install-hint pattern). Not started; carried for Phase-2
sequencing.

## Epic R6: Advanced Surrogate Methods — *Phase 2*

> **Phase 2 — opens a second Phase-2 track (parallel to R5's solver-native readers).** v1 shipped
> three methods (`polynomial`, `kriging`, `rbf`) plus the `reduced-field` reducer and **no neural
> network**. This epic begins the *advanced methods* track. Like every method, it rides the spine
> theorem (AR-R1): a new method is a single `surroforge.trainers` entry point + `_BUILTIN` line —
> the engine builds it via `trainer_cls(**hyperparameters)` (`ModelConfig` is `extra="allow"`), so
> engine/config/services/CLI/export/compare/reproduce/conformance stay **unchanged**, and any
> `BaseTrainer` is automatically usable as a `reduced-field` inner.

### Story R6.1: Neural-network surrogate — built-in `mlp` trainer (scikit-learn)

As a methods engineer,
I want a neural-network surrogate method (`model.method: mlp`) that trains a multi-layer
perceptron, compares against the existing methods, and exports to ONNX,
So that I can fit smooth nonlinear response surfaces with a genuine NN without the Core taking on a
heavy deep-learning framework or losing its reproducibility/export guarantees.

**Acceptance Criteria:** (full BDD in the story file)

**Given** a config with `model.method: mlp` (or `model.inner.method: mlp` under `reduced-field`)
**When** the run trains
**Then** an `MLPTrainer(BaseTrainer)` backed by `sklearn.neural_network.MLPRegressor` (input
`StandardScaler`, `solver="lbfgs"` default) fits and predicts behind the unchanged trainer
contract (2-D predict, native multi-output), passes `check_trainer()` unmodified, declares a small
positive `reproduction_tolerance` (iterative optimiser) with a negative reproduce test, and exports
to **ONNX** via skl2onnx (the second ONNX-capable method)
**And** it adds **no new dependency** (scikit-learn is Core, BSD → NFR8) and changes only the
trainer leaf + the two registration sites (AR-R1).

**Backend decision:** scikit-learn `MLPRegressor`, **not** PyTorch. A real MLP with zero new deps,
permissive, and compatible with the sklearn-style `random_state` reproducibility model. PyTorch is
heavy + SGD-stochastic and, per the Golden Path `[MUST]`, framework plugins are entry-points-only
(a separate distribution) — so a PyTorch/DeepONet/FNO track stays available on that external seam
for the large, strongly-nonlinear field regime, but is out of Core scope.

**Status:** ✅ delivered — Phase 2 (new-method track, AR-R1; story `r6-1-neural-network-mlp-trainer.md`).
`src/surroforge/trainers/mlp.py` (sklearn MLPRegressor, StandardScaler + lbfgs, native multi-output,
ONNX via skl2onnx → 2nd ONNX-capable method); registered in both sites; added to conformance
BUILTINS; works as a `reduced-field` inner (POD-NN). Zero new dependency. 3-layer adversarial review
(6 patches applied / 3 dismissed). Suite 400 passed / 7 skipped; all gates green.

## Open Questions (carried from PRD-ROM §9)

- **OQ-1 — Edition:** confirm Core (default) vs. a Pro differentiator. *Recommendation: Core* (the reproduce() change must be Core).
- **OQ-2 — Field schema:** glob/prefix in `data.outputs` (default) vs. a separate field manifest vs. a field dataset type at `data add`.
- **OQ-3 — Probe selection:** simple energy-weighted set in v1 vs. Q-DEIM/QR picker (#49/#84) now.
- **OQ-4 — Verdict tolerance:** concrete field-energy-relative default and how it enters the spec hash.
- **OQ-5 — Research track:** evaluate the coordinate-channel approach (#63) as a parallel spike vs. modal reduction.

## References

- Feature PRD: [`prd-field-rom-v1.md`](prd-field-rom-v1.md)
- Brainstorm + ADR-ROM-001: [`../brainstorming/brainstorming-session-2026-06-04-231637.md`](../brainstorming/brainstorming-session-2026-06-04-231637.md)
- Phase-1 epics (format reference): [`epics.md`](epics.md)
- Architecture (seams, plugin authoring, reproducibility patterns): [`architecture.md`](architecture.md)
