---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
lastStep: 8
status: 'complete'
completedAt: '2026-06-04'
inputDocuments:
  - PRD.md
  - SurroForge-UX-Briefing.md
  - SurroForge-Tech-Spikes.md
  - _bmad-output/planning-artifacts/epics.md
  - README.md
workflowType: 'architecture'
project_name: 'SurroForge'
user_name: 'Georg'
date: '2026-06-04'
projectMode: 'brownfield — Phase 1 MVP already implemented (working CLI, 116 passing tests); this document makes the existing architecture explicit and authoritative for AI agents'
---

# Architecture Decision Document — SurroForge

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements (21):**
The FRs describe one end-to-end workflow surfaced through two equal interfaces.
They cluster into: project/data lifecycle (FR1–FR3, FR14), training & validation
(FR4–FR5, FR16–FR17), run analysis & use (FR6–FR7, FR10–FR12), output &
interoperability (FR8–FR9, FR18), and the extensibility/contract layer (FR13,
FR15, FR19–FR21). Architecturally they demand a shared core invoked identically
by a thin CLI (Typer) and a first-class Python API — not a CLI with an API
bolted on.

**Non-Functional Requirements (9):**
Architecture-shaping NFRs: NFR2 reproducibility (no opt-out, automatic lineage) —
the core differentiator and a cross-cutting invariant; NFR1 performance (bounded
training/hashing times); NFR5 security/privacy (no outbound network in Core, no
telemetry without opt-in); NFR8 permissive-only dependencies (no GPL/LGPL in Core);
NFR4 SemVer + backward-compatible CLI/API contracts from 1.0; NFR9 error UX as a
first-class, tested concern. NFR3 (cross-platform, Py3.10+) and NFR6 (English
output) are environmental constraints.

**Scale & Complexity:**
Single-process, local, offline-capable CLI tool with a matching library API. No
multi-tenancy, real-time, network, or GUI surface.

- Primary domain: CLI tool + Python library (scientific-computing / CAE tooling)
- Complexity: **low infrastructural, high correctness/numerical-determinism** (see
  Caveats) — concentrated in reproducibility and the Core↔commercial plugin seam
- Estimated architectural components (PRD 6.2): ~10 packages — cli / core / data /
  trainers / validators / exporters / reporters / persistence / plugins / api

### Technical Constraints & Dependencies

- **Pre-resolved by Tech-Spikes** (binding decisions):
  - ONNX export only for scikit-learn-based methods; SMT Kriging/RBF use pickle +
    a clear error when ONNX is requested (Spike 1).
  - Plugin/edition separation via entry points; commercial editions attach as
    internal plugins with a license check; Core runs standalone with graceful
    degradation when optional packages are absent (Spike 2).
  - Reproducibility is differentiated: bit-identical under deterministic threading,
    otherwise within a documented tolerance, verified by a prediction fingerprint
    (SHA-256 over predictions) (Spike 3).
  - Data hashing over a canonical in-memory representation (values + column names +
    dtypes), stable across lossless re-encodings; CSV acknowledged lossy (Spike 4).
- **Mandated stack (PRD 6.1):** Python 3.10+, Typer, YAML + Pydantic, SMT +
  scikit-learn, Pandas/PyArrow/h5py, SHA-256/JSON/Pickle, skl2onnx/onnxruntime,
  Jinja2, pytest + hypothesis, ruff + mypy.
- **Licensing:** Core is Apache 2.0; only permissive (Apache/MIT/BSD) dependencies
  (NFR8).
- **Optional dependencies:** distributed as extras (`[smt]`, `[onnx]`, `[formats]`,
  `[all]`); must degrade gracefully.
- **Run-id format:** sequential, human-readable (`run-001`).
- **Atomic persistence:** project metadata and lineage written atomically (temp file
  + replace).

### Cross-Cutting Concerns Identified

1. **Reproducibility & lineage (behavioural, verified, environment-conditional)** —
   The unit of reproducibility is the prediction fingerprint (behavioural
   equivalence), not byte-identity of the model artifact. `reproduce` is a
   *verification* operation: it must compare and emit a verdict — match /
   within-tolerance / diverged. The invariant is conditional: reproducible **given**
   the recorded environment, so captured dependency versions are a *required
   reproduction input*. Spans data, trainers, validators, persistence.
2. **Edition-agnostic plugin registry; license gating layered on top** —
   Built-in, third-party, and commercial plugins load through the identical
   entry-point path; the Core registry holds no *named* edition logic (it may host a
   generic, edition-neutral license-gate hook). Standalone Core completeness is
   primary; gating is the edge case. Per Spike 2 the gate's job is an unambiguous,
   well-behaved licensed state and a frictionless honest path — not bypass prevention.
3. **Plugin-interface stability IS the commercial boundary** —
   Commercial and third-party modules are written against Core's plugin interfaces,
   so SemVer-stable backward compatibility of those interfaces (NFR4) — not the
   license check — is what protects the Open-Core model. CLI and Python API are twin
   first-class consumers of the same core, vocabulary- and behaviour-identical,
   SemVer-stable from 1.0.
4. **Error UX & tone** — helpful, tested, fix-oriented messages; no tracebacks for
   domain errors; precise/sober, no emojis or marketing.
5. **Determinism & threading control** — reproducibility depends on controlled
   BLAS/optimizer threading at the trainer boundary.
6. **Optional-dependency degradation** — missing backends produce guidance, never
   crashes.
7. **Permissive-only dependency hygiene** — protects the Open-Core licensing model.

### Invariant Enforcement (Architectural Guards)

Several invariants cannot be enforced by design intent alone — they are only
catchable mechanically, so the architecture treats them as first-class artifacts:

- **Reproduction verdict is asserted:** `reproduce` computes the fingerprint over the
  *recorded* inputs and exits non-zero on divergence; environment drift downgrades a
  match to "reproduced under changed environment."
- **Edition-agnostic registry, proven:** no Core source references edition names;
  plugin loading is lazy and failure-isolated.
- **CLI↔API parity is structural:** all behaviour lives in the core service layer;
  CLI and API are thin adapters.
- **Domain-error boundary:** a domain-error hierarchy is caught at the CLI edge and
  rendered as guidance; raw tracebacks only under `--verbose`.
- **Capability probes over cryptic ImportErrors:** an uninstalled extra yields an
  actionable install hint (e.g. `surroforge[smt]`).
- **CI-enforced license hygiene** over the full dependency tree (NFR8).
- **Pinned-threading reproducibility tests** in CI, with per-method tolerance
  documented (Spike 3).

### Settled Decisions & Rejected Alternatives

Recorded so they are not re-litigated; formal rationale lives in the decision
sections below.

- **Plugin discovery: entry points.** Rejected: hand-rolled registry (Core must know
  its plugins), directory scan (fragile/insecure). Keeps the registry
  edition-agnostic and Phase 2/3 additive.
- **Reproducibility unit: prediction fingerprint.** Rejected: byte-identical artifact
  (unachievable for Kriging), trust-the-seed (unverifiable). Achievable, verifiable,
  method-agnostic.
- **Business logic in a shared core service layer**; CLI and API are thin adapters.
  Rejected: CLI-first with API wrapping it, parallel implementations. Makes parity
  structural.

### Genuinely Open Decisions (for the decision phase)

- **Config format (PRD §12):** YAML is de-facto (README, FRs, examples). Whether to
  also accept TOML is deferred to explicit decision.
- Other PRD §12 architecture-touching questions: Pro/Enterprise command visibility in
  Core help, and the plugin-API stability version. (Run-id `run-001` already
  de-facto chosen.)

### Caveats on This Analysis (Brownfield Honesty)

- **Descriptions vs. prescriptions.** Several invariants above are stated as intent
  and MUST be verified against the implemented source during the decision phase;
  until verified, treat them as *intended invariants*, not confirmed facts.
- **Complexity by axis.** Infrastructural complexity is low; correctness/numerical-
  determinism complexity is high. "Medium" overall must not invite under-investment
  in reproducibility and the plugin seam.
- **License boundary nuance:** "no *named* edition logic in Core," not "Core unaware
  licensing exists" — Core may host a generic edition-neutral gate hook.
- **Open soundness question — fingerprint sample set:** predictions on *training
  inputs* may under-detect divergence for interpolating methods (Kriging predicts
  training points near-exactly). Whether to use a held-out/grid sample is an open
  decision-phase question bearing on the reproducibility guarantee's strength.
- **Audience altitude:** this analysis is principle-level; the AI-agent audience needs
  rule-level guardrails (concrete first mistakes: forgetting entry-point
  registration, top-level hard imports of optional backends, skipping the
  fingerprint). The decision sections must descend to that altitude.

### Brownfield Note

Phase 1 (MVP / Core) is already implemented and passing 116 tests. This document
captures and hardens the architecture that exists so AI agents extending it
(Phase 2 Professional, third-party plugins) implement consistently with the
established seams rather than re-deriving or contradicting them.

## Starter Template Evaluation

### Primary Technology Domain

CLI tool + Python library (scientific-computing / CAE tooling). This is **not** a
greenfield scaffold decision: the stack is mandated by PRD §6.1 and the project is
already scaffolded and implemented (Phase 1, 116 passing tests). This section records
the **existing scaffolding baseline** — the project-init equivalent of a "starter" —
verified against the actual repository.

### Starter Options Considered

No external project-generator/starter applies. For a packaged Python CLI library the
relevant "starter" decisions are build backend, project layout, and tooling config —
all already made and validated in the repo:

- **Build backend:** `hatchling` (PEP 517), `src/` layout. Modern, standards-based,
  minimal config.
- **Rejected:** Poetry/PDM (heavier, lock-centric — unnecessary for a library), flat
  layout (`src/` layout avoids import-shadowing during tests).

### Selected Baseline: existing `hatchling` + `src/` layout (as built)

**Rationale for Selection:** Already in place, standards-based, consistent with the
permissive-only, PyPI-distributed posture (NFR3, NFR8). No re-scaffolding warranted.

**Initialization Command (for fresh clones):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all,dev]'   # editable install with all backends + dev tooling
```

**Architectural Decisions Provided by the Baseline (verified against repo):**

- **Language & Runtime:** Python ≥3.10 (`requires-python = ">=3.10"`).
- **Packaging & Distribution:** `hatchling`; wheel packages `src/surroforge`;
  PyPI name `surroforge`. Core runtime deps are permissive-only (numpy, scipy,
  pandas, scikit-learn, pyyaml, pydantic, typer, jinja2). Heavy/optional backends are
  **extras**: `[smt]`, `[onnx]`, `[formats]`, `[all]`, `[dev]`.
- **Plugin Registration (load-bearing):** built-in trainers/exporters/reporters are
  declared as entry points in `pyproject.toml` under `surroforge.trainers`,
  `surroforge.exporters`, `surroforge.reporters` — the identical mechanism third-party
  and future commercial plugins use. CONFIRMS the edition-agnostic-registry invariant.
- **CLI Entry Point:** `surroforge = "surroforge.cli.main:main"`.
- **Code Organization (matches PRD §6.2, verified):**
  `src/surroforge/{cli, core, data, trainers, validators, exporters, reporters,
  persistence, plugins, api}`, plus top-level `engine.py`, `services.py`, `errors.py`.
  - `services.py` → shared core service layer; CLI and API are thin adapters
    (CONFIRMS the structural CLI↔API-parity invariant).
  - `errors.py` → domain-error hierarchy (CONFIRMS the domain-error-boundary guard).
- **Tooling (verified):** ruff (line-length 100; rules E/F/I/UP/B/W; scoped per-file
  ignores), mypy (py310, `ignore_missing_imports`), pytest (`testpaths=["tests"]`).
  Tests are organized one file per concern (config, data, trainers, training,
  cross_validation, metrics, exporters, reporters, persistence, plugins, reproduce,
  api, cli, project, scenarios).
- **Version currency (verified Jun 2026):** SMT 2.14.x, Typer 0.26.7, Pydantic 2.13.4
  are current; pinned floors (`smt>=2.0`, `typer>=0.12`, `pydantic>=2.5`) are valid
  and inside the current majors. Note: **Typer is pre-1.0** (API may shift; it vendors
  Click since 0.26) — the thin-CLI-over-`services.py` design insulates against this.
  The Spike-1 finding holds at SMT 2.14: no ONNX export path for SMT models.

### Gaps Found Against the Intended Architecture

- **No CI yet** (`.github/workflows/` absent). The step-2 "CI-enforced" guards —
  full-tree license-hygiene scan (NFR8) and pinned-threading reproducibility tests
  (Spike 3) — are **not yet implemented.** Carry forward as architecture work, not a
  settled fact. Because the project is already scaffolded, the equivalent "first
  story" is **standing up CI to enforce the invariants**, not project initialization.

### Core-vs-Commercial Packaging: Deliberately Deferred, Seam Locked

A Time Traveler Council (past MVP intent / present state / Phase 2 hindsight)
converged on a precise line: **commit to the seam contract now; defer all commercial
content and machinery.** The failure mode that actually bites at Phase 2 is
*interface drift*, not packaging.

- **Do now (cheap, high-leverage):** treat the plugin interfaces as a *versioned
  contract* with **contract tests** that fail on signature/behaviour drift; reserve a
  **generic, edition-neutral license-gate hook** in Core that is a documented **no-op
  in Phase 1**.
- **Do NOT do now (premature):** create `surroforge-professional` /
  `surroforge-enterprise` packages, license/activation logic, or any edition-named
  code — empty shells with no Phase-1 consumer (contradicts PRD §1.5 MVP discipline).

### Minimum Seam Scaffolding (Reverse-Engineered from Phase 2)

Working backward from the Phase 2 Professional release confirmed the registry,
metric, and fingerprint mechanisms are already method-agnostic, and surfaced seam
requirements that are cheap to reserve now and expensive to retrofit (see the tiered
list below for the final, self-consistency-corrected classification):

- Trainer contract must let a method declare its **reproducibility characteristics**
  (deterministic vs. tolerance).
- The **license-gate hook needs a defined call site and signature now**
  (`gate.check(plugin_metadata)`, always-allow in Phase 1) — not merely "a hook
  exists" — to avoid a later edit to Core's hot path and an edition-name leak.
- **Exporter selection must be open/data-driven** — a method declares its own export
  capability; Core's dispatch reads that declaration.
- The plugin-facing API must be an **explicitly marked, SemVer-stable public surface**
  (NFR4), distinct from internal code, so refactors cannot silently break dependents.

### Pre-mortem: What Could Still Break Phase 2 (Implementation Requirements)

A pre-mortem assuming Phase 2 failed *despite* the seam contract surfaced these
implementation requirements:

- **Gate purpose, written at the hook:** unambiguous licensed state + frictionless
  honest path, NOT bypass-prevention (Spike 2) — so implementers don't over-invest in
  un-hardenable DRM.
- **Plugin provenance in lineage:** every run records the *resolving plugin* name +
  version (a sharpening of FR15's dependency capture), so a faulty plugin's output is
  traceable to its source rather than blamed on Core.
- **Help/discovery composed from installed plugins** (not hard-coded). → promoted to
  Genuinely Open Decisions: resolve before Phase 2.
- **Cut a versioned Core release as the dependency anchor** (1.0 / pinned pre-release
  with a frozen plugin API). → promoted to Genuinely Open Decisions.

### Scaffolding Commitments, Tiered

To avoid over-scaffolding, the "do now" items are tiered by necessity. Tier A is
required for a correct Phase 1; Tier B reserves *shape only* (cheap now, expensive to
retrofit) and is an explicit judgment call, not an MVP requirement.

**Tier A — required now (has a Phase-1 consumer, verified or mandatory):**

- Entry-point registration for built-ins (exists), `src/` + hatchling (exists),
  `services.py` shared layer (exists), domain-error boundary (exists).
- **Trainer reproducibility-characteristics field** — Phase 1's own methods force it
  (polynomial/RBF deterministic, Kriging tolerance-based per Spike 3); Core's
  reproduce-verdict must distinguish these for its three methods. *Built now, not
  reserved.*
- **Exporter capability declaration (data-driven)** — Phase 1 already mixes
  ONNX-capable (polynomial) and ONNX-incapable (Kriging/RBF → pickle) methods
  (Spike 1); the method→exporter mapping is a present-tense behavior. *Built now.*
- **Plugin-interface contract tests** — a Tier-A *quality gate* (keeps the three
  built-in trainers from drifting); in CI from the start.
- CI standing up: test run + license-hygiene scan + pinned-threading reproducibility
  test. (Currently absent — the first post-scaffolding story.)

**Tier B — reserve shape only (no Phase-1 consumer; cheap now / costly later):**

- License-gate **call site** in the registry with a trivial always-allow default;
  gate *decision logic* ships with the commercial package, NOT now. (Resolves the
  "don't build the gate / do build the gate" tension: call site yes, logic no.)
- Lineage records the **resolving plugin name + version** specifically.

**Deliberately NOT now (no shape reserved):** commercial packages, activation/
licensing logic, trainer cost/applicability declarations, multi-output, HTML/PDF
reporters. Neither a Phase-1 consumer nor a retrofit cost that justifies pre-work.

**Note:** Because the project is already scaffolded, "project initialization as the
first story" does not apply. The equivalent first story is **standing up CI to
enforce the Tier-A invariants above.**

## Core Architectural Decisions

> Categories are reframed for a local, single-process CLI library. Web/SaaS
> categories (database, auth, REST/GraphQL, frontend state, hosting/scaling) are
> **Not Applicable** and marked as such. Most heavy decisions were settled in code or
> the Tech-Spikes; this section records them and resolves the genuinely open ones.

### Decision Priority Analysis

**Critical Decisions (block correct implementation):**
- Reproducibility fingerprint sample set → **held-out + grid sample** (corrects a
  soundness gap; see Data Architecture).
- Pickle deserialization safety posture → **load + loud provenance warning**.
- Plugin-interface contract is versioned and frozen at **1.0** (PRD §4.11, §11.2).

**Important Decisions (shape the architecture):**
- Configuration format → **YAML only** (TOML deferred, not adopted).
- CI/CD → **GitHub Actions**, gating the Tier-A invariants (PRD §13).
- Pro/Enterprise command visibility → **visible with upgrade hint**, help composed
  from installed plugins + a static edition catalog.
- Core-anchor release → publish **Core 1.0 with a frozen plugin API** before Phase 2.

**Deferred Decisions (post-MVP, no shape reserved):**
- TOML config support, commercial packaging/activation, trainer cost/applicability
  declarations, multi-output models, HTML/PDF reporters, DVC integration (PRD §12).

### Data Architecture

- **Config validation:** Pydantic v2 models; a single YAML file is the complete run
  spec (FR13). **YAML only** — TOML deferred. Rationale: one validation/docs surface,
  matches what CAE engineers expect and what is already built.
- **Data hashing:** SHA-256 over a canonical in-memory representation (values +
  column names + dtypes), stable across lossless re-encodings; CSV acknowledged lossy
  (Spike 4). *Decided/implemented.*
- **Run storage:** filesystem layout per PRD §7.1 (`runs/<id>/` with
  config.snapshot.yaml, model.onnx/pkl, metrics.json, lineage.json, log.txt);
  atomic writes (temp + replace). Sequential human-readable run ids (`run-001`).
- **Reproducibility fingerprint → held-out + grid sample (DECISION + CORRECTION).**
  The fingerprint (SHA-256 over predictions) is computed over a fixed held-out/grid
  sample recorded per run, **not** over training inputs alone. Rationale: interpolating
  methods (Kriging) predict training points near-exactly, so a training-only
  fingerprint can report "match" for models that diverge between samples. **Brownfield
  action:** if the implemented fingerprint currently uses training inputs, it must be
  migrated; lineage gains the recorded sample set. Affects: persistence, validators,
  `reproduce`.

### Authentication & Security

- **Network posture:** no outbound network in Core except an opt-in update check
  (NFR5). *Decided.*
- **License gating:** edition-neutral no-op hook with a reserved call site in Phase 1;
  decision logic ships with the commercial package (Spike 2). Purpose is an
  unambiguous licensed state + frictionless honest path, **not** bypass prevention.
- **Pickle deserialization → load + loud provenance warning (DECISION).** Pickle
  round-trip is required for SMT/sklearn models, but unpickling is an arbitrary-code-
  execution surface. Core loads pickles but **warns when loading an artifact not
  produced on this machine/project** (provenance recorded in lineage) and documents
  the risk. ONNX remains the preferred *interchange* format where supported (sklearn
  methods); pickle is the same-environment round-trip format. Affects: exporters,
  persistence, lineage (provenance), `predict`/`reproduce`/`export`.

### API & Communication Patterns

- **CLI ↔ API parity is structural:** all behaviour lives in `services.py` (core
  service layer); CLI (Typer) and Python API are thin adapters (FR19). *Decided/
  verified.*
- **Plugin interfaces** (`surroforge.trainers/exporters/reporters`) are the public,
  SemVer-stable contract, **frozen at 1.0** (PRD §4.11, §11.2), distinct from internal
  code. Contract tests gate drift. Tier-A contract fields (decided here, specified as
  patterns in step 5): a trainer declares its **reproducibility characteristics**
  (deterministic vs. tolerance); an exporter capability is **data-driven** (a method
  declares which exporters it supports).
- **Error handling standard:** a domain-error hierarchy (`errors.py`) is caught at the
  CLI boundary and rendered as guidance with fix suggestions; raw tracebacks only
  under `--verbose` (NFR9, UX-DR5). *Decided/verified.*
- **Help/discovery:** composed from installed plugins; **Pro/Enterprise commands shown
  with an upgrade hint** ("requires Professional") via a static edition catalog, so
  paid features are discoverable without installing them (resolves PRD §12).

### Frontend Architecture

**Not Applicable.** SurroForge is CLI-only in the MVP and by design principle
("CLI before GUI, always" — UX §7). No state management, components, routing, or
bundling. Any future visualization is out of scope (PRD §8).

### Infrastructure & Deployment

- **Distribution:** PyPI (`pip install surroforge`), hatchling, `src/` layout; optional
  backends as extras (NFR3). *Decided/verified.*
- **CI/CD → GitHub Actions (DECISION, fills the gap).** Pipeline gates the Tier-A
  invariants: test matrix (Linux/macOS/Windows × Python 3.10+), ruff, mypy,
  **full-tree license-hygiene scan** (no GPL/LGPL — NFR8), **pinned-threading
  reproducibility test** (Spike 3), and **plugin-interface contract tests**. This is
  the first post-scaffolding story.
- **Versioning & release:** SemVer from 1.0 (NFR4). **Core-anchor release:** publish a
  Core 1.0 (or pinned pre-release) with a frozen plugin API as the dependency anchor
  `surroforge-professional` will pin to.
- **Observability:** local logging only (`runs/<id>/log.txt`); no telemetry without
  opt-in (NFR5).

### Decision Impact Analysis

**Implementation Sequence:**
1. Stand up CI (GitHub Actions) with the five gates — enforces everything below.
2. Apply the fingerprint correction (held-out + grid) and migrate lineage.
3. Formalize the plugin-interface contract + contract tests; add the
   reproducibility-characteristics and exporter-capability fields.
4. Add pickle provenance capture + warning.
5. Add edition catalog + composed help with upgrade hints.
6. Freeze plugin API, cut Core 1.0 anchor release.

**Cross-Component Dependencies:**
- Fingerprint correction touches persistence + validators + `reproduce` together.
- Pickle provenance depends on lineage already recording run origin.
- Exporter-capability declaration and the ONNX-vs-pickle behavior are the same
  mechanism (Spike 1) — implement once.
- Plugin-API freeze gates the Core-anchor release, which gates Phase 2.

## Implementation Patterns & Consistency Rules

> Reframed for a CLI library. The audience is an AI agent adding a trainer, exporter,
> reporter, or command — or building Phase 2. Rules are derived from and verified
> against the existing Phase-1 source, so new code matches what is already there.

### Critical Conflict Points Identified

Where independent agents would otherwise diverge: (1) how a new method/exporter/
reporter plugs in, (2) where business logic lives (service vs. CLI vs. API), (3) how
errors surface, (4) how optional backends are imported, (5) how reproducibility is
computed and asserted. Each is pinned below.

### Naming & Module Conventions (verified)

- **Module header:** every module opens with `from __future__ import annotations` and
  a one-line docstring citing the relevant PRD section (e.g. `(PRD 4.6)`).
- **Casing:** `PascalCase` classes, `snake_case` functions/variables, leading `_` for
  module-private helpers. Registry/`--format`/`model.method` identifiers are
  lowercase strings (`"polynomial"`, `"kriging"`, `"onnx"`).
- **Class `name` attribute** on every plugin must equal its registry key and, for
  trainers, the `model.method` config value.
- **Type hints everywhere** (mypy py310); arrays are `np.ndarray`, paths are
  `pathlib.Path`.

### Plugin Authoring Patterns (the core extension surface)

**Trainer** (`surroforge.trainers`, subclass `BaseTrainer`):
- Set class attrs `name` and `supports_onnx`. Keep `__init__` cheap; accept
  hyperparameters as keyword args and store them via `super().__init__(**hp)`.
- **Reject unknown hyperparameters** with `TrainerError` listing the supported ones
  (see `PolynomialTrainer`). Validate ranges in `__init__`, not in `fit`.
- Implement `fit(x, y)` (both 2-D) and `predict(x) -> (n, n_outputs)`; normalize with
  the inherited `_as_2d`. Raise `TrainerError("Model is not fitted.")` if `predict`
  precedes `fit`.
- The trainer **must be picklable** (it is the persisted artifact).
- Override `to_onnx(n_features)` only if `supports_onnx = True`; otherwise the base
  raises `ExportError`.

**Exporter** (`surroforge.exporters`, subclass `BaseExporter`): set `name` and
`extension` (with dot); implement `export(trainer, destination, *, n_features) -> Path`.

**Reporter** (`surroforge.reporters`): set `name` and `extension`; implement
`render(run) -> str`.

**Registration:** declare the class as an entry point in `pyproject.toml` under the
matching `surroforge.*` group **and** (for built-ins only) add it to the `_BUILTIN`
fallback table in `plugins/registry.py` (the fallback keeps source checkouts working
before `pip install`). Third-party/commercial plugins use entry points **only** — they
never touch `_BUILTIN`. Installed entry points take precedence over built-ins.

### Service-Layer & CLI/API Patterns

- **All behaviour lives in `services.py`** (or `engine.py`/`core`), returning plain
  dataclasses or dicts (e.g. `ReproduceResult`). The CLI and Python API are **thin
  adapters** that call services and format output. **Never** put domain logic in a
  Typer command or an API wrapper — that breaks CLI↔API parity (FR19).
- **CLI commands** are decorated with `_guard` (catches `SurroForgeError` → clean
  `Error: …` line; full traceback only under `--verbose`). New commands MUST use it.
- Global options live on the Typer root callback (`--verbose/-v`, `--quiet/-q`,
  `--config-dir`); output goes through `_info`/quiet-aware helpers, never bare
  `print`.

### Error-Handling Patterns

- Raise a subclass of `SurroForgeError` (`ProjectError`, `ConfigError`, `DataError`,
  `TrainerError`, `PluginError`, `ExportError`, `RunNotFoundError`,
  `ReproducibilityError`) — never a bare `Exception`/`ValueError` for domain failures.
- Messages are **actionable**: state what failed and the fix (e.g. missing-column
  lists, install hints). No emojis, no marketing (UX-DR4/5).
- A new failure category gets a new `SurroForgeError` subclass in `errors.py`.

### Optional-Dependency (Graceful-Degradation) Patterns

- **Never import an optional backend at module top.** Import inside the method that
  needs it (see `PolynomialTrainer._build`/`to_onnx`, which import sklearn/skl2onnx
  locally). This lets the registry list methods without the backend installed.
- Translate an optional-backend `ImportError` into a domain error with an install
  hint: `raise ExportError("… requires the 'onnx' extra. Install: pip install
  'surroforge[onnx]'") from exc`. Same pattern for `[smt]`, `[formats]`. **Note:**
  use the *extras* syntax for Core optional backends; future *commercial* plugins are
  separate distributions (`pip install surroforge-professional`), so their hints must
  name the distribution, not an extra.

### Reproducibility Patterns (with the step-4 correction)

- Persist the full lineage automatically (config snapshot, data hash, seed, versions)
  — no opt-out.
- **Fingerprint over a recorded held-out + grid sample** (step-4 decision), NOT over
  `config.data.inputs`. **Brownfield action:** `services.py:reproduce()` currently
  diffs predictions on the data inputs — migrate it to the recorded sample set and
  store that sample in lineage.
- **`reproduce` must emit a verdict, not just data.** Fold `assert_reproducible`'s
  check into the reproduce path so a divergence beyond tolerance is a non-zero exit /
  raised `ReproducibilityError`, and an environment-version mismatch is reported as
  "reproduced under changed environment" — rather than leaving the assertion opt-in.
- A trainer declares its **reproducibility characteristics** (deterministic vs.
  tolerance) so the verdict logic can pick the right comparison (Tier-A; not yet on
  `BaseTrainer` — add it).

### Exporter-Capability Pattern (generalization note)

Today capability is a binary `BaseTrainer.supports_onnx` + `to_onnx()`. For Phase 2
(more methods, more formats) generalize to a **queried** declaration — the exporter
asks the trainer "do you support format X?" (capability is *queried, not assumed*),
so a Phase-2 exporter introducing a new format needs no change to Core's service
layer. Keep `supports_onnx` working as a shim until generalized.

### Enforcement Guidelines

**All AI agents MUST:**
- Put domain logic in services/engine/core, never in CLI/API adapters.
- Subclass `SurroForgeError` for domain failures and write actionable messages.
- Register plugins via entry points (third parties: entry points only).
- Lazy-import optional backends and translate `ImportError` to a domain error + hint.
- Keep trainers picklable and `__init__` import-free.

**Enforcement status (honest):** CI does **not exist yet** (`.github/workflows/` is
absent), so none of the gates below are currently active. Until the CI story lands,
every item is a **convention enforced by manual review and the existing pytest suite
only** — treat "enforced by CI" anywhere in this document as *planned, not active*.

| Gate | Mechanism | Status |
|---|---|---|
| ruff + mypy | lint/type CI job | **planned — not active** |
| plugin-interface contract tests | parametrized pytest suite (see Core Decisions) | **planned — suite not yet written** |
| pinned-threading reproducibility test | subprocess CI job | **planned — not active** |
| full-tree license-hygiene scan (NFR8) | dependency-tree scan in CI | **planned — not active** |

Standing up CI to move these from *planned* to *active* is the first post-scaffolding
story (see Infrastructure & Deployment). Until then, a "pattern violation" is caught
in review, not by automation that isn't there.

### Good Examples vs Anti-Patterns

- ✅ `PolynomialTrainer`: cheap `__init__`, unknown-hyperparameter `TrainerError`,
  lazy sklearn import, `supports_onnx = True` with a real `to_onnx`.
- ❌ Importing `smt`/`torch` at module top (breaks `available_trainers()` without the
  extra).
- ❌ A Typer command that loads data, trains, and formats inline (logic belongs in a
  service; the API can't reuse it).
- ❌ Raising `ValueError("bad config")` (use `ConfigError` with the offending field).
- ❌ Fingerprinting reproduction on training inputs (under-detects Kriging divergence).

## Project Structure & Boundaries

### Complete Project Directory Structure

> Verified against the repository (2026-06-04). `(planned)` marks items decided in
> this document but not yet present; everything else exists today.

```
surroforge/
├── pyproject.toml              # hatchling build, deps, extras, entry points, tool config
├── README.md
├── LICENSE                     # Apache-2.0
├── .gitignore
├── .importlinter               # (planned) layering + core-never-outward contracts
├── .github/
│   └── workflows/
│       └── ci.yml              # (planned) test matrix, ruff, mypy, import-linter, license scan, repro gate
├── docs/                       # (planned) MkDocs Material site (PRD §6.1, §13)
├── src/
│   └── surroforge/
│       ├── __init__.py         # package version, public re-exports (Project, Trainer)
│       ├── errors.py           # SurroForgeError hierarchy — verified leaf (no intra-package imports)
│       ├── engine.py           # run_training: pipeline execution + leaf composition
│       ├── services.py         # use-case verbs: reproduce, compare, predict, export, report
│       ├── conformance/        # (planned) SHIPPABLE parametrized plugin contract suite (importable by out-of-tree plugins)
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py         # Typer app; thin adapter; _guard error boundary; global options
│       ├── api/
│       │   ├── __init__.py
│       │   └── highlevel.py    # Python API (Project, Trainer); thin adapter, peer to CLI
│       ├── core/              # domain center — verified: imports nothing outward (cli/api/services/engine)
│       │   ├── config.py       # Pydantic v2 config models + load_config (YAML only)
│       │   ├── dataset.py      # Dataset domain object (schema, registered data)
│       │   ├── project.py      # Project: open, dirs, dataset(), run(), resolve_source()  (imports data, persistence)
│       │   └── run.py          # Run: lineage, metrics, load_model(), run_dir  (imports persistence; lazy exporters.load_pickle)
│       ├── data/
│       │   ├── hashing.py      # canonical in-memory data hash (Spike 4)
│       │   └── loader.py       # load_table: CSV / Parquet / HDF5 (extras-gated)
│       ├── trainers/          # extension surface — register via entry points
│       │   ├── base.py         # BaseTrainer ABC (published plugin contract)
│       │   ├── polynomial.py   # scikit-learn; supports_onnx = True
│       │   ├── kriging.py      # SMT; pickle only
│       │   └── rbf.py          # SMT; pickle only
│       ├── validators/
│       │   ├── cross_validation.py  # k-fold CV
│       │   └── metrics.py           # RMSE, R², MAE
│       ├── exporters/
│       │   ├── base.py         # BaseExporter ABC
│       │   ├── pickle_exporter.py
│       │   └── onnx_exporter.py     # extras-gated ([onnx])
│       ├── reporters/
│       │   ├── base.py         # BaseReporter ABC (render(run) -> str)
│       │   └── markdown.py
│       ├── persistence/
│       │   └── store.py        # run storage, atomic writes (temp+replace), lineage I/O
│       └── plugins/
│           └── registry.py     # entry-point + _BUILTIN-fallback resolver (sole DYNAMIC name→class path)
└── tests/
    ├── conftest.py
    ├── test_architecture.py    # (planned) AST import-scan: no Core module imports a commercial package
    ├── test_config.py  test_data.py  test_trainers.py  test_training.py
    ├── test_cross_validation.py  test_metrics.py  test_exporters.py
    ├── test_reporters.py  test_persistence.py  test_plugins.py
    ├── test_reproduce.py  test_project.py  test_api.py  test_cli.py
    └── test_scenarios.py        # the three UX scenarios end-to-end (UX-DR8)
```

### Architectural Boundaries (verified against source 2026-06-04)

**Verified dependency layering — confirmed by import scan, not aspirational:**

- **Adapters** `cli`, `api` — top of the graph; may import anything below; nothing
  imports them. (`cli/main.py` calls `services`, `engine.run_training`, AND `core`
  directly — adapters use any lower layer; they are not funneled through `services`.)
- **Orchestration** `services` → `engine`. Verified: `engine` does **not** import
  `services` (no back-edge). `services` = use-case orchestration (opens projects,
  resolves plugins via the registry, maps errors); `engine` = pipeline execution +
  leaf composition. **Litmus for which file new code goes in:** *calls `plugins.get_*`
  or opens a project → `services`; runs a training-loop step → `engine`.*
- **Domain & leaves**, verified internal order:
  `core → {data, persistence, exporters(lazy `load_pickle`)}`;
  `engine → {core, data, validators, trainers(via registry), exporters(direct),
  persistence, plugins}`; `errors` is a pure leaf.

**The load-bearing invariant (verified TRUE):** nothing below the adapters imports
`cli`/`api`; `core`/`engine`/`services` never import upward. This keeps CLI and API
interchangeable peers.

**Sole-resolver rule — with its one verified exception:** dynamic name→class
resolution (`entry_points`, `import_module`) lives only in `plugins/registry.py`.
**Exception:** `engine.py` imports the two built-in exporters (`ONNXExporter`,
`PickleExporter`) directly to write the default train-time artifacts; only
*user-requested* `export` flows through `get_exporter`. **Cleanup candidate:** route
train-time artifact writing through the registry too, so a Phase-2 commercial
exporter is reachable on the training path.

**Interface (contract) boundary — the published plugin surface:** `BaseTrainer`,
`BaseExporter`, `BaseReporter`, the `SurroForgeError` hierarchy, the three
`surroforge.*` entry-point groups, and the `Project`/`Trainer` API. SemVer-stable
from 1.0; internal modules (`engine`, `services` internals, `persistence.store`) are
not part of the contract.

**Data boundary:** all run state lives under `runs/<id>/` in the *user's project
directory*. Persistence is the only writer; everything else reads through
`Run`/`Project`.

### Boundary Enforcement (how the rules become real)

- **`import-linter`** `layers` + `forbidden` contracts encode the layering and the
  `core`-never-outward rule transitively. **(planned — ships with `ci.yml`.)**
- **AST import-scan test** (`tests/test_architecture.py`) for "no Core module imports
  a commercial package" — AST, not regex; runs without the commercial package
  installed. **Define it now so a Phase-2 violation goes red on its first commit.**
- These import-scan checks are the cheapest possible CI gate (no fixtures, ms,
  deterministic) — the natural first gate to wire, and a pre-commit-hook candidate.

### Requirements to Structure Mapping

| FRs | Lives in |
|---|---|
| FR1 init, FR3 data list, project metadata (FR2, §7.3) | `core/project.py`, `cli/main.py` |
| FR2 data add, FR14 formats, hashing (Spike 4) | `data/loader.py`, `data/hashing.py`, `core/dataset.py` |
| FR4 train, FR13 YAML config | `engine.py`, `core/config.py`, `trainers/*` |
| FR5 validate, FR17 CV/metrics | `validators/*`, `services.py` |
| FR6 compare | `services.py:compare_runs` |
| FR7 predict | `services.py:predict_*` |
| FR8 export, FR18 artifacts | `exporters/*`, `services.py:export_model` |
| FR9 report | `reporters/*`, `services.py:generate_report` |
| FR10 list, FR12 info | `core/project.py`, `core/run.py`, `cli/main.py` |
| FR11 reproduce, FR15 lineage, NFR2 | `services.py:reproduce`, `persistence/store.py` |
| FR16 methods | `trainers/{polynomial,kriging,rbf}.py` |
| FR19 Python API | `api/highlevel.py` |
| FR20 plugins | `plugins/registry.py`, the three ABCs |
| FR21 global options | `cli/main.py` root callback |

**Cross-cutting concerns → locations:**
- Reproducibility/lineage → `persistence/store.py` + `services.py:reproduce` +
  `core/run.py`; **(planned)** held-out+grid fingerprint sample recorded here.
- Error/UX tone → `errors.py` + `_guard` in `cli/main.py`.
- Optional-dependency degradation → lazy imports inside `data/loader.py`,
  `trainers/*`, `exporters/onnx_exporter.py`.
- Edition gating → **(planned)** no-op `gate.check` call site in `plugins/registry.py`,
  invoked *after* resolution.

### Integration Points & Data Flow

**Walking-skeleton flow (`init → data add → train → report`):**
`cli/main.py` → `core/project.py` (create dirs) → `data/loader.py` +
`data/hashing.py` (register) → `engine.run_training` (resolve trainer via
`plugins/registry.py`, fit, cross-validate via `validators/*`, write default
pickle/ONNX artifacts) → `persistence/store.py` (write run + lineage atomically) →
`services.py:generate_report` → `reporters/markdown.py`. The Python API
(`api/highlevel.py`) drives the identical path through `services`/`engine`.

**External integrations:** none at runtime (no network in Core — NFR5). The only
external surfaces are the filesystem (project dir) and PyPI (distribution).

### Phase-2 Reservations (cheap now, expensive after commercial packages exist)

- **Freeze the contract surface as the only supported import path** (listed above);
  everything else is importable-but-unsupported. Guards against the *reverse* leak (a
  commercial package reaching into `engine`/`services`/`core` internals).
- **Add `api_version` to the plugin ABCs, checked at registration** in
  `plugins/registry.py` — turns a Core/plugin version mismatch into a clean message
  instead of an `AttributeError` deep in `engine`.
- **Reserve a precedence-policy seam + decide gate-vs-resolve ordering** in the
  registry now (even unused): Core + commercial dist registering the same name must
  resolve deterministically across install orders; `gate.check` runs *after*
  resolution.
- **Dogfood the seam in Phase 1:** verified — `get_trainer` already resolves all three
  built-ins via the entry-point path, proving the mechanism before money rides on it.
- **Commercial code stays a separate top-level distribution** — no `commercial/`/`pro/`
  placeholder inside this tree.

### Test Organization

- **Ship the conformance suite as library code:** `src/surroforge/conformance/`
  exposes a parametrized base/fixtures taking the class-under-test as a parameter.
  Core's `tests/` parametrize it with the three built-ins; an out-of-tree
  commercial/third-party author imports the *same* suite. A suite under `tests/`
  can't be run by an out-of-tree plugin — hence it must ship. **(planned — package
  boundary reserved now.)**
- **Mark the reproducibility test** (`@pytest.mark.slow`/`repro`) in its own file so
  the CI split is `pytest -m "not slow"` on push vs. the threading-pinned repro gate
  on a cadence.
- Otherwise keep `tests/` flat (one file per concern + `test_scenarios.py`) until a
  marker split genuinely hurts — no premature unit/integration/contract taxonomy.

### Development Workflow Integration

- **Dev:** `pip install -e '.[all,dev]'`; run `pytest -q`, `ruff`, `mypy` locally.
- **Build:** `hatchling` builds the wheel from `src/surroforge`.
- **CI (planned):** `.github/workflows/ci.yml` runs the Tier-A gates on a platform
  matrix; first gate is a single-platform run with `import-linter` + the
  pinned-threading reproducibility test, expanded to a cross-platform matrix later.
- **Distribution:** PyPI; Core 1.0 anchor release freezes the plugin API for Phase 2.

## Architecture Validation Results

> Calibrated against a brownfield reality (Phase 1 built, 116 tests) and pressure-
> tested in a multi-perspective review. The validation distinguishes the
> *architecture document* (complete, coherent, source-verified) from *implementation
> deltas* (scheduled work), and uses a **two-horizon** readiness verdict: Phase-1 MVP
> vs. Phase-2 commercial.

### Coherence Validation ✅

**Decision Compatibility:** All choices cohere. YAML+Pydantic config, entry-point
registry, the `services`/`engine` split, filesystem run storage, and the
fingerprint-based reproducibility invariant reinforce rather than conflict. Library
versions were web-verified compatible and inside current majors (SMT 2.14, Typer
0.26.7, Pydantic 2.13.4). The one config open-question (TOML) was closed (YAML only).

**Pattern Consistency:** Patterns were derived from and verified against the source
(BaseTrainer/Exporter/Reporter ABCs, lazy optional imports, `SurroForgeError` +
`_guard`), so they describe the real code. The one divergence — `engine` importing
concrete exporters — is documented as an exception with a cleanup path, not hidden.

**Structure Alignment:** The dependency layering was import-scanned and confirmed:
`core`/`engine`/`services` never import upward; `engine` has no back-edge to
`services`; `errors` is a leaf; dynamic resolution lives only in the registry.

### Requirements Coverage Validation ✅

**Functional Requirements:** All 21 FRs map to concrete modules (Requirements-to-
Structure table). No orphan FRs.

**Non-Functional Requirements:** NFR2 (reproducibility) — addressed in design, but the
implementation has a Critical defect (see gaps). NFR1 (performance) — bounded-time
targets recorded; optimization deferred by design (MVP). NFR3/4/5/6/8 — addressed.
NFR9 (error UX) — addressed via the error hierarchy + `_guard`. UX-DR1–8 — addressed,
incl. `test_scenarios.py` for the three end-to-end scenarios.

### Implementation Readiness Validation ⚠️ (one Critical gap blocks; rest scheduled)

Decision, structure, and pattern documentation are complete and source-verified.
However, a confirmed correctness defect in the headline feature (below) means the
implementation is **not** ready until that one item is closed.

### Gap Analysis Results

**Critical Gaps (block the MVP — must close before "ready"):**

1. **`reproduce()` fingerprints the TRAINING inputs, not held-out predictions.**
   This is a *present, confirmed* defect in `services.py`, not a future risk. Because
   the verdict is emitted from this comparison, two models that differ only by
   seed/thread/BLAS — identical training inputs, divergent predictions — would be
   reported as **REPRODUCED when they are not.** For a product whose headline
   differentiator is reproducibility, a check that passes when reproduction failed is
   a feature that lies. It passes the current 116 tests and would surface only when a
   (paying, Phase-2) user finds their "reproduced" model gives different numbers.
   **Concrete gate (the 117th test):** *given two models trained from the same config
   but with a deliberately perturbed nondeterminism source (seed or thread count),
   `reproduce()` MUST return NOT-REPRODUCED.* Write it, watch it fail against current
   `services.py`, fix the code to fingerprint a recorded held-out + grid sample,
   *then* finalize tolerance/grid parameters. (The parameter design — bit-exact in a
   pinned-thread env + cross-platform `allclose` with a stated tolerance — remains an
   Important sub-task, but the wrong-sample-set defect is the Critical part.)

**MVP-Gating (must hold for the Phase-1 Definition of Done — three external users,
three scenarios):**

2. **Cross-environment-install CI.** "116 tests passing" is passing *on the author's
   machine.* The MVP DoD is satisfied on a stranger's laptop, across platforms and
   Python versions (NFR3). For a reproducibility tool, an unreproducible build is
   self-refuting. CI must verify a clean install + the three scenarios on the support
   matrix. This is the defined first story.

**Important — Phase-2 readiness, NOT MVP-gating (deliberately deferred):**

These four are Phase-2 commercial enablers. None is touched by the MVP DoD; gating the
MVP on them would be scope-creep. They are scheduled for Phase-2, with one nuance
(below) about *when* the contract must freeze.

3. Tier-A contract fields on `BaseTrainer` (reproducibility-characteristics; generalized
   queried exporter capability). *Note: these are forced by the current method mix —
   build them when the fingerprint work lands, since the verdict logic needs the
   determinism declaration.*
4. Conformance suite (`src/surroforge/conformance/`) — required before any external
   plugin can self-verify.
5. Registry **seam contract** — `api_version` check, name-collision/precedence policy,
   post-resolution `gate.check` ordering.
6. Pickle provenance warning; edition catalog + composed help.

**Phase-2 contract caveat (the one cross-perspective tension, resolved):** building the
commercial machinery is correctly deferred (MVP discipline), BUT the registry seam
*contract* (#5) is irreversible-expensive once an external author binds to it — you
cannot migrate third-party/commercial plugins unilaterally later. So the contract must
be **declared/frozen before the first external plugin binds**, which is a Phase-2
*precondition*, not an MVP gate. This presumes the Phase-2 plugin shapes are known well
enough to freeze the right contract; **they are not yet specified to that level**, so
the document's "Phase 2 is additive" claim is currently **provisional** — validate it
by specifying at least one concrete commercial plugin before freezing.

**Nice-to-Have:** `docs/` MkDocs site; wiring import-linter + the AST architecture
test; routing train-time artifact writing through the registry (the `engine`
exporter-bypass cleanup).

### Architecture Completeness Checklist

> These items assess the architecture *document's* completeness (all satisfied). The
> open Critical Gap is an *implementation* defect, which drives Overall Status below.

**Requirements Analysis**
- [x] Project context thoroughly analyzed
- [x] Scale and complexity assessed (corrected: low infra / high determinism)
- [x] Technical constraints identified
- [x] Cross-cutting concerns mapped

**Architectural Decisions**
- [x] Critical decisions documented with versions
- [x] Technology stack fully specified
- [x] Integration patterns defined
- [x] Performance considerations addressed (identified; optimization deferred by design)

**Implementation Patterns**
- [x] Naming conventions established
- [x] Structure patterns defined
- [x] Communication patterns specified
- [x] Process patterns documented

**Project Structure**
- [x] Complete directory structure defined (verified)
- [x] Component boundaries established (verified by import scan)
- [x] Integration points mapped
- [x] Requirements to structure mapping complete

### Architecture Readiness Assessment

**Overall Status (two-horizon):**
- **Architecture document:** COMPLETE & COHERENT (16/16 checklist items; source-verified).
- **Phase-1 MVP implementation:** NOT READY until the one Critical gap (fingerprint
  correctness) is closed and cross-environment CI exists. Once those two land, Phase-1
  is READY. The remaining MVP work is small and well-scoped.
- **Phase-2 (commercial):** GATED on freezing the registry seam contract before the
  first external plugin binds — and that freeze is itself gated on specifying at least
  one concrete commercial plugin (the additivity claim is provisional until then).

**Confidence Level:** High — the assessment is grounded in a built, tested codebase,
and the riskiest claims were verified against source or the web during this workflow.

**Key Strengths:**
- Decisions grounded in real, tested code; boundary invariants were import-scan-verified.
- The Open-Core seam is designed once, with reservations that keep Phase 2 additive.
- Reproducibility honesty: the fingerprint defect is named and gated, not glossed.

**Areas for Future Enhancement:**
- Close the fingerprint defect + finalize tolerance/grid/threading parameters.
- Stand up cross-environment CI; make the Tier-A gates real.
- Specify a concrete Phase-2 plugin, then freeze the seam contract.

### Implementation Handoff

**AI Agent Guidelines:**
- Follow the documented decisions and patterns exactly; treat the import-verified
  boundary invariants as hard constraints.
- Treat "(planned)" / "Brownfield action" / Critical-gap items as *work to do*, not
  existing behavior — verify against source before assuming.
- Put domain logic in `services`/`engine`/`core`; register plugins via entry points;
  lazy-import optional backends; raise `SurroForgeError` subclasses.

**First Implementation Priority (ordered):**
1. Write the fingerprint perturbation test (117th test); watch it fail.
2. Fix `reproduce()` to fingerprint a recorded held-out + grid sample; emit the verdict.
3. Stand up cross-environment CI (`.github/workflows/ci.yml`) with import-linter + a
   pinned-threading reproducibility gate, then expand to the platform matrix.
