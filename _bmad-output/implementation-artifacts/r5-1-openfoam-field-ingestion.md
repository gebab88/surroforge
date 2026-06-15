---
baseline_commit: bed83fa6b33909b393cd110eaeff9c96b63e8ef2
---

# Story R5.1: OpenFOAM field ingestion (solver-native, mesh-free)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->
<!-- Phase-2 / Epic R5 — first story of the "Solver-Native & Mesh Readers" track. Field ROM v1
     (R1–R4) is feature-complete; this opens Phase 2 on the reserved surroforge[rom] seam. -->

## Story

As a methods engineer with OpenFOAM CFD results,
I want a first-class library reader that turns an OpenFOAM case sweep into the wide
`(n_samples × n_dof)` snapshot matrix SurroForge already trains on,
so that I can build a field surrogate straight from solver output without hand-writing a
CSV exporter — and without SurroForge ever reading a mesh.

## Context & Rationale

The prototype already exists and runs end-to-end in `examples/05_openfoam_field_rom.py`
(`read_openfoam_scalar_field`, `read_openfoam_dimensioned_scalar`, `latest_time_field`,
`ingest_cases`). This story **promotes that prototype into the package** as a hardened,
typed, tested, reusable reader so it is callable on real cases — not copy-pasted from an
example.

This is **Phase-2 work**: `epics-field-rom-v1.md` lists *"solver-native & mesh readers"* as
out of scope for v1 (line 22) and names `surroforge[rom]` as their Phase-2 home. It extends
the v1 ingestion contract (FR-R1: a field is "columns that move together", mesh-free) to a
solver-native source. It rides the **spine theorem** (AR-R1): the reader only produces an
ordinary wide DataFrame, so the engine, registry, `reduced-field` trainer, fingerprint, and
reproduce path are **untouched**.

## Acceptance Criteria

**AC1 — Scalar `internalField` reader.**
**Given** an OpenFOAM `volScalarField` file with `internalField nonuniform List<scalar> N ( … )`
**When** I call `read_openfoam_scalar_field(path)`
**Then** a 1-D `np.ndarray` of length `N` is returned, with C/C++ block (`/* … */`) and line
(`// …`) comments stripped first
**And** `internalField uniform <value>;` is supported (returns the single value)
**And** a declared-vs-parsed count mismatch, a missing `internalField`, or unbalanced
parentheses each raise a `DataError` that names the offending file.

**AC2 — Vector field flattening.**
**Given** a `volVectorField` with `internalField nonuniform List<vector> N ( (x y z) … )`
**When** I read it as a field
**Then** it flattens to a length-`3N` vector in a **documented, deterministic interleave**
(cell-major: `[x0 y0 z0 x1 y1 z1 …]`) with balanced nested-paren handling
**And** the chosen interleave is reflected in the emitted column names (AC3) so the field is
glob-addressable.

**AC3 — Sweep → snapshot matrix.**
**Given** a sweep directory (one case sub-directory per DOE point) and a DoE table mapping
each case to its parameters
**When** I call the sweep-ingestion helper for field `p`
**Then** a wide `pandas.DataFrame` of shape `(n_cases, n_params + n_dof)` is returned with a
**stable, deterministic column order** — parameters first, then the field column-group
(`p_0000 … p_NNNN`, vector components suffixed `_x/_y/_z`) in a fixed case/DOF order
**And** because `data/hashing.py:hash_dataframe` is column- *and* row-order-sensitive, the
column and row order is reproducible across runs (sorted by case name, native DOF order)
**And** a case whose `n_dof` differs from the others raises a `DataError` naming that case.

**AC4 — Time resolution + parameter recovery from the case.**
**Given** a case with multiple numeric time directories and a `constant/transportProperties`
dictionary
**When** I resolve the field and recover a parameter
**Then** `latest_time_field(case, field)` returns the field in the highest-numbered time dir
**And** `read_openfoam_dimensioned_scalar(path, keyword)` recovers a dimensioned scalar
(e.g. `nu [0 2 -1 0 0 0 0] 1.5e-05;`) so a sweep parameter can come from the case itself, not
only a DoE table; a missing keyword raises a `DataError`.

**AC5 — Mesh-free and Core-pure (faithful pattern).**
**Given** any field file
**When** it is read
**Then** only the `internalField` values are read — **no coordinates, no connectivity, no
`polyMesh`** (the mesh-free invariant, FR-R1 / AR-R5)
**And** ASCII parsing is **pure-Python (stdlib `re`)** and adds **no new dependency** — Core
stays lean and permissively licensed (NFR8)
**And** binary OpenFOAM format and any `meshio`/`pyvista`-backed reader are **explicitly
deferred** to the `surroforge[rom]` extra, to be added later with the lazy-import +
`DataError` install-hint pattern used by `[smt]`/`[onnx]`/`[formats]` (documented, not built).

**AC6 — Integrates through the unchanged pipeline; example reuses the package.**
**Given** a DataFrame produced by the reader, written to CSV and registered as a dataset
**When** I train with `model.method: reduced-field` and `data.outputs: ['p_*']`
**Then** it flows through the existing engine/registry/trainer with **zero** engine, config,
or services changes (AR-R1)
**And** `examples/05_openfoam_field_rom.py` is refactored to **import the package reader**
instead of defining its own copies, and still runs to completion (the `tests/test_examples.py`
smoke test stays green).

**AC7 — Errors as a feature; discriminating tests.**
**Given** malformed inputs
**When** they are read
**Then** every domain failure raises a `DataError` (never a bare `ValueError`/`Exception`)
with an actionable message naming the offending file or case
**And** `tests/test_openfoam.py` covers: scalar round-trip, vector interleave, `uniform`,
comment stripping, count mismatch, missing `internalField`, unbalanced parens, latest-time
selection, dimensioned-scalar recovery, ragged sweep, and stable column/row order — each
negative test pinned to the **specific** wrong implementation it catches (no vacant pins).

## Tasks / Subtasks

- [x] **Task 1 — Create `src/surroforge/data/openfoam.py`** (AC1, AC2, AC4, AC5)
  - [x] Module opens with `from __future__ import annotations` + a one-line docstring citing
        the source (Story R5.1, FR-R1 extension).
  - [x] `read_openfoam_field(path) -> np.ndarray` — strip comments, handle
        `nonuniform List<scalar|vector>` via a balanced-paren scan, handle `uniform`, validate
        the declared count, raise `DataError` on every malformation naming the file.
        (Renamed from the prototype's `read_openfoam_scalar_field`: it always handled vectors,
        so "scalar" was a misnomer — see Completion Notes.)
  - [x] Flatten `vector` lists to `3N` cell-major; expose the interleave (`_VECTOR_AXES`) used
        by the column-namer.
  - [x] `latest_time_field(case, field) -> Path` and `read_openfoam_dimensioned_scalar(path,
        keyword) -> float`.
  - [x] No top-level optional/heavy import (pure-Python `re`; numpy/pandas are Core deps).
- [x] **Task 2 — Sweep ingestion helper** (AC3)
  - [x] `read_openfoam_sweep(case_root, *, field, params=<mapping or callable>, time=None)
        -> pd.DataFrame` returning the wide snapshot matrix with deterministic column/row order
        (sorted case names; native DOF order; params first, then the `<field>_<dof>` group).
  - [x] Raise `DataError` naming the case on inconsistent `n_dof`.
- [x] **Task 3 — Export from the data package** (AC6)
  - [x] Added the four public names to `src/surroforge/data/__init__.py` `__all__`. `data`
        stays a leaf (lint-imports 3/3 kept).
- [x] **Task 4 — Refactor the example onto the package** (AC6)
  - [x] `examples/05_openfoam_field_rom.py` imports the reader from `surroforge.data` and
        deletes its private copies; the synthetic-case *writer* stays in the example.
  - [x] `python examples/05_openfoam_field_rom.py` exits 0 (R²=1.0, rel err 1.3e-10).
- [x] **Task 5 — Tests** (AC7)
  - [x] `tests/test_openfoam.py` — 21 tests covering all AC7 cases; vector fixtures use distinct
        per-component values (interleave has teeth), negative tests assert the named `DataError`,
        sweep order pinned via `assert_frame_equal` + `hash_dataframe` equality.
- [x] **Task 6 — Gates & docs** (all ACs)
  - [x] `ruff check src tests examples`, `mypy src`, `lint-imports`, `pytest` all green
        (350 passed, 3 skipped).
  - [x] CHANGELOG `[Unreleased]` entry added (Solver-native ingestion / OpenFOAM).

### Review Findings (code-review 2026-06-14)

3-layer adversarial review (Blind Hunter / Edge Case Hunter / Acceptance Auditor). Verdict:
ship-ready, no Critical/Major correctness bug on well-formed input; findings bite on
realistic-but-untested inputs. Deduped across layers; 5 dismissed (see bottom).

**Resolution (2026-06-14):** lead chose strict latest-time + clearer diagnostic, and "apply all
patches". All 1 decision + 11 patches applied and pinned by tests (`tests/test_openfoam.py` now
32 tests). Full suite **361 passed / 3 skipped**; ruff / mypy / lint-imports green.

**Decision needed:**

- [x] [Review][Decision] Latest-time semantics when a field is missing in the newest time dir
      — `latest_time_field` uses strictly the highest-numbered time dir; if `p` exists in `100/`
      but a newer `200/` lacks it (restarted/cleaned run, or `200/` holds only `U`), the case is
      silently excluded and the sweep reports a misleading "No cases found". [openfoam.py:_has_field/latest_time_field]

**Patches (unchecked):**

- [x] [Review][Patch] Sweep param schema taken from first case only — extra param keys present
      in later cases are silently dropped (silent data loss); missing keys error. Validate an
      identical key set across all cases. [src/surroforge/data/openfoam.py:read_openfoam_sweep]
- [x] [Review][Patch] `params` callable raising a non-`DataError` leaks a raw exception with no
      case name (the example's own `params_for` calls `doe.loc[...]`/dict reader, both can raise).
      Wrap → `DataError` naming the case. [src/surroforge/data/openfoam.py:_resolve_params]
- [x] [Review][Patch] `params` callable returning a non-`Mapping` → cryptic `TypeError` deep in
      the build loop. Add an `isinstance(result, Mapping)` guard → `DataError`. [src/surroforge/data/openfoam.py:_resolve_params]
- [x] [Review][Patch] `_floats` silently accepts `nan`/`inf` tokens (a diverged solver dump)
      → non-finite values poison the snapshot matrix/POD. Add a `np.isfinite` guard → `DataError`.
      [src/surroforge/data/openfoam.py:_floats]
- [x] [Review][Patch] Ragged vector list (`((1 2 3)(4 5)(6 7 8 9))`) with a matching flat count
      silently mis-groups components across cells (only the flat total is checked). Validate
      per-vector-group arity == n_components. [src/surroforge/data/openfoam.py:_read_field]
- [x] [Review][Patch] Non-finite time-dir names (`nan`/`inf`) pass `_is_time_dir` and poison
      `max(..., key=float)`. Reject non-finite (`math.isfinite`). [src/surroforge/data/openfoam.py:_is_time_dir]
- [x] [Review][Patch] Unsupported list kinds (`symmTensor`/`tensor`) report a misleading
      "no internalField found"; raise a precise "unsupported field type" `DataError` instead.
      [src/surroforge/data/openfoam.py:_read_field]
- [x] [Review][Patch] `_field_columns` `:04d` zero-pad breaks lexical sortability for ≥10000 DOF
      (common in CFD); DataFrame order is fine but names mislead. Use a width derived from
      `n_cells`. [src/surroforge/data/openfoam.py:_field_columns]
- [x] [Review][Patch] AC6 pipeline integration is proven only by the example smoke test; add a
      focused test that trains `reduced-field` on reader output and asserts `n_outputs` + reproduce.
      [tests/test_openfoam.py]
- [x] [Review][Patch] Malformed `uniform` value (e.g. `uniform (1 2);`, size 2) raises but is
      unpinned by a test; add a negative test. [tests/test_openfoam.py]
- [x] [Review][Patch] `read_openfoam_sweep(params=...)` typed as `object`; tighten to the
      `Mapping | Callable | None` union for a first-class typed reader. [src/surroforge/data/openfoam.py]

**Dismissed (5):** dimensioned-scalar regex grabbing a dimension exponent (Blind self-withdrew;
Edge+Auditor verified the `]` before `;` blocks bracket digits — safe); `_matching_paren`/`_LIST_HEADER`
mismatch (Blind withdrew); double resolution of `latest_time_field` (negligible perf); empty-list-with-declared-count
message ordering (still a named `DataError`); the `read_openfoam_scalar_field`→`read_openfoam_field`
rename (deliberate, documented — not a violation).

## Dev Notes

### Architecture compliance (project-context.md — non-negotiable)

- **Layer:** the reader is pure **data-layer**. `data` is a leaf — it may import `errors` and
  (narrowly) sibling data modules, and must **not** import `cli`/`api`/`services`/`engine`.
  `lint-imports` enforces this; keep the reader free of project/engine coupling. It returns a
  DataFrame; *registration* stays in `services`/`api` where it already lives.
- **Errors:** raise `DataError` (a `SurroForgeError` subclass) for every failure — never a bare
  `ValueError`. Messages must state the fix and name the offending file/case (Errors-as-a-feature).
- **Optional-dependency discipline:** ASCII parsing needs no backend, so there is **no** lazy
  import in v1. If/when a `meshio` binary reader is added, it imports **inside** the function and
  translates `ImportError` → `DataError` with `pip install 'surroforge[rom]'` — never a
  top-level import.
- **Determinism / hashing `[GOTCHA]`:** `hash_dataframe` iterates `frame.columns` in order and
  hashes row bytes — column **and** row order change the dataset hash. The reader MUST emit a
  fixed order (sorted case rows; params-then-field columns; native DOF order) so a re-ingested
  sweep registers with the same hash and stays reproducible.
- **Conventions:** `from __future__ import annotations`; full type hints (mypy py310); arrays
  `np.ndarray`, paths `pathlib.Path`; ruff line length 100.

### What to reuse (do NOT reinvent)

- **Glob expansion already exists.** `loader.py:expand_field_columns` resolves
  `data.outputs: ['p_*']` to the concrete column group in frame order (Story R1.1). The reader's
  only job is to emit columns that this matches — pick a name pattern with a glob-friendly stem
  (`p_0000…`, vector `U_0000_x`). Do not add a second expansion path.
- **The trainer/engine are unchanged.** `reduced-field` (`trainers/reduced_field.py`) consumes
  the wide matrix as an ordinary `(n, n_dof)` target. No engine/config/registry change (AR-R1) —
  if you find yourself editing the engine, stop: the design intent is reader-only.
- **The prototype is the reference implementation.** Port the four functions from
  `examples/05_openfoam_field_rom.py` verbatim in behaviour; harden (typing, `DataError`s,
  edge cases) rather than redesign.

### Faithful-pattern decision (record it, like r4-4)

ASCII OpenFOAM parsing is genuinely dependency-free, so the v1 reader lives in **Core**, not
behind `[rom]`. The `[rom]` extra remains the documented home for the *future* binary/meshio
readers. State this in the module docstring and the CHANGELOG so the next reader knows the seam
exists but is intentionally empty for ASCII — mirroring the r4-4 "ship the seam shape, not a
dependency we don't need" decision.

### Scope boundary (kept tight on purpose)

R5.1 is the **library reader + example refactor + tests**. A `surroforge data add-openfoam
<case-dir>` CLI command and binary/`meshio` support are **follow-on stories** (R5.2+), not this
one. Keeping R5.1 reader-only preserves the one-coherent-slice rule and the spine theorem.

### Testing standards summary

- Flat `tests/`, one file per concern → `tests/test_openfoam.py`.
- `[GOTCHA] Vacant pins` (R3 retro): for each discriminating assertion name the wrong
  implementation it catches; avoid (a) aliasing (scope to the value, not whole-text substrings),
  (b) value degeneracy (choose fixtures where scalar vs. vector interleave actually differ),
  (c) presence-without-structure (assert exact arrays, not "contains").
- Round-trip with teeth: write a vector field with **distinct** per-component values
  (`(1 2 3)(4 5 6)`) so a wrong interleave (component-major vs. cell-major) is detected.
- No network, no external OpenFOAM install — synthesize case files in `tmp_path`.

### Project Structure Notes

- New file: `src/surroforge/data/openfoam.py` (leaf, alongside `loader.py`).
- Touch: `src/surroforge/data/__init__.py` (exports), `examples/05_openfoam_field_rom.py`
  (import the package), `tests/test_openfoam.py` (new), optionally `docs/` + `CHANGELOG.md`.
- No change to `engine.py`, `core/config.py`, `services.py`, `plugins/`, `trainers/` (AR-R1).

### References

- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md#Requirements Inventory] — FR-R1
  (mesh-free snapshot ingest), AR-R1 (basis-inside-trainer / zero downstream touchpoints), AR-R5
  (no-mesh invariant), and the explicit "solver-native & mesh readers → Phase 2 / `[rom]`" scope.
- [Source: src/surroforge/data/loader.py#expand_field_columns] — the existing `data.outputs`
  glob expansion the reader's column names must match.
- [Source: src/surroforge/data/hashing.py] — column/row-order-sensitive `hash_dataframe`
  (determinism constraint on emitted order).
- [Source: _bmad-output/project-context.md] — layer rules, DataError discipline,
  optional-dependency discipline, vacant-pin taxonomy.
- [Source: pyproject.toml#optional-dependencies] — the intentionally-empty `rom` extra and the
  `[smt]`/`[onnx]`/`[formats]` lazy-import + install-hint precedent.
- [Source: examples/05_openfoam_field_rom.py] — the working prototype to promote.
- [Source: _bmad-output/implementation-artifacts/epic-r4-retro-2026-06-14.md] — the r4-4
  faithful-pattern precedent (ship the seam shape, not an unneeded dependency).

## Dev Agent Record

### Agent Model Used

claude-opus-4-8[1m] (Claude Opus 4.8, 1M context)

### Debug Log References

- One test regex was initially too strict (`case_00 has 3` vs. the quoted `'case_00' has 3`);
  loosened to `case_00.*has 3`. No production change.

### Completion Notes List

- Promoted the working prototype from `examples/05_openfoam_field_rom.py` into
  `src/surroforge/data/openfoam.py` as a first-class, typed, `DataError`-disciplined reader,
  exported from `surroforge.data`.
- **API refinement vs. the story's literal names:** the prototype's `read_openfoam_scalar_field`
  already flattened vector fields, so it was renamed `read_openfoam_field` (handles scalar→`n`
  and vector→`3n` cell-major). Documented in the module + Task 1.
- **Mesh-free, Core-pure (faithful pattern):** ASCII parsing uses only stdlib `re` + numpy/pandas
  (already Core deps) — **no new dependency**, no top-level optional import. Binary/`meshio`
  readers remain deferred to the empty `surroforge[rom]` extra (recorded in the module docstring
  + CHANGELOG). `[rom]` was intentionally left empty.
- **Determinism for hashing:** `read_openfoam_sweep` emits a fixed layout (rows sorted by case
  name; parameter columns first in first-case key order; field group in native cell-major DOF
  order) so the registered dataset hash is reproducible — verified by a test asserting equal
  `hash_dataframe` across two ingestions.
- **Glob compatibility:** field columns are named `<field>_0000…` (`_x/_y/_z` for vectors) so the
  existing `expand_field_columns` resolves `outputs: ['p_*']` with no second expansion path
  (tested). The example config now uses the glob.
- **Zero downstream change (AR-R1):** no edit to `engine.py`, `core/config.py`, `services.py`,
  `plugins/`, or `trainers/`. The example trains the unchanged `reduced-field` (R²=1.0,
  reconstruction rel-err 1.3e-10). lint-imports confirms `data` stays a leaf.
- Scope held to R5.1 (library reader + example + tests). `surroforge data add-openfoam` CLI and
  binary/`meshio` readers remain R5.2 (backlog).
- Validation: **350 passed, 3 skipped** (smt-only Kriging/RBF paths); ruff / mypy(42 files) /
  lint-imports(3 kept) all green; `examples/05` smoke test green.

### File List

- `src/surroforge/data/openfoam.py` (new) — the reader module.
- `src/surroforge/data/__init__.py` (modified) — export the four public names.
- `examples/05_openfoam_field_rom.py` (modified) — import the package reader; drop private copies;
  config uses `outputs: ['p_*']`.
- `tests/test_openfoam.py` (new) — 21 unit tests.
- `CHANGELOG.md` (modified) — `[Unreleased]` OpenFOAM ingestion entry.

## Change Log

| Date | Change |
|---|---|
| 2026-06-14 | R5.1 implemented: OpenFOAM ASCII field reader promoted into `surroforge.data` (scalar/vector, sweep→snapshot matrix), example refactored onto it, 21 tests added. All gates green. Status → review. |
| 2026-06-14 | Code review (3-layer): 1 decision + 11 patches applied — param-key validation, nan/inf value + time-dir guards, ragged-vector arity check, unsupported-tensor message, dynamic column width, clearer missing-field diagnostic, AC6 pipeline test, malformed-uniform test, tightened `params` type. Tests 21→32; suite 361 passed. Status → done. |
