---
baseline_commit: b60c7a04a061da8d94064fb359604506109bd687
---

# Story R5.2: `surroforge data add-openfoam` CLI + binary/meshio readers

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->
<!-- Phase-2 / Epic R5, story 2. Follows r5-1 (the library-level OpenFOAM ASCII reader, done). -->

## Story

As a methods engineer with OpenFOAM CFD results,
I want a `surroforge data add-openfoam` command that registers a case sweep as a dataset in one
step, and the option to read binary/mesh formats via `meshio`,
so that I can go from a solver run to a registered, trainable dataset without hand-writing a CSV
or being limited to ASCII fields.

## Context & Rationale

r5-1 shipped the library reader (`surroforge.data.read_openfoam_sweep` / `read_openfoam_field`,
ASCII, mesh-free, Core-pure) and is `done` (commit `b60c7a0`). R5.2 puts a **CLI surface** on it
and opens the deferred **binary/mesh** track on the `surroforge[rom]` extra — the two items the
r5-1 review and the R5 epic explicitly scoped out of r5-1.

It rides the spine theorem (AR-R1): the command writes an ordinary tabular dataset and registers
it exactly like `surroforge data add`, so the engine/registry/trainer/`reduce`/`reproduce` paths
stay **unchanged**. The only new code is one CLI command + a lazy `meshio` reader in the `data`
leaf; `[rom]` stops being empty and becomes the documented home for mesh readers (as the r4-4
notes and `pyproject.toml` already foreshadow).

## Acceptance Criteria

**AC1 — `surroforge data add-openfoam` registers a case sweep.**
**Given** an OpenFOAM case-root directory (one case sub-dir per DOE point) and a field name
**When** I run `surroforge data add-openfoam <case-root> --field p --samples samples.csv --name cfd`
**Then** the sweep is ingested via the **existing** `read_openfoam_sweep`, written to a tabular
file under the project, and registered as a dataset — identical downstream to `data add` (schema
+ SHA-256 content hash), printing rows/cols/hash via `_success`/`_info`
**And** the command lives under the existing `data` Typer group (`data_app`), is decorated with
`@_guard`, and keeps CLI logic thin (no numerics in the CLI — it delegates to the library reader).

**AC2 — Options and parameter sourcing.**
**Given** the command
**When** I pass its options
**Then** `--field` (required) selects the field; `--samples <csv>` (a table with a `case` column
plus parameter columns) supplies the parameter columns; `--name` names the dataset; `--output
<path>` chooses the written file (a deterministic default is used when omitted); `--time <t>`
pins a fixed time directory (else latest)
**And** an invalid combination (missing `--field`, a `--samples` table whose `case` values don't
match the case dirs, a non-existent case-root) raises a clear `SurroForgeError` (`DataError`/
`ConfigError`) rendered as `Error: …` — never a traceback (only under `--verbose`).

**AC3 — Binary/mesh reading behind the `[rom]` extra (meshio).**
**Given** a binary OpenFOAM field or another `meshio`-supported mesh/result file
**When** I read it via the new `meshio`-backed reader
**Then** `meshio` is imported **lazily inside the function**, and its absence raises a `DataError`
with an actionable `pip install 'surroforge[rom]'` hint (mirroring `[smt]`/`[onnx]`)
**And** `pyproject.toml` `[rom]` gains `meshio` (it is no longer empty) and is also added to the
`all` extra; `meshio` is verified permissively licensed (MIT) so NFR8 (no GPL/LGPL in Core) holds
**And** the snapshot it produces is **mesh-free in the same sense**: only field values populate the
`(n_samples × n_dof)` matrix; coordinates/connectivity are not part of the trained columns (AR-R5).

**AC4 — Determinism.**
**Given** the same case sweep
**When** I run `data add-openfoam` twice
**Then** the written file and the registered dataset content hash are identical (the reader already
emits a deterministic row/column order; the chosen `--output` format must not reintroduce order
or precision nondeterminism — prefer a lossless default where available).

**AC5 — `data` stays a leaf; no engine change.**
**Given** the new code
**When** `lint-imports` runs
**Then** the `meshio` reader lives in the `data` leaf (no `cli`/`api`/`services`/`engine` import;
lazy backend import only); the CLI command is the only new `cli` code; there is **no** change to
`engine.py`, `core/config.py`, `services.py`, `plugins/`, or `trainers/` (AR-R1). The 3 import
contracts stay green.

**AC6 — Errors-as-a-feature; discriminating tests.**
**Given** failure modes
**When** they occur
**Then** each raises a `SurroForgeError` subclass with an actionable message; and tests cover:
the CLI happy path and error paths via the Typer runner (`tests/test_cli.py`), the `meshio`
install-hint path by monkeypatching the import to raise `ImportError` and asserting the hint
(closing the 6-3 deferred follow-up pattern), and a `meshio` happy-path test `skipif` meshio is
absent — each negative test pinned to the specific wrong implementation it catches (no vacant pins).

## Tasks / Subtasks

- [x] **Task 1 — `data add-openfoam` CLI command** (AC1, AC2)
  - [x] Added `@data_app.command("add-openfoam")` + `@_guard` in `src/surroforge/cli/main.py`,
        beside `data add`. Args/options: `case_root` (arg, `exists=True, file_okay=False`),
        `--field` (req), `--name` (req), `--samples` (optional), `--output` (optional), `--time`.
  - [x] Body parses `--samples` into a `{case: {param: value}}` mapping, calls
        `read_openfoam_sweep`, writes via the existing `_save_frame` (default `data/<name>.csv`),
        then `project.add_dataset`; reports rows/cols/hash via `_success`/`_info`.
  - [x] CLI stays thin — all parsing/numerics delegated to `surroforge.data`.
- [x] **Task 2 — samples-table → params helper** (AC2)
  - [x] `_openfoam_params(samples)` turns a DoE CSV (`case` column + parameter columns) into the
        mapping; raises a named `DataError` on a missing `case` column, a duplicate case, or a
        non-numeric parameter.
- [x] **Task 3 — meshio-backed reader behind `[rom]`** (AC3, AC5)
  - [x] `read_mesh_field(path, field, *, location)` in the `data` leaf (`data/openfoam.py`)
        **lazily** imports `meshio` via `_require_meshio()`, translating `ImportError` →
        `DataError` with `pip install 'surroforge[rom]'`. Exported from `surroforge.data`.
  - [x] Maps point/cell data to the flat cell-major layout (vectors `[x0 y0 z0 …]`); finite-guard
        + named errors for unknown/empty fields; reads only field values (mesh-free).
- [x] **Task 4 — packaging** (AC3)
  - [x] `pyproject.toml`: `rom = ["meshio>=5.3"]` (no longer empty); `meshio` added to `all`.
        meshio is MIT (permissive) — NFR8 holds. `[rom]` comment updated.
- [x] **Task 5 — tests** (AC6, AC4)
  - [x] `tests/test_cli.py`: `add-openfoam` happy path (registers + prints rows/cols/hash),
        deterministic-hash across two projects (AC4), bad-`--samples` error path (`Error:` exit 1).
  - [x] `tests/test_openfoam.py`: meshio install-hint via `monkeypatch` of `sys.modules` (runs
        regardless of meshio presence); meshio happy-path + unknown-field tests `importorskip`.
- [x] **Task 6 — gates & docs** (all ACs)
  - [x] `ruff`, `mypy src` (42 files), `lint-imports` (3 kept), `pytest` all green
        (365 passed, 5 skipped).
  - [x] CHANGELOG `[Unreleased]` entries added (CLI command + meshio reader).

### Review Findings (code-review 2026-06-14)

3-layer adversarial review (Blind / Edge / Auditor). Verdict: ship-able; scope/leaf/licensing
clean (AR-R1 respected, meshio MIT verified, `data` stays a leaf). Two real bugs found + verified
live; rest are robustness/test gaps. Deduped; 6 dismissed (bottom).

**Resolution (2026-06-14):** lead chose Parquet-when-available default + "apply all". 1 decision +
8 patches applied and test-pinned (clobber-guard, integer-case keys, early `--output` validation +
OSError wrap, mesh tensor/empty-block guards, samples empty/zero-param guards, 6 new CLI error
tests, pragma narrowed). Suite **371 passed / 6 skipped**; ruff / mypy / lint-imports green.

**Decision needed:**

- [x] [Review][Decision] Default snapshot output format when `--output` is omitted — CSV is lossy
      at ULP for arbitrary floats (spec advised "prefer lossless where available"); options:
      Parquet-when-pyarrow-available else CSV (lossless, but the file needs pyarrow to re-read) /
      always CSV / CSV + a "lossy" note. [cli/main.py:data_add_openfoam]

**Patches (unchecked):**

- [x] [Review][Patch] Name-collision **corrupts an existing dataset**: default output writes
      `data/<name>.csv` *before* `add_dataset` rejects the duplicate name, so the prior dataset's
      backing file is overwritten then the command aborts. Check the name (and output collision)
      before writing. [cli/main.py:data_add_openfoam] (blind+edge, verified)
- [x] [Review][Patch] Integer-like `case` names rejected: `str(row["case"])` via `iterrows()`
      upcasts an int `case` column to float → keys `'0.0'` ≠ dir `'0'` → "No parameters provided".
      Build keys from `table["case"].astype(str)`, not the upcast row. [cli/main.py:_openfoam_params]
      (blind+edge, verified)
- [x] [Review][Patch] Bad `--output` escapes `_guard`: an unsupported suffix is reported with the
      "predictions" noun and only *after* the full sweep is read; a non-writable path raises a raw
      `OSError`. Validate the suffix early (snapshot-appropriate message) and wrap the write →
      `DataError`. [cli/main.py:data_add_openfoam] (blind+edge)
- [x] [Review][Patch] `read_mesh_field` accepts tensor/higher-rank fields silently (`reshape(-1)`
      on `(n,9)`), unlike `_read_field`. Validate components ∈ {1,3} → `DataError`.
      [data/openfoam.py:read_mesh_field] (blind+edge)
- [x] [Review][Patch] `read_mesh_field` `location='cell'` with zero blocks → raw
      `np.concatenate([])` ValueError; the `np.asarray(...,float)` conversion is outside the
      try. Guard empty blocks + wrap conversion → `DataError`. [data/openfoam.py:read_mesh_field] (edge)
- [x] [Review][Patch] `_openfoam_params` silently accepts an empty table or a `case`-only
      (zero-parameter) table; the non-numeric error doesn't name the column. Add guards + name the
      column. [cli/main.py:_openfoam_params] (edge+auditor)
- [x] [Review][Patch] AC2/AC6 error paths under-tested — add CLI runner tests for a samples/case
      mismatch and a bad `--time`. [tests/test_cli.py] (auditor)
- [x] [Review][Patch] `# pragma: no cover` on `_require_meshio` over-broadly masks the
      genuinely-tested install-hint guard; narrow or remove it. [data/openfoam.py:_require_meshio] (edge+auditor)

**Dismissed (6):** `Path = typer.Option(None)` "non-Optional" annotation (matches the existing
`data add` convention); broad `except Exception` around `meshio.read` (acceptable external-library
boundary, re-raised as `DataError`); samples-rows-without-a-matching-dir ignored (the missing
direction already errors); `--time`/`--field`-absent message nuance (already handled); vector
default-CSV column count (verified fine); CSV run-to-run hash determinism (verified stable).

## Dev Notes

### What to reuse (do NOT reinvent)

- **The reader already exists.** `surroforge.data.read_openfoam_sweep(case_root, *, field, params,
  time)` returns the wide, deterministic snapshot DataFrame and raises named `DataError`s. The CLI
  must call it, not re-parse OpenFOAM. [Source: src/surroforge/data/openfoam.py]
- **The CLI registration pattern is fixed.** Model `add-openfoam` on `data add`
  (`src/surroforge/cli/main.py:209`): `@data_app.command(...)` + `@_guard`, open the project with
  `_project()`, call `project.add_dataset(file, name=…)`, report with `_success`/`_info`. Do not
  invent a new registration path. [Source: src/surroforge/cli/main.py:209-220]
- **Writing the frame: reuse `_save_frame`** (`cli/main.py:121`) — it already dispatches CSV /
  Parquet / HDF5 and emits actionable `DataError`s + install hints for the optional writers. The
  `--output` default should pick a deterministic, ideally lossless format; note that a CSV
  round-trip is not hash-stable to the last ULP (project-context Methods/Data `[GOTCHA]`), so
  prefer Parquet when `[formats]` is present or document the trade-off. [Source: src/surroforge/cli/main.py:121-193]
- **Lazy-backend + install-hint precedent.** `_save_frame`'s h5py branch and the `[smt]`/`[onnx]`
  trainers show the exact pattern for AC3's meshio guard: import inside the function, translate
  `ImportError` → `DataError` naming the extra. [Source: src/surroforge/cli/main.py:131-137]

### Architecture compliance (project-context.md — non-negotiable)

- **Layers:** the CLI is the top adapter (may import `data`/`api`/`services`); the `meshio` reader
  stays in the `data` leaf (imports only `errors` + siblings; **lazy** meshio import; no
  `cli`/`api`/`services`/`engine`). `lint-imports` enforces this.
- **Optional-dependency discipline `[GOTCHA]`:** never import `meshio` at module top — import it
  inside the reader so the registry/CLI load without it; translate the missing-backend
  `ImportError` into a `DataError` + `pip install 'surroforge[rom]'`.
- **Errors & CLI output `[MUST]`:** new CLI command uses `@_guard`; output via `_info`/`_success`
  (quiet-aware) — never bare `print`; tone precise/sober/English, no emojis. Domain failures raise
  `SurroForgeError` subclasses with actionable messages.
- **Determinism / hashing `[GOTCHA]`:** `hash_dataframe` is column- and row-order-sensitive; the
  reader already fixes order, so just don't reorder when writing. Prefer a lossless `--output`.
- **Licensing `[MUST]`:** Core stays permissive — `meshio` is MIT (verify before adding); it goes
  in the `[rom]` extra, never Core deps.
- **Conventions:** `from __future__ import annotations`; full type hints; Typer options in
  parameter defaults are allowed (the `B008` per-file ignore already covers `cli/main.py`).

### Previous-story intelligence (r5-1, done — review-hardened)

- r5-1's review produced 12 fixes; carry the lessons: **per-case parameter-key validation**
  (the samples-table helper must produce a consistent key set), **non-finite guards**, **named
  `DataError` for every failure**, and **vacant-pin-proof tests** (distinct fixtures; assert exact
  values, name the wrong impl each negative test catches). [Source: r5-1-openfoam-field-ingestion.md]
- The reader's `params` accepts a `{case: {param: value}}` mapping or a callable — the samples
  helper should build the mapping form. `read_openfoam_dimensioned_scalar` is available if a
  parameter should come from a case dict instead of the table.
- r5-1 deliberately deferred exactly this story's two items; no design surprise expected.

### Scope boundary

R5.2 = the CLI command + the `meshio` reader + packaging + tests. It does **not** add new trainer
behavior, mesh-aware reconstruction, or coordinate-channel features (those remain later Phase-2 /
research-track work). If a change reaches into `engine`/`trainers`, stop — the design is
adapter-and-leaf only.

### Project Structure Notes

- Touch: `src/surroforge/cli/main.py` (new command + helper), `src/surroforge/data/openfoam.py`
  (or a new `src/surroforge/data/mesh.py`) for the meshio reader + `data/__init__.py` export,
  `pyproject.toml` (`[rom]`/`all`), `tests/test_cli.py` + a meshio test, `CHANGELOG.md`, optional
  `docs/`.
- No change to `engine.py`, `core/config.py`, `services.py`, `plugins/`, `trainers/` (AR-R1).

### References

- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md#Epic R5] — R5 scope, R5.2 stub,
  AR-R1/AR-R5, the `[rom]` mesh-reader deferral.
- [Source: _bmad-output/implementation-artifacts/r5-1-openfoam-field-ingestion.md] — the library
  reader this builds on + its review-hardened patterns.
- [Source: src/surroforge/cli/main.py] — `data add` (209), `_guard` (85), `_project` (79),
  `_save_frame` (121), output helpers (65-72).
- [Source: src/surroforge/data/openfoam.py] — `read_openfoam_sweep` / `read_openfoam_field`.
- [Source: _bmad-output/project-context.md] — layer rules, optional-dependency + error discipline,
  hashing determinism, licensing, vacant-pin taxonomy.
- [Source: pyproject.toml#optional-dependencies] — the `[rom]` extra (currently empty) + the
  `[smt]`/`[onnx]`/`[formats]` lazy-import precedent.

## Dev Agent Record

### Agent Model Used

claude-opus-4-8[1m] (Claude Opus 4.8, 1M context)

### Debug Log References

- Two E501 long lines (CLI help string + a meshio test); wrapped. No logic change.

### Completion Notes List

- **CLI:** `surroforge data add-openfoam <case-root> --field … --samples … --name …` registers a
  sweep as a dataset via the existing `read_openfoam_sweep` → `_save_frame` → `project.add_dataset`
  path (default output `data/<name>.csv`). Thin wrapper, `@_guard`, `_success`/`_info` — no
  numerics in the CLI (AR-R1; engine/registry/trainers untouched).
- **meshio reader:** `read_mesh_field` (data leaf) lazily imports `meshio`; absence → `DataError`
  with `pip install 'surroforge[rom]'`. `[rom]` is no longer empty (`meshio>=5.3`, MIT → NFR8).
  meshio is **not** installed in this venv, so the happy-path tests `importorskip`; the install-hint
  test forces absence via `monkeypatch.setitem(sys.modules, "meshio", None)` so it runs everywhere
  (closes the 6-3 deferred "test the missing-backend hint" follow-up for this path).
- **Determinism (AC4):** a CLI-level test registers the same sweep in two projects and asserts
  equal dataset hashes; the reader's stable order does the work.
- **Layering:** `lint-imports` 3/3 — `data` stays a leaf (lazy meshio import only); the CLI is the
  only new adapter code.
- **Help-text convention:** the command docstring cites `(PRD-ROM 4.2)` like the other commands,
  not the story id (kept out of user-facing `--help`).
- Validation: **365 passed, 5 skipped** (3 smt, 2 meshio happy-path); ruff / mypy(42) /
  lint-imports(3) green.

### File List

- `src/surroforge/cli/main.py` (modified) — `data add-openfoam` command + `_openfoam_params` helper
  + `read_openfoam_sweep`/`load_table` imports.
- `src/surroforge/data/openfoam.py` (modified) — `read_mesh_field` + `_require_meshio`.
- `src/surroforge/data/__init__.py` (modified) — export `read_mesh_field`.
- `pyproject.toml` (modified) — `rom = ["meshio>=5.3"]`; `meshio` in `all`.
- `tests/test_cli.py` (modified) — `add-openfoam` happy / deterministic-hash / bad-samples tests.
- `tests/test_openfoam.py` (modified) — meshio install-hint + happy-path + unknown-field tests.
- `CHANGELOG.md` (modified) — `[Unreleased]` CLI + meshio entries.

## Change Log

| Date | Change |
|---|---|
| 2026-06-14 | R5.2 implemented: `surroforge data add-openfoam` CLI (sweep → registered dataset) + `read_mesh_field` meshio reader behind a non-empty `surroforge[rom]` extra (lazy import + install hint). Tests added (CLI happy/deterministic/error; meshio hint + skipif happy-path). Suite 365 passed / 5 skipped; all gates green. Status → review. |
| 2026-06-14 | Code review (3-layer): 1 decision + 8 patches applied — clobber-before-write guard, integer-case-name keys, early `--output` suffix validation + OSError wrap, Parquet-when-available default, mesh tensor/empty-block guards, samples empty/zero-param guards, narrowed meshio pragma, +6 CLI error-path tests + tensor-reject test. Suite 371 passed / 6 skipped. Status → done. |
