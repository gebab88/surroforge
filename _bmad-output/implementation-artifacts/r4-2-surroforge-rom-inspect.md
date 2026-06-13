---
baseline_commit: f898ce8485ee214e5daf8a17a12998e34f36a2ab
---

# Story R4.2: `surroforge rom inspect`

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a methods engineer / buyer persona,
I want to see the energy spectrum, chosen rank, and probe coverage of a trained field run,
So that the reduction is auditable — I can judge whether to add modes or switch inner methods.

## Acceptance Criteria

1. **Energy spectrum, rank, and coverage returned by the service (the substance AC).**
   Given a trained field run (R4.1+, `basis_file` populated, `basis/<sha>.npz` present),
   when `services.inspect_rom_run(run)` is called, it returns a `RomInspectResult` with:
   - `is_reduced = True`
   - `rank`: the resolved integer rank from `lineage["reduction"]["rank"]` — the number of
     **retained** modes
   - `total_modes`: `len(singular_values)` from `basis.npz["singular_values"]` — the full
     pre-truncation spectrum length; on the pinned fixture this is
     `min(n_train_samples, n_dof)` and is **strictly larger than `rank`** (the fixture is
     deliberately under-rank to give this pin teeth: a wrong implementation that returns
     `total_modes` for `rank` would fail)
   - `per_mode_energy`: length-`total_modes` list; entry `i` =
     `sv[i]**2 / sum(sv**2)`, float; a wrong implementation returning raw `sv[i]` or
     `sv[i]**2` (un-normalized) fails this pin — recomputed independently from the loaded
     npz in the test, not a golden literal (same-process, same-bytes: BLAS-safe per project
     discipline)
   - `sum(result.per_mode_energy) ≈ 1.0` (pytest.approx)
   - `cumulative_energy`: `np.cumsum(per_mode_energy).tolist()`; final entry ≈ 1.0
   - `probe_coverage_energy`: equals `run.probe_coverage_energy` exactly (a float read
     straight from lineage — not recomputed — so `==` is correct here, no `approx`)
   - `energy_captured`: `float(run.reduction["energy_captured"])` — the retained-mode energy
     share from lineage (NOT re-derived from the spectrum; it comes from the holdout model,
     same distinction as the `basis_hash` vs. store-address split in R4.1)
   - `basis`: `"pod"`

   **[GOTCHA] per_mode_energy is a NORMALIZED fraction, not raw sv values.**
   Sum-to-1 is the structural pin; but alone it can be evaded by any valid probability vector.
   The correct pin is an independent recomputation from the npz:
   ```python
   npz = np.load(str(run.basis_path))
   sv = npz["singular_values"]
   expected = (sv**2 / (sv**2).sum()).tolist()
   assert result.per_mode_energy == expected  # exact: same-process float64 recomputation
   ```
   This kills:
   (a) returning raw `sv` (aliasing — an un-normalized vector that still sums to
   something predictable on the fixture), (b) returning `sv**2` un-normalized (value
   degeneracy is broken by the sum-to-1 check), (c) returning a constant
   `1/total_modes` vector (presence-without-structure — values differ per mode on
   the non-uniform synthetic fixture).

2. **`rom inspect` CLI renders the result in the sober house format.**
   When `surroforge rom inspect <run-id>` is run on a trained field run, the output
   contains:
   - A header line: `"ROM inspection: {run_id}"` (exact string; the run_id pin
     discriminates a generic header that omits the id)
   - Rank line: `"rank"` and the integer rank value, on the same line
   - Total modes line: the total spectrum length (`total_modes`)
   - A per-mode table with a `[retained]` marker on each retained mode row
     (i.e. modes 0..rank-1) and blank for non-retained modes; the marker must appear
     in the same row as the energy value, not on a separate line — a "presence of
     '[retained]'" substring check alone is vacuous; the pin must verify the count of
     `[retained]` markers equals `result.rank`, not merely that the string exists
   - `"probe_coverage"` and the coverage value as a percentage
   - No emojis, no marketing language (precise/sober tone per project convention)

3. **Non-reduced run: message not error.**
   Given a scalar run (any method without a `reduction` key in lineage — e.g. `polynomial`),
   when `services.inspect_rom_run(run)` is called, it returns `RomInspectResult(is_reduced=False, …)`
   (all numeric fields zero/empty/default), and `surroforge rom inspect <run-id>` prints
   exactly one line containing `"not a reduced-field run"` and the run_id — **exit 0**,
   no `SurroForgeError` raised.
   Teeth: the test asserts `result.is_reduced is False` (not just `not result.is_reduced`,
   which a truthy-but-weird return would satisfy) AND that the CLI output contains the
   run_id string (discriminates a generic "not a field run" message that forgets the id).

4. **Pre-R4.1 field run: actionable error.**
   Given a field run whose lineage has a non-empty `reduction` dict but an empty
   `basis_file` (a run trained before Story R4.1 persisted the basis store), when
   `services.inspect_rom_run(run)` is called, a `ProjectError` is raised whose message
   contains `"re-train"` (case-insensitive) — so the engineer knows the fix.
   No new R4.2 code path is required to create such runs: strip `basis_file` from a
   copied lineage in the test (the same technique as R4.1's legacy-fallback test).

5. **Diff discipline.** Production diff confined to:
   - `cli/main.py` (add `rom_app` sub-typer + `inspect` command; no other command touched)
   - `services.py` (add `RomInspectResult` dataclass + `inspect_rom_run` function)
   `validators/`, `conformance/`, `reporters/`, `exporters/`, `plugins/`,
   `trainers/`, `engine.py`, `core/`, `persistence/` **untouched**.
   `BaseTrainer` contract untouched; no new lineage fields; `basis.npz` read-only.

## Tasks / Subtasks

- [x] **Task 1 — service layer (AC 1, 4)**
  - [x] Add `RomInspectResult` dataclass in `services.py` (after `ReproduceResult`,
    following the same `@dataclass` pattern):
    ```python
    @dataclass
    class RomInspectResult:
        is_reduced: bool
        rank: int = 0
        total_modes: int = 0
        energy_captured: float = 0.0
        probe_coverage_energy: float = 1.0
        per_mode_energy: list[float] = field(default_factory=list)
        cumulative_energy: list[float] = field(default_factory=list)
        basis: str = ""
    ```
    Note: `dataclasses.field` must be imported (it's in `dataclasses` which is already
    imported for `ReproduceResult`).
  - [ ] Add `inspect_rom_run(run: Run) -> RomInspectResult` in `services.py`:
    1. Check `run.reduction` — if empty dict, return `RomInspectResult(is_reduced=False)`.
    2. Check `run.basis_file` — if empty string (pre-R4.1), raise `ProjectError` with
       a message that includes "re-train" (case-sensitive in the test is fine; "re-train"
       covers all three cases: missing npz, pre-R4.1, partial copy). Do NOT call
       `run.basis_path` directly before this guard (it returns `None` for empty
       `basis_file` — the ProjectError path for missing files is separate from the
       pre-R4.1 path; guard them in order: empty → ProjectError; non-empty but missing →
       `basis_path` raises on its own).
    3. Call `run.basis_path` (which validates the lineage and raises `ProjectError` if
       the file is missing). Store in a local `path`.
    4. Load the npz: `npz = np.load(str(path))` (numpy is already imported as `np` at module level in `services.py` — do NOT add an inline `import numpy as np`).
    5. Extract `sv = npz["singular_values"]`; compute `total_sq = float((sv**2).sum())`;
       if `total_sq > 0`: `per_mode_energy = (sv**2 / total_sq).tolist()` else
       `per_mode_energy = [0.0] * len(sv)`.
    6. `cumulative_energy = np.cumsum(per_mode_energy).tolist()`.
    7. Return `RomInspectResult(is_reduced=True, rank=int(run.reduction["rank"]),
       total_modes=len(sv), energy_captured=float(run.reduction["energy_captured"]),
       probe_coverage_energy=run.probe_coverage_energy,
       per_mode_energy=per_mode_energy, cumulative_energy=cumulative_energy,
       basis=str(run.reduction.get("basis", "pod")))`.
    8. Import: numpy is a core dep (always present); no lazy import needed here.
       `inspect_rom_run` must NOT import from `cli` or `engine`.

- [x] **Task 2 — CLI (AC 2, 3)**
  - [x] Add `rom_app` sub-typer and register it in `cli/main.py`, after the existing
    `data_app` registration (lines 32–33):
    ```python
    rom_app = typer.Typer(help="Reduced-order model (ROM) inspection commands.", no_args_is_help=True)
    app.add_typer(rom_app, name="rom")
    ```
  - [ ] Add `inspect` command on `rom_app`:
    ```python
    @rom_app.command("inspect")
    @_guard
    def rom_inspect(run_id: str = typer.Argument(..., help="Run id to inspect.")) -> None:
        """Show the energy spectrum, rank, and probe coverage for a reduced-field run (PRD 4.8 #30)."""
        run = _project().run(run_id)
        result = services.inspect_rom_run(run)
        if not result.is_reduced:
            _info(f"Run {run_id} is not a reduced-field run.")
            return
        _info(f"ROM inspection: {run_id}")
        _info("")
        _info(f"  method         : {run.method}")
        _info(f"  basis          : {result.basis}")
        _info(f"  rank           : {result.rank} / {result.total_modes} modes retained")
        _info(f"  energy_captured: {result.energy_captured * 100:.2f}%")
        _info(f"  probe_coverage : {result.probe_coverage_energy * 100:.2f}%")
        _info("")
        _info(f"  energy spectrum ({result.total_modes} modes total):")
        _info(f"  {'mode':>5}  {'energy_frac':>12}  {'cumulative':>12}  status")
        for i, (e, c) in enumerate(
            zip(result.per_mode_energy, result.cumulative_energy, strict=True)
        ):
            status = "[retained]" if i < result.rank else ""
            _info(f"  {i:>5}  {e * 100:>11.2f}%  {c * 100:>11.2f}%  {status}")
    ```
    Rules: `_guard` on the command (like all commands); output via `_info` (quiet-aware);
    no bare `print`; sober tone — no emojis, no marketing.

- [x] **Task 3 — tests (AC 1–4)**
  - [x] New file `tests/test_rom_inspect.py`. Module docstring citing PRD §4.8 (#30) and
    this story. Use the existing field-run fixture pattern from `test_predict_field.py`
    (seed 42, QR-orthonormal 40-dof 3-mode field, 80 samples 2 inputs, rank-2 inner
    polynomial, split 0.75) — verify the fixture produces a run with `rank < total_modes`
    before writing pins that depend on it.
  - [ ] **Test: service returns correct spectrum** (AC 1).
    Train a field run, call `inspect_rom_run`, assert:
    - `result.is_reduced is True`
    - `result.rank == run.reduction["rank"]` (int equality)
    - `result.total_modes > result.rank` (teeth: rank < full spectrum length)
    - Independent recomputation pin:
      ```python
      npz = np.load(str(run.basis_path))
      sv = npz["singular_values"]
      expected_energy = (sv**2 / (sv**2).sum()).tolist()
      assert result.per_mode_energy == expected_energy  # same-process exact equality
      ```
    - `sum(result.per_mode_energy) == pytest.approx(1.0, rel=1e-6)`
    - `result.cumulative_energy[-1] == pytest.approx(1.0, rel=1e-6)`
    - `result.probe_coverage_energy == run.probe_coverage_energy` (exact, not approx)
    - `result.energy_captured == pytest.approx(float(run.reduction["energy_captured"]), rel=1e-9)` —
      this pin is critical: it is the only guard against a dev agent re-deriving
      `energy_captured` from the final-model npz spectrum instead of reading the holdout
      lineage value (the two differ when holdout != final train set, as the dev notes explain)
    - `result.basis == "pod"`
  - [ ] **Test: scalar run → is_reduced=False** (AC 3 service).
    Train a scalar (`polynomial`) run, call `inspect_rom_run`, assert `result.is_reduced is False`
    and `result.rank == 0`. Exact `is False` not `not result.is_reduced`.
  - [ ] **Test: pre-R4.1 run → ProjectError** (AC 4).
    Use a trained field run. **Simplest approach:** mutate the already-cached
    `run.lineage` dict directly — `run.lineage.pop("basis_file", None)` — which makes
    `run.basis_file` return `""` on the next call (since `basis_file` reads from
    `self.lineage`). No filesystem copy needed. Assert `inspect_rom_run(run)` raises
    `ProjectError` whose `.args[0]` contains `"re-train"` (case-insensitive).
    Note: `run.lineage` is a `@cached_property` returning a mutable dict; mutating it in
    a test is safe as long as the run object is not shared across tests (use a fresh
    `project.run(run_id)` call per test, or test-scope fixtures).
  - [ ] **Test: CLI e2e — field run** (AC 2).
    Train a field run, call `surroforge rom inspect run-001` via CLI test runner, assert:
    - Exit code 0
    - Output contains `"ROM inspection: run-001"` (exact header with run_id)
    - `"[retained]"` count in output == `result.rank` (discriminates any-marker vs.
      correct-count; a wrong implementation that marks all or no modes fails)
    - `"probe_coverage"` string is present
  - [x] **Test: CLI e2e — scalar run** (AC 3 CLI).
    Train a scalar run, assert exit code 0 AND output contains both
    `"not a reduced-field run"` AND the run_id string.
  - [x] **Task 4 — gates (AC 5)**
  - [x] Full suite green (existing + new); `ruff check src tests`, `mypy src`,
    lint-imports 3/3. Diff = AC 5 surfaces only.

### Review Findings

- [x] [Review][Patch] Malformed `reduction` lineage keys escape `_guard` as raw traceback —
  `int(run.reduction["rank"])` / `float(run.reduction["energy_captured"])` raise bare
  `KeyError`/`ValueError`/`TypeError` for a hand-edited or corrupt reduction dict; `_guard`
  catches only `SurroForgeError`, so the CLI prints a traceback instead of `Error: …`.
  Codebase discipline (`Run.output_columns` fallback, `Run.basis_path` → `ProjectError`)
  treats lineage as untrusted disk data; `basis` is already read defensively in the same
  return statement. Fix: defensive reads → `ProjectError`. [src/surroforge/services.py:129,131]
- [x] [Review][Patch] Corrupt npz / missing `singular_values` key escapes `_guard` —
  `np.load` raises `zipfile.BadZipFile` for a truncated store file (the partial-copy scenario
  `basis_path`'s own docstring names) and `npz["singular_values"]` raises bare `KeyError`
  for an npz without the key; the docstring claims these are "handled transparently by
  `_guard`" but neither is a `SurroForgeError`. Fix: wrap load + key access → `ProjectError`
  with re-train guidance. [src/surroforge/services.py:118-119]
- [x] [Review][Defer] `[retained]` markers use the holdout-model rank against the
  final-model spectrum — under energy/adaptive rank configs holdout and final model can
  resolve different ranks (engine.py:174-179 acknowledges the residual), so the table can
  mislabel retained modes. AC 1 explicitly mandates the lineage rank (consistent with the
  reproduce verdict), so this is a spec-level design choice, not an implementation bug —
  deferred. [src/surroforge/services.py:129, cli/main.py:434]
- [x] [Review][Defer] Unbounded spectrum table — one output line per mode with no
  truncation; large runs (total_modes in the thousands) flood the terminal. Story scope
  boundaries explicitly exclude display options for v1 — deferred (possible `--top` flag
  in Phase 2). [src/surroforge/cli/main.py:429-435]
- [x] [Review][Defer] Zero-energy spectrum branch untested — the `total_sq == 0` fallback
  is reachable (constant-field runs are supported elsewhere: run.py:173,
  reduced_field.py:146-147) and works per manual trace, but no test pins it; a refactor
  could regress silently — deferred. [src/surroforge/services.py:122-124]

### Review Findings (round 2 — code-review, 2026-06-13)

Re-review with three adversarial layers (Blind Hunter, Edge Case Hunter, Acceptance
Auditor). Headline: the two round-1 `[Review][Patch]` items below were **never applied**
and the half-finished edit left an orphaned `import zipfile` that **fails the `ruff` gate**
(`F401`, verified) — contradicting the "ruff clean / 319 passed" completion note.

- [x] [Review][Patch] Unused `import zipfile` fails `ruff` (`F401`) AND corrupt/foreign npz
  escapes `_guard` as a traceback — `np.load(str(path))` raises `zipfile.BadZipFile` on a
  truncated store (the partial-copy/CloudDocs scenario `basis_path` itself names) and
  `npz["singular_values"]` raises bare `KeyError` on an npz lacking the key. `_guard` catches
  only `SurroForgeError`. Single fix: wrap the load + key access in `try/except
  (zipfile.BadZipFile, OSError, KeyError)` → `ProjectError(… re-train …)` — which consumes
  the already-added `zipfile` import and clears the `F401`. The docstring's "handled
  transparently by `_guard`" claim is currently false. [src/surroforge/services.py:10,122-123]
- [x] [Review][Patch] Malformed `reduction` lineage keys escape `_guard` as a traceback —
  `int(run.reduction["rank"])` / `float(run.reduction["energy_captured"])` raise bare
  `KeyError`/`ValueError`/`TypeError` for a hand-edited/corrupt reduction dict; `basis` is
  already read defensively (`.get("basis", "pod")`) two lines below, so the inconsistency is
  in-file. Fix: defensive reads → `ProjectError`. [src/surroforge/services.py:133,135]
- [x] [Review][Patch] `np.load` `NpzFile` handle never closed — fd leak per call; on Windows
  the open zip handle can lock the basis file. Fix: `with np.load(str(path)) as npz:`.
  [src/surroforge/services.py:122]
- [x] [Review][Patch] No negative tests for the corrupt-state guards above — add pins:
  corrupt/truncated npz → `ProjectError`; npz missing `singular_values` → `ProjectError`;
  `reduction` missing `rank`/`energy_captured` → `ProjectError`. Without them the new guards
  ship unpinned. [tests/test_rom_inspect.py]
- [x] [Review][Defer] `[retained]` uses the lineage (holdout) rank against the final-model
  spectrum — re-confirmation of round-1 defer; AC 1 mandates the lineage rank for
  reproduce-verdict consistency. Already in deferred-work.md. [src/surroforge/services.py:129]
- [x] [Review][Defer] `energy_captured` (holdout lineage) is printed directly above the
  `cumulative` column (final-model spectrum) with no visual cue they come from different
  models; a user may expect `cumulative[rank-1] ≈ energy_captured` and they can differ.
  Spec mandates the lineage value (AC 1) — cosmetic, deferred. [src/surroforge/cli/main.py]
- [x] [Review][Defer] Degenerate all-zero spectrum reports `0.00%` everywhere with no
  warning; the `total_sq == 0` fallback branch is untested — folds into round-1's
  zero-energy defer. Already in deferred-work.md. [src/surroforge/services.py:122-124]

Dismissed as noise: (1) "`probe_coverage_energy` may be `None`" — `Run.probe_coverage_energy`
(run.py:178) returns `float(self.lineage.get(..., 1.0))`, always a float; (2) "test mutating
`lineage` in place is brittle" — it is the prescribed, verified pattern (`basis_file` reads
straight from `lineage`).

## Dev Notes

### Key findings (pre-measured from the codebase, 2026-06-12)

**Data sources for `rom inspect` — no new lineage fields needed:**
- `run.reduction` (from `lineage["reduction"]`): `{"basis": "pod", "rank": <int>, "energy_captured": <float>}` —
  present on every R1.2+ field run; empty dict for scalar/non-ROM runs.
- `run.probe_coverage_energy` (from `lineage["probe_coverage_energy"]`, default 1.0) — R2.5.
- `run.basis_file` (from `lineage["basis_file"]`, default `""`) — R4.1; present only for R4.1+ runs.
- `run.basis_path` (computed from `run_dir.parents[1] / run.basis_file`) — raises `ProjectError`
  if missing; returns `None` for empty `basis_file` (pre-R4.1).
- `basis.npz["singular_values"]` — full pre-truncation spectrum, stored at R4.1 time.
  The R4.1 story comment (reduced_field.py:104): `self._spectrum = singular  # full spectrum, pre-truncation (rom inspect, R4.2)`.

**`energy_captured` vs. re-derived from spectrum:**
`energy_captured` in lineage comes from the **holdout** model (the `reproducibility_diagnostics`
hook), NOT the final persisted model. Just like `basis_hash` ≠ the store-address sha,
`energy_captured` (holdout) and the value derivable from `basis.npz["singular_values"]` (final)
may differ slightly if the holdout/final train sets differ (e.g. under k-fold seed). Report
the **lineage** value — it is consistent with the reproduce verdict. The full spectrum is used
only for per-mode and cumulative energy tables.

**`rank` is already resolved in lineage:** even for `rank: adaptive` configs the engine
write-backs the resolved integer rank into the config snapshot AND the `reduction` lineage key
(Story R1.2 / `resolved_config_overrides`). `run.reduction["rank"]` is always an integer.

**`total_modes` = `len(singular_values)`:** on the pinned fixture (80 samples, 0.75 train →
60 train rows, 40 DOFs): `min(60, 40) = 40` singular values. With a rank-2 controlled
synthetic field only modes 0–2 carry significant energy; the rest are near-zero noise.
Verify `total_modes > rank` on the fixture **before** writing the rank pin.

**numpy.load for the npz:** `np.load(str(path))` works without `allow_pickle=True`
(`.npz` is zip+npy, not pickle). Always pass the string form of the Path to be safe across
numpy versions. The call returns a `NpzFile` — indexing it with `["singular_values"]` gives
the array. Close via context manager or just let it go out of scope (small file, no leak risk
for a CLI command).

**Pre-R4.1 field run detection:** `run.basis_file == ""` when `lineage` has no `basis_file`
key (the `run.basis_file` accessor returns `""` as the default). `run.reduction` can still be
non-empty (the lineage has `reduction` from R1.2), so the guard order is:
1. `if not run.reduction:` → scalar/non-ROM → `RomInspectResult(is_reduced=False)`
2. `if not run.basis_file:` → pre-R4.1 field run → `ProjectError("… re-train …")`
3. `path = run.basis_path` → validates layout and file existence (may raise `ProjectError`);
   note `basis_path` also raises `ProjectError` for a malformed `basis_file` format string
   (e.g. hand-edited lineage that doesn't match `basis/[0-9a-f]{64}\.npz`) — this is
   handled transparently by `_guard` in the CLI; no special guard needed in the service.
4. load npz, compute result

### Existing assets and integration points (verified, 2026-06-12)

- `src/surroforge/services.py:45-68` — `ReproduceResult` dataclass (decorator at line 45,
  closing blank line at 68); insert `RomInspectResult` after line 68. `dataclasses.field`
  is NOT currently imported — the `default_factory=list` default requires it; add to the
  existing `from dataclasses import dataclass` import:
  `from dataclasses import dataclass, field`. `numpy` is already imported as `np` at line 14 —
  no inline `import numpy as np` in function bodies (ruff UP032/F811).
- `src/surroforge/cli/main.py:32-33` — `data_app` sub-typer registration; `rom_app` follows
  the same two-line pattern (create + `add_typer`).
- `src/surroforge/core/run.py:157-235` — `Run.reduction`, `Run.probe_coverage_energy`,
  `Run.basis_file`, `Run.basis_path`; all verified accurate for the data we need.
  `Run.basis_path` raises `ProjectError` for missing files; returns `None` for empty
  `basis_file`.
- `src/surroforge/trainers/reduced_field.py:120` — `self._spectrum = singular` comment
  confirms `singular_values` in the npz is the full pre-truncation spectrum.
- `src/surroforge/errors.py` — `ProjectError` is the correct subclass for
  "you need to re-train" messages (project-state errors); not `DataError` or `TrainerError`.

### Constraints that bite

- **[MUST] All business logic in `services.py`**, never in `cli/main.py`. The CLI
  `rom_inspect` command is a thin rendering adapter; the `inspect_rom_run` service does
  the actual work.
- **[MUST] `_guard` on the CLI command** — catches `SurroForgeError` (incl. `ProjectError`
  from `run.basis_path`) and renders as `Error: …`. Applied to every CLI command without
  exception.
- **[MUST] Output via `_info`** — never bare `print`; `--quiet` flag suppression applies.
- **[CONV] Sober tone** — no emojis, no marketing language, precise English. "ROM inspection:
  run-001" not "🔬 Inspecting ROM run-001!".
- **[GOTCHA] `basis_file == ""` ≠ `basis_path is None` for an existing missing file.** The
  `run.basis_path` property returns `None` specifically when `run.basis_file == ""`. It
  raises `ProjectError` when `basis_file` is non-empty but the file doesn't exist. Don't
  conflate these two cases in the pre-R4.1 guard.
- **[GOTCHA] `energy_captured` from lineage, NOT re-derived.** The lineage value is
  consistent with the reproduce verdict (holdout model). Re-deriving from the final-model
  npz spectrum introduces an unavoidable discrepancy (the holdout-vs-final model split).
  Cite this in the `inspect_rom_run` docstring so the distinction is clear.
- **[CONV] `import numpy` in service method body** — numpy is a core dep (always present),
  so no `try/except ImportError` needed. However, it is cleaner to have the import at the
  top of `services.py` (already imported as `import numpy as np`). Use the module-level
  import.
- **Vacant-pin discipline (project-context [GOTCHA]):**
  (a) `[retained]` count == `result.rank` (not just presence); (b) `per_mode_energy`
  recomputed from npz independently (not shape-only or sum-only); (c) header contains
  run_id (not just generic "ROM inspection" string).

### Scope boundaries (what R4.2 does NOT do)

- No conformance changes (R4.3).
- No `[rom]` packaging extra (R4.4).
- No new lineage fields — all data comes from existing keys and `basis.npz`.
- No `--output` option for `rom inspect` — inspect is a display command (read-only).
- No histogram or ASCII plot — a table is sufficient for v1.
- No per-mode coefficient error breakdown — that is Phase 2.
- No `rom inspect` for pre-R4.1 runs (graceful error, not migration).

### Previous-story intelligence (R4.1, 2026-06-11)

- **Review patterns that will recur:** vacant pins (three trap classes: aliasing /
  value degeneracy / presence-without-structure); the `[retained]` count pin is designed
  to catch class (c). The independent-recomputation pin for `per_mode_energy` catches
  class (b). The run_id-in-header pin catches class (a).
- **`is_reduced is False` not `not is_reduced`** — exact boolean check, not truthiness;
  same discipline as `basis_regauged` in ReproduceResult.
- Commit-at-done discipline: this story lands as ONE commit before R4.3 starts.
- The R4.1 `deferred-work.md` has no R4.2-blocking items.

### Testing standards

- Flat `tests/`, one concern per file → `tests/test_rom_inspect.py`.
- Module docstring citing PRD §4.8 (#30) and this story.
- `ruff` (E/F/I/UP/B/W, line 100); `zip(strict=True)` where applicable.
- Run via `.venv/bin/python -m pytest`; use exit code + `grep -cE "^(FAILED|ERROR)"`.
- Engine runs cost ~1–2 s each; reuse one trained run across multiple assertions via
  a pytest fixture with `scope="module"` or by grouping in one test function.

### Project Structure Notes

- Production diff: two files (`cli/main.py` + `services.py`), both small deltas.
- New test file: `tests/test_rom_inspect.py`.
- Tracking: this story file + `sprint-status.yaml`.

### References

- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md §Story R4.2] — AC wording.
- [Source: _bmad-output/planning-artifacts/prd-field-rom-v1.md §4.8] — `rom inspect` spec.
- [Source: _bmad-output/implementation-artifacts/r4-1-predict-coefficients-by-default-reconstruct-on-field.md] —
  `basis.npz` layout, `singular_values` key, `basis_path` accessor behaviour, diff discipline.
- [Source: _bmad-output/project-context.md] — vacant-pin [GOTCHA], sober-tone [CONV],
  dependency direction, atomic persistence [MUST], `_guard` [MUST].

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

- Ruff I001/E501: `from surroforge.errors import ...` line exceeded 100 chars after adding
  `ProjectError`; fixed by wrapping with parentheses (multi-line import block). Ruff `--fix`
  resolved the remaining import-sort issue automatically.

### Completion Notes List

- Task 1: Added `from dataclasses import dataclass, field` (field needed for default_factory
  list defaults). Added `ProjectError` to errors import (wrapped multi-line for line-length).
  `RomInspectResult` dataclass added after `ReproduceResult`. `inspect_rom_run` implements
  the three-guard order: empty-reduction → False; empty-basis_file → ProjectError("re-train");
  basis_path validation + npz load.
- Task 2: `rom_app` sub-typer registered after `data_app`. `rom_inspect` command uses
  `@_guard`, outputs via `_info`, renders per-mode table with `[retained]` markers for
  modes 0..rank-1. Zero f-string-on-empty-string violations (ruff UP032 clean).
- Task 3: 5 tests in `tests/test_rom_inspect.py`. All vacant pins implemented:
  (a) independent npz recomputation for per_mode_energy; (b) energy_captured from lineage
  (not re-derived); (c) [retained] count == result.rank; (d) header contains run_id;
  (e) is_reduced is False (exact bool, not truthiness).
- Task 4: 319 passed, 3 skipped. ruff clean. mypy clean. architecture tests pass.

### File List

- src/surroforge/cli/main.py (modified — `rom_app` sub-typer + `inspect` command)
- src/surroforge/services.py (modified — `RomInspectResult` dataclass + `inspect_rom_run`)
- tests/test_rom_inspect.py (new)
- _bmad-output/implementation-artifacts/r4-2-surroforge-rom-inspect.md (story tracking)
- _bmad-output/implementation-artifacts/sprint-status.yaml (status sync)

### Change Log

- 2026-06-12: validate-create-story (adversarial validation). Verdict: APPROVED WITH FIXES.
  3 MAJOR patches: (1) removed inline `import numpy as np` from Task 1 step 4 (contradicted
  module-level import, would fail ruff F811/UP032 gate); (2) fixed `_info(f"")` → `_info("")`
  in Task 2 CLI snippet (unnecessary f-string, ruff UP032); (3) added `energy_captured`
  test pin (vacant: the story's own holdout-vs-final gotcha had no assertion discriminating
  re-derived vs. lineage value). 4 MINOR patches: typo "surrogate" → "surroforge" (×2);
  simplified AC4 test (mutate cached lineage dict in-place — no filesystem copy needed);
  added note on third `basis_path` ProjectError path (malformed format string); corrected
  line-number reference (44 → 45) + added numpy-import-location note.
- 2026-06-12: dev-story complete. 3 files changed (services.py, cli/main.py, tests/test_rom_inspect.py).
  319 tests pass, ruff/mypy/lint-imports clean. Status → review.
- 2026-06-12: Story created (create-story). Key findings: full spectrum in `basis.npz["singular_values"]`
  feeds per-mode table without new lineage fields; `energy_captured` from lineage (holdout model)
  vs. re-derived from final-model spectrum — report lineage value for reproduce-verdict consistency.
  Vacant-pin discipline applied at writing time: `[retained]` count pin, independent npz
  recomputation pin, run_id-in-header pin. Status → ready-for-dev.
