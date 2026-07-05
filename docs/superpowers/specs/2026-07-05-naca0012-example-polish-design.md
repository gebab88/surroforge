# NACA 0012 Example Polish Design

## Goal

Polish the existing real-CFD example `examples/naca0012_openfoam/` without changing what
it does: slim the code (delete a dead export path, extract the VTK I/O into a reusable
module), consolidate the documentation (one tracked English README as the single
reference), and tidy the working tree. **No new scenario, no new features, no committed
visual output.** The example's behaviour — same console output, same artifacts, same fit
quality — is preserved and proven against the real 110-case sweep on disk.

## Scope & File Structure (Section 1, approved)

Approach A: the example stays a thin script pair and keeps its ParaView story, with the
VTK/HDF plumbing extracted into a sibling module.

```
examples/naca0012_openfoam/
├─ run_sweep.py            # untouched
├─ ingest_and_train.py     # slimmed: 616 → ~300 lines
├─ surrogate_vtk.py        # NEW: pure VTK/VTKHDF I/O module (~230 lines)
├─ README.md               # consolidated single reference (English, tracked)
└─ LOCAL_SETUP.md          # gitignored, shrunk to machine-local facts only
```

`examples/09_naca0012_openfoam_surrogate.py` (the synthetic CI companion) is untouched.

## Code Architecture (Section 2, approved)

### Dead code — deleted, not moved

`main()` exports ParaView output solely through `write_surrogate_vtkhdf`. The old
"one `.vtu` per case + `.pvd` collection" path is unreachable and is removed (~98 lines):

- `write_surrogate_vtk_cases`
- `write_surrogate_internal_vtu`
- `_vtk_float32_payload`
- `_write_pvd_collection`

### New module `surrogate_vtk.py`

Pure VTK/HDF I/O, **no SurroForge imports** — reusable against any OpenFOAM sweep laid
out the same way. Receives the live VTKHDF chain unchanged:

- Public: `write_surrogate_vtkhdf(run_dir, output, samples, fields, *, field_name="p") -> Path`
- Private: `VtuMesh` (NamedTuple), `_read_vtu_mesh`, `_decode_vtk_data_array`,
  `_vtk_dtype`, `_data_array`, `_internal_vtu_template`, `_step_offsets`
- Imports: `base64`, `struct`-free (`struct` was only used by the dead path),
  `xml.etree.ElementTree`, `numpy`, `pandas`, `pathlib`, `typing`
- `h5py` stays a lazy import with the existing
  `pip install 'surroforge[formats]'` error message
- The misindented closing paren on the `raise ValueError(...)` (old line ~381) is fixed
  in the move

### Slimmed `ingest_and_train.py`

Keeps the CFD-specific logic and the ex09-mirroring `main()` flow:

- Kept: `CL_WINDOW`/`CL_RTOL`/`CL_ATOL`, `CoeffReading`, `read_coeffs`, `_nn`,
  `_candidates`, `_key`, `main`, `_report`, `_best_label`, argparse CLI
- New: `from surrogate_vtk import write_surrogate_vtkhdf` (sibling import; works because
  Python puts the script's directory on `sys.path[0]`)
- Removed: the four dead functions plus the now-unused `base64`/`struct`/
  `xml.etree.ElementTree` imports
- `main()` is functionally unchanged: same ingestion, same surrogate loops, same
  exports, same VTKHDF output path

## Documentation Consolidation (Section 3, approved)

### Known contradictions to resolve

README and LOCAL_SETUP currently disagree; the acceptance run (Section 5) regenerates
the numbers and becomes the single truth in the README:

1. **Convergence:** README says "`--iters 1000` → 90/100 converge, unconverged
   |AoA| ≳ 11–12°"; LOCAL_SETUP documents the actual verified run: `--iters 1500` →
   **90/110**, unconverged +12/+14/+15/+16° across all 5 speeds (+13° converges).
2. **Fit numbers:** README quotes field torch-mlp R² ≈ 0.998 and scalar mlp ≈ 0.975;
   LOCAL_SETUP's results table says **0.986** and **0.994**.

### Target state

**`README.md`** (tracked, English, ~180 lines) absorbs the reusable, machine-agnostic
reference from LOCAL_SETUP §§2–12:

- New section **"Case setup in detail"**: BC table (`freestreamVelocity` inlet/outlet,
  `noSlip` walls, `empty` frontAndBack), `forceCoeffs` configuration (rotated
  lift/drag directions, `CofR (0.25 0 0)`, `writeInterval 1`), numerics
  (`residualControl 1e-8`, linear-solver tolerance `1e-10`, `endTime` cap), fluid
  properties + Reynolds range (`nu = 1e-5` → Re 2·10⁶ … 10⁷)
- **"Verified"** gains the concrete DoE table (22 angles × 5 speeds = 110 cases), the
  exact repo-relative commands of the verified run, the results table (best model per
  task, test R²/RMSE from the acceptance run), and the output directory tree
- "How this maps onto SurroForge" mentions `surrogate_vtk.py` as the standalone
  ParaView-export helper

**`LOCAL_SETUP.md`** (stays gitignored, German, user-owned) shrinks to the genuinely
machine-local core: install paths, `source` command, verification commands, and a
pointer "everything substantive lives in README.md" — no more duplicate that can drift.

## Tree Tidying (Section 4, approved)

Pure on-disk hygiene, no tracked content affected (produces no commit):

- Delete the four verified-empty untracked root dirs: `design-artifacts/`,
  `field_autoencoder_rmse10/`, `field_autoencoder_tuned/`, `field_energy_9999/`
  (0 bytes, only empty subdirectories, invisible to git)
- Delete the three `.DS_Store` files under `examples/naca0012_openfoam/` (gitignored)
- Deliberately untouched: `run/` (~514 MB real CFD data, needed for acceptance),
  `ex09_real/`, `scratch/` (741 MB, user workspace), `__pycache__/` (regenerates),
  `.gitignore` (already complete)

## Verification & Acceptance (Section 5, approved)

The naca directory is **not in CI** (pytest smoke-tests only top-level
`examples/[0-9]*.py`), so the refactor gets its own three-stage acceptance:

1. **Static** (after each step): `ruff check examples/naca0012_openfoam/` (examples are
   outside CI's `src tests` ruff scope) and `python -m py_compile` on both files.
2. **Behavioral acceptance** (the equivalence proof): run the slimmed
   `ingest_and_train.py examples/naca0012_openfoam/run --out <scratch>/ex09_real_check`
   against the real 110 on-disk cases. Success criteria: clean exit; all three model
   artifacts exported (`.onnx`, `.pt`, `.pkl`); `surrogate_vtk/surrogate_fields.vtkhdf`
   with 110 time steps + `samples.csv` written; fit table plausible against the
   verified values (field torch-mlp and scalar mlp each best, R² in the same range —
   bitwise equality is not the criterion due to NN training nondeterminism). A quick
   h5py structural check of the written file (NSteps=110, cell data `p`) stands in for
   opening it in ParaView. **These fresh numbers become the README's verified table**,
   resolving the Section-3 contradictions.
3. **Regression**: full `scripts/ci.sh` once at the end (nothing in the package is
   touched, so all gates stay green), and top-level example 09 still exits 0.

Environment preconditions verified on 2026-07-05: `h5py 3.16.0` in `.venv`;
`run/samples.csv` + 110 `case_*` dirs on disk.

## Commit Plan

Small, single-purpose commits directly to `main` (solo project):

1. `refactor(examples): extract surrogate_vtk.py and drop the dead .vtu/.pvd path`
2. `docs(examples): consolidate the NACA 0012 reference into the tracked README`

(Tree tidying and LOCAL_SETUP.md changes touch no tracked files — no commit.)
