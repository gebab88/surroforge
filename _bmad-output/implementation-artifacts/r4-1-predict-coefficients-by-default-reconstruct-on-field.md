# Story R4.1: Predict coefficients by default; reconstruct on `--field`

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a methods engineer,
I want predictions to be compact coefficients unless I ask for the full field,
So that querying a million-node surrogate stays KB-sized.

## Acceptance Criteria

1. **Coefficients by default (the #35 flip).** Given a trained reduced-field run,
   `surroforge predict <run-id> --input new.csv` (and
   `services.predict_from_file(run, path)`) returns the input columns plus compact
   modal coefficients `mode_0 … mode_{r-1}` — NOT the reconstructed field. Pinned
   bitwise (same process, same arrays — no BLAS exposure):
   the returned values equal `run.load_model().predict_coefficients(x)` exactly,
   and `coeffs @ basis.T + mean` equals `model.predict(x)` exactly
   (engine-measured: bitwise True on the pinned fixture). This deliberately
   changes the r1-3-era behavior of `predict` on field runs (full field) —
   ratified, pre-release, mandated by PRD §4.7 #35; no existing test pins the old
   behavior (verified: no field-run predict coverage in
   test_scenarios/test_cli/test_api).
2. **`--field` reconstructs and writes the full field with round-trippable column
   names.** `surroforge predict <run-id> --input new.csv --field` (service:
   `field=True`) returns/writes the reconstructed field, bitwise equal to
   `model.predict(x)`, under the ORIGINAL expanded snapshot column names
   (`f0 … f39` on the fixture) taken from new lineage `output_columns` — never the
   `output_{i}` fallback. Teeth (vacant-pin check: discriminates the fallback):
   the written CSV re-ingests through `expand_field_columns` with the run's own
   `outputs: ["f*"]` pattern and matches all n_dof columns. Write formats: `.csv`
   / `.parquet` (existing `_save_frame`) **plus `.h5`/`.hdf5`** via lazy `h5py`
   import (one dataset per column, column-group layout), missing backend →
   actionable `DataError` hinting `pip install 'surroforge[formats]'` (CI installs
   `[all]`, local venv has h5py 3.16 — testable both places).
3. **Scalar and legacy behavior unchanged.** A scalar run (`method: polynomial`)
   predicts exactly as today with AND without `--field` (the flag is meaningless
   there and is accepted silently — ratified UX decision: scalar output already IS
   the full field; an error would be noise). The duck-typed gate is
   `hasattr(model, "predict_coefficients")` — scalar trainers never grow it.
   Legacy field runs (pre-R4.1 pickles) gain the coefficients default for free
   (pickle restores state; methods come from the new class) and fall back to the
   existing naming when `output_columns` is absent from lineage.
4. **Content-addressed basis store + lazy-coefficient artifact (#32/#22/#47).**
   Training a reduced-field run writes `basis/<sha256>.npz` (project-level store,
   atomic) containing `basis` (n_dof × r), `mean` (1 × n_dof), and
   `singular_values` (the FULL spectrum — retained at fit, feeds R4.2's energy
   display without reopening persistence), and records lineage
   `basis_file: "basis/<sha>.npz"` (POSIX form — built by construction from the
   store name, which contains no separators; the original `as_posix()` wording
   named a mechanism, this is the equivalent outcome).
   - The address is sha256 over the **contiguous array bytes in their native
     dtype, each framed with key/dtype/shape, in sorted key order** — NEVER the
     npz file bytes (zip embeds timestamps → non-deterministic) and **NEVER
     lineage `basis_hash`** (see the Dev Notes key finding: holdout vs final).
     (Amended in review round 1: the original "float64 cast, fixed order"
     wording left the encoding non-injective across shapes/dtypes and desynced
     from what the npz actually stores; round 2 added key-charset + numeric-
     dtype contract validation. For the built-in trainer the address inputs are
     unchanged in practice: float64, and sorted order == the fixed trio.)
   - The npz `basis` is bitwise equal to the final pickled model's basis
     (`np.array_equal`); columns orthonormal (`approx`).
   - **Dedup pin:** two identical trainings in one test process → exactly ONE file
     in `basis/`, both lineages reference it (measured: same-process retrain gives
     a bit-identical basis).
   - **Scope note (ratified deviation):** the epic AC AND the PRD (§6 seam row
     "basis.npz … im Run-Dir", §7 "Run-Dir: basis.npz (über Runs per Hash
     dedupliziert)") say "run dir". A per-run copy cannot be "deduplicated across
     runs"; the deviation — project-owned store + lineage reference — is the only
     reading that satisfies the dedup clause both sources also mandate. Flag in
     review if contested.
   - Scalar runs: no store write, no new lineage keys, lineage byte-unchanged
     (hook absent; the R2.4 scalar-identity family must stay green unmodified).
5. **Run dir smaller than the raw snapshots (#47), pinned.** On the pinned fixture
   (40 dof × 80 samples), the run dir total INCLUDING its share of the basis store
   (~2 KB npz) is strictly smaller than the source CSV — engine-measured ratio
   0.432 without / ~0.46 with the npz (>2× headroom; at 400 dof the ratio drops to
   0.356 — the claim strengthens with size). Assert `run_size + npz_size <
   csv_size`. The "lazy coefficients" of #22 are the pickled inner surrogates
   (x → coefficients on demand) — already the artifact; this AC pins the size
   consequence, it does not add coefficient persistence.
6. **Diff discipline.** Production diff confined to: `trainers/reduced_field.py`
   (spectrum retention, `predict_coefficients`, `reduction_artifacts`),
   `persistence/store.py` (`write_npz`) plus the `persistence/__init__.py`
   re-export (eight files — the original "seven" omitted the `__init__`),
   `engine.py` (store write + 2 lineage keys inside the hook branch),
   `core/project.py` (`basis_dir`), `core/run.py`
   (accessors), `services.py` (`field=` kwarg + duck-typed default),
   `cli/main.py` (`--field`, `_save_frame` h5). `validators/*`, `conformance/*`,
   `reporters/*`, `exporters/*`, `plugins/*`, `core/config.py` untouched (PRD §6
   seam table). `BaseTrainer` contract untouched: `predict` still returns
   `(n, n_dof)`; conformance suite green unmodified.

## Tasks / Subtasks

- [x] **Task 1 — trainer surface (AC 1, 4)**
  - [x] `reduced_field.py`: retain the full spectrum at fit
    (`self._spectrum = singular` before truncation; cheap — min(n, n_dof)
    float64). Initialize `self._spectrum: np.ndarray | None = None` in
    `__init__`'s fitted-state block (reduced_field.py:98-103) and gate the
    unfitted raise on `is None` like `_basis` — otherwise an unfitted instance
    gives `AttributeError` instead of the promised `TrainerError`, and legacy
    pickles (restored via `__dict__`) lack the attribute entirely.
    Promote `_coefficients` to public `predict_coefficients(x)`
    (raises `TrainerError` when unfitted, mirroring `predict`); update the two
    internal callers (`predict`, `reproducibility_diagnostics`).
  - [x] New public hook `reduction_artifacts() -> dict[str, np.ndarray]`:
    `{"basis": V, "mean": mean, "singular_values": spectrum}`; raises
    `TrainerError` when unfitted. Docstring: storage artifact of the PERSISTED
    final model, consumed by the engine persist step; explicitly NOT a
    fingerprint input — `reproduce()` never reads it (retro action item 3:
    basis.npz is storage, the R2 spec superseded basis.npz only as a VERDICT
    input; field-primary verdict unchanged).
  - [x] Unit tests (in the new test file): `predict_coefficients` shape `(n, r)`,
    bitwise reconstruction identity, unfitted raises, artifacts dict keys/shapes,
    spectrum length == min(n_samples, n_dof).
- [x] **Task 2 — persistence + engine (AC 4, 5, 6)**
  - [x] `persistence/store.py`: `write_npz(path, **arrays)` — atomic via
    `tempfile` in the destination dir + `np.savez` + `os.replace` (same pattern
    as `_atomic_write`; binary mode). [MUST] never hand-roll `open(...,'w')`.
  - [x] `core/project.py`: `basis_dir` property (`self.root / "basis"`). Do NOT
    add it to `Project.init`'s dir tuple — the store is created lazily on first
    field-run write (`mkdir(parents=True, exist_ok=True)`), so scalar-only and
    legacy projects grow no empty dir.
  - [x] `engine.py` persist step (after the pickle export, before the lineage
    write): duck-type `hasattr(final_model, "reduction_artifacts")` — note
    deliberately the FINAL model, see Dev Notes key finding; compute the address
    (sha256 over `np.ascontiguousarray(a, dtype=np.float64).tobytes()` for
    basis, mean, singular_values in that fixed order), write
    `project.basis_dir / f"{sha}.npz"` only if absent, set lineage
    `basis_file` + `output_columns`
    (`expand_field_columns(frame, config.data.outputs)` — engine already holds
    `frame`; import from `data.loader`, same module as `select_columns`, legal
    direction). Both keys land inside the hook branch ONLY.
  - [x] `core/run.py`: accessors `basis_file: str` (lineage, `""` default),
    `output_columns: list[str]` (lineage, `[]` default), `basis_path: Path | None`
    (resolves `run_dir.parents[1] / basis_file`; None when absent — the runs/
    layout assumption is already baked into Run).
  - [x] Tests: store file exists + loads + bitwise basis equality with the final
    pickle's basis + orthonormality; dedup (two identical trainings → exactly one
    npz, both lineages point at it); the size pin (AC 5, include the npz);
    address ≠ lineage `basis_hash` on the fixture (teeth against wiring the
    store name to the holdout diagnostic — measured: they differ); **the address
    spec itself (validation Major):** load the npz back and assert the filename
    stem equals sha256 over `np.ascontiguousarray(a, np.float64).tobytes()` for
    (basis, mean, singular_values) IN THAT ORDER, recomputed in the test from the
    loaded arrays — recomputed-not-golden and same-process/same-bytes (BLAS-safe),
    and it decisively fails file-bytes addressing (zip timestamps make the dedup
    pin alone evadable: back-to-back runs often share the 2-second DOS-timestamp
    window), partial-array addressing (basis-only also satisfies ≠ basis_hash),
    and wrong-order addressing; scalar run → no `basis/` dir created,
    `basis_file`/`output_columns` absent from lineage.
- [x] **Task 3 — services + CLI (AC 1, 2, 3)**
  - [x] `services.predict_with_run(run, frame, *, field=False)` /
    `predict_from_file(run, path, *, field=False)`: load model once; if
    `hasattr(model, "predict_coefficients")` and not `field` → result frame =
    input copy + `mode_{k}` columns from `predict_coefficients(x)`; if hook and
    `field` → existing full-prediction path, but name columns from
    `run.output_columns` when its length matches `preds.shape[1]`, else the
    existing fallback chain; no hook → exactly today's code path (`field`
    ignored).
  - [x] `cli/main.py` `predict`: add
    `field: bool = typer.Option(False, "--field", help="Reconstruct and write the full field (default: compact modal coefficients for reduced runs).")`,
    pass through. Extend `_save_frame` with `.h5`/`.hdf5`: lazy `import h5py`
    inside the branch ([GOTCHA] never top-level), one dataset per column
    (column-group layout, str column names), missing backend → `DataError` with
    the `pip install 'surroforge[formats]'` hint (6-3 degradation discipline).
  - [x] Tests: coefficients default through the real engine + services path
    (bitwise vs `predict_coefficients`); `--field` CSV round-trip teeth
    (re-ingest written file with the run's own `outputs: ["f*"]` via
    `expand_field_columns` → all 40 columns match — discriminates the
    `output_{i}` fallback); scalar run identical frames with and without
    `field=True` (`DataFrame.equals`); CLI e2e: `predict run-001 -i new.csv -o
    out.csv` writes mode_* columns, `--field -o field.csv` writes f0..f39; h5
    write (importorskip h5py) → re-read datasets match; legacy fallback: strip
    `output_columns` from a copied lineage → `--field` falls back to `output_{i}`
    names without crashing.
- [x] **Task 4 — gates (AC 6)**
  - [x] Full suite green (299 existing + new); the R2.4 scalar-identity family
    (`test_reproduce_scalar_identity.py`) and conformance suite UNMODIFIED and
    green. `ruff check src tests`, `mypy src`, `lint-imports` (3/3).
  - [x] Diff discipline: `git diff --stat -- src/` shows exactly the seven files
    of AC 6; `core/config.py`, `validators/`, `conformance/`, `reporters/`
    untouched.

### Review Findings

_Code review 2026-06-11 (Blind Hunter + Edge Case Hunter + Acceptance Auditor;
3 layers, 24 raw findings → 17 unique; 5 dismissed as noise/ratified).
All 13 patches applied 2026-06-11: h5 writer validates columns (numeric-only,
no '/', no duplicates) and writes atomically; the store address now frames
every array with key/dtype/shape in sorted key order over native dtypes (the
recompute pin in tests matches); legacy-pickle guard via getattr; coefficient
ndim normalization; `_models` guard; `basis_path` validates the lineage value
against the exact engine layout (regex) + parent-depth guard; engine comment
corrected; size pin recursive; suffix message complete; write_npz fd window
closed; `output_columns` from canonical_config; 4 new tests (missing-h5py
DataError + hint, non-numeric/slash/duplicate columns, tampered basis_file →
ProjectError). Gates re-run green: full suite exit 0, ruff, mypy,
lint-imports 3/3._

- [x] [Review][Decision] PRD §6/§7 deviation: basis.npz lives in a project-level
  `basis/` content-addressed store + lineage reference, not "im Run-Dir" — the
  story ratifies this as the only reading satisfying the dedup clause and
  instructs "Flag in review if contested". **Resolved 2026-06-11: ratified
  reading confirmed by Georg in review — project-level store stands.**
- [x] [Review][Patch] h5 writer robustness: non-numeric (object-dtype) input
  columns, `/` in column names (silent group nesting), and duplicate column
  names all crash with raw non-SurroForgeError tracebacks; `h5py.File(path,
  "w")` truncates immediately so failures leave a partial/corrupt file (no
  tmp+replace, unlike the diff's own `write_npz`) [src/surroforge/cli/main.py]
- [x] [Review][Patch] Content-address hash has no shape/dtype/key framing (raw
  concatenated bytes are non-injective across shapes/boundaries) and hashes
  float64 casts while `write_npz` stores original dtypes; missing artifact keys
  from a duck-typed plugin trainer raise bare `KeyError` mid-training
  [src/surroforge/engine.py]
- [x] [Review][Patch] Pre-R4.1 legacy pickle: `reduction_artifacts()` raises
  `AttributeError` (pickle restores `__dict__` without `_spectrum`) instead of
  `TrainerError` — exact path R4.2 `rom inspect` will take; use
  `getattr(self, "_spectrum", None)` [src/surroforge/trainers/reduced_field.py]
- [x] [Review][Patch] Duck-typed `predict_coefficients` return not
  ndim-normalized: a plugin returning `(n,)` for rank-1 hits `IndexError`; the
  adjacent `predict` path reshapes 1-D, the new path doesn't
  [src/surroforge/services.py]
- [x] [Review][Patch] `predict_coefficients` guards `_basis` but uses
  `_models` — proxy guard; `_basis` set with `_models` empty silently returns
  an `(n, 0)` frame [src/surroforge/trainers/reduced_field.py]
- [x] [Review][Patch] `Run.basis_path`: `run_dir.parents[1]` raises `IndexError`
  on shallow run dirs; lineage `basis_file` is joined unvalidated (absolute
  path in lineage discards the left operand entirely) [src/surroforge/core/run.py]
- [x] [Review][Patch] Misleading comment: "the arrays predict and `rom inspect`
  run against" — predict uses the arrays inside the pickle, never reads the
  npz back [src/surroforge/engine.py]
- [x] [Review][Patch] Size-pin test uses non-recursive `iterdir()` — counts
  directory-entry sizes if the run dir ever gains subdirs, making the AC-5
  bound vacuous [tests/test_predict_field.py]
- [x] [Review][Patch] Unsupported-suffix error message advertises ".csv,
  .parquet, or .h5" but the writer also accepts `.pq`/`.hdf5`
  [src/surroforge/cli/main.py]
- [x] [Review][Patch] `write_npz` fd-leak window: `mkstemp` fd leaks if
  `os.fdopen` itself raises (cleanup only unlinks the path)
  [src/surroforge/persistence/store.py]
- [x] [Review][Patch] `output_columns` expanded from raw `config.data.outputs`
  while snapshot/`config_hash` use `canonical_config` — the one new lineage
  field read off the pre-canonical config; latent divergence
  [src/surroforge/engine.py]
- [x] [Review][Patch] AC 2's missing-backend path untested: no test blocks the
  h5py import and pins the `DataError` + `pip install 'surroforge[formats]'`
  hint — vacant spot by the project's own pin discipline
  [tests/test_predict_field.py]
- [x] [Review][Patch] `reduction_artifacts()` returns live internal arrays by
  reference — caller mutation corrupts the fitted model and could desync
  hash-then-write; return read-only views
  [src/surroforge/trainers/reduced_field.py]
- [x] [Review][Defer] `mode_k`/output columns silently overwrite same-named
  input columns [src/surroforge/services.py] — deferred, pre-existing: the
  scalar predict path has applied the same overwrite convention since Epic 3;
  changing it is a cross-cutting design decision.
- [x] [Review][Defer] Orphaned basis npz when training fails after the store
  write; deleting runs never GCs the shared store [src/surroforge/engine.py] —
  deferred: content-addressed store semantics (retry re-uses the artifact);
  GC policy is an R4.4/post-MVP design question.
- [x] [Review][Defer] Predict-written `.h5` read back through `load_table`
  surfaces a raw pytables error, not `DataError`
  [src/surroforge/data/loader.py] — deferred, pre-existing: the story ratifies
  non-re-ingestability (raw h5py column groups, pandas read_hdf ships in no
  extra); the missing error translation is in the unchanged loader.

### Review Findings — Round 2

_Re-review 2026-06-11 after the 13 round-1 patches (same 3 layers; 31 raw
findings → unique set below; 7 dismissed as noise/already-ratified/already-
deferred). Auditor verdict: all 13 round-1 patches correctly applied, no
behavioral AC violation, gates independently re-verified green._

_All 9 round-2 patches applied 2026-06-11: artifact-contract validation
(key charset incl. reserved `file`, numeric dtype) ahead of hashing;
dedup-aware log line + gate comment; h5 writer rejects empty/`.`/`..` names,
duplicate message corrected, `track_order=True` (order pinned by list-equality
in the CLI test), output perms honor umask; `--output` help complete;
`(n, r)` contract enforced on duck-typed coefficients; `output_columns`
type-guarded; `basis_path` raises ProjectError with re-train guidance on a
missing store file; docstring claim softened; story AC 4/AC 6/Dev Notes/File
List text amended to the ratified implementation. 3 new tests (missing store
file, corrupt output_columns fallback, coefficient shape contract) + 1
extended (empty h5 name). Gates green: full suite exit 0, ruff, mypy,
lint-imports 3/3._

- [x] [Review][Patch] Engine validates artifact keys' presence but not their
  form: a plugin key named `file` crashes `np.savez` (parameter collision), a
  key containing `:`/`;` makes the hash framing ambiguous, and an object-dtype
  value hashes memory addresses (dedup silently never fires) — validate
  key charset and numeric dtype next to the existing missing-keys check
  [src/surroforge/engine.py]
- [x] [Review][Patch] Misleading log on dedup hit ("Reduction basis stored"
  when nothing was written) and stale "hook-gated" comment (the actual gate is
  duck-typed `hasattr`) [src/surroforge/engine.py]
- [x] [Review][Patch] h5 writer residual gaps: empty/`"."` column names pass
  every guard and crash with raw h5py `ValueError`; duplicate-check message
  misdiagnoses distinct columns that stringify identically; column order is
  lost without `track_order=True`; `mkstemp`'s 0600 perms leak onto the
  user-requested output file (csv/parquet honor umask)
  [src/surroforge/cli/main.py]
- [x] [Review][Patch] CLI `--output` help string still says
  `.csv/.parquet/.h5` while the writer (and its error message) accept
  `.pq`/`.hdf5` too [src/surroforge/cli/main.py]
- [x] [Review][Patch] Duck-typed `predict_coefficients` return is normalized
  for 1-D only: a 0-d scalar hits raw `IndexError`, ≥3-D misuses axis 1, a row
  count ≠ len(frame) dies inside pandas — validate the `(n, r)` contract with
  a clean `TrainerError` [src/surroforge/services.py]
- [x] [Review][Patch] `Run.output_columns` type-launders corrupted lineage
  (`"f0"` → `['f', '0']`) while the sibling `basis_path` treats lineage as
  hostile — return `[]` unless the value is a list [src/surroforge/core/run.py]
- [x] [Review][Patch] `Run.basis_path` validates the lineage string but not
  file existence: a deleted/missing store npz returns a valid-looking path and
  consumers get raw `FileNotFoundError` instead of the property's own
  "re-train" guidance (re-training does restore the store — dedup rewrites a
  missing address) [src/surroforge/core/run.py]
- [x] [Review][Patch] `reduction_artifacts` docstring overclaims: "callers can
  never corrupt the fitted state", but the read-only view's `.base` is the
  original writable array — soften to what the freeze actually guarantees
  [src/surroforge/trainers/reduced_field.py]
- [x] [Review][Patch] Story text stale after round-1 patches: AC 4 still
  mandates the superseded address spec (float64 cast, fixed order, literal
  `as_posix()`), AC 6 says "exactly seven files" (it is eight incl. the
  `persistence/__init__.py` re-export), File List says 11 tests (now more) —
  amend the normative text to the ratified implementation [story file]
- [x] [Review][Defer] Existing store npz is trusted by filename: a
  zero-byte/truncated file (interrupted copy/cloud sync) is never repaired
  because the dedup check fires before any write [src/surroforge/engine.py] —
  deferred: store content-verification policy, same bucket as the GC question.
- [x] [Review][Defer] Persist-window failure (incl. the new artifact-store
  exceptions) strands a run dir with model.pkl but no lineage.json — invisible
  to list/run() yet consuming the run id [src/surroforge/engine.py] —
  deferred: pre-existing persist-failure semantics, not introduced here.
- [x] [Review][Defer] Rank-1/duck-typed plugin coefficient shapes have no
  conformance backing [src/surroforge/conformance/] — deferred to R4.3 (the
  reconstruction round-trip conformance probe is that story's scope).

## Dev Notes

### Key finding (engine-measured 2026-06-11): holdout basis ≠ final basis — always

On the pinned fixture, lineage `basis_hash` is `db5ec8c8…` while the sha of the
FINAL model's basis bytes is `8e331457…`. This is structural, not incidental: the
lineage hash is the R2.3 diagnostic from the **holdout** model (fit on the 60-row
train split, then sign+order-canonicalized by `canonicalize()`), while basis.npz
must serve `predict`/`rom inspect` on the **persisted final** model (fit on all 80
rows). The two bases differ in content and in gauge treatment.

Consequences the implementation MUST honor:
- The store address is computed from the FINAL model's arrays at persist time —
  never reuse lineage `basis_hash` as the filename.
- The store address is an **artifact identity**, not a reproducibility gauge: two
  gauge-equivalent bases get different addresses, and that is correct for storage.
- This does NOT violate the project-context holdout-vs-final [GOTCHA]: that rule
  governs *fingerprint/reproducibility* reads (`reproducibility_diagnostics`,
  `resolved_config_overrides` — both stay on the holdout model). `reduction_artifacts`
  is a storage hook for the shipped artifact; `reproduce()` never reads basis.npz
  (retro action item 3 — say this in the hook docstring AND assert it stays true:
  the reproduce-family tests run unmodified).
- Test teeth: assert the store filename's sha ≠ lineage `basis_hash` on the
  fixture (wiring the name to the holdout diagnostic fails decisively).

### Engine-measured fixture (2026-06-11, real `run_training` path)

Fixture family = r3 standard (seed 42, QR-orthonormal modes 40×3, x uniform(−1,1)
80×2, amps `[5·x0, 2·x1², 1.8·x0·x1]`, split 0.25/seed 1, rank-2 pod +
polynomial degree-2 inner, cv 3, field metrics):

| Quantity | Value |
|---|---|
| csv size / run-dir size / ratio | 67455 B / 29108 B / **0.432** |
| run-dir breakdown | lineage.json 24360 (holdout fingerprint JSON dominates), model.pkl 2640, metrics.json 1248, snapshot 390, log 470 |
| npz size (basis+mean+spectrum) | ~2046 B |
| n_dof=400 variant ratio | **0.356** (claim strengthens with size) |
| `coeffs @ basis.T + mean == predict(x)` | bitwise True |
| same-process retrain basis bytes | bit-identical (dedup pin is deterministic) |
| content address over array bytes | stable across same-process retrains (verified; no hex literal here — the address is recomputed in tests, never golden) |
| npz BytesIO/savez round-trip | exact (`np.array_equal`) |
| `expand_field_columns(frame, ["f*"])` | `['f0' … 'f39']` (40) |
| lineage basis_hash vs final-basis sha | `db5ec8c8…` ≠ `8e331457…` (see key finding) |

Bitwise pins are safe here BECAUSE every comparison is same-process between two
reads of the same persisted arrays (or one recomputation on identical bytes) —
no cross-run golden literals, no BLAS exposure (project discipline). Anything
cross-run stays a dedup/count/reference assertion.

### Existing assets and integration points (verified against source, 2026-06-11)

- `trainers/reduced_field.py:157-163` — `_coefficients(x)` already exists (the
  pre-reconstruction step); promotion is a rename + docstring, callers at :154
  (`predict`) and :187 (`reproducibility_diagnostics`). Spectrum: `fit` discards
  `singular` today (:118-124) — retain it before truncation.
- `services.py:334-360` — `predict_with_run` / `predict_from_file`. Column logic
  to preserve as the fallback: `outputs if preds.shape[1] == len(outputs) else
  output_{i}` (:349-351). Input-column validation and the frame-copy result
  pattern stay byte-identical for scalar runs.
- `cli/main.py:117-125` — `_save_frame` (csv/parquet + `DataError` on other
  suffixes); :235-251 — the `predict` command (`--input/-i`, `--output/-o`,
  `_guard`, `_info`/`_success` quiet-aware output [MUST]).
- `engine.py:196-237` — persist block: snapshot → pickle export → (onnx) →
  metrics → lineage → log. The store write goes after the pickle export; the two
  new lineage keys join the `field_extras` update (already conditional — scalar
  lineage untouched, R2.4). `frame` is in scope from step 1; `expand_field_columns`
  import joins the existing `data.loader` import.
- `persistence/store.py:46-60 + _atomic_write` — the atomicity pattern to copy
  for `write_npz` (tempfile in destination dir + `os.replace`); [MUST] all run
  state through persistence helpers.
- `core/project.py:78-96` — dir properties; `runs_dir` precedent for `basis_dir`.
  Run-dir layout `project/runs/run-XXX` ⇒ `run_dir.parents[1]` is the project
  root (used by `Run.basis_path`).
- `core/run.py:145-177` — the lineage-accessor pattern (`basis_hash`,
  `reduction`, defaults for scalar/legacy) to mirror for `basis_file` /
  `output_columns`.
- Pre-release compat: legacy field-run pickles restore against the NEW class →
  `predict_coefficients` works on old artifacts (methods live on the class);
  `_spectrum`/`output_columns` only matter at train/persist time, so legacy
  predict needs only the naming fallback (AC 3).
- CI installs `[all]` (ci.yml:30) → h5py paths exercised in CI; local venv has
  h5py 3.16 + pyarrow 24 (no skips locally).

### Constraints that bite

- **[MUST] Scalar path byte-identical** (R2.4): every new engine/lineage behavior
  lives inside the `reduction_artifacts` hook branch; the scalar-identity and
  reproduce families run UNMODIFIED.
- **[MUST] `predict` keeps the `(n, n_outputs)` 2-D contract** — the coefficients
  default is a SERVICES-level behavior; the trainer contract, conformance suite,
  metrics, and engine internals never see coefficients as "the prediction".
- **[MUST] No top-level `h5py` import; missing backend → domain error + extra
  hint** (`surroforge[formats]`), matching the 6-3 pattern. Do NOT use
  `frame.to_hdf`/`pd.read_hdf` — pandas HDF needs **pytables**, which ships in NO
  extra (verified: `pd.read_hdf` fails with "Import pytables failed" even under
  `[all]`); write with raw h5py, one dataset per column. The written column-group
  `.h5` is deliberately NOT re-ingestable by `load_table` (accepted v1
  read/write asymmetry; the test re-reads via h5py).
- **[MUST] Atomic persistence via `persistence/store.py` helpers**; lineage paths
  in POSIX form (`as_posix()` — the cross-OS matrix already caught one backslash
  bug).
- **Address = array bytes, not file bytes**: npz is a zip; zips embed timestamps;
  hashing the file would break dedup. ~~Fixed array order (basis, mean,
  singular_values) IS the address spec~~ — amended in review: the address spec is
  key/dtype/shape-framed native-dtype array bytes in sorted key order (injective
  over what the npz stores; see AC 4 and the Review Findings) — documented at the
  hash site.
- **Vacant-pin discipline (project-context [GOTCHA], R3 retro — design teeth at
  writing time):** (a) the `--field` column-name pin MUST be the re-ingest
  round-trip (an `output_{i}` fallback passes a "has 40 columns" assert);
  (b) the store-name pin MUST assert ≠ lineage `basis_hash` (aliasing the two
  hashes is THE likely wrong implementation); (c) coefficients pins are bitwise
  equality against `predict_coefficients`, not shape-only.
- **No golden literal hashes** (Accelerate BLAS, standing rule): never pin a
  concrete sha string; pin equalities, inequalities, file counts, references.
- Story file edits during dev: only Tasks checkboxes, Dev Agent Record, File
  List, Change Log, Status.

### Scope boundaries (what R4.1 does NOT do)

- No `rom inspect` (R4.2 — but `singular_values` in the npz is its data source;
  that is why the spectrum ships now).
- No conformance probes (R4.3), no packaging extra (R4.4), no VTK/mesh writers,
  no ONNX field export (Phase 2).
- No surgery on model.pkl (it stays self-contained incl. basis — the size AC is
  already met at 0.432× without removing anything; deduplicating the basis OUT of
  the pickle would break artifact self-containment and the conformance
  picklability contract for zero needed bytes).
- No coefficient persistence at train time — "lazy coefficients" (#22) is the
  existing pickled mapper; see AC 5.

### Previous-story intelligence (R3 epic, 2026-06-11)

- **Review patterns that will recur:** vacant pins (now a project-context
  [GOTCHA] with three named trap classes — the reviewers will check every pin
  against them; the teeth listed above are designed-in, cite them), BLAS-margin
  discipline (bitwise only same-process/same-bytes — stated explicitly in the
  measurement table), prose-coupling dismissals, symmetric-guard probing (the
  services duck-type gate is single-sided by design: hook present/absent — there
  is no partial state).
- The r3-3 review strengthened substring asserts to exact full-line equality —
  prefer exact frame/array equality over column-presence asserts here from the
  start.
- Engine runs cost ~1-2 s each; the dedup test needs TWO runs (unavoidable);
  keep other assertions sharing one run per test function where possible.
- Commit-at-done discipline: this story lands as ONE commit before R4.2 starts.

### Testing standards

- Flat `tests/`, one file per concern → `tests/test_predict_field.py` (predict
  default/`--field`/CLI) and the store/persistence asserts may live in the same
  file (one story, one concern: the R4.1 predict-and-storage surface) — do NOT
  touch `test_reduced_field.py`/`test_two_source_report.py`.
- Module docstring citing PRD §4.7 (#35, #22, #21, #47, #32) and this story.
- `ruff` (E/F/I/UP/B/W, line 100); `zip(strict=True)` where applicable;
  `importorskip("h5py")` for the h5 test (CI has `[all]`).
- Run via `.venv/bin/python -m pytest`; pytest summary may not print locally —
  use exit code + `grep -cE "^(FAILED|ERROR)"` (established pattern).

### Project Structure Notes

- Production diff: the seven files of AC 6 — the widest R-story so far; the
  breadth is horizontal plumbing for one feature, not new subsystems. Each file's
  delta is small and listed in its task.
- New test file: `tests/test_predict_field.py`.
- Tracking: this story file + `sprint-status.yaml`.

### References

- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md §Story R4.1] —
  AC wording: coefficients default, `--field`, content-addressed basis.npz,
  run dir smaller than raw snapshots.
- [Source: _bmad-output/planning-artifacts/prd-field-rom-v1.md §4.4, §4.7, §4.8,
  §5, §6, §7] — predict delta (#35), lazy storage (#22/#47), write-back (#21),
  basis.npz dedup (#32), seam table (which files may move), lineage deltas.
- [Source: _bmad-output/implementation-artifacts/epic-r3-retro-2026-06-11.md] —
  action item 3 (basis.npz = storage artifact, NOT a fingerprint gate; verdict
  stays field-primary), vacant-pin taxonomy, pipeline agreement.
- [Source: _bmad-output/implementation-artifacts/spec-field-rom-fingerprint.md] —
  the SUPERSEDED note demoting basis.npz as a verdict input (the storage role
  here is new and orthogonal); frozen block untouched as always.
- [Source: _bmad-output/project-context.md] — holdout-vs-final [GOTCHA], vacant
  pins [GOTCHA], optional-dependency discipline, atomic persistence [MUST],
  dependency direction.

## Dev Agent Record

### Agent Model Used

Claude Fable 5 (claude-fable-5[1m])

### Debug Log References

- RED: `tests/test_predict_field.py` written first; failed on
  `AttributeError: predict_coefficients` (surface absent), as expected.
- GREEN: Tasks 1-3 implemented; 11/11 new tests after one test-side fix — the
  bitwise pins initially compared in-memory `np.linspace` floats against the
  CSV-parsed values the service reads (ulp drift in the CSV round-trip); fixture
  now re-reads `x_new` from the written CSV so both sides parse identical bytes
  (the pins stay bitwise and honest).
- Gates: full suite exit 0, 0 failures, 310 collected (299 + 11 new); ruff "All
  checks passed!", mypy clean (41 files), lint-imports 3/3 kept. Diff
  discipline: `git diff --stat -- src/` = exactly the AC-6 surfaces
  (persistence = store.py + its `__init__.py` re-export line). R2.4
  scalar-identity family and conformance suite green UNMODIFIED.

### Completion Notes List

- AC 1: coefficients default in `services.predict_with_run` (duck-typed
  `hasattr(model, "predict_coefficients")`, `mode_{k}` columns); pinned bitwise
  vs `predict_coefficients(x)` and via reconstruction identity vs `predict(x)`.
- AC 2: `--field` reconstructs under lineage `output_columns` (recorded at train
  time from `expand_field_columns`); round-trip teeth green (written frame
  re-ingests under the run's own `f*` pattern); `.h5` column-group write via raw
  h5py (NOT `to_hdf`/pytables) with the `[formats]` install hint; CLI e2e for
  csv coefficients/field + h5.
- AC 3: scalar run frames `DataFrame.equals` with and without `field=True`; no
  store dir, no new lineage keys on scalar runs; legacy fallback test (lineage
  without `output_columns` → `output_{i}` naming, no crash).
- AC 4: content-addressed store `basis/<sha>.npz` written by the engine from the
  FINAL model's `reduction_artifacts()`; address pinned by recomputing sha256
  over the loaded arrays in fixed order == filename stem (validation-Major pin);
  store sha ≠ lineage `basis_hash` (holdout) pinned; dedup: two runs, one file,
  same reference; npz arrays bitwise equal to the final model's.
- AC 5: `run_size + npz_size < csv_size` green (measured ratio ~0.46, >2×
  headroom).
- AC 6: production diff = the listed surfaces only; `validators/`,
  `conformance/`, `reporters/`, `exporters/`, `plugins/`, `core/config.py`
  untouched; `BaseTrainer.predict` contract unchanged.

### File List

- src/surroforge/trainers/reduced_field.py (modified — `_spectrum` fitted state +
  retention, `predict_coefficients` public, `reduction_artifacts` hook)
- src/surroforge/persistence/store.py (modified — atomic `write_npz`)
- src/surroforge/persistence/__init__.py (modified — `write_npz` re-export)
- src/surroforge/engine.py (modified — content-addressed store write on the
  FINAL model, lineage `basis_file` + `output_columns` inside the hook branch)
- src/surroforge/core/project.py (modified — `basis_dir` property)
- src/surroforge/core/run.py (modified — `basis_file`/`output_columns`/
  `basis_path` accessors)
- src/surroforge/services.py (modified — `field=` kwarg, coefficients default,
  recorded-columns naming)
- src/surroforge/cli/main.py (modified — `--field` option, `_save_frame` h5
  column group)
- tests/test_predict_field.py (new — 11 tests at dev time: trainer surface,
  coefficients default, --field round-trip, legacy fallback, scalar unchanged,
  store address spec, dedup, size pin, CLI csv/h5; review rounds 1+2 added
  more — degradation/validation/contract tests — see Review Findings)
- _bmad-output/implementation-artifacts/r4-1-predict-coefficients-by-default-reconstruct-on-field.md
  (story tracking)
- _bmad-output/implementation-artifacts/sprint-status.yaml (status sync)

### Change Log

- 2026-06-11: Implemented (dev-story, red-green-refactor). RED: new test file
  fails on the absent `predict_coefficients` surface. GREEN: trainer surface
  (spectrum retained pre-truncation, public coefficients step, storage-artifacts
  hook with the storage-vs-fingerprint docstring), atomic `write_npz`,
  engine store write keyed to the FINAL model with the array-bytes address,
  lineage `basis_file`/`output_columns` hook-gated (scalar lineage untouched),
  services coefficients default + `--field` reconstruction under recorded
  columns, CLI flag + h5 column-group writer. One test-side fix (CSV-roundtrip
  ulp vs in-memory floats — fixture re-reads the CSV). Gates: 310 tests exit 0,
  ruff/mypy/lint-imports clean, diff = AC-6 surfaces only, R2.4 family green
  unmodified. Status → review.

- 2026-06-11: validate-create-story (fresh-context adversarial validation):
  verdict APPROVED WITH FIXES, all applied. **Major (1):** the address spec
  ("array bytes, not file bytes") had no test pin of its own — a file-bytes
  implementation usually survives the dedup pin (back-to-back runs share the
  2-second zip-timestamp window → identical bytes → one file; otherwise an
  intermittent flake), and a basis-only hash satisfies the ≠ basis_hash pin; fix:
  mandatory test recomputing the address from the loaded npz arrays (fixed order
  basis/mean/singular_values) and asserting it equals the filename stem —
  recomputed-not-golden, kills file-bytes / partial-array / wrong-order wiring
  (the vacant-pin taxonomy applied to the story's own key spec, 4th round).
  Minor: stale content-address hex literal removed from the measurement table
  (not reproducible under the final spec); pytables trap documented (pandas
  to_hdf/read_hdf ship in NO extra — raw h5py only; .h5 write is deliberately not
  load_table-re-ingestable); the "run dir" deviation note now cites PRD §6/§7 as
  well as the epic AC; `_spectrum` initialization in `__init__`'s fitted-state
  block with `is None` gating (legacy pickles lack the attribute); citation drift
  fixed (cli :117-125/:235-251, engine :196-237, project :78-96). Validator
  reproduced empirically: size ratios 0.432/0.462/0.356, bitwise reconstruction,
  bit-identical retrain basis, basis_hash ≠ final sha (both prefixes
  character-exact), fnmatch discrimination of the output_{i} fallback, no
  existing field-predict pins, no name collisions, no custom pickling, h5py/CI
  extras present, import direction legal, 299 tests collected.
- 2026-06-11: Story created (create-story). Empirically measured through the real
  engine path: run dir already 0.432× the raw CSV (size AC pins with >2×
  headroom; lineage.json dominates), coefficient reconstruction bitwise-exact,
  same-process retrain basis bit-identical (dedup pin deterministic), npz
  round-trip exact, expanded output names available at the engine persist point.
  **Key finding: lineage `basis_hash` (holdout, canonicalized) ≠ final-basis sha
  — structurally, not incidentally** — the store address must come from the FINAL
  model's arrays (artifact identity, not a gauge), with a test pin against
  aliasing the two hashes. Ratified interpretations: project-level `basis/` store
  + lineage reference (a per-run copy cannot dedup), `mode_{k}` coefficient
  column names, scalar+`--field` silently identical, spectrum persisted now to
  feed R4.2. Carried from the R3 retro: basis.npz is a storage artifact, never a
  fingerprint input. Status → ready-for-dev.
