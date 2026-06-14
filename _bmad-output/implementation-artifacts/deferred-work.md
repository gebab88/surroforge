# Deferred Work

Findings surfaced during reviews but intentionally out of scope for the triggering change.

## Deferred from: r4-4-surroforge-rom-packaging-extra (2026-06-14)

- **The `[rom]` lazy guard cannot fire in a real v1 install.** `_require_rom_support()`
  (`trainers/reduced_field.py`) gates on `import scipy.linalg`, but scipy is a hard Core
  dependency (and transitively pulled in by scikit-learn), so a working SurroForge always has
  it — the actionable install hint is structurally present and unit-tested (via a simulated
  `sys.modules["scipy"] = None`) but unreachable in practice. This is the accepted consequence
  of the Faithful-pattern decision (honour PRD-ROM §4.10 "gleiches Muster wie [smt]/[onnx]"
  literally; v1 adds no new heavy dependency). It becomes live the moment Phase-2 adds the first
  genuine optional ROM dependency (meshio/pyvista mesh readers): they join this guard and
  populate the currently empty `rom = []` extra. No action in v1.
- **`np.linalg.svd` vs. the PRD-named "SVD aus scipy".** The v1 reduction uses numpy's SVD
  (the field fingerprint is frozen on it, R2.x), while the `[rom]` guard imports scipy and PRD
  §4.10 names scipy as the SVD source. Deliberately not reconciled: switching the numeric path
  to `scipy.linalg.svd` would churn field-run identity (a frozen-contract renegotiation). If a
  Phase-2 reduction ever genuinely needs scipy's SVD (e.g. sparse / randomized variants for
  million-DOF meshes), introduce it behind a new spec version, not by mutating the v1 path.

## Deferred from: code review of r4-3-reconstruction-round-trip-no-mesh-conformance-probe (2026-06-13)

- **`reconstruction_round_trip` tests basis orthonormality only — `mean` and `predict()`
  are not cross-checked** (`conformance/trainer.py`). `recovered = (coeffs@Bᵀ+mean−mean)@B`
  cancels `mean` algebraically, and the check never compares `trainer.predict(x)` to
  `coeffs@Bᵀ+mean`, so a wrong stored `mean` or a `predict()`/`predict_coefficients()`
  inconsistency (a likely real ROM bug) passes. AC 1 scopes the check to `Bᵀ·B = I`
  (faithful projector), which it does correctly. Strengthening it to assert
  `predict(x) ≈ coeffs@Bᵀ+mean` would exercise `mean` and the central ROM invariant in one
  shot, but it needs its own negative fixture and re-scopes the `_GeometryDependentReducer`
  test (whose post-hoc ramp would then also trip the round-trip). A deliberate suite-quality
  pass, not an R4.3 AC.

- **Declared `reproduction_tolerance` has no upper bound in the conformance band**
  (`conformance/trainer.py` `_resolve_tolerance`, also the pre-existing `_deterministic`).
  `tol = max(floor, declared)` lets a trainer declare an arbitrarily large tolerance and pass
  the determinism / round-trip / mesh-free checks trivially. `declares_capabilities` enforces
  finite & ≥ 0 but no ceiling. A sane upper bound (or a warning) is a suite-wide policy
  decision affecting every band-comparing check, not R4.3-specific — revisit when the first
  external ROM/tolerant plugin binds (folds in with the standing engine-ROM-hook hardening).

## Deferred from: code review of r3-1-field-metrics-as-registered-callables (2026-06-11)

- **engine.py:123 rewraps every CV `ValueError` as `DataError` — including numerical
  failures.** The `except ValueError` around `cross_validate` exists for KFold/data
  problems, but `np.linalg.LinAlgError` subclasses `ValueError`, so an SVD
  non-convergence (e.g. NaN predictions from a diverged inner surrogate reaching
  `truncation_error`) surfaces as a `DataError` blaming the dataset. Pre-existing
  behavior, merely made more reachable by SVD-based metrics. Fix shape when touched:
  catch `LinAlgError` separately (re-raise as `TrainerError`/numerical error) or
  narrow the handler to the KFold call.

## Deferred from: code review of r1-4-built-in-methods-work-as-inner-surrogates-and-pass-conformance-unmodified (2026-06-10)

- **`blas_is_bit_reproducible` is optimistic for unknown BLAS backends.** tests/conftest.py
  returns `True` on any exception or any non-Accelerate backend — e.g. MKL without CNR mode
  is not bit-reproducible, yet the guard would let `requires_reproducible_blas` tests run
  (and potentially fail at 1e-9) instead of skipping. Pre-existing policy inherited from
  test_conformance.py's use of the same guard; R1.4's kriging-inner reproduce test merely
  doubles the exposure (more kriging fits judged at 1e-9). Revisit if CI ever adds a
  non-OpenBLAS runner: extend the probe to an allow-list (OpenBLAS, Accelerate-detected)
  rather than default-True.

## Deferred from: code review of r1-2-configure-reduction-in-a-model-reduction-block (2026-06-10)

- **RATIFIED DECISION (lead, 2026-06-10): pre-R1.2 field-run snapshots are not migrated.**
  r1-3-era persisted snapshots with `reduction: {basis: pod}` / `reduction: {}` /
  `inner: {<hyper>}` (no `method`) no longer re-parse through the strict R1.2 models —
  `Run.config` raises `ConfigError` on such runs (reproduce/predict/compare/report), and
  re-parsed old field snapshots hash differently (`basis: pod` default injected,
  `energy: 1` → `1.0`). Accepted because the product is pre-release with no shipped
  users; snapshots are canonical from R1.2 on. If a migration ever becomes necessary
  (a real user's persisted runs), add a lenient legacy re-parse in `Run.config`
  (default `method: polynomial`, map neither-set to `energy: 0.999`), keyed off a
  missing `schema_version`.
- **Engine ROM-hook `resolved_config_overrides` robustness for external plugins.** The
  engine deep-merges the duck-typed hook's return without shape validation: a truthy
  non-dict return crashes `_deep_merge` (`AttributeError: items`); a merged dict that
  fails `parse_config` raises `ConfigError` labeled `<resolved config>` AFTER holdout
  fit + CV + final fit (the completed run is discarded); an override touching
  `experiment`/`method` would change snapshot/`config_hash` while lineage still reads
  those keys from the pre-merge config. The built-in `ReducedFieldTrainer` hook returns
  only `{"model": {"reduction": {"rank": <int>}}}`, so none of this can fire in v1.
  Folds into the standing R2.3/R2.5 item "Engine ROM-hook input validation for external
  plugins" — harden all hook entry points together when the first external ROM plugin binds.
- **Declared `reduction`/`inner` config fields shadow third-party trainer hyperparameters.**
  A plugin trainer (entry-point seam) with its own `inner`- or `reduction`-named
  hyperparameter of a different shape (e.g. `inner: {radius: 2}`, no `method`) now fails
  strict config parsing (`InnerConfig` requires `method`; `ReductionConfig` is
  extra=forbid/pod-only), where pre-R1.2 these flowed through as opaque extras. Accepted
  consequence of the R1.2 architectural decision (structured blocks in the generic
  config); revisit naming/namespacing with the seam hardening when an external plugin binds.

## Deferred from: code review of r1-1-ingest-a-field-as-a-snapshot-matrix-mesh-free (2026-06-10)

- **`expand_field_columns` non-string column labels.** Pattern matches are returned as
  `str(col)` (data/loader.py); a frame whose `.columns` are non-string (e.g. a headerless
  read with integer `RangeIndex` labels, or numeric-only headers) would then mis-resolve
  against the original-dtype `frame.columns` in `select_columns` (`c not in frame.columns`
  / `frame[resolved]`), yielding a false "Columns not found". **Latent — not reachable on
  any current path:** `load_table` reads CSV *with* a header (`pd.read_csv`), Parquet, and
  HDF5, all of which yield string column labels, and the dataset schema is keyed on
  `{str(col): …}`. Harden (return the original column label for pattern matches; `str()`
  only for messages) if a headerless / numeric-label reader is ever added.
- **Glob run field-column naming in `predict`.** A run trained with a glob `data.outputs`
  stores the pattern verbatim, so `predict_with_run` (services.py:346-349) falls to generic
  `output_0…` names (the field still predicts at `(n, n_dof)`). Field-aware naming /
  reconstruction is **Story R4.1**; this is the R1.1 scope boundary working as intended,
  recorded here only so the R4.1 author re-expands the stored pattern against the loaded
  frame (or a recorded resolved-column list) when naming reconstructed-field columns.

## Deferred from: code review of r2-5-record-probe-coverage-prefer-a-global-energy-norm (2026-06-09)

- **Engine ROM-hook `probe_coverage_energy` direct index.** `engine.run_training` reads
  `float(diag["probe_coverage_energy"])` (engine.py) by direct index; a third-party ROM
  trainer whose `reproducibility_diagnostics` hook omits the key would raise `KeyError` and
  abort the run. This is consistent with the existing R2.3 contract — the sibling keys
  (`basis_hash`/`holdout_coefficients`/`reduction`) are all direct-indexed — and folds into
  the **already-recorded R2.3 deferral "Engine ROM-hook input validation for external
  plugins"** (above). The built-in `ReducedFieldTrainer` always returns the key, so it
  cannot fire in v1. Harden (validate keys / fail-soft to full coverage 1.0) together with
  that R2.3 item when the first external ROM plugin binds.

## Deferred from: code review of r2-2-sign-order-canonicalization-with-degeneracy-handling (2026-06-07)

- **`degenerate_clusters` single-linkage chaining.** Each candidate mode is compared to
  the previous accepted member (`sigma[cur[-1]]`), so a chain of near-equal σ can merge
  into one cluster whose extremes differ by more than `rtol`. This is byte-identical to
  the frozen R2.1 spike (`spec-field-rom-fingerprint.md`), so changing it is a contract
  renegotiation, not a fix. Revisit if a real field produces a problematic σ-chain.
- **`degenerate_clusters` `max(σ, 1.0)` tolerance floor.** For σ<1 the relative tolerance
  degrades to an absolute gap of `rtol`, over-clustering small (tail) singular values.
  Also a faithful port of the frozen spike. Tail modes are usually truncated by the energy
  threshold anyway; revisit alongside R2.5 (global energy-norm) if it ever bites.

## Deferred from: code review (re-review) of r2-2-sign-order-canonicalization-with-degeneracy-handling (2026-06-07)

- **`round(σ, 9)` absolute sort-key vs relative `rtol` cluster metric.** `canonicalize`
  orders modes on an absolute-rounded σ key while `degenerate_clusters` groups by a
  *relative* tolerance, so the two disagree about "equal σ" for values differing by
  <5e-10 that are not clustered — the per-mode order can flip. Byte-identical to the
  frozen spike: the `round(,9)` quantizer deliberately absorbs BLAS noise in the
  ordering, and the field-primary verdict (R2.3) is the safety net for any residual
  per-mode instability. Revisit with R2.5 (global energy-norm), which re-touches σ scaling.
- **`subspace_reproduces` cluster-index bounds.** An out-of-range `cluster` index raises a
  raw numpy `IndexError` rather than a clear `ValueError`. In-contract the cluster always
  comes from `degenerate_clusters` of same-rank coeffs, so it cannot fire today. The
  clean-error guard belongs with R2.3, which establishes the real caller relationship.

## Deferred from: code review of r2-3-field-primary-verdict-in-reproduce (2026-06-09)

- **Engine ROM-hook input validation for external plugins.** `engine.run_training` calls
  the duck-typed `reproducibility_diagnostics(x)` hook and indexes its dict
  (`basis_hash`/`holdout_coefficients`/`reduction`) directly; a malformed dict (missing
  key, ragged coefficients) from a *third-party* ROM trainer would raise and abort the
  training run. The built-in `ReducedFieldTrainer` hook is well-formed, so this cannot
  fire in v1. Harden (validate keys / fail-soft to no field_extras) when an external ROM
  plugin first binds.
- **Annotate the frozen fingerprint spec's stale non-frozen prose.** `spec-field-rom-
  fingerprint.md` lines 76–81 (the "reproduce() Contract (proposed shape)" section, which
  is OUTSIDE the `<frozen-after-approval>` block) lists `reduction`/`energy_captured` as
  "part of the hashed spec" and a persisted `basis.npz`. The shipped R2.3 design folds
  only `config + tolerance` into `spec_hash` (energy_captured is ULP-sensitive) and uses a
  coefficient-subspace diagnostic instead of `basis.npz` — both sound, Auditor-confirmed.
  Add a one-line "superseded by R2.3 implementation" note so a future reader does not
  treat the proposed shape as binding.

## From: fix-reproduce-fingerprint (2026-06-04 review)

- **Thread-pinned, `@pytest.mark.slow` reproducibility test.** The reproduction tests
  do not pin BLAS threads (`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS=1`) and are not
  marked slow. The project-context reproducibility discipline calls for this, but it
  is flagged `(planned)` and belongs with the CI story (it needs a CI matrix to be
  meaningful). Pick up when standing up CI.

## From: code review of r2-4-scalar-path-stays-byte-identical (2026-06-09)

- **Over-tolerance scalar verdict not pinned end-to-end through `reproduce()`.** AC1 names
  "verdict"; R2.4 pins the happy-path `reproduced` end-to-end and the divergent verdict via
  a hand-built `ReproduceResult`, but no real scalar run is driven to over-tolerance
  `diverged` through `reproduce()`. Well-backstopped (`_verdict` is unit-tested across
  tolerance bands in test_reproduce.py:134–140; `data_changed` end-to-end at
  test_reproduce.py:51–66) and a real polynomial run reproduces bit-identically by design,
  so a genuine over-tolerance scalar run is awkward to construct. Pick up if a future
  scalar method has a non-zero tolerance worth exercising end-to-end.
- **`metrics.json` / config-snapshot byte-shape not asserted by the R2.4 guard.** Not an AC
  requirement (AC1 enumerates lineage + fingerprint + `ReproduceResult` only); the broader
  run-dir shape is backstopped by `test_reproduce.py` and the full suite. If "byte-identical
  to pre-ROM" is ever read to include the whole run dir, add a metrics/config-snapshot
  shape pin to the scalar-identity guard.

## Deferred from: code review of r3-2-two-source-error-split-in-the-report (2026-06-11)

- **`two_source_split` overflows/NaNs for FIELD_ENERGY_FLOOR-dominated relative errors.**
  A persisted field run whose reference field is identically zero gets
  `energy_l2 ≈ ‖diff‖/1e-300` (by design of the floor); `energy_l2**2` then raises
  `OverflowError` at report-render time (Python float `**`), or — via numpy floats —
  `inf − inf = nan` renders a silent `nan` row. The ratio formulation
  `e·sqrt(max(1 − (t/e)², 0))` is overflow-free and algebraically identical. Not patched
  now: the input is degenerate (an all-zero target field), the metric values themselves
  are already meaningless at that magnitude, and the main metrics table renders the same
  absurd numbers. Pick up if a robustness pass over render-time failure modes happens.

## Deferred from: code review of r4-1-predict-coefficients-by-default-reconstruct-on-field (2026-06-11)

- **`mode_k`/output columns silently overwrite same-named input columns**
  (`services.py` predict paths). Pre-existing convention: the scalar predict path
  has overwritten output-named input columns since Epic 3; the new `mode_k` path
  merely follows it. Changing it (collision detection, suffixing, or a warning)
  is a cross-cutting design decision over both paths.
- **Orphaned basis npz on failed runs; no GC for the shared `basis/` store**
  (`engine.py` basis-store block). The npz is written before metrics/lineage; a
  later failure leaves an unreferenced artifact, and deleting run dirs never
  removes store entries. Benign for a content-addressed store (a retry re-uses
  the artifact), but a GC/refcount policy is undefined — revisit with R4.4
  packaging or post-MVP.
- **Predict-written `.h5` re-read via `load_table` raises a raw pytables error,
  not `DataError`** (`data/loader.py:_read_hdf`, unchanged by R4.1). The story
  ratifies non-re-ingestability (raw h5py column groups; pandas `read_hdf` ships
  in no extra), but the loader's missing-backend/wrong-format error translation
  is rough — pick up if a loader robustness pass happens.

## Deferred from: code review round 2 of r4-1-predict-coefficients-by-default-reconstruct-on-field (2026-06-11)

- **Store entries trusted by filename only** (`engine.py` dedup check): a
  zero-byte or truncated `basis/<sha>.npz` (interrupted copy, cloud partial
  sync — the project lives in CloudDocs) is never repaired, because
  `store_path.exists()` short-circuits the write forever. A content/size
  verification before reuse belongs to the same store-policy pass as the GC
  question above.
- **Persist-window failure strands a half-written run dir** (`engine.py`
  between `model.pkl` and `lineage.json`): the run id is consumed, the dir
  fails `Run.exists()` and is invisible to `list`/`run()`. Pre-existing
  sequence semantics; the R4.1 store block merely added new exception sources
  inside the window. Revisit with a persist-transaction/cleanup pass.
- **Duck-typed `predict_coefficients` has no conformance backing** (rank-1
  shapes, return contract): R4.3's reconstruction round-trip conformance probe
  is the natural home — add a `predict_coefficients` shape/identity check
  there.

## Deferred from: code review (round 2) of r4-2-surroforge-rom-inspect (2026-06-13)

- **`energy_captured` vs. `cumulative` column from different models, no visual cue**
  (`cli/main.py` rom_inspect): the printed `energy_captured` is the holdout-model lineage
  value (consistent with the reproduce verdict, AC 1), while the per-mode `cumulative`
  column is derived from the final-model `basis.npz` spectrum. A reader may expect
  `cumulative[rank-1] ≈ energy_captured` and be confused when they differ slightly (holdout
  ≠ final train set). Spec mandates the lineage value; a display-polish note (e.g. a
  footnote "energy_captured from holdout model") is the natural Phase-2 fix. Same family as
  the round-1 holdout-vs-final `[retained]` defer below.

## Deferred from: code review of r4-2-surroforge-rom-inspect (2026-06-12)

- **`[retained]` markers use the holdout-model rank against the final-model
  spectrum** (`services.py` inspect_rom_run / `cli/main.py` rom_inspect): under
  energy/adaptive rank configs the holdout and final model can resolve
  different ranks (the engine's own residual, engine.py:174-179), so the
  spectrum table can mislabel retained modes. AC 1 mandates the lineage rank
  for reproduce-verdict consistency — a deliberate spec choice. If it bites,
  the final rank is available as `npz["basis"].shape[1]`; reconcile in a
  display-polish pass.
- **Unbounded spectrum table in `rom inspect`** (`cli/main.py` rom_inspect):
  one line per mode, no truncation; thousands of modes flood the terminal.
  V1 scope excludes display options — a `--top` flag is the natural Phase-2
  addition.
- **Zero-energy spectrum branch untested** (`services.py` inspect_rom_run,
  `total_sq == 0` fallback): reachable for constant-field runs and correct per
  manual trace, but no regression pin exists. Add a constant-field test when
  the branch is next touched.
