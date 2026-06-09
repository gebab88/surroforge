# Deferred Work

Findings surfaced during reviews but intentionally out of scope for the triggering change.

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
