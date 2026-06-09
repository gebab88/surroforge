# Deferred Work

Findings surfaced during reviews but intentionally out of scope for the triggering change.

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
