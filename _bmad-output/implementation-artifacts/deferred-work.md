# Deferred Work

Findings surfaced during reviews but intentionally out of scope for the triggering change.

## From: fix-reproduce-fingerprint (2026-06-04 review)

- **Thread-pinned, `@pytest.mark.slow` reproducibility test.** The reproduction tests
  do not pin BLAS threads (`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS=1`) and are not
  marked slow. The project-context reproducibility discipline calls for this, but it
  is flagged `(planned)` and belongs with the CI story (it needs a CI matrix to be
  meaningful). Pick up when standing up CI.
