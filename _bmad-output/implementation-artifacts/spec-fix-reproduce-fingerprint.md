---
title: 'Fix reproducibility fingerprint: out-of-sample sample + emitted verdict'
type: 'bugfix'
created: '2026-06-04'
status: 'done'
baseline_commit: 'NO_VCS'
context:
  - '{project-root}/_bmad-output/project-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The reproducibility fingerprint (`engine._prediction_fingerprint`) is computed from the final model's predictions on the **training inputs**. Interpolating methods (Kriging) reproduce training points near-exactly regardless of how the model diverges, so two genuinely different models can produce the same fingerprint — `reproduce()` can report success when reproduction actually failed. Reproducibility is SurroForge's headline guarantee, so a fingerprint that can't detect divergence is a feature that lies.

**Approach:** Compute the fingerprint from **out-of-sample** predictions — the holdout model (fit on `x_train` only) evaluated on the held-out `x_test` — so model divergence actually moves the fingerprint. Persist that holdout prediction sample, fold a tolerance-based verdict into `reproduce()` against a single named tolerance constant, and add a sensitivity (negative) test proving two divergent models produce different fingerprints.

## Boundaries & Constraints

**Always:** Fingerprint must be **out-of-sample** (holdout model on `x_test`), never the final model on its own training data. Lineage capture stays automatic/no-opt-out. The reproduction tolerance is a **single named module-level constant**, not per-call-site literals. Sampling stays deterministic from the existing `split.seed` (no new RNG). Reuse the already-computed `holdout_model`/`x_test` from `run_training` step 3 — no extra model fitting.

**Ask First:** Renaming or removing existing lineage fields (`prediction_hash`) or changing the public `ReproduceResult` shape beyond additive fields. Changing the default tolerance value from the existing `1e-9`.

**Never:** No generated grid sampling (deferred — held-out only, per decision). No new `BaseTrainer` field (deferred). Do not weaken/loosen the check to make a flaky test pass. No new dependencies. Do not touch the training math, methods, or CLI command surface beyond verdict rendering.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Honest re-run | Same config, deterministic env | `reproduce()` verdict = reproduced; `max_abs_diff` ≤ tolerance; no raise | N/A |
| Divergent models (sensitivity) | Two runs, different `seed` (or hyperparam) | Holdout fingerprints **differ**; tolerance comparison flags divergence | N/A |
| Underlying data changed | Source file edited between runs | `data_hash_match` is False | `assert_reproducible` / verdict raises `ReproducibilityError` |
| Diverged predictions | Reproduction preds exceed tolerance | verdict = diverged | raise `ReproducibilityError` |

</frozen-after-approval>

## Code Map

- `src/surroforge/engine.py` — `_prediction_fingerprint` + `run_training` step 3/5; source of the defect; compute fingerprint from `holdout_model.predict(x_test)` and persist the holdout prediction sample.
- `src/surroforge/services.py` — `reproduce()`, `ReproduceResult`, `assert_reproducible`; add named tolerance constant + verdict; compare stored holdout samples instead of final-model-on-all-`x`.
- `src/surroforge/core/run.py` — lineage/metrics field accessors; expose the stored holdout sample.
- `tests/test_reproduce.py` — existing reproduction tests; add the sensitivity (negative) test; keep existing assertions valid.
- `tests/conftest.py` — `poly_config`, `kriging` config, `make_samples`, `project` fixtures (reuse).

## Tasks & Acceptance

**Execution:**
- [x] `src/surroforge/engine.py` -- fingerprint computed from `holdout_test_pred` (out-of-sample holdout model on `x_test`); `holdout_fingerprint` persisted in lineage; `_prediction_fingerprint` now takes a predictions array.
- [x] `src/surroforge/services.py` -- added `REPRODUCTION_TOLERANCE` constant; `max_abs_diff` computed from the two runs' stored holdout samples (shape-mismatch → inf); added `verdict` field + `_verdict()`; `assert_reproducible` routes through the constant.
- [x] `src/surroforge/core/run.py` -- added `holdout_fingerprint` accessor over lineage.
- [x] `tests/test_reproduce.py` -- added `test_fingerprint_is_sensitive_to_model_divergence` (two Kriging kernels, same split → out-of-sample fingerprints differ) + `test_reproduce_emits_verdict` + verdict assertion on the data-change test. Also removed the now-orphaned `_warn` CLI helper.

**Acceptance Criteria:**
- Given a trained run, when the persisted fingerprint is compared to the final model's in-sample predictions on the same test points, then they **differ by > tolerance** — proving the fingerprint is the holdout model's *out-of-sample* predictions (the teeth: this fails if the fingerprint reverts to in-sample).
- Given an honest re-run, when `reproduce()` runs, then verdict = reproduced and `assert_reproducible` does not raise.
- Given a reproduction whose holdout predictions exceed `REPRODUCTION_TOLERANCE`, when evaluated, then `reproduce()` surfaces a diverged verdict and `assert_reproducible` raises `ReproducibilityError`.
- Given a run with no/empty holdout fingerprint (legacy), when reproduced, then verdict = `unverifiable` and `assert_reproducible` raises a clear "cannot verify" error (never a false pass or false divergence).
- Given the full suite, when `pytest` runs, then all prior reproduction tests still pass.

## Spec Change Log

- **2026-06-04 review (patches, no loopback).** Three adversarial reviewers confirmed
  the core logic but surfaced refinements, all applied as patches:
  (1) **legacy/empty fingerprint** now yields an `unverifiable` verdict (previously a
  false "diverged" for pre-change runs, and a false "reproduced" when both empty);
  (2) **lineage bloat** bounded via `_FINGERPRINT_MAX_ROWS = 512`;
  (3) **sensitivity test replaced** with `test_fingerprint_is_out_of_sample` (real
  teeth — fails if the fingerprint reverts to in-sample) plus a direct over-tolerance
  raise test. Deferred: thread-pinned `@pytest.mark.slow` repro test (→ CI story).

## Design Notes

The critical insight: the persisted artifact (`final_model`) is fit on **all** data, so any sample drawn from that data is in-sample for it — and Kriging interpolates in-sample points near-exactly even when the underlying model diverges. Fingerprinting `final_model` on `x_test` therefore does NOT fix the bug. The holdout model (`holdout_model.fit(x_train, y_train)`, already built in `run_training` step 3) has never seen `x_test`, so divergent models give divergent `x_test` predictions — that is what makes the fingerprint sensitive. Both runs persist this holdout sample, so `reproduce()` compares stored arrays with the named tolerance and never needs to refit. Grid sampling over input bounds (for fingerprinting the final model out-of-sample) is the deferred follow-up, not part of this change.

## Verification

**Commands:**
- `pytest -q tests/test_reproduce.py` -- expected: all pass, including the new sensitivity test.
- `pytest -q` -- expected: full suite green (no regressions in the other 110+ tests).
- `ruff check src tests && mypy src` -- expected: clean.

## Suggested Review Order

**The fix — fingerprint over out-of-sample holdout predictions**

- Start here: why in-sample fingerprinting was blind to divergence (the core insight).
  [`engine.py:128`](../../src/surroforge/engine.py#L128)

- The holdout test predictions captured at split time — the out-of-sample sample.
  [`engine.py:99`](../../src/surroforge/engine.py#L99)

- Fingerprint helper now hashes a predictions array, not a model.
  [`engine.py:185`](../../src/surroforge/engine.py#L185)

- Persisted sample capped so lineage stays a lean metadata file.
  [`engine.py:54`](../../src/surroforge/engine.py#L54)

**The verdict — comparison, tolerance, and legacy handling**

- Comparability guard: empty/legacy or shape-mismatched fingerprints → not comparable.
  [`services.py:62`](../../src/surroforge/services.py#L62)

- Single named tolerance constant (one source of truth).
  [`services.py:28`](../../src/surroforge/services.py#L28)

- Verdict classifier, incl. the `unverifiable` outcome for legacy runs.
  [`services.py:75`](../../src/surroforge/services.py#L75)

- `assert_reproducible` raises on data-change, unverifiable, or over-tolerance.
  [`services.py:90`](../../src/surroforge/services.py#L90)

**Surfacing the verdict**

- Lineage accessor for the stored holdout sample.
  [`run.py:86`](../../src/surroforge/core/run.py#L86)

- CLI renders the verdict and exits non-zero on divergence/data-change/unverifiable.
  [`main.py:291`](../../src/surroforge/cli/main.py#L291)

**Tests**

- The teeth: fingerprint must be out-of-sample — fails if reverted to in-sample.
  [`test_reproduce.py:74`](../../tests/test_reproduce.py#L74)

- Direct over-tolerance divergence → raises.
  [`test_reproduce.py:104`](../../tests/test_reproduce.py#L104)
