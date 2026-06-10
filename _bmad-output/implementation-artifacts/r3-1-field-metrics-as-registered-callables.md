# Story R3.1: Field metrics as registered callables

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a methods engineer,
I want field-appropriate metrics (`energy_l2`, `truncation_error`) available like any other metric,
so that they flow through CV, comparison, and the report with no special handling (#41, FR-R5, AR-R4).

> **Scope realization (read first).** The story stands or falls on ONE design question: every
> entry in `AVAILABLE_METRICS` is a pure `Callable[[np.ndarray, np.ndarray], float]`
> (metrics.py:29), and `truncation_error` seems to need the model's basis — which the signature
> does not carry. **The design below resolves this** (subspace-from-predictions; empirically
> validated 2026-06-10, numbers in Dev Notes): both metrics fit the two-argument shape, so the
> entire delivery is **registry entries + tests + one constant relocation**. Expected diff:
> `validators/metrics.py` (+2 metrics, +1 relocated constant), `services.py` (import line only),
> `tests/test_field_metrics.py` (new). If implementing any AC seems to require touching
> `engine.py`, `cross_validation.py`, `reporters/`, `compare`, or `core/config.py` — STOP:
> that contradicts "no special handling" and means the design assumption broke.

## Acceptance Criteria

1. **Given** a config with `validation.metrics: [energy_l2, truncation_error, rmse]` **when** the run executes through the **unchanged** engine **then** both new metrics are resolved from `AVAILABLE_METRICS` and computed for train/test/CV exactly like `rmse` today (present in `run.metrics["train"]/["test"]` and in `cross_validation` with mean/std/per_fold) — with **zero modifications** to `engine.py`, `validators/cross_validation.py`, `reporters/`, or `core/config.py`.
2. **Given** an unknown metric name in `validation.metrics` **then** the existing clear `ConfigError` path fires unchanged (`validate_metric_names`, metrics.py:36-41; engine validates **early** at engine.py:86) — regression-pinned, and the error's "Available:" list now includes the two new names.
3. **Given** `energy_l2` **then** it is the relative field-energy error `‖y_pred − y_true‖_F / max(‖y_true‖_F, FIELD_ENERGY_FLOOR)` — **definitionally aligned** with the R2.3 reproduce-verdict norm (`_field_energy_relative_diff`, services.py:202-210), with `FIELD_ENERGY_FLOOR` relocated to `validators/metrics.py` as the single source of truth and `services.py` importing it from there (value `1e-300` unchanged; behavior byte-identical).
4. **Given** `truncation_error` **then** it returns the orthogonal-residual fraction of `y_true` w.r.t. the affine subspace the predictions inhabit (definition in Dev Notes), satisfying the Pythagorean two-source split `energy_l2² = truncation_error² + (in-subspace surrogate error)²` (exact up to the `matrix_rank` tolerance cut — a mode with `σ ≤ σ_max·max(shape)·eps` leaves a ~1e-14-relative remainder; `pytest.approx` absorbs it), returning ≈0 for a full-rank or scalar model, and exposing a truncated basis (the **pinned** rank-2-model-on-rank-3-field fixture in Dev Notes yields truncation ≈ 0.153 while the in-span error is ~0).
5. **Given** `surroforge compare` and the Markdown report **then** both new metrics flow through without code change: `_select_best` already defaults unknown names to lower-is-better (services.py:325 — correct for both), and the report template renders `metric_names` generically (markdown.py:41) — pinned by the e2e test, not by edits.

## Tasks / Subtasks

- [x] **Task 1 — `energy_l2` + `FIELD_ENERGY_FLOOR` relocation** (AC: 3)
  - [x] Move `FIELD_ENERGY_FLOOR = 1e-300` (+ its comment) from `services.py:34` to `validators/metrics.py`; in `services.py` replace the definition with `from surroforge.validators.metrics import FIELD_ENERGY_FLOOR` (the only `services.py` change; its use at services.py:210 stays untouched). Import direction is legal: `services` already sits above `validators` (it imports `engine`, which imports `validators`); no import-linter contract forbids it — re-run `lint-imports` to confirm 3/3.
  - [x] Implement `_energy_l2(y_true, y_pred)` in `validators/metrics.py`: `float(np.linalg.norm(y_pred - y_true) / max(np.linalg.norm(y_true), FIELD_ENERGY_FLOOR))`. Frobenius norm over the whole `(n, n_outputs)` block — deliberately a **global energy norm**, not sklearn's per-column `uniform_average` (that asymmetry vs. rmse/mae/r2 is the point: it is the verdict's norm; document in the docstring with a pointer to services.py `_field_energy_relative_diff`).
- [x] **Task 2 — `truncation_error` (subspace-from-predictions)** (AC: 4)
  - [x] Implement `_truncation_error(y_true, y_pred)` exactly as validated (Dev Notes has the reference implementation): center `y_pred` by its own column mean; SVD of the centered block; keep right-singular vectors with `σ > σ_max · max(centered.shape) · eps` (numpy `matrix_rank` default tolerance — deterministic, no new constant); the metric is `‖resid − (resid @ B) @ B.T‖_F / max(‖y_true‖_F, FIELD_ENERGY_FLOOR)` with `resid = y_true − mean(y_pred)`. Guard the all-zero-predictions case (`σ_max == 0` or empty) with an empty basis (the metric then returns the full relative residual).
  - [x] **Docstring honesty (semantics):** it measures the floor of the subspace the predictions **actually inhabit** — for a `reduced-field` run that is `span(V)` (predictions are `mean + C·Vᵀ`, and `mean(y_pred)` lies in the same affine space, so centering does not leave it), but if an inner surrogate predicts a constant for a retained mode, that mode's true variation counts as truncation (the model effectively does not use the mode — arguably the honest answer). For a full-rank/scalar model the centered predictions span the full output space ⇒ ≈0. State both in the docstring. Also note the 2-D contract: the module docstring (metrics.py:1-4) already declares it, the existing sklearn metrics merely *tolerate* 1-D, but `_truncation_error` raises `LinAlgError: 1-dimensional array given` on 1-D input (every engine/CV path passes 2-D — `select_columns` and all trainers' `predict` return 2-D — so no guard; just the sharper failure mode, documented).
- [x] **Task 3 — register + unknown-metric regression** (AC: 1, 2)
  - [x] Add `"energy_l2"` and `"truncation_error"` to `AVAILABLE_METRICS` (metrics.py:29-33). NO other registry/validation change — `validate_metric_names` and the early engine check (engine.py:86) work as-is.
  - [x] Regression test: an unknown name still raises `ConfigError` via `pytest.raises(ConfigError, match=...)` and the message's available-list contains both new names (recompute `sorted(AVAILABLE_METRICS)` in the assertion — no golden string literal).
- [x] **Task 4 — unit tests with teeth** (AC: 3, 4) — new `tests/test_field_metrics.py` (house style: one file per concern, like `test_field_ingest.py`; `tests/test_metrics.py` stays untouched as the scalar regression net)
  - [x] `energy_l2`: definitional check against a hand-computed value (recomputed, not golden); zero-field guard (`energy_l2(0, 0) == 0.0`, no division error); agreement with the verdict norm shape.
  - [x] `truncation_error` discrimination (the story's teeth, **pinned fixture from Dev Notes — copy it verbatim, incl. seed 42 and amplitudes `[5, 2, 2]`**): rank-2 `ReducedFieldTrainer` on the rank-3 field → `truncation_error > 0.1` (measured 0.153 on the pinned fixture; 0.16–0.30 across 20 seeds — real headroom, not a 5% margin) AND in-span error `sqrt(max(energy_l2² − truncation², 0)) < 1e-12` (polynomial inner is exact on the fixture); rank-3 (full-rank) model on the same field → `truncation_error < 1e-12`; plain `PolynomialTrainer` (scalar, 2 outputs) → `< 1e-12`; Pythagorean identity `energy_l2² ≈ truncation² + in_span²` via `pytest.approx`.
  - [x] Edge cases: all-zero predictions (empty basis path); single-sample input (centered block is zero ⇒ full residual — defined, not crashing); margins not bit-equality for every SVD-derived number (BLAS discipline — fixture singular values are well-separated by construction).
- [x] **Task 5 — e2e: metrics flow through engine, CV, report, compare unchanged** (AC: 1, 5)
  - [x] One e2e in `tests/test_field_metrics.py`: a `reduced-field` run (polynomial inner — fast, deterministic, no smt/BLAS guard needed) with `validation.metrics: [energy_l2, truncation_error, rmse]` and `cross_validation: 3` through `run_training`; assert all three appear in `train`, `test`, and `cross_validation` (with `mean`/`std`/`per_fold` of length 3), and that the test-split values satisfy the Pythagorean relation. Reuse the inline low-rank-field + CSV + `Project.init` scaffold pattern (replicate locally like R1.4 did — do not import from another test module).
  - [x] Report + compare in the same e2e: `generate_report` (or the existing reporter entry point) renders rows for both new names (assert substring presence of the metric names in the Markdown — the template iterates `metric_names` generically); `compare_runs([run], metric="energy_l2")` returns that run as best (pins the lower-is-better default without editing `_LOWER_IS_BETTER` — deliberately zero-touch, the default is already correct).
- [x] **Task 6 — gates green**
  - [x] `ruff check src tests`, `mypy src`, `lint-imports` (3 contracts — re-check after the constant relocation). Full `pytest` (the existing 272-test suite must stay green: metrics are additive; `test_metrics.py`, `test_cross_validation.py`, reporter/compare tests unchanged; no test pins the metric set or the "Available:" string — verified).

### Review Findings

- [x] [Review][Patch] In-span assertion `< 1e-12` is BLAS-flaky and the Pythagorean assertion is a tautology — `in_span` is DERIVED from energy/trunc (`sqrt(max(e²−t²,0))`), so `e² ≈ t² + in_span²` is algebra, not verification; and under 1-ulp rounding noise `in_span` reaches ~6.5e-9 (verified empirically), so `< 1e-12` only passes when the `max(…,0)` clamp happens to fire — a latent flake on the next BLAS/numpy change, violating the project's own margins-over-bit-equality discipline. Fix: assert the robust forms `trunc <= energy + 1e-12` (mathematical inequality) + `trunc == pytest.approx(energy, abs=1e-9)` (exact inner ⇒ all error is truncation) + `in_span < 1e-6` (still 5 orders below the 0.1 floor) [tests/test_field_metrics.py:107-116]
- [x] [Review][Patch] AC5's compare "pin" is vacuous — best-of-ONE run is best under ANY ordering, so a flipped lower-is-better default is undetectable; the test comment claims protection it does not provide (the story Task 5 itself prescribed the single-run call — spec-internal weakness faithfully reproduced). Fix: train a second (rank-3) run in the e2e and assert compare picks it (lower energy_l2) as best [tests/test_field_metrics.py:219-220]
- [x] [Review][Patch] New metrics lack the input contract the sklearn-backed entries enforce: shape mismatch silently broadcasts to a plausible-but-wrong number, 1-D input raises a raw `LinAlgError`, NaN/inf yields silent `nan` (energy_l2, `max(nan, floor)` keeps nan) or an SVD convergence error (truncation), empty arrays return 0.0 — a "perfect score" for no data with only a RuntimeWarning. rmse/mae/r2 raise `ValueError` on every one of these via sklearn's check_array. Fix: shared `_check_field_metric_inputs` guard (matching 2-D shapes, non-empty, all-finite → clear `ValueError`, mirroring the sklearn entries) + tests for each rejection; docstring updated (LinAlgError note obsolete) [src/surroforge/validators/metrics.py:47-78]
- [x] [Review][Patch] Wording/completeness (Low, merged): (a) docstring should state truncation_error is a BATCH-level metric (depends on the evaluation block's empirical mean/SVD — per-fold values are not per-sample averages); (b) fixture comment conflates ranges — 0.153 is the pinned seed 42, the 0.16–0.30 range is seeds 0–19; (c) e2e should assert `mean`/`std` keys for the new metrics (Task 5 claims it, test only checks per_fold) [tests/test_field_metrics.py + validators/metrics.py docstrings]
- [x] [Review][Defer] engine.py:123 wraps every `ValueError` from the CV path as `DataError` (str(exc), "blaming the dataset") — `np.linalg.LinAlgError` subclasses `ValueError`, so numerical failures (SVD non-convergence on NaN predictions from a diverged inner) surface as a misleading data error. Pre-existing engine behavior, not introduced by r3-1 [src/surroforge/engine.py:123] — deferred, pre-existing

## Dev Notes

### The design decision this story turns on (validated 2026-06-10)

`AVAILABLE_METRICS` callables are `(y_true, y_pred) -> float` and the AC mandates "no new
plumbing" — but the truncation floor seems to need the model's basis `V`. **Resolution:** the
predictions themselves reveal the basis. Every `reduced-field` prediction is `mean + c·Vᵀ`, so
the centered prediction block lies exactly in `span(V)`; an SVD of centered `y_pred` recovers
the inhabited subspace, and `mean(y_pred)` is a valid anchor of the affine prediction space
(the orthogonal distance is anchor-independent). The orthogonal residual of `y_true` against
that subspace IS the truncation floor; the in-span residual is the surrogate error; the split
is exactly Pythagorean because the two components are orthogonal.

Reference implementation (empirically validated on the dev machine, copy as the starting point):

```python
def _truncation_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean = y_pred.mean(axis=0, keepdims=True)
    centered = y_pred - mean
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    if s.size and s[0] > 0.0:
        tol = s[0] * max(centered.shape) * np.finfo(float).eps  # matrix_rank default
        basis = vt[s > tol].T
    else:
        basis = np.zeros((y_pred.shape[1], 0))
    resid = y_true - mean
    ortho = resid - (resid @ basis) @ basis.T
    return float(np.linalg.norm(ortho) / max(np.linalg.norm(y_true), FIELD_ENERGY_FLOOR))
```

**Pinned fixture** (copy verbatim into the tests — validate-create-story found that a prose-only
fixture description is seed-fragile: a faithful replication with amplitudes `[5,2,1]` measured a
floor of 0.074, FAILING a `> 0.1` assertion; the third amplitude is therefore `2`, not `1`):

```python
rng = np.random.default_rng(42)
n_dof, n = 40, 80
modes = np.linalg.qr(rng.standard_normal((n_dof, 3)))[0]   # orthonormal spatial modes
x = rng.uniform(-1.0, 1.0, size=(n, 2))
amps = np.column_stack([5.0 * x[:, 0], 2.0 * x[:, 1] ** 2, 2.0 * x[:, 0] * x[:, 1]])
y = amps @ modes.T
x_train, x_test, y_train, y_test = x[:60], x[60:], y[:60], y[60:]
```

Measured on this pinned fixture (polynomial-degree-2 inner — exact on these amplitudes):

| Case | energy_l2 | truncation_error | in-span |
|---|---|---|---|
| rank-2 model (truncated) | 1.535e-1 | **1.535e-1** | 0.0 |
| rank-3 model (full rank) | 5.0e-15 | 5.8e-16 | 5.0e-15 |
| scalar `PolynomialTrainer` | — | 1.8e-16 | — |
| tiny fold (n=3, rank-2 model) | 2.14e-1 | 2.14e-1 | defined |
| zero field | 0.0 | 0.0 | no crash |

Seed-robustness of the rank-2 floor: 0.16–0.30 across seeds 0–19 — the `> 0.1` threshold has
real headroom on ANY faithful replication, not a 5% margin. The truncated case is the "great R²
from a tiny basis doing nothing" detector r3-2 will build on: surrogate error ~0, floor 0.153.
The in-span margin (0.153 vs. <1e-12) is BLAS-safe by construction (well-separated σ).

### Existing assets and integration points (verified against source, 2026-06-10)

- **Registry:** `AVAILABLE_METRICS` (metrics.py:29-33), `validate_metric_names` (:36-41 —
  raises `ConfigError` naming the unknowns and `sorted(AVAILABLE_METRICS)`),
  `compute_metrics` (:44-51, coerces to float arrays). Adding entries is the whole hookup.
- **Engine flow (zero changes):** early validation engine.py:86; train/test computation
  :108-109 (holdout model); CV :119 via `cross_validate` (cross_validation.py:23-62, per-fold
  fresh factory — the basis is honestly re-derived per fold already, which is why the CV
  numbers for the new metrics are meaningful; r3-3 will *assert* honesty, do not do that here).
- **Verdict-norm alignment:** `_field_energy_relative_diff` (services.py:202-210) is
  `‖Δ‖_F / max(‖parent‖_F, FIELD_ENERGY_FLOOR)`. `energy_l2` uses the same form with
  `y_true` as the reference. Layer constraint: `validators` must NOT import `services`
  (would cycle: services → engine → validators) — hence the **constant relocation downward**,
  not an import upward. `FIELD_ENERGY_FLOOR` is referenced ONLY at services.py:34 (def) and
  :210 (use) — relocation is two edits + one import.
- **Compare:** `_LOWER_IS_BETTER = {"rmse": True, "mae": True, "r2": False}` (services.py:294)
  with `.get(metric, True)` at :325 — lower-better default is already correct for both new
  metrics. **Deliberately zero-touch**; the e2e pins the behavior so a future default flip
  cannot silently invert best-run selection for field metrics.
- **Report:** template iterates `metric_names` from `config.validation.metrics`
  (markdown.py:41,94) — new names render with no reporter change.
- **Conformance:** `conformance/trainer.py` computes its own checks; a growing registry does
  not affect it. `tests/test_metrics.py` / `test_cross_validation.py` pin the scalar metrics —
  they must pass unchanged (AC1's "zero modifications" net).

### Constraints that bite

- **[MUST] Production diff = `validators/metrics.py` + one `services.py` import line.** Any
  edit to engine/CV/report/compare/config contradicts the AC ("no special handling") — STOP
  and consult the lead instead of patching around it.
- **[MUST] No golden literals; margins over bit-equality** for SVD-derived assertions
  (project BLAS discipline). The fixture's singular-value separation makes every threshold
  in Task 4 safe by orders of magnitude.
- **[GOTCHA] Defaults are non-discriminating (R1.4 retro lesson):** the e2e must use a
  fixture where `truncation_error` is decisively non-zero for the truncated case (rank-2 on
  rank-3, NOT energy 0.999 on a rank-2 field, where the floor would be ~0 and the metric
  indistinguishable from noise).
- **[GOTCHA] `energy_l2` is a global Frobenius norm, NOT a per-column average** — that is
  intentional (verdict alignment) and differs from rmse/mae/r2's `uniform_average`. Do not
  "fix" it to match sklearn conventions.
- **[GOTCHA] Metrics do not enter the fingerprint/spec-hash** — no reproduce/pinned-thread
  ceremony needed beyond the normal suite run; the e2e needs no BLAS guard (polynomial inner).
- **[GOTCHA] Known cosmetic, do NOT fix:** the CLI metrics table formats the name column as
  `{'metric':<8}` (cli/main.py:105); `truncation_error` (16 chars) shifts that row's columns.
  Purely cosmetic, breaks nothing — widening it would violate the diff discipline above.
  Leave it; note it in Completion Notes if it bothers you.
- **[SCOPE] Not in this story:** `rmse@mode_k` (PRD #41 mentions it; the epic AC does not —
  per-mode metrics are Phase 2 / r3-2 territory), the two-source report **table** (r3-2),
  basis-variance surfacing and leaky-CV comparison (r3-3), any `_LOWER_IS_BETTER` edit.

### Previous-story intelligence (R1.4 / Epic R1 retro, 2026-06-10)

- Pin discriminating values: assert the truncated-case floor `> 0.1` AND the in-span error
  `< 1e-12` — either alone could pass under a broken implementation; together they pin the
  orthogonal decomposition.
- Verify empirically before pinning: the reference implementation above was run against all
  five cases in the table — re-run it as the first dev step (teeth check) before writing tests.
- Engine introspection lesson (project-context GOTCHA) does not apply here — metrics see only
  `(y_true, y_pred)`, never the model. That is the point of the design.
- Process: commit this story at `done` (after review) before r3-2.

### Testing standards

pytest, flat `tests/`, new file `tests/test_field_metrics.py` (mirror `test_field_ingest.py`
precedent: replicate small scaffolds locally, do not import other test modules). Deterministic
fixtures (seeded `default_rng`, polynomial inner). `pytest.approx` for the Pythagorean
identity; `match=` regexes for the `ConfigError` path recomputed from `AVAILABLE_METRICS`.

### Project Structure Notes

- Files: `src/surroforge/validators/metrics.py` (modified), `src/surroforge/services.py`
  (import line), `tests/test_field_metrics.py` (new). Nothing else.
- r3-1 is the first R3 story — `epic-r3` flips to `in-progress` with this story's creation.

### References

- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md#Story R3.1] — ACs (FR-R5, AR-R4; #41).
- [Source: _bmad-output/planning-artifacts/prd-field-rom-v1.md §4.6] — metric-as-key (#41), two-source error (#25, r3-2), honest-basis CV (#40, r3-3); §"Seam"-table row `validators/metrics.py`.
- [Source: src/surroforge/validators/metrics.py:17-51] — registry, validation, computation (the file to extend).
- [Source: src/surroforge/validators/cross_validation.py:23-62] — per-fold fresh factory (unchanged).
- [Source: src/surroforge/engine.py:86,108-109,119] — early validation + train/test/CV call sites (unchanged).
- [Source: src/surroforge/services.py:34,202-210,294,321-327] — `FIELD_ENERGY_FLOOR` (relocating), verdict norm (alignment), `_LOWER_IS_BETTER` + `_select_best` (zero-touch).
- [Source: src/surroforge/reporters/markdown.py:41,94] — generic metric rendering (unchanged).
- [Source: tests/test_field_ingest.py / tests/test_reduced_field.py] — scaffold patterns to replicate (CSV + `Project.init` + `run_training`; local helper replication precedent).
- [Source: _bmad-output/implementation-artifacts/epic-r1-retro-2026-06-10.md] — open design question resolved by this story's Dev Notes; lessons applied above.

## Dev Agent Record

### Agent Model Used

claude-fable-5 (Claude Fable 5)

### Debug Log References

- RED confirmed before implementation: `tests/test_field_metrics.py` fails on
  `ImportError: cannot import name 'FIELD_ENERGY_FLOOR' from 'surroforge.validators.metrics'`.
- GREEN: all 12 new tests pass after implementing Tasks 1–3 (no iteration needed — the
  reference implementation from Dev Notes worked verbatim, as pre-validated).
- One full-suite invocation flagged `test_probe_coverage.py::test_render_warnings_passes_through_foreign_warnings`
  as FAILED — artifact of a diagnostic `-p no:warnings` flag (the test uses pytest warning
  capture), not a regression; passes in isolation and in the normal full run (exit 0).

### Completion Notes List

- Production diff exactly as scoped: `validators/metrics.py` (+`FIELD_ENERGY_FLOOR`,
  +`_energy_l2`, +`_truncation_error`, 2 registry entries) and `services.py` (definition
  replaced by `from surroforge.validators.metrics import FIELD_ENERGY_FLOOR` + relocation
  comment). Zero changes to engine/CV/reporters/compare/config — AC1/AC5's "no special
  handling" held with no STOP condition triggered.
- `services.FIELD_ENERGY_FLOOR` remains importable (re-export via the import), pinned by
  `test_field_energy_floor_relocated_single_source` asserting object identity with the
  validators constant — single source of truth, byte-identical behavior (AC3).
- The pinned fixture from Dev Notes reproduced exactly: truncated rank-2 floor 0.153
  (asserted `> 0.1`), in-span `< 1e-12`, full-rank and scalar `< 1e-12`, all-zero-predictions
  → 1.0, single-sample defined (== energy_l2), zero field → 0.0 (AC4).
- e2e: `validation.metrics: [energy_l2, truncation_error, rmse]` through the unchanged
  engine — all three in train/test/CV (3 folds, per_fold length 3), test-split Pythagoras
  holds (`trunc ≈ energy`, exact polynomial inner), report text contains both names,
  `compare_runs(..., metric="energy_l2")` picks the run via the lower-is-better default
  (AC1, AC2, AC5).
- Gates: ruff clean, mypy clean (41 files), lint-imports 3/3 contracts KEPT (re-checked
  after the relocation), full suite green — 284 tests collected (272 existing + 12 new),
  exit 0, no regressions.
- Known cosmetic left as instructed: CLI metrics-table column width (cli/main.py:105) shifts
  for the 16-char `truncation_error` name — out of scope per the story's diff discipline.

### File List

- src/surroforge/validators/metrics.py (modified — FIELD_ENERGY_FLOOR relocated here; _energy_l2 + _truncation_error added and registered)
- src/surroforge/services.py (modified — FIELD_ENERGY_FLOOR definition replaced by import from validators.metrics)
- tests/test_field_metrics.py (new — 12 tests: registration/regression, energy_l2 unit, truncation_error discrimination + edges, e2e engine/CV/report/compare)
- _bmad-output/implementation-artifacts/r3-1-field-metrics-as-registered-callables.md (story tracking)
- _bmad-output/implementation-artifacts/sprint-status.yaml (status sync)

### Change Log

- 2026-06-11: Adversarial code review (3 layers: Blind Hunter / Edge Case Hunter / Acceptance Auditor; 24 raw findings → 4 patches, 1 defer, 9 dismissed). Auditor: all 5 ACs satisfied, diff discipline exactly honored, pinned fixture verbatim, gates re-verified independently. Patches applied: (1) flaky in-span assertion `< 1e-12` replaced with robust margins (`trunc ≤ energy + 1e-12`, `approx(abs=1e-9)`, `in_span < 1e-6`) — the derived gap amplifies 1-ulp noise to ~1e-8 (verified empirically), the old threshold passed only via the max-clamp; tautological Pythagoras assert removed; (2) compare pin made discriminating — second full-rank run, best-selection now actually detects a lower-is-better default flip; (3) `_check_field_inputs` guard on both new metrics (matching 2-D, non-empty, finite → clear ValueError, mirroring sklearn's check_array; kills silent broadcasting, raw LinAlgError on 1-D, silent nan, perfect-score-on-empty) + parametrized rejection tests; (4) batch-level-metric docstring sentence, fixture comment disambiguated (0.153 = seed 42; 0.16–0.30 = seeds 0–19), e2e asserts mean/std keys. Deferred (pre-existing): engine.py:123 rewraps CV ValueErrors (incl. LinAlgError) as DataError → deferred-work.md. Gates after patches: ruff/mypy/lint-imports (3/3) clean, full suite 286 tests exit 0. Status → done.
- 2026-06-11: Implemented (dev-story, red-green-refactor). RED: new test file fails on missing imports. GREEN: `FIELD_ENERGY_FLOOR` relocated to `validators/metrics.py` (services imports it — only services change); `_energy_l2` (global Frobenius, verdict-aligned) and `_truncation_error` (subspace-from-predictions, matrix_rank default tolerance, empty-basis guard) implemented per the validated reference and registered. 12 new tests (unit + edge + e2e through unchanged engine/CV/report/compare). Gates: ruff/mypy/lint-imports (3/3) clean, 284 tests pass, production diff = the two scoped files only. Status → review.
- 2026-06-10: validate-create-story (fresh-context adversarial validation): verdict APPROVED WITH FIXES, all applied. **Critical:** the discriminating fixture was prose-only and seed-fragile (faithful replication with amplitudes `[5,2,1]` measured floor 0.074 < the prescribed 0.1 threshold) — fixture now **pinned as code** (seed 42, amplitudes `[5,2,2]`, QR-orthonormal modes), floor 0.153 on the pin, 0.16–0.30 across 20 seeds; table re-measured. Minor: suite count 269→272; 1-D `LinAlgError` failure mode documented (engine paths verified all-2-D: `select_columns` + every trainer `predict` return 2-D); CLI `{'metric':<8}` column overflow flagged as known-cosmetic-do-not-fix; AC4 "exact" Pythagorean claim footnoted with the `matrix_rank` tolerance cut. Validator confirmed: all cited line numbers exact, `FIELD_ENERGY_FLOOR` reference inventory exact (services.py:34/:210 only), no test pins the metric set, `ValidationConfig.metrics` is a free `list[str]`, import direction legal (3/3 contracts unaffected), `compare_runs` accepts ≥1 run, report renders generically.
- 2026-06-10: Story created (create-story). **Resolved the Epic-R1-retro design question:** `truncation_error` fits the pure `(y_true, y_pred)` callable shape via subspace-from-predictions (centered `y_pred` spans the model's basis; orthogonal residual = truncation floor; exact Pythagorean split with `energy_l2`) — empirically validated on five cases incl. the discriminating rank-2-on-rank-3 fixture (floor 0.105, in-span 0). `energy_l2` aligns definitionally with the R2.3 verdict norm; `FIELD_ENERGY_FLOOR` relocates downward to `validators/metrics.py` (services imports it — only services change). Compare/report/engine/CV verified zero-touch (lower-better default, generic template rendering, early name validation). Status → ready-for-dev.
