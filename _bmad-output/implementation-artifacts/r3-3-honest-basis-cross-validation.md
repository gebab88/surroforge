# Story R3.3: Honest-basis cross-validation

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a methods engineer,
I want each CV fold to re-derive the basis from only its training snapshots,
So that my CV score isn't inflated by test snapshots leaking into the basis.

## Acceptance Criteria

1. **Leaky CV is demonstrably more optimistic (the honesty pin).** Given the pinned
   a3 = 1.8 exact-inner fixture (Dev Notes) run through the REAL engine
   (`run_training`, rank-2 POD, polynomial degree-2 inner, 3-fold CV), when the test
   computes a deliberately LEAKY cross-validation — global mean + POD basis derived
   from ALL snapshots (so every fold's test snapshots leak into the basis), with
   IDENTICAL KFold splits and IDENTICAL per-mode inner fits — then the leaky mean
   `energy_l2` is strictly below the honest engine-measured mean with margin:
   `honest_mean > leaky_mean * 1.04` AND `honest_mean − leaky_mean > 0.007`
   (engine-measured: honest 0.18453 vs leaky 0.17047, ratio 1.082, diff 0.0141 —
   2× headroom on both). The honest path's per-fold surrogate error is `< 1e-6`
   each (exact inner ⇒ total ≡ truncation floor per fold), so the comparison
   isolates basis leakage — nothing else differs.
2. **Basis variance across folds is a first-class report number.** Given a
   reduced-field run with both field metrics configured, the Two-Source Error Split
   section's reduction context renders a labeled basis-variance line — per-fold
   truncation floors plus their std — e.g. (on the DISCRIMINATING degree-1 / a3=0.8
   fixture, see below)
   `Basis variance across folds (honest CV): per-fold truncation floors 0.093964, 0.080809, 0.085696 (std 0.0054296).`
   The rendered values match per-fold-recomputed numbers from `run.metrics`
   (recomputed, not golden): each floor is `cv["truncation_error"]["per_fold"][k]`
   at the report's `{:.5g}`, and the std equals `np.std` over those floors (the same
   number as the table's truncation CV-std cell — the line's job is to label it and
   expose the per-fold spread the table's aggregate hides). **The pin MUST be
   line-scoped and MUST use the degree-1 / a3=0.8 fixture** (validation findings):
   the std string already occurs 4× elsewhere in the report (CV-std cells), so a
   whole-report substring assert is vacant; and on the exact-inner a3=1.8 fixture
   floors ≡ energies bitwise, so a template wired to `energy_l2` per_fold or
   `cv_std.total` would render a byte-identical line. On degree-1/a3=0.8 the floors
   `[0.093964, 0.080809, 0.085696]` differ from the energies
   `[0.20827, 0.21425, 0.22951]` and `std(trunc)=0.0054296` from
   `std(energy)=0.008944` at 5g — wrong wiring fails.
3. **No leak into non-reduced or scalar reports.** An unreduced-method field run
   (`method: polynomial` + both field metrics — no `lineage["reduction"]`) renders
   the section WITHOUT the basis-variance line (there is no per-fold basis to speak
   of); a scalar run and a field run without the metrics render byte-identically to
   pre-R3.3 (the existing splice-point pin `"-fold.\n\n## Configuration"` and the
   r3-2 absence tests stay green unmodified).
4. **No production change outside the reporter.** `validators/cross_validation.py`
   is UNTOUCHED (the epic states the honest basis comes from the existing per-fold
   fresh-factory contract — `cross_validate` already calls `factory()` per fold and
   `ReducedFieldTrainer.fit` re-derives mean/basis from the fold's training block
   only; this story pins that, it does not build it). `validators/metrics.py`,
   `engine.py`, and `trainers/reduced_field.py` are untouched. Production diff =
   `src/surroforge/reporters/markdown.py` only.

## Tasks / Subtasks

- [x] **Task 1 — the honesty pin: leaky CV vs the engine (AC 1, 4)**
  - [x] New test file `tests/test_honest_basis_cv.py` (flat-tests convention; module
    docstring cites PRD 4.6 / #40). Local pinned fixture builder (copy of the r3-2
    family, parameterized over `a3` AND `degree` like r3-2's `_field_run` — seed 42,
    QR-orthonormal modes 40×3, x uniform(−1,1) 80×2, amps
    `[5·x0, 2·x1², a3·x0·x1]`, `test_fraction 0.25, seed 1`, rank-2 POD +
    polynomial inner, `cross_validation: 3`, metrics
    `[energy_l2, truncation_error, rmse, r2]`) driven through `run_training`.
    Task 1 uses `a3=1.8, degree=2` (exact inner); Task 2 uses `a3=0.8, degree=1`
    (discriminating floors-vs-energies).
  - [x] Leaky-CV helper INSIDE the test module (never in src/): global
    `mean = y.mean(axis=0, keepdims=True)`, global basis = top-2 right-singular
    vectors of `y − mean`; `KFold(n_splits=3, shuffle=True, random_state=1)` —
    **`random_state` is `split.seed` (= 1), NOT the config seed 42** (engine.py:121
    passes `seed=split.seed`); per fold fit `get_trainer("polynomial")(degree=2)`
    per mode on `(x[train], coeffs_train[:, k:k+1])`, reconstruct
    `pred_coeffs @ basis.T + mean`, score with `AVAILABLE_METRICS["energy_l2"]`.
  - [x] Assertions: honest per-fold floors/energies from
    `run.metrics["cross_validation"]` — the engine-measured values
    `[0.19243, 0.17422, 0.18693]` go in a COMMENT, asserted via
    `pytest.approx(rel=1e-3)`, never 5g-string equality on SVD-derived engine
    outputs (validation finding: a 5th-digit flip across BLAS implementations is a
    known gotcha; the honesty AC is carried by the margins, which are BLAS-robust
    because honest and leaky move together). Exact-inner isolation: per-fold
    `two_source_split(e, t)["surrogate"] < 1e-6` (use the helper — never re-derive
    the clamp inline; measured exactly 0.0 per fold). The optimism gap:
    `honest_mean > leaky_mean * 1.04` and `honest_mean − leaky_mean > 0.007`
    (measured ratio 1.082, diff 0.0141 — 2× headroom both).
- [x] **Task 2 — basis variance as a first-class report line (AC 2, 3, 4)**
  - [x] `reporters/markdown.py` `render()`: inside the existing `two_source` block
    (the r3-2 symmetric guard already guarantees both `per_fold` lists exist,
    non-empty, equal length), add the formatted per-fold floors to the context —
    hoist the `fmt` lambda above the `two_source` derivation and reuse it:
    `"truncation_per_fold": ", ".join(fmt(t) for t in cv["truncation_error"]["per_fold"])`
    (single formatting authority; do NOT duplicate a second `":.5g"` literal).
  - [x] Template: add the line INSIDE `{% if two_source.reduction %}` (second line,
    after `Reduction: rank …`):
    `Basis variance across folds (honest CV): per-fold truncation floors {{ two_source.truncation_per_fold }} (std {{ fmt(two_source.cv_std.truncation) }}).`
    Both lines stay inside the reduction-if, so unreduced/scalar output is
    byte-unchanged (AC 3) — re-verify the splice-point pin and the r3-2 absence
    tests WITHOUT modifying them.
  - [x] Tests (in `tests/test_honest_basis_cv.py`), **on the DISCRIMINATING
    degree-1 / a3=0.8 run** (NOT the exact-inner a3=1.8 run — there floors ≡
    energies bitwise and the line test would be vacant; validation finding):
    **line-scoped extraction is mandatory** — `splitlines()`, select lines starting
    with `Basis variance across folds (honest CV)`, assert EXACTLY ONE matches,
    then assert WITHIN that extracted line: each
    `f"{v:.5g}"` for `v in cv["truncation_error"]["per_fold"]` (recomputed —
    `[0.093964, 0.080809, 0.085696]` on the pin) appears, and the std equals
    `f"{np.std(per_fold):.5g}"` (`0.0054296` on the pin). A whole-report substring
    assert is VACANT: the std string occurs 4× in the report's CV-std cells. The
    discrimination: `std(energy)=0.008944` and energies `[0.20827, 0.21425,
    0.22951]` all differ at 5g — wrong-key wiring (`energy_l2` per_fold or
    `cv_std.total`) and a hardcoded template both fail. Plus the absence pin: a
    `method: polynomial` field run renders the SECTION but NOT the basis-variance
    label (mirror of r3-2's
    `test_unreduced_method_renders_split_without_reduction_line`, which must stay
    green untouched; write the new assertion in the NEW file).
- [x] **Task 3 — gates green (AC 4)**
  - [x] `ruff check src tests`, `mypy src`, `lint-imports` (3/3). Full `pytest` —
    296 existing tests stay green; `tests/test_two_source_report.py` and
    `tests/test_reporters.py` UNMODIFIED. Diff discipline: production diff =
    `reporters/markdown.py` only; `git diff --stat` shows nothing else under
    `src/`; `validators/cross_validation.py` byte-identical.

### Review Findings

- [x] [Review][Patch] Report-line pin tolerates scrambled/duplicated floors — per-value
  substring asserts (`f"{v:.5g}" in line`) verify presence, not structure; a prefix
  collision or reordered join would pass. Strengthen to exact full-line equality
  against the recomputed expected line (same values, same format — BLAS-safe)
  [tests/test_honest_basis_cv.py:155-158] (blind+edge) — APPLIED: the test now
  builds the expected line from `run.metrics` and asserts equality with the
  extracted line (vacant-pin hardening, round 3 of the same anti-pattern).
- [x] [Review][Patch] Hardcoded line-number refs in test comments go stale
  (`engine.py:121`, `config.py:184`) — cite the symbol/behavior instead
  [tests/test_honest_basis_cv.py:89-90] (blind) — APPLIED: comment now cites
  `seed=split.seed` / `CrossValidationConfig.shuffle` by symbol.
- [x] [Review][Patch] Test hygiene: unused `proj` binding in the honesty test and
  never-varied `degree` parameter on the `_leaky_cv_energies` helper
  [tests/test_honest_basis_cv.py:77,107] (blind) — APPLIED: `_proj` underscore
  convention; helper signature reduced to `(a3)` with the degree-2 inner inlined
  (it replicates the a3=1.8 fixture's exact inner — never varied).

**Dismissed (15, verified):** unguarded `truncation_per_fold` KeyError (the r3-2
symmetric guard guarantees both per_fold lists non-empty + equal length — Blind
Hunter saw only the diff); std-vs-floors "two data paths" (`two_source_split`
passes `truncation` through unchanged — `cv_std.truncation` ≡ `np.std(per_fold)`
by construction, and the test pins it); pre-joined context string (Task 2 mandates
it); missing `Any` import (exists, markdown.py:10); empty/single per_fold (guard
requires truthy per_fold; np.std of 1 value is 0.0); ddof inconsistency (both
sides ddof=0, ratified in r3-2); leaky-vs-engine split parity unasserted (the
honest per-fold `approx(rel=1e-3)` pin is the tripwire — any engine split drift
fails it loudly; validator reproduced parity empirically); leaky CV over all 80
samples (verified: engine passes FULL `x, y` to `cross_validate` — holdout is
separate); `approx(rel=1e-3)` on SVD outputs brittle (validation-mandated; BLAS
drift is ~1e-8, margin 10^5); dual optimism thresholds redundant (AC 1 mandates
both); unscaled 1e-6 surrogate bound (story-mandated; measured exactly 0.0);
missing negative coverage for CV-disabled/metric-absent (covered transitively by
the r3-2 degradation suite — the line lives inside the guarded block); heavyweight
engine tests (story mandates the real `run_training` path); per_fold-vs-folds
count mismatch on hand-edited run dirs (outside the supported surface; rendering
the actual list is faithful); unbounded fold count on one line (spec mandates the
full per-fold list; capping would contradict "first-class means decomposed").

## Dev Notes

### The design decision this story turns on (validated 2026-06-11)

**Honest-basis CV already exists structurally — this story pins it and surfaces
it.** `cross_validate` (validators/cross_validation.py:46-52) calls `factory()` per
fold; the engine's factory (engine.py:95-96) builds a fresh `ReducedFieldTrainer`,
whose `fit` (reduced_field.py:107-133) derives mean + POD basis from ONLY the data
it is handed — the fold's training block. No production change can make CV "more
honest"; what is missing is (a) a test with teeth proving the honest path is
genuinely less optimistic than a leaky one, and (b) the report surfacing the
per-fold basis spread instead of hiding it in an aggregate.

**Basis variance = the per-fold truncation-floor spread (prediction-observable).**
The metrics see only `(y_true, y_pred)` per fold; capturing fold bases would require
changing `cross_validation.py` (forbidden by the epic AC) or new plumbing. But
r3-1's `truncation_error` recovers the inhabited subspace FROM the predictions —
each fold's floor is that fold's basis quality measured on that fold's test block.
Its per-fold spread IS the honest, prediction-observable footprint of per-fold basis
re-derivation, consistent with the field-primary philosophy (measure what the
predictions inhabit, never trainer internals). The std the line renders is
deliberately the SAME number as the table's truncation CV-std cell — first-class
means labeled and decomposed (per-fold list), not new math.

**The leaky comparison isolates the basis because the inner is exact.** The fixture
amplitudes (`5·x0`, `2·x1²`, `1.8·x0·x1`) are all degree ≤ 2 polynomials in x, so
the degree-2 inner fits every retained mode's coefficients exactly — engine-measured
per-fold energy EQUALS the truncation floor (surrogate < 1e-6 per fold). Honest and
leaky runs use identical splits and identical inner fits; the ONLY difference is
whose snapshots shaped the basis. a3 = 1.8 makes mode-2/mode-3 energies comparable
(centered: `var(2·x1²) ≈ 0.356` vs `1.8²/9 = 0.36` for x ~ U(−1,1)), so the rank-2
basis is genuinely contested per fold — the regime where leakage pays the most.
Measured optimism: honest mean 0.18453 vs leaky 0.17047 (ratio 1.082); leaky is
more optimistic on every individual fold, but assert the MEAN with margins (per-fold
margins are as thin as 0.006).

### Engine-measured fixture (2026-06-11, real `run_training` path)

| Quantity | Value |
|---|---|
| honest per-fold `energy_l2` = `truncation_error` | `[0.19243, 0.17422, 0.18693]` |
| honest CV mean / std (floors) | `0.18453` / `0.0076276` |
| leaky per-fold `energy_l2` | `[0.16818, 0.16237, 0.18085]` |
| leaky CV mean | `0.17047` |
| optimism: ratio / diff | `1.082` / `0.0141` |
| holdout `test` energy = floor / r2 / energy_captured | `0.20809` / `0.82159` / `0.96874` |

Assertion thresholds carry 2× headroom (validator-confirmed: 2.06× / 2.0×):
`ratio > 1.04`, `diff > 0.007`.

**Report-line fixture (degree-1 / a3=0.8 — the discriminating one, validator-measured):**

| Quantity | Value |
|---|---|
| per-fold truncation floors | `[0.093964, 0.080809, 0.085696]`, std `0.0054296` |
| per-fold `energy_l2` (differs — degree-1 inner underfits) | `[0.20827, 0.21425, 0.22951]`, std `0.008944` |

All floor/energy values and both stds differ pairwise at `{:.5g}` — the
report-line assertions discriminate wrong-key wiring and hardcoded templates.

### Existing assets and integration points (verified against source, 2026-06-11)

- `validators/cross_validation.py:46-52` — the fresh-factory loop. **[MUST] DO NOT
  TOUCH** (epic AC: "no change to cross_validation.py"). `per_fold` persists for
  every metric (lines 54-61, `arr.std()` = ddof 0).
- `engine.py:95-96` factory; `engine.py:119-121` — `cross_validate(factory, x, y,
  folds=cv_cfg.folds, shuffle=cv_cfg.shuffle, seed=split.seed, …)`. **Replication
  gotcha:** KFold seed is `split.seed` (fixture: 1), not the run-config seed 42;
  `shuffle` defaults to `True` (config.py:184).
- `reporters/markdown.py` — r3-2's `two_source` block. The symmetric guard
  (post-review) guarantees `cv["energy_l2"]["per_fold"]` and
  `cv["truncation_error"]["per_fold"]` exist, are non-empty, and have equal lengths
  whenever `two_source` is truthy — the new context entry needs NO extra guard. The
  `fmt` lambda currently sits in the context dict (markdown.py:170); hoist it to a
  local before the `two_source` derivation so the per-fold join reuses it.
- Template structure at the insertion point (r3-2, post-review):
  `{% if two_source.reduction %}Reduction: rank …, energy captured ….\n{% endif %}{% endif %}` —
  the new line goes after the `Reduction:` line, before `{% endif %}`. Everything
  stays inside both ifs ⇒ scalar/unreduced byte-identity is structural; the existing
  splice-point pin (`SPLICE = "-fold.\n\n## Configuration"`) and r3-2's absence
  tests must pass unmodified.
- `validators/metrics.py:two_source_split` — use it for the exact-inner isolation
  assertion (the clamp is load-bearing; never re-derive `sqrt(max(e²−t²,0))`
  inline in a test).
- `tests/test_two_source_report.py::_field_run` — the fixture family to copy
  (including the `model=` override added in the r3-2 review for the unreduced-method
  case). Tests do not import across test files — duplicate the ~30-line helper into
  the new file rather than importing it.
- `plugins.get_trainer("polynomial")(degree=2)` — the inner-fit replica for the
  leaky helper (same class the reduced-field trainer instantiates internally,
  reduced_field.py:131).

### Constraints that bite

- **[MUST] `cross_validation.py` byte-identical** — the epic AC names it; the
  honest mechanism is the existing contract, and any "improvement" there voids the
  story's premise.
- **[MUST] No golden literal hashes / no exact equality on SVD-derived floats**
  (Accelerate BLAS, project memory): pins are `{:.5g}`-rendered comparisons or
  margins; surrogate thresholds 1e-6, never 1e-12.
- **Leaky helper lives in the test module only.** Shipping a leaky CV path in src/
  — even "for comparison" — is an attractive nuisance; the epic wants it
  *demonstrated in tests*.
- **Single formatting authority:** the report's `{:.5g}` exists once (`fmt`);
  the per-fold join must reuse it (r3-1 lesson: never inline a named constant
  twice).
- **Jinja whitespace:** both new-line and `Reduction:` line live inside
  `{% if two_source.reduction %}…{% endif %}` with the trailing newline INSIDE the
  block — adding a second line cannot disturb scalar output, but re-run the splice
  pin anyway (r3-2 lesson: verify, don't reason).
- **Vacant-pin check (r3-1/r3-2 recurring anti-pattern; sharpened by this story's
  validation):** TWO confirmed vacancy traps in the report-line test. (a) Aliasing:
  the std string occurs 4× in the report (CV-std cells of the metrics table and the
  two-source table) — assertions MUST be scoped to the single extracted
  `Basis variance across folds (honest CV)` line, never whole-report substrings.
  (b) Value degeneracy: on the exact-inner a3=1.8 fixture, floors ≡ energies
  bitwise (surrogate exactly 0.0) — a template wired to `energy_l2` per_fold or
  `cv_std.total` renders a byte-identical line. The report-line test therefore runs
  on the degree-1 / a3=0.8 fixture, where floors, energies, and both stds all
  differ at 5g, with expectations recomputed from `run.metrics`.
- Story file edits during dev: only Tasks checkboxes, Dev Agent Record, File List,
  Change Log, Status.

### Previous-story intelligence (r3-2, 2026-06-11)

- **Review patterns that will recur:** symmetric-guard reasoning (the reviewers
  WILL probe `per_fold` edge cases — the r3-2 guard already closes them; cite it),
  vacant pins (see above), prose-coupling dismissals (label assertions are fine
  when the prose IS the AC deliverable), BLAS-margin thresholds with stated
  headroom.
- The r3-2 review added `test_incomplete_cv_records_degrade_to_legacy_report` and
  `test_unreduced_method_renders_split_without_reduction_line` — both must stay
  green untouched; the new unreduced-method assertion (no basis-variance label)
  goes in the NEW test file.
- Defer ledger: `two_source_split` overflow on floor-dominated relative errors is
  already in `deferred-work.md` — do not "fix" it in passing here.
- Engine runs in tests cost ~1-2 s each; reuse ONE fixture run per test where the
  assertions allow (the honesty test and the report-line test can share a single
  `_field_run` if structured as one test or a module-scoped helper — keep pytest's
  tmp_path semantics in mind; simplest is one run per test, matching r3-2's style).

### Testing standards

- Flat `tests/`, one file per concern → `tests/test_honest_basis_cv.py`.
- Module docstring citing PRD 4.6 / #40 and the story.
- `ruff` (E/F/I/UP/B/W, line 100): `zip(..., strict=True)` where applicable (B905).
- Run via `.venv/bin/python -m pytest` (Python 3.14 venv; temp-FS can be tight —
  redirect long output to a repo-local file and delete it, established pattern).

### Project Structure Notes

- Production diff: `src/surroforge/reporters/markdown.py` only.
- New test file: `tests/test_honest_basis_cv.py`.
- Tracking: this story file + `sprint-status.yaml`.

### References

- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md §Story R3.3] — AC
  wording: fresh-factory contract, basis variance first-class, leaky CV more
  optimistic in tests.
- [Source: _bmad-output/planning-artifacts/prd-field-rom-v1.md §4.6] — Honest-Basis-CV
  (#40): per-fold basis from training snapshots only; global basis inflates R²;
  basis variance becomes a first-class number. §6 seam table:
  `validators/cross_validation.py` **Unverändert**. NOTE: the same table lists
  `reporters` as "Unverändert in v1" — that tension was already resolved in r3-2
  (report content is MANDATED by §4.6 #25/#40; the seam row describes plumbing, not
  templates; ratified in the r3-2 story scope note). Expected diff
  `reporters/markdown.py` is scope, not an oversight.
- [Source: _bmad-output/implementation-artifacts/r3-2-two-source-error-split-in-the-report.md]
  — two_source context shape, symmetric guard, splice-point pin, review findings.
- [Source: _bmad-output/implementation-artifacts/r3-1-field-metrics-as-registered-callables.md]
  — truncation_error semantics ("the subspace the predictions actually inhabit"),
  "r3-3 will *assert* honesty" note.
- [Source: _bmad-output/project-context.md] — BLAS/threading gotchas, named-tolerance
  discipline, flat-tests convention, dependency direction.

## Dev Agent Record

### Agent Model Used

Claude Fable 5 (claude-fable-5[1m])

### Debug Log References

- RED: `tests/test_honest_basis_cv.py` written first; exactly
  `test_report_renders_per_fold_basis_variance_line` failed (0 label lines found) —
  the honesty pin and the absence test pass against the existing system, as expected
  for pins of existing behavior.
- GREEN: `markdown.py` — `fmt` hoisted to a named local (`def fmt`) above the
  `two_source` derivation (the context's former lambda now references it; single
  formatting authority), `truncation_per_fold` joined string added to the
  `two_source` dict, template line added inside `{% if two_source.reduction %}`.
  13/13 (3 new + 10 r3-2 unmodified) on first run after the edit.
- Gates: full suite exit 0, 0 failures, 299 collected (296 + 3 new); ruff "All
  checks passed!"; mypy clean (41 files); lint-imports 3/3 kept. Diff discipline:
  `git diff --stat -- src/` = `reporters/markdown.py | 12 +++++++++++- (11+/1-)`
  ONLY — `cross_validation.py`, `metrics.py`, `engine.py`, `reduced_field.py`
  untouched (AC 4).

### Completion Notes List

- AC 1: honesty pin in `test_leaky_global_basis_cv_is_more_optimistic` — engine run
  (a3=1.8, exact degree-2 inner) vs test-local leaky CV (global mean+basis from ALL
  snapshots, identical `KFold(3, shuffle=True, random_state=1)` splits, identical
  per-mode polynomial fits). Asserted: per-fold surrogate < 1e-6 via
  `two_source_split` (isolation), honest per-fold ≈ [0.19243, 0.17422, 0.18693]
  (`pytest.approx(rel=1e-3)`, literals in comment), `honest_mean > leaky_mean*1.04`
  and `diff > 0.007` (measured ratio 1.082, diff 0.0141 — 2× headroom).
- AC 2: report line `Basis variance across folds (honest CV): per-fold truncation
  floors … (std …).` rendered inside the reduction context;
  `test_report_renders_per_fold_basis_variance_line` pins it on the discriminating
  degree-1/a3=0.8 run with mandatory line-scoped extraction (exactly one label
  line; floors and std asserted WITHIN the line; fixture discrimination
  pre-asserted: floors-vs-energies 5g-disjoint, stds differ at 5g).
- AC 3: `test_unreduced_method_gets_no_basis_variance_line` (section present, label
  absent, `"reduction" not in run.lineage` precondition); scalar/legacy
  byte-identity carried by the r3-2 splice-point and absence tests — all 10 green
  unmodified.
- AC 4: production diff = `reporters/markdown.py` only (+11/−1);
  `validators/cross_validation.py` byte-identical.

### File List

- src/surroforge/reporters/markdown.py (modified — basis-variance template line
  inside the reduction-if; `fmt` hoisted to a named local; `truncation_per_fold`
  context entry)
- tests/test_honest_basis_cv.py (new — 3 tests: leaky-vs-honest optimism pin,
  line-scoped basis-variance report pin on the discriminating fixture,
  unreduced-method absence)
- _bmad-output/implementation-artifacts/r3-3-honest-basis-cross-validation.md
  (story tracking)
- _bmad-output/implementation-artifacts/sprint-status.yaml (status sync)

### Change Log

- 2026-06-11: Adversarial code review (3 layers: Blind Hunter diff-only, Edge Case
  Hunter with repo trace, Acceptance Auditor with empirical verification). Auditor:
  ALL 4 ACs verified empirically, zero findings (diff discipline via git, template
  wording character-exact, 17 reporter-family tests green, no leaky path in src/).
  Triage of 18 unique findings (20 raw, 2 merged): 3 patches applied (exact
  full-line equality for the report-line pin — vacant-pin hardening round 3;
  symbol-based comment refs; test hygiene `_proj`/helper signature), 0 deferred,
  15 dismissed with source-verified reasoning (key verifications: engine passes
  FULL x,y to cross_validate so the leaky replication population is correct;
  two_source_split passes truncation through unchanged so cv_std.truncation ≡
  np.std(per_fold) by construction). Gates after patches: 299 tests exit 0,
  0 failures; ruff/mypy/lint-imports (3/3) clean. Status → done.
- 2026-06-11: Implemented (dev-story, red-green-refactor). RED: new test file —
  exactly the report-line test fails (no label line); honesty pin and absence test
  pass against the existing system (they pin existing behavior, as designed).
  GREEN: basis-variance line in `reporters/markdown.py` (template line inside the
  reduction-if, `truncation_per_fold` context entry, `fmt` hoisted to a named local
  — single formatting authority preserved). 3 new tests, all validation-mandated
  vacancy guards in place (line-scoped extraction, discriminating degree-1/a3=0.8
  fixture with pre-asserted 5g disjointness). Gates: ruff/mypy/lint-imports (3/3)
  clean, 299 tests (296 + 3) exit 0, production diff = reporters/markdown.py only
  (+11/−1), cross_validation.py byte-identical. Status → review.
- 2026-06-11: validate-create-story (fresh-context adversarial validation): verdict
  APPROVED WITH FIXES, all applied. **Major (2):** (1) the report-line std assertion
  was vacant as drafted — the std string `0.0076276` occurs 4× in the report's
  CV-std cells, so a whole-report substring assert passes with the line absent or
  wrong; fix: mandatory line-scoped extraction (exactly one matching line, assert
  within it). (2) The exact-inner a3=1.8 fixture is value-degenerate for the line
  test — floors ≡ energies bitwise (surrogate exactly 0.0), so wrong-key wiring
  (`energy_l2` per_fold, `cv_std.total`) renders a byte-identical line; fix: the
  line test runs on the degree-1/a3=0.8 fixture (validator-measured: floors
  [0.093964, 0.080809, 0.085696] std 0.0054296 vs energies [0.20827, 0.21425,
  0.22951] std 0.008944 — all differ at 5g). Minor: engine per-fold literals
  demoted to comments + `pytest.approx(rel=1e-3)` (no 5g-string equality on
  SVD-derived outputs, BLAS gotcha); `fmt` line ref corrected to markdown.py:170;
  the PRD §6 "reporters Unverändert" seam-table tension explicitly cited as
  resolved in r3-2. Validator confirmed empirically: all honest/leaky pins
  reproduce exactly (ratio 1.082, headroom 2.06×/2.0×, leaky more optimistic on
  every fold), honesty test genuinely discriminates (leaky cross_validate → ratio
  1.0 fails decisively), all line refs correct, blast radius clean (296 tests,
  substring-only assertions elsewhere).
- 2026-06-11: Story created (create-story). **Key analysis result: honest-basis CV
  already exists structurally** (fresh factory per fold → fold-local mean/basis);
  the story pins it (leaky comparison, engine-measured: honest 0.18453 vs leaky
  0.17047, ratio 1.082 — leaky more optimistic on every fold) and surfaces basis
  variance as a labeled report line (per-fold truncation floors + std, derived from
  data the r3-2 guard already guarantees; same number as the table's std cell,
  decomposed). Exact-inner fixture (a3 = 1.8, degree-2 amps) makes the comparison a
  pure basis-leakage measurement; a3 = 1.8 chosen so mode-2/3 energies are
  comparable (contested rank-2 basis — leakage regime). Production diff =
  reporters/markdown.py only; cross_validation.py [MUST] untouched per epic AC.
  Status → ready-for-dev.
