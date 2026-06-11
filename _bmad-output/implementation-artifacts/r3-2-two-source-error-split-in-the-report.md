# Story R3.2: Two-source error split in the report

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a methods engineer,
I want the report to separate the projection floor from surrogate-added error,
so that I know whether to add modes or add data (#25, #42, FR-R5).

> **Scope realization (read first).** The epic AC asks for "coefficient-space and
> reconstructed-field errors in one table" — which sounds like new persistence (coefficients
> are not stored). **It is not.** Two facts make this a pure report-time derivation from
> r3-1's metrics: (1) the Pythagorean split `energy_l2² = truncation² + surrogate²` makes
> `surrogate = sqrt(max(energy_l2² − truncation², 0))` computable from the two persisted
> metric values; (2) the POD basis is **orthonormal**, so by Parseval the in-span (surrogate)
> field error EQUALS the coefficient-space error — verified empirically 2026-06-11
> (in-span 7.193e-14 vs. coefficient-space 6.639e-14 on the pinned fixture). One table with a
> Parseval footnote IS the #42 two-pass view. **Deliberate narrowing of the epic AC:** the
> epic says "Given a completed field run" unconditionally; this story renders the table only
> when both metrics are configured — the consequence of #41's metric-as-key design plus
> zero-persistence (a run without the metrics has no numbers to derive from). Ratified
> scope, not an oversight. Expected diff: `reporters/markdown.py`
> (conditional section), `validators/metrics.py` (one pure helper `two_source_split`, NOT an
> `AVAILABLE_METRICS` entry), `tests/test_two_source_report.py` (new). If implementing any AC
> seems to require touching `engine.py`, any trainer, `services.py`, `core/config.py`, or
> persisting new run data — STOP: the design assumption broke; consult the lead.

## Acceptance Criteria

1. **Given** a completed `reduced-field` run with `validation.metrics` containing BOTH
   `energy_l2` and `truncation_error` **when** I generate its Markdown report **then** it
   contains a two-source table with rows **total field error (energy_l2)** / **truncation
   floor (basis)** / **surrogate error (in-span ≡ coefficient-space)** and columns Train /
   Test / CV mean / CV std — where the CV columns are mean/std over **per-fold-derived**
   surrogate values (`sqrt(max(e_k² − t_k², 0))` per fold k; never `sqrt` of aggregated
   means — Jensen), and the rendered numbers satisfy `total² ≈ truncation² + surrogate²`.
2. **Given** the table **then** it states the Parseval identity (orthonormal POD basis ⇒
   in-span error ≡ coefficient-space error) — satisfying #42's "coefficients vs.
   reconstructed field in one table" without persisting coefficients — plus the #25
   action guidance (floor dominates ⇒ add modes; surrogate dominates ⇒ add data) and the
   reduction context (`rank`, `energy_captured`) read from `lineage["reduction"]`.
3. **Given** the pinned "deception" fixture (Dev Notes; third amplitude **0.8**) trained at
   rank 2 through the unchanged engine **then** the run *looks* great on every classic
   signal — `r2 > 0.85` (measured 0.949) and `energy_captured > 0.98` (measured 0.9936) —
   while the table exposes the truth: truncation floor `> 0.05` (measured 0.102) carrying
   ~100% of the total error and surrogate `< 1e-6` (measured 2.95e-9) — the "great R² from
   a tiny basis doing nothing" case, asserted in the e2e against the rendered report text
   AND the underlying numbers.
4. **Given** a scalar run, or any run lacking either metric (legacy field runs included),
   **then** the report is **unchanged** — the section renders ONLY when both names are in
   `config.validation.metrics` and present in `metrics["test"]`; existing
   `tests/test_reporters.py` stays untouched and green as the scalar regression net.
5. **Given** the implementation **then** the production diff is exactly
   `reporters/markdown.py` + `validators/metrics.py` (the `two_source_split` helper only;
   `AVAILABLE_METRICS` untouched) — zero changes to engine/trainers/services/config, no new
   persisted fields, `lint-imports` 3/3 (the new `reporters → validators` edge is
   leaf-to-leaf; no contract forbids it).

## Tasks / Subtasks

- [x] **Task 1 — `two_source_split` helper** (AC: 1)
  - [x] Implement `two_source_split(energy_l2: float, truncation_error: float) -> dict[str, float]`
    in `validators/metrics.py` returning `{"total": e, "truncation": t, "surrogate": sqrt(max(e² − t², 0))}`.
    Pure float algebra — NOT registered in `AVAILABLE_METRICS` (it is not a
    `(y_true, y_pred)` callable). The clamp is load-bearing: rounding noise can push
    `t` infinitesimally above `e` (r3-1 review lesson — the derived gap amplifies 1-ulp
    noise to ~1e-8). Docstring: Pythagoras source (truncation ⊥ in-span), Parseval
    (orthonormal basis ⇒ surrogate ≡ coefficient-space error), and the Jensen warning
    (derive per fold, then aggregate — never split aggregated means).
  - [x] Unit tests (in the new test file): recomputed algebra on hand values (no goldens);
    clamp case `t > e` returns surrogate 0.0 (not nan); zero case (0, 0) → all zeros.
- [x] **Task 2 — conditional report section** (AC: 1, 2, 4)
  - [x] In `reporters/markdown.py`: render() computes a `two_source` context object ONLY
    when both `"energy_l2"` and `"truncation_error"` are in `config.validation.metrics`
    AND in `metrics["test"]`; otherwise `two_source = None` and the template block
    (`{% if two_source %}`) emits nothing — scalar/legacy output byte-identical (AC 4).
  - [x] Rows from `two_source_split` on train and test values; CV mean/std from per-fold
    derivation: `splits = [two_source_split(e_k, t_k) for e_k, t_k in zip(cv["energy_l2"]["per_fold"], cv["truncation_error"]["per_fold"])]`,
    then mean/std of each component across folds (float() everything; reuse the existing
    `fmt` 5g formatter).
  - [x] Section text: heading "## Two-Source Error Split"; the table; one Parseval line
    ("orthonormal POD basis ⇒ the in-span surrogate error IS the coefficient-space error
    **in the basis the predictions actually inhabit** — this table is the
    coefficient-vs-field view"); one guidance line ("truncation floor
    dominates ⇒ add modes; surrogate dominates ⇒ add data"); one context line with
    `rank` and `energy_captured` from `lineage.get("reduction")` (guard: if the lineage
    key is absent — not expected for field runs since R2.3 — omit the context line, do
    not crash).
- [x] **Task 3 — tests with teeth** (AC: 1, 3, 4) — new `tests/test_two_source_report.py`
  (house style: one file per concern; `tests/test_reporters.py` stays untouched as the
  scalar net, exactly like r3-1 left `test_metrics.py` untouched)
  - [x] Replicate the pinned field scaffold locally (CSV + `Project.init` + `run_training`,
    pattern from `tests/test_field_metrics.py`) with the amplitude as a parameter
    (Dev Notes fixture; seed 42, QR-orthonormal modes, amplitudes `[5, 2, a3]`).
  - [x] **Table correctness** (a3 = 2.0, the r3-1 discrimination fixture): report contains
    the section heading + all three row labels; parse or recompute: rendered test-column
    AND train-column values consistent with `two_source_split` on the respective split's
    metrics (AC 1 names all four columns — the train column needs its own assertion;
    train e = t = 0.07681 on the deception pin); Pythagoras `total² ≈ truncation² +
    surrogate²` via `pytest.approx`; `rank` and `energy_captured` rendered.
  - [x] **Deception exposure e2e** (a3 = 0.8, AC 3): assert `r2 > 0.85`,
    `lineage["reduction"]["energy_captured"] > 0.98`, `truncation > 0.05`,
    `surrogate < 1e-6` (NOT 1e-12 — BLAS margin discipline, r3-1 review lesson), and the
    report text contains the section with the floor value rendered.
  - [x] **Absence regression** (AC 4): a scalar `polynomial` run (metrics `[rmse, r2]`) →
    report does NOT contain the section heading; a field run with metrics `[rmse]` only →
    no section either (legacy-run path).
  - [x] **CV derivation has teeth** (the discriminating fixture is mandatory — see the
    [MUST] below): a THIRD run on the a3 = 0.8 fixture with **inner polynomial
    `degree: 1`** (underfitting inner ⇒ real per-fold in-span error; engine-measured
    per-fold surrogates `[0.1859, 0.1984, 0.2129]`). Assert the rendered CV-mean
    surrogate equals mean-of-per-fold-splits (recomputed from `run.metrics` via
    `two_source_split` per fold — 0.19907 on the pin) AND differs from the Jensen-wrong
    `sqrt(max(mean_e² − mean_t², 0))` (0.19925 on the pin) — the two are distinguishable
    at the report's `{:.5g}` rendering. Do NOT try to pin this via the truncation or
    total rows (their CV means equal the persisted `cv[name]["mean"]` regardless of how
    the surrogate is aggregated) or via the exact-inner fixtures (per-fold surrogates
    all 0.0 ⇒ both aggregations give 0 — vacuous).
- [x] **Task 4 — gates green**
  - [x] `ruff check src tests`, `mypy src`, `lint-imports` (3/3 — re-check after the new
    `reporters → validators` import). Full `pytest` (286 existing tests stay green:
    `test_reporters.py`, `test_field_metrics.py`, `test_scenarios.py` unchanged). Diff
    discipline: production diff = `reporters/markdown.py` + `validators/metrics.py`;
    nothing else under `src/`.

### Review Findings

- [x] [Review][Patch] Render-Guard asymmetrisch: nur `metrics["test"]` geprüft, dann `metrics["train"]` und `cv[...]["per_fold"]` unbedingt indiziert — Kommentar verspricht "rendered only when both metrics carry data", aber ein Run mit fehlenden train/cv-Einträgen oder ungleich langen `per_fold`-Listen crasht den Report (KeyError/ValueError) statt zum Legacy-Report zu degradieren [src/surroforge/reporters/markdown.py:106-118] (blind+edge) — gepatcht: Guard prüft train/test/cv symmetrisch inkl. non-empty + gleich langer `per_fold`; Test `test_incomplete_cv_records_degrade_to_legacy_report` pinnt die drei exklusiv neuen Pfade (per_fold ungleich/leer/fehlend). Hinweis: fehlende train/test-Skalare crashen bereits die Haupt-Metriktabelle (vorbestehend, nicht durch R3.2 eingeführt).
- [x] [Review][Patch] Zeilenlabel weicht vom AC-1-Wortlaut ab: Template rendert `surrogate error (in-span)`, AC 1 verlangt `surrogate error (in-span ≡ coefficient-space)` [src/surroforge/reporters/markdown.py:56] (auditor) — gepatcht, Tests prefix-stabil.
- [x] [Review][Patch] CV-std-Spalten ungepinnt: kein Test assertet eine std-Zelle — ein Bug nur in der std-Spalte (z. B. 0 oder die falsche Komponente) bestünde die Suite; degree-1-Fixture hat echte per-fold-Streuung zum Pinnen [tests/test_two_source_report.py] (auditor+blind) — gepatcht: alle drei std-Zellen gegen per-fold-recomputed `np.std` gepinnt (rel=1e-3, abs=1e-6).
- [x] [Review][Patch] `reduction is None`-Pfad ungetestet, obwohl seit r3-1 erreichbar: Feld-Metriken auf method=polynomial (kein `lineage["reduction"]`) → Sektion muss ohne "Reduction:"-Zeile rendern; kein Test deckt das ab [src/surroforge/reporters/markdown.py:63] (blind+edge) — gepatcht: `test_unreduced_method_renders_split_without_reduction_line` (Precondition `"reduction" not in run.lineage` assertet).
- [x] [Review][Defer] `two_source_split` overflowt/NaNt bei FIELD_ENERGY_FLOOR-dominierten Relativfehlern (~>1e154, Null-Energie-Referenzfeld): `energy_l2**2` → OverflowError zur Render-Zeit; Ratio-Formulierung `e·sqrt(1−(t/e)²)` würde es beheben [src/surroforge/validators/metrics.py:122] — deferred, degenerierter Input (identisch null persistiertes Feld), Metrikwerte selbst sind dann bereits bedeutungslos

## Dev Notes

### The design decision this story turns on (validated 2026-06-11)

The epic asks for "coefficient-space and reconstructed-field errors in one table" — but
coefficients are never persisted, and the reporter sees only a `Run`. **Resolution: the
table is pure algebra over r3-1's two persisted metrics.**

1. **Pythagoras (r3-1):** truncation (orthogonal residual) ⊥ in-span (surrogate) error ⇒
   `surrogate = sqrt(max(energy_l2² − truncation², 0))` — per split, per fold.
2. **Parseval:** the POD basis `V` is orthonormal (SVD right-singular vectors), so for any
   error vector its in-span component satisfies `‖P_∥(pred − true)‖_F = ‖C_pred − C_true‖_F`
   — the in-span field error IS the coefficient-space error. The algebra is exact (the
   prediction-mean anchor cancels in the coefficient difference; independently re-derived
   during validation); the degree-1 Jensen fixture exercises it at ~0.2 magnitude, not just
   at the noise floor. One honesty caveat: the metric's subspace is recovered FROM the
   predictions, so the identity holds w.r.t. the basis the predictions **actually inhabit**
   — if an inner predicts a constant for a retained mode, that mode's coefficient error is
   reclassified as truncation (consistent with `_truncation_error`'s documented semantics).

So one table + a Parseval footnote delivers #42's two-pass view with zero new persistence
and zero engine/trainer changes. The reporter needs only `run.metrics` (both metric values
incl. `per_fold` lists) and `lineage["reduction"]` (`rank`, `energy_captured` — persisted
since R2.3 for every field run, see engine.py:147-161).

**Pinned fixture** (parameterize r3-1's fixture by the third amplitude; everything else
identical — seed 42, QR-orthonormal modes, 80×40, engine split `test_fraction 0.25, seed 1`,
rank-2 POD + polynomial-degree-2 inner, `cross_validation: 3`):

```python
rng = np.random.default_rng(42)
n_dof, n = 40, 80
modes = np.linalg.qr(rng.standard_normal((n_dof, 3)))[0]   # orthonormal spatial modes
x = rng.uniform(-1.0, 1.0, size=(n, 2))
amps = np.column_stack([5.0 * x[:, 0], 2.0 * x[:, 1] ** 2, a3 * x[:, 0] * x[:, 1]])
y = amps @ modes.T
```

Measured THROUGH THE ENGINE (run_training, pinned seed 42, split seed 1) on 2026-06-11:

| a3 | inner degree | role | r2 (test) | energy_l2 | truncation | surrogate | energy_captured |
|---|---|---|---|---|---|---|---|
| 0.8 | 2 | **deception** (AC 3) | 0.9492 | 0.1020 | 0.1020 | 2.95e-9 | 0.9936 |
| 2.0 | 2 | table correctness | 0.8601 | 0.1587 | 0.1587 | 0.0 | 0.9644 |
| 0.8 | **1** | Jensen/CV-aggregation pin | — | — | — | per fold [0.1859, 0.1984, 0.2129] | — |

Deception CV truncation per fold: `[0.094, 0.081, 0.086]` (differing values — use this row
to pin per-fold aggregation); surrogate per fold: all 0.0. The deception case is the story's
point: **r2 = 0.949 and energy_captured = 0.9936 both look great, yet 10% RMS field error
is pure truncation floor — the surrogate does essentially nothing wrong, the basis is just
too small.** Headroom on the pin: truncation 0.102 vs. threshold 0.05 (2×), r2 0.949 vs.
0.85, surrogate 2.95e-9 vs. 1e-6. Note the measured surrogate 2.95e-9 is NOT < 1e-12 —
exactly the derived-gap noise amplification the r3-1 review flagged; thresholds chosen
accordingly.

### Existing assets and integration points (verified against source, 2026-06-11)

- **Reporter:** `reporters/markdown.py` — jinja `_TEMPLATE` (:14-64), `render()` builds the
  context (:71-109); metrics table loop at :41 iterates `metric_names`; `fmt` = `{:.5g}`
  lambda (:107). The new section slots after "## Validation Metrics" (after :43). Reporters
  resolve via `plugins.get_reporter` — `MarkdownReporter.render(run)` is the only surface.
- **Run data:** `run.metrics["train"/"test"]` are flat name→float dicts;
  `run.metrics["cross_validation"][name]` = `{"mean", "std", "per_fold"}`
  (cross_validation.py:54-62). `run.lineage["reduction"]` =
  `{"basis": "pod", "rank": int, "energy_captured": float}` for every field run since R2.3
  (engine.py:147-161, trainer hook reduced_field.py:202-204); scalar runs lack the key —
  hence the guard in Task 2.
- **Helper placement:** `two_source_split` goes in `validators/metrics.py` next to the two
  metrics it derives from (it is metric algebra, reused by r3-3 and `rom inspect` R4.2).
  `reporters → validators` is a NEW leaf-to-leaf import edge: legal — `.importlinter`'s 3
  contracts constrain cli/api, engine→services, and errors only; project-context allows
  leaves importing "narrowly each other".
- **r3-1 metrics guarantee finite values:** `_check_field_inputs` rejects NaN/inf at compute
  time, so persisted metric values are finite — the reporter needs no nan guards.
- **Scalar net:** `tests/test_reporters.py` (render/path tests, scalar fixtures) and
  `tests/test_scenarios.py` must pass UNCHANGED — they pin AC 4's "report unchanged".

### Constraints that bite

- **[MUST] Production diff = `reporters/markdown.py` + `validators/metrics.py`.** No engine,
  trainer, services, or config change; no new persisted run data. The report derives —
  it never computes from arrays (the reporter has no predictions, only the Run).
- **[MUST] Per-fold first, aggregate second (Jensen).** `sqrt(mean(e²) − mean(t²))` ≠
  `mean(sqrt(e_k² − t_k²))`. The CV columns MUST aggregate per-fold splits — and the ONLY
  fixture that can pin this is the degree-1-inner run (per-fold surrogates
  `[0.1859, 0.1984, 0.2129]`; mean-of-splits 0.19907 vs. Jensen-wrong 0.19925,
  distinguishable at `{:.5g}`). The truncation/total rows CANNOT discriminate (their CV
  means are the persisted `cv[name]["mean"]` either way) and neither can the exact-inner
  fixtures (all-zero per-fold surrogates ⇒ both aggregations give 0) — validate-create-story
  caught exactly this vacuous-pin trap in the story's first draft.
- **[MUST] Margins over bit-equality (BLAS discipline + r3-1 review lesson):** surrogate
  thresholds are `< 1e-6`, never `< 1e-12` — the derived gap `sqrt(e² − t²)` amplifies
  1-ulp noise on ~0.1-magnitude values to ~1e-8 (empirically verified in the r3-1 review).
- **[GOTCHA] The clamp in `two_source_split` is correctness, not cosmetics:** rounding can
  make `t > e`; without `max(…, 0)` the sqrt returns nan and the report renders "nan".
- **[GOTCHA] Section condition is BOTH names in config metrics AND in `metrics["test"]`** —
  a field run configured with only `rmse` (legacy) must render exactly the old report. Do
  not key on `method == "reduced-field"`: the metrics, not the method, carry the data (a
  third-party reducing trainer with the metrics configured gets the table for free).
- **[GOTCHA] `lineage["reduction"]` is absent on scalar runs** and on hypothetical reducing
  trainers without the diagnostics hook — omit the context line, never KeyError.
- **[GOTCHA] Jinja whitespace drift:** a naive `{% if two_source %}\n…\n{% endif %}` block
  injects blank lines into the SCALAR rendering too. Use jinja whitespace control (the
  template-level `{% if %}` must collapse to nothing when false) and pin it: the scalar
  test asserts the report still contains `"-fold.\n\n## Configuration"` — i.e. the text
  between the CV line and the next section is exactly what it was.
- **[SCOPE] Not in this story:** per-mode metrics / `rmse@mode_k` (Phase 2), basis-variance
  across folds and leaky-CV comparison (r3-3), energy spectrum display (`rom inspect`,
  R4.2), any CLI table change, any `AVAILABLE_METRICS` entry.

### Previous-story intelligence (r3-1, 2026-06-11)

- **The derived-gap noise lesson (review High):** asserting `sqrt(e² − t²)`-derived numbers
  below 1e-12 is a coin-flip on the `max(…, 0)` clamp. This story's thresholds (1e-6) and
  the helper's clamp both come from that finding — do not "tighten" them.
- **Vacuous pins (review Medium):** best-of-one compare pinned nothing; this story's
  analogue is the CV-aggregation pin — only the degree-1-inner fixture (per-fold
  surrogates genuinely nonzero AND differing) discriminates the Jensen-wrong
  implementation; every other row/fixture combination passes either way.
- **Verify empirically before pinning:** every number in the fixture table above was
  measured through the real engine path (not a manual split — the engine's
  `test_fraction 0.25, seed 1` split gives DIFFERENT values than a 60/20 slice; the r3-1
  unit tests use the manual split, this story's e2e numbers use the engine split).
- **Defaults are non-discriminating (R1.4):** a3 = 0.8 was tuned so the deception case has
  real headroom on the pinned seed (r2 0.949 > 0.85, floor 0.102 > 0.05); a3 = 1.0 gives
  r2 0.92 (thin vs. a 0.9 threshold), a3 = 0.5 gives floor 0.059 (thin vs. 0.05).
- Process: commit this story at `done` (after review) before r3-3.

### Testing standards

pytest, flat `tests/`, new file `tests/test_two_source_report.py` (replicate the scaffold
locally — pattern in `tests/test_field_metrics.py`'s e2e; do not import other test modules).
Deterministic fixtures (seeded `default_rng`, polynomial inner — no smt/BLAS guards needed).
`pytest.approx` for algebra; substring assertions against the rendered Markdown for
presence/absence; recompute expected values from `run.metrics` + `two_source_split`, no
golden numbers.

### Project Structure Notes

- Files: `src/surroforge/reporters/markdown.py` (modified), `src/surroforge/validators/metrics.py`
  (helper added), `tests/test_two_source_report.py` (new). Nothing else.
- r3-2 is the second R3 story; epic-r3 is already `in-progress`.

### References

- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md:261-274] — Story R3.2 ACs (FR-R5; #25, #42).
- [Source: _bmad-output/planning-artifacts/prd-field-rom-v1.md §4.6:141-144] — two-source error (#25), two-pass table (#42), metric-as-key (#41, shipped in r3-1).
- [Source: src/surroforge/reporters/markdown.py:14-64,71-109] — template + render context (the file to extend).
- [Source: src/surroforge/validators/metrics.py] — `_energy_l2`/`_truncation_error` (r3-1), `_check_field_inputs` (finite guarantee), home of the new helper.
- [Source: src/surroforge/engine.py:147-161 / trainers/reduced_field.py:202-204] — `lineage["reduction"]` provenance (`rank`, `energy_captured`).
- [Source: src/surroforge/validators/cross_validation.py:54-62] — per-fold structure the CV derivation consumes.
- [Source: tests/test_field_metrics.py] — pinned-fixture + engine-e2e scaffold pattern to replicate; r3-1 review lessons baked into its assertions.
- [Source: _bmad-output/implementation-artifacts/r3-1-field-metrics-as-registered-callables.md] — Change Log: review findings this story's constraints derive from.

## Dev Agent Record

### Agent Model Used

claude-fable-5 (Claude Fable 5)

### Debug Log References

- RED confirmed before implementation: `tests/test_two_source_report.py` fails on
  `ImportError: cannot import name 'two_source_split' from 'surroforge.validators.metrics'`.
- GREEN: all 8 new tests pass first try after Tasks 1–2 — including the splice-point pin
  (jinja whitespace exactly as the story prescribed), the Jensen discrimination
  (degree-1 fixture), and the deception exposure.
- One ruff iteration: B905 `zip()` without `strict=` at both new zip sites — fixed with
  `strict=True` (the two per_fold lists must be equally long; failing loudly is correct).

### Completion Notes List

- Production diff exactly as scoped: `reporters/markdown.py` (conditional
  "## Two-Source Error Split" section — template block + render-time derivation) and
  `validators/metrics.py` (`two_source_split` helper with the load-bearing clamp; NOT an
  `AVAILABLE_METRICS` entry). Zero engine/trainer/services/config changes, no new
  persisted data — no STOP condition triggered (AC 5).
- Section renders ONLY when both `energy_l2` and `truncation_error` are in
  `config.validation.metrics` AND in `metrics["test"]`; keyed on the metrics, not the
  method. Scalar and legacy-field reports byte-identical at the splice point — pinned by
  the `"-fold.\n\n## Configuration"` assertion in both absence tests (AC 4).
- Table: total/truncation/surrogate × Train/Test/CV mean/CV std; CV columns aggregate
  per-fold splits (`zip(..., strict=True)` over the per_fold lists), never the
  aggregated means — pinned by the degree-1 fixture where mean-of-splits (0.19907)
  differs from the Jensen-wrong value (0.19925) at the report's 5g rendering (AC 1).
- Parseval line ("coefficient-space error in the basis the predictions actually
  inhabit"), #25 guidance (add modes vs. add data), and the reduction context line
  (`rank`, `energy_captured` from `lineage["reduction"]`, guarded with `.get`) all
  rendered and asserted (AC 2).
- Deception case verified end-to-end (AC 3): r2 0.949 > 0.85 and energy_captured
  0.9936 > 0.98 both look great while the rendered floor is 0.102 > 0.05 with
  surrogate 2.9e-9 < 1e-6.
- Gates: ruff clean (after the B905 fix), mypy clean (41 files), lint-imports 3/3 KEPT
  (the new `reporters → validators` leaf edge violates nothing), full suite green —
  294 tests (286 + 8 new), exit 0, no regressions; r3-1's field-report e2e still passes
  with the new section present (substring assertions, as the validation predicted).

### File List

- src/surroforge/reporters/markdown.py (modified — conditional Two-Source Error Split section: template block + render-time derivation)
- src/surroforge/validators/metrics.py (modified — two_source_split helper added; AVAILABLE_METRICS untouched)
- tests/test_two_source_report.py (new — 10 tests: helper unit/clamp/zero, table correctness incl. train column, deception exposure, Jensen/per-fold CV pin incl. std cells, scalar + legacy-field absence with splice-point pin, incomplete-CV-record degradation, unreduced-method section without Reduction line)
- _bmad-output/implementation-artifacts/deferred-work.md (defer entry: two_source_split overflow on floor-dominated relative errors)
- _bmad-output/implementation-artifacts/r3-2-two-source-error-split-in-the-report.md (story tracking)
- _bmad-output/implementation-artifacts/sprint-status.yaml (status sync)

### Change Log

- 2026-06-11: Adversarial code review (3 layers: Blind Hunter 15 / Edge Case Hunter 11 / Acceptance Auditor: alle 5 ACs PASS). Triage: 4 patch, 1 defer, 13 dismissed. Alle 4 Patches angewendet: (1) Render-Guard symmetrisch (train/test/cv inkl. non-empty + gleich langer per_fold; Degradation zum Legacy-Report statt KeyError/zip-ValueError/nan — neuer Test pinnt die drei exklusiv neuen Pfade; fehlende train/test-Skalare crashen die Haupttabelle vorbestehend); (2) Zeilenlabel → `surrogate error (in-span ≡ coefficient-space)` per AC-1-Wortlaut; (3) alle drei CV-std-Zellen gegen per-fold-recomputed np.std gepinnt; (4) reduction-loser Pfad (method=polynomial, kein lineage["reduction"]) e2e getestet. Defer: two_source_split-Overflow bei FIELD_ENERGY_FLOOR-dominierten Relativfehlern (Null-Energie-Referenzfeld) → deferred-work.md. Dismissed u. a.: recomputed-vs-persistierte CV-Werte (verifiziert: cross_validation.py nutzt arr.std() = ddof 0, identische Algebra), Prosa-Kopplung der Tests (Prosa ist AC-2-Deliverable), Schwellen-Fragilität (2× Headroom by design). Gates nach Patches: 296 Tests (294 + 2 neue) exit 0, ruff/mypy/lint-imports (3/3) grün. Status → done.
- 2026-06-11: Implemented (dev-story, red-green-refactor). RED: new test file fails on the missing helper import. GREEN: `two_source_split` in validators/metrics.py (Pythagoras + clamp + Parseval/Jensen docstring); conditional report section in reporters/markdown.py (per-fold-first CV aggregation, `lineage["reduction"]` context line, scalar output byte-identical at the splice point). 8 new tests; one ruff B905 fix (`zip(strict=True)`). Gates: ruff/mypy/lint-imports (3/3) clean, 294 tests pass, production diff = the two scoped files only. Status → review.

- 2026-06-11: validate-create-story (fresh-context adversarial validation): verdict APPROVED WITH FIXES, all applied. **Critical (C1):** the Jensen/CV-aggregation pin in the first draft was vacuous — the truncation row's CV mean is the persisted `cv[name]["mean"]` regardless of surrogate aggregation, and the exact-inner fixtures have all-zero per-fold surrogates (both aggregations give 0); a Jensen-wrong implementation would have passed every test. Fix (engine-verified by the validator): third fixture variant a3=0.8 + **degree-1 inner** — per-fold surrogates [0.1859, 0.1984, 0.2129], mean-of-splits 0.19907 vs. Jensen-wrong 0.19925, distinguishable at `{:.5g}`. Minor: Parseval "verified empirically" softened (the 7e-14-vs-7e-14 check verified 0≈0; the algebra was independently re-derived and is exact, the degree-1 fixture exercises it at ~0.2) + footnote now says "in the basis the predictions actually inhabit" (constant-inner degenerate case); Train-column assertion added (train e = t = 0.07681 on the pin); the deliberate epic-AC narrowing (table only when both metrics configured) made explicit in the scope note. Validator confirmed: fixture table reproduces exactly through the engine, all line refs correct, scalar splice-point pin (`"-fold.\n\n## Configuration"`) verified against real output, jinja whitespace guidance correct, lineage["reduction"] present/absent as claimed, per_fold survives JSON, blast radius safe (only r3-1's e2e renders a field report — substring-only assertions), single reporter, import edge legal, 286 tests exact.
- 2026-06-11: Story created (create-story). **Resolved the "coefficient-space" question:**
  no new persistence needed — Pythagoras (r3-1) derives the surrogate error from the two
  persisted metrics, and Parseval (orthonormal POD basis) makes the in-span surrogate error
  IDENTICAL to the coefficient-space error (verified empirically: 7.193e-14 vs. 6.639e-14),
  so one table + footnote satisfies #42's two-pass view. Both fixtures measured through the
  REAL engine path: deception pin (a3=0.8) r2=0.9492 / energy_captured=0.9936 / floor=0.1020 /
  surrogate=2.95e-9 — the "great R² from a tiny basis doing nothing" case with 2× threshold
  headroom; table-correctness pin (a3=2.0) floor=0.1587. CV truncation per fold differs
  (0.094/0.081/0.086) — chosen as the discriminating row for the Jensen/per-fold-aggregation
  pin (r3-1 lesson: vacuous pins). Surrogate thresholds 1e-6, clamp mandatory (r3-1 review:
  derived-gap noise ~1e-8). Status → ready-for-dev.
