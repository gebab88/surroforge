# Story R1.4: Built-in methods work as inner surrogates and pass conformance unmodified

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a plugin author / maintainer,
I want the three Core methods (`polynomial`, `kriging`, `rbf`) to serve as inner surrogates with no special-casing,
so that ROM is genuinely "one trainer on top of the existing stack" (spine theorem #36) — proven, not assumed.

> **Scope realization (read first).** This is a **proof story, not a feature story**: the per-mode
> design already places every inner fit squarely inside the SMT MVP envelope (each mode is a
> single-output `(n, 1)` target — `reduced_field.py:126` fits `coefficients[:, k:k+1]`; Kriging/RBF
> raise `TrainerError` only on `y.shape[1] != 1`, which the reduced trainer can never trigger).
> The expected diff is **tests + one docstring correction**. If implementing any AC seems to require
> touching `polynomial.py`/`kriging.py`/`rbf.py`, the engine, or the conformance suite — STOP: that
> contradicts the story's own claim ("unmodified") and means a design assumption broke.

## Acceptance Criteria

1. **Given** a `reduced-field` config whose `model.inner.method` is `polynomial`, `kriging`, or `rbf` **when** the run trains through the **unchanged** engine (config path: `model.inner` block per R1.2) **then** it trains end-to-end on a low-rank field fixture (test R² above a fixed threshold; one inner instance per retained mode) — with **zero modifications** to the three inner trainers (the story's diff touches no file under `src/surroforge/` except the Task-3 docstring line in `reduced_field.py`).
2. **Given** `check_trainer(ReducedFieldTrainer, hyperparameters={"reduction": …, "inner": {"method": <m>, …}})` for each of the three built-ins **then** the report passes — with the **documented Accelerate carve-out** for a kriging inner: on a non-bit-reproducible BLAS only the `deterministic` check may fail, nothing else (mirror `tests/test_conformance.py:54-64`).
3. **Given** the existing conformance suite and BUILTINS tests **then** they pass unchanged (the `BaseTrainer` contract is untouched; `tests/test_conformance.py` needs no edits to its suite/expected-checks — only new parametrized cases may be ADDED in `tests/test_reduced_field.py`).
4. **Given** a kriging-inner field run **when** it is reproduced under a bit-reproducible BLAS with pinned threads **then** `reproduce()` returns a reproducing verdict and `assert_reproducible` does not raise (test guarded by `requires_reproducible_blas` **and** the smt skip marker). The polynomial-inner reproduce path is already pinned by R2.3/R1.2 suites — do not duplicate it.

## Tasks / Subtasks

- [x] **Task 1 — e2e: each built-in as inner method through the unchanged engine** (AC: 1)
  - [x] In `tests/test_reduced_field.py` add a parametrized e2e test (pattern: existing `test_end_to_end_through_unchanged_engine`, reuse `low_rank_field` + the CSV/`Project.init` scaffold): `model.inner = {"method": m}` for `m ∈ {polynomial, kriging, rbf}`; polynomial passes `degree: 2`, kriging/rbf use defaults. Assert `run.metrics["n_outputs"]` correct, `run.metrics["test"]["r2"] > 0.9`, and `run.load_model().predict(...)` shape `(n, n_dof)`.
  - [x] **The discriminating assertion (without it the test proves nothing):** metrics/shape are IDENTICAL for the default-polynomial fallback (verified empirically — r²=1.0 either way), so a silently ignored `inner.method` would still pass everything above. After `model = run.load_model()`, also assert the inner method was honored: `assert len(model._models) == model.rank` and `assert all(isinstance(im, get_trainer(m)) for im in model._models)` (`get_trainer` is already imported at test_reduced_field.py:18; `isinstance` against the registry class pins the binding, not a name string).
  - [x] Mark kriging/rbf params with the smt skip marker (copy the `_needs_smt` pattern from `tests/test_conformance.py:25-28` — `importlib.util.find_spec("smt") is None`). No BLAS guard here: e2e training asserts no bit-identity.
  - [x] **Keep it fast:** SMT fits dominate runtime. Use a small fixture (`low_rank_field(n_samples=60, n_dof=6, k_true=2, …)`, `energy: 0.99` ⇒ r≈2, `cross_validation: 3`) so a kriging run is ~10 inner fits, not 25+.
- [x] **Task 2 — conformance: `check_trainer` with each built-in as inner** (AC: 2, 3)
  - [x] Extend `test_passes_conformance_suite` in `tests/test_reduced_field.py` (or add a sibling parametrized test) with `check_trainer(ReducedFieldTrainer, hyperparameters={"reduction": {"energy": 0.99}, "inner": {"method": "kriging"}})` and `{"method": "rbf"}` (smt-marked). The **polynomial inner is already covered** at test_reduced_field.py:120-123 — ADD only kriging/rbf, do not duplicate it. For the **kriging inner** apply the Accelerate carve-out verbatim from `tests/test_conformance.py:54-64`: if `not blas_is_bit_reproducible()`, assert `{r.name for r in report.failures} <= {"deterministic"}`; else `report.passed`. (Carve-out scope verified empirically for the WRAPPED case, 5/5 runs on Accelerate: only `deterministic` fails — fit-to-fit divergence 4e-6..4e-5; `picklable` passes because it re-predicts ONE fitted model, which is bit-stable. rbf-inner passes all 10 checks.)
  - [x] Do NOT modify `src/surroforge/conformance/trainer.py` or `tests/test_conformance.py`'s `BUILTINS`/`EXPECTED_CHECKS` — AC3 pins them as the regression net. The conformance suite drives the trainer **directly** (bypassing config), so this task exercises the trainer-kwargs path; Task 1 exercises the config path. Both are required (R1.2 made config strict, but the API/conformance path has no config in front of it).
- [x] **Task 3 — docstring honesty: the reduced-field tolerance comment** (AC: 1)
  - [x] `src/surroforge/trainers/reduced_field.py:45-47` claims "a tolerant inner (SMT) inherits its own looseness at fit time" — **no such mechanism exists** (`reproduction_tolerance = 1e-9` is a static class attribute; nothing sets an instance value at fit). It is coincidentally harmless because all three built-ins declare ≤ 1e-9 (polynomial 0.0, kriging 1e-9, rbf 1e-9 — equal to reduced-field's own band). Reword the comment to state reality, e.g.: "all built-in inners declare ≤ 1e-9, matching this band; an inner with a looser declared tolerance would need this widened (revisit when one exists)."
  - [x] **Do NOT implement dynamic tolerance widening** (e.g. `self.reproduction_tolerance = max(...)` at fit). It is a no-op for every built-in today (max(1e-9, 1e-9) = 1e-9), and `reproduction_tolerance` is folded into the field `spec_hash` (engine.py `_spec_hash`) — touching how it is derived risks churning field-run identity for zero behavioral gain. YAGNI; note it as deliberately skipped in the Dev Agent Record.
- [x] **Task 4 — guarded reproducibility: kriging-inner field run reproduces** (AC: 4)
  - [x] Add a test (in `tests/test_reduced_field.py` or `tests/test_reduction_config.py`) that trains a kriging-inner field run and asserts `reproduce(...)` verdict ∈ {reproduced, reproduced_within_tolerance} + `assert_reproducible(result)`. **Use the same small fixture sizing as Task 1** (60 samples, k_true=2, energy 0.99 ⇒ rank 2) — reproduce re-trains, doubling the fit count; the `test_reduction_config._field_config` scaffold's 80/k_true=3 sizing would mean ~30 kriging fits. Guard with BOTH `requires_reproducible_blas` (from `tests/conftest.py:34`) and the smt marker — gradient-MLE Kriging is legitimately non-bit-reproducible on Apple Accelerate, and an unguarded version becomes the flaky test the project's reproducibility discipline forbids. **Never widen a tolerance or unpin threads to make it pass.**
- [x] **Task 5 — gates green**
  - [x] `ruff check src tests`, `mypy src`, `lint-imports` (3 contracts). Full `pytest` (locally kriging/rbf tests may SKIP without smt or on Accelerate — that is correct behavior, not failure; CI's smt extra + pinned-threads job exercises them for real). Run the reproduce family + new tests under `OMP/OPENBLAS/MKL_NUM_THREADS=1`.
  - [x] Verify the story's diff discipline: `git diff --stat` shows only test files + the one `reduced_field.py` docstring line. This IS the deliverable of AC1's "unmodified" claim.

### Review Findings

- [x] [Review][Patch] e2e under-discriminates: r² floor 0.9 lies below the fixture's single-mode energy share (0.945) and rank is only pinned `>= 1` — a rank-collapse regression or a garbage mode-2 fit would pass; pin `model.rank == 2` and raise the floor to 0.99 (mode-1 energy 0.943–0.945 across all subset sizes, far from the 0.99 threshold — empirically stable) [tests/test_reduced_field.py:228,234]
- [x] [Review][Patch] Polynomial param row is non-discriminating and inner-hyper pass-through is untested: `degree: 2` equals both `DEFAULT_INNER_METHOD` and the polynomial default, so a silently dropped `inner` block would pass; use `degree: 3` and assert `model._models[0].hyperparameters["degree"] == 3` (verified: inner exposes hyperparameters; degree-3 reconstruction error ~9e-16) [tests/test_reduced_field.py:188,225-236]
- [x] [Review][Patch] Tolerance comment reads as an unconditional guarantee while the same diff's tests document Accelerate divergence; add the platform caveat (band holds under bit-reproducible BLAS; Accelerate is guarded in tests/CI) [src/surroforge/trainers/reduced_field.py:45-49]
- [x] [Review][Patch] Dev Agent Record mislabels one pre-existing skip: "2 pre-existing smt/onnx skips" — one of them is the BLAS-guarded kriging reproduce in tests/test_reproduce.py, not an smt skip [story file, Debug Log]
- [x] [Review][Defer] `blas_is_bit_reproducible` returns True for any unknown/non-Accelerate BLAS (e.g. MKL without CNR), so the new kriging reproduce test could fail rather than skip on such a runner [tests/conftest.py:27-29] — deferred, pre-existing policy inherited from test_conformance.py

## Dev Notes

### Why this story exists (and why it is small)
Epic R1's promise — DoD of "ROM is one trainer on top of the existing stack" (AR-R1, #36) — is currently proven only for the polynomial inner (`test_passes_conformance_suite`, `test_end_to_end_through_unchanged_engine`). R1.4 closes the loop for kriging/rbf, which are the methods whose **single-output MVP envelope** the per-mode design was explicitly built around (#11): `KrigingTrainer.fit`/`RBFTrainer.fit` raise `TrainerError` on `y.shape[1] != 1` (kriging.py:57-60, rbf.py:53-56), and `ReducedFieldTrainer.fit` always feeds `coefficients[:, k:k+1]` — shape `(n, 1)` (reduced_field.py:124-127). If these tests pass without source changes, the spine theorem holds; if they cannot, the design is broken and the story must HALT for the lead, not patch around it.

### Existing assets to reuse (do not reinvent)
- **Fixture:** `low_rank_field(...)` in `tests/test_reduced_field.py:22` + the CSV/`Project.init`/`run_training` scaffold of `test_end_to_end_through_unchanged_engine` (:128-162). `tests/test_reduction_config.py:_field_project/_field_config` is the same scaffold parameterized — either is fine; do not write a third variant from scratch.
- **smt skip marker:** `_needs_smt` in `tests/test_conformance.py:25-28`. **Replicate locally (4 lines)** — importing from `tests.test_conformance` works mechanically, but replication keeps cross-test-module coupling to the single established `low_rank_field` precedent instead of importing a whole test module for one marker.
- **BLAS guard:** `requires_reproducible_blas` + `blas_is_bit_reproducible` in `tests/conftest.py:12-36`. The Accelerate carve-out idiom for kriging conformance is at `tests/test_conformance.py:54-64` — copy its logic AND its comment rationale.
- **Conformance entry point:** `check_trainer(trainer_cls, hyperparameters=...)` (`src/surroforge/conformance/trainer.py:90`). Note `_make_data` (:70-76) generates a smooth single-output target; with a reduced-field trainer the "field" is that 1-column output — POD of a 1-DOF field is degenerate-but-valid (r=1, constant-field guard at `reduced_field.py:137-139` covers zero-variance). This already works for the polynomial inner; expect identical behavior for kriging/rbf.

### Hyperparameter envelopes (exact, for test configs)
- `polynomial`: `degree`, `include_bias`, `interaction_only` (polynomial.py) — use `{"method": "polynomial", "degree": 2}`.
- `kriging`: `corr`, `poly`, `theta0`, `nugget` (kriging.py:17) — defaults suffice; `{"method": "kriging"}`.
- `rbf`: `d0`, `poly_degree`, `reg` (rbf.py:15) — defaults suffice; `{"method": "rbf"}`.
- The trainer-kwargs form is `inner={"method": m, **hyper}` (trainer resolves `method` via `get_trainer` lazily at fit, `reduced_field.py:102`); the config form is the `model.inner` block (R1.2's `InnerConfig` requires `method` explicitly at config level).

### Constraints that bite (read before writing tests)
- **[MUST] No source changes to the three inner trainers, engine, conformance suite.** That is the AC, not a preference.
- **[MUST] Reproducibility discipline:** never trade thread-count against tolerance; the kriging reproduce test is double-guarded (BLAS + smt) and runs pinned in CI's reproducibility job. A same-seed round-trip test alone is weak — but the negative side (perturbed run fails) is already owned by R2.3's suite; R1.4 does not need a new negative test (no numerics change).
- **[GOTCHA] Local venv (py3.14): smt builds locally but Accelerate BLAS is non-reproducible** — locally the kriging conformance determinism check fails legitimately (carve-out) and the Task-4 test SKIPS. Green-with-skips locally is the expected outcome; real coverage is CI (OpenBLAS, pinned threads, smt extra).
- **[GOTCHA] SMT fit cost:** each mode × each CV fold × holdout × final is a separate SMT fit. The Task-1 fixture sizing (r≈2, 3 folds) keeps one kriging e2e at ~10 fits. Measured on the dev machine: kriging-inner e2e ≈ 12 s, kriging-inner `check_trainer` ≈ 5 s (3 kriging fits), rbf negligible (≈ 1 s). Do not copy R1.2's 80-sample/k_true=3/0.999-energy config verbatim for SMT methods.
- **[GOTCHA] `reproduction_tolerance` is in the field `spec_hash`.** Leave the attribute and its derivation alone (Task 3 changes only the comment).
- Engine queries `resolved_config_overrides()` on the **holdout model** (R1.2 review fix, engine.py) — irrelevant to this story unless a test uses `rank: adaptive`; prefer `energy` in fixtures to stay off that path.

### Previous-story intelligence (R1.2, done 2026-06-10)
- R1.2's `InnerConfig` made the config path strict (`method` required, null extras rejected) — but `check_trainer` bypasses config entirely, which is exactly why AC2 (trainer-kwargs path) and AC1 (config path) are separate ACs. Keep both.
- R1.2 review patterns that apply here: empirically verify a fixture really exercises the intended branch before pinning it (R1.2's straddling-spectrum test); prefer recomputation/threshold assertions over golden values; BLAS-sensitive assertions get markers, not loose tolerances.
- Process: commit this story at `done` (after review) before the Epic R1 retrospective / Epic R3.

### Testing standards
pytest, flat `tests/`; parametrize with `pytest.param(..., marks=...)` (BUILTINS style). Thresholds: R² > 0.9 on the low-rank fixture is robust for all three methods (polynomial is exact on the polynomial amplitudes; kriging/rbf interpolate 60 smooth samples easily). Run the new tests + reproduce family under pinned threads for the Task-5 gate.

### Project Structure Notes
- New/changed files: `tests/test_reduced_field.py` (extend), optionally `tests/test_reduction_config.py` (Task 4 alt-location), `src/surroforge/trainers/reduced_field.py` (ONE docstring line). Nothing else.
- Completing r1-4 completes Epic R1 → set `epic-r1: done` in sprint-status only after this story reaches `done`; the Epic R1 retrospective is then `optional`.

### References
- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md#Story R1.4] — ACs (FR-R3, AR-R1; supports DoD).
- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md#Additional Requirements] — AR-R1 (basis-inside-trainer, zero downstream touchpoints).
- [Source: src/surroforge/trainers/kriging.py:17,57-60 / rbf.py:15,53-56] — supported hyperparameters + single-output MVP guard.
- [Source: src/surroforge/trainers/reduced_field.py:124-127] — per-mode `(n, 1)` inner fit; :45-47 — the stale tolerance comment (Task 3); :102 — lazy `get_trainer`.
- [Source: src/surroforge/conformance/trainer.py:70-76,90-98,187-214] — `_make_data` single-output target, `check_trainer` signature, determinism check honoring declared tolerance.
- [Source: tests/test_conformance.py:25-28,30-36,54-64] — `_needs_smt`, BUILTINS, the Accelerate determinism carve-out idiom.
- [Source: tests/conftest.py:12-38] — `blas_is_bit_reproducible` / `requires_reproducible_blas`.
- [Source: tests/test_reduced_field.py:22,118-123,129-160] — fixture, existing conformance + e2e patterns to extend.

## Dev Agent Record

### Agent Model Used

claude-fable-5 (Claude Code)

### Debug Log References

- Teeth check before relying on the discriminating assertion: trained a kriging-inner
  reducer directly and asserted its per-mode models are `KrigingTrainer` and NOT
  `PolynomialTrainer` — confirms the `isinstance(im, get_trainer(m))` assertion would
  fail under a silent default-polynomial fallback.
- `tests/test_reduced_field.py`: 18 passed, 1 skipped (23 s) — the single skip is the
  BLAS-guarded kriging reproduce test on Accelerate, exactly as designed.
- Full suite: 269 passed, 3 skipped (1 pre-existing onnx skip, 1 pre-existing BLAS-guarded
  kriging reproduce skip in test_reproduce.py, + the new guarded test).
- Pinned-threads family (`OMP/OPENBLAS/MKL_NUM_THREADS=1`): reproduce/canonicalize/field/
  scalar-identity/probe-coverage + reduced-field + reduction-config → 75 passed, 2 skipped.
- Gates: ruff clean, mypy clean (41 files), lint-imports 3 contracts kept.
- Diff discipline (`git diff --stat`): ONLY `tests/test_reduced_field.py` (+110) and the
  `reduced_field.py` tolerance comment block — no inner trainer, no engine, no conformance
  suite touched. AC1's "unmodified" claim is proven by the diff itself.

### Completion Notes List

- **Task 1:** parametrized `test_each_builtin_trains_as_inner_through_unchanged_engine`
  (polynomial/kriging/rbf; SMT params `_needs_smt`-marked, marker replicated locally per
  story guidance). Shared `_inner_field_run` scaffold with the prescribed small fixture
  (60 samples, k_true=2, energy 0.99 ⇒ rank 2, 3 folds). Includes the discriminating
  per-mode assertion `len(model._models) == model.rank` + `isinstance(im, get_trainer(m))`.
- **Task 2:** `test_conformance_with_smt_inner` (kriging/rbf) via `check_trainer` with
  trainer kwargs (config bypassed — complements Task 1's config path). Kriging inner
  carries the Accelerate carve-out verbatim (`failures <= {"deterministic"}`), with the
  empirically-verified rationale in the comment. Conformance suite + test_conformance.py
  untouched (AC3).
- **Task 3:** reworded the false "inherits its own looseness at fit time" comment to state
  reality (all built-ins ≤ 1e-9; static until a looser inner exists; spec_hash coupling
  documented in place). **Deliberately skipped** dynamic tolerance widening — a no-op for
  every built-in today and a field-run-identity risk via `_spec_hash` (story Task 3 YAGNI).
- **Task 4:** `test_kriging_inner_field_run_reproduces`, double-guarded (`_needs_smt` +
  `requires_reproducible_blas`), small fixture. Skips on the local Accelerate machine
  (where an unguarded run measurably diverges); CI's pinned-threads OpenBLAS job runs it.
- **Task 5:** all gates green; diff discipline verified (see Debug Log).
- Epic R1 is complete with this story — sprint-status `epic-r1` set to `done` when this
  story reaches `done` (after review).

### File List

- tests/test_reduced_field.py (modified — all new tests)
- src/surroforge/trainers/reduced_field.py (modified — tolerance comment block only)
- _bmad-output/implementation-artifacts/sprint-status.yaml (modified)
- _bmad-output/implementation-artifacts/r1-4-built-in-methods-work-as-inner-surrogates-and-pass-conformance-unmodified.md (modified)

### Change Log

- 2026-06-10: Adversarial code review (Blind Hunter / Edge Case Hunter / Acceptance Auditor) → PASS with 4 patches, 1 defer, 10 dismissed. Auditor: all 4 ACs SATISFIED, every Dev Agent Record claim reproduced. Patches applied: (1) e2e pins `model.rank == 2` + r² floor 0.99 (mode-1 energy 0.945 made the 0.9 floor non-discriminating against rank collapse); (2) polynomial inner uses non-default `degree: 3` + hyper pass-through assertion (`degree: 2` equaled both fallback defaults); (3) tolerance comment gained the bit-reproducible-BLAS caveat; (4) Debug-Log skip labeling corrected. Deferred: `blas_is_bit_reproducible` unknown-BLAS optimism (pre-existing, deferred-work.md). Re-verified: test file 18 passed / 1 skipped, ruff/mypy/lint-imports green. Status → done; Epic R1 complete.
- 2026-06-10: Implemented (dev-story). All 5 tasks complete; 6 new tests (3 e2e + 2 conformance + 1 guarded reproduce); full suite 269 passed / 3 skipped; gates green; diff touches only the test file + the tolerance comment — the "unmodified" proof holds. Status → review.
- 2026-06-10: Validated (validate-create-story, independent fresh-context) → ACCEPT-WITH-FIXES. Every load-bearing claim survived empirical verification on the dev machine: the AC2 Accelerate carve-out holds for the WRAPPED kriging case (5/5 runs: only `deterministic` fails, 4e-6..4e-5 fit-to-fit; `picklable` is bit-stable), rbf-inner passes all 10 checks, the Task-1 fixture trains kriging-inner in ~12 s at r²=1.0, an unguarded kriging-inner reproduce on Accelerate measurably returns `diverged` (the double guard is mandatory, not defensive), and the `_make_data` 1-DOF/rank-1 path works. **Critical fix applied:** Task 1's original assertions could not discriminate the inner method — the default-polynomial fallback produces identical metrics/shape (verified), so a silently ignored `inner.method` would pass; added the `isinstance(im, get_trainer(m))` per-mode assertion as the test's actual teeth. Also applied: Task-4 fixture-sizing instruction (reproduce doubles fit count), measured timings in the SMT-cost gotcha, "polynomial already covered, ADD only kriging/rbf" in Task 2, empirical carve-out basis recorded, marker guidance sharpened to replicate-not-import, citation touch-ups (:128-162, :30-36, :12-38).
- 2026-06-10: Story created (create-story). Scoped as a **proof story**: tests + one docstring correction, zero changes to inner trainers/engine/conformance suite (that absence IS the deliverable — AR-R1's "zero downstream touchpoints"). Key context baked in: SMT single-output envelope vs per-mode `(n,1)` fits, smt/BLAS double-guarding (Accelerate carve-out idiom), SMT fit-cost-aware fixture sizing, and the deliberate YAGNI on dynamic `reproduction_tolerance` widening (no-op for all built-ins; spec_hash stability). Status → ready-for-dev.
