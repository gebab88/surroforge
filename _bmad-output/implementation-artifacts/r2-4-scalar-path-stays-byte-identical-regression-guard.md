# Story R2.4: Scalar path stays byte-identical (regression guard)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a maintainer,
I want the generalized field-primary fingerprint to leave **scalar runs bit-for-bit unchanged**, pinned by a dedicated CI regression test,
so that adding (or later evolving) ROM can **never** silently regress the existing reproducibility guarantee — "reduced-field shipped" must not weaken "scalar reproduce" by a single byte.

## Acceptance Criteria

1. **Given** the pre-ROM scalar fingerprint baseline **when** the field-primary fingerprint code path runs on a **scalar run** (no reduction / identity basis ⇒ the field *is* the prediction) **then** the produced fingerprint and verdict are **byte-identical** to the pre-ROM behaviour: the scalar `lineage.json` carries **none** of the field keys (`basis_hash`, `holdout_coefficients`, `reduction`, `spec_hash`), the persisted `holdout_fingerprint` bytes + `prediction_hash` equal the raw out-of-sample holdout predictions (the pre-ROM algorithm), and the `ReproduceResult` is scalar-shaped (`field_relative_diff is None`, `basis_hash_match is None`, `basis_regauged is False`, `coefficient_diagnostic is None`, verdict driven by the absolute `max_abs_diff`).
2. **Given** this equivalence **when** CI runs **then** a dedicated regression test **pins** it under pinned BLAS/OpenMP threads (the existing `reproducibility (pinned threads)` job), in the spirit of the existing profiling/boundary gates — a future change to the field code path that perturbs a scalar run is caught by a failing gate, not discovered in production.

> **Scope boundary (do not exceed).**
> - **R2.4 is expected to be a *test + CI* story with ZERO production changes.** R2.3 already made the scalar path byte-identical *by construction* (the field branch in `engine.run_training` is gated on a duck-typed hook that scalar trainers do not expose; `services.reproduce` discriminates on `parent.basis_hash` and leaves the absolute-diff path untouched), and `tests/test_reproduce.py` passes verbatim today. R2.4 *pins* that invariant; it must **not** refactor `services.py`/`engine.py`. If — and only if — a genuine byte-divergence is discovered, fix it minimally and document it (that is the story's whole point), but do not "improve" the scalar path otherwise.
> - **Do NOT hardcode a golden literal hash.** A pinned `prediction_hash` string is cross-platform-fragile (LAPACK/BLAS bytes differ across OpenBLAS/Accelerate/MKL/Windows), which is exactly the flaky-gate anti-pattern the project forbids ("never trade thread-count against tolerance"). Pin byte-identity by **same-process recomputation/equivalence** (re-derive the holdout predictions with the pre-ROM algorithm and assert the persisted bytes match), not by a frozen constant.
> - **Do NOT** widen scope into R2.5 (`probe_coverage_energy`, global energy-norm), the physics/bitwise two-verdict (#73), or per-mode bands (#72) — all deferred.
> - **Do NOT** change the scalar default tolerance (`1e-9`) — "Ask First" in the frozen contract.
> - **Do NOT** make `engine.py` import `reduced_field` / any ROM module (spine theorem #36; import-linter).

## Tasks / Subtasks

- [x] **Task 1 — Confirm the scalar path is byte-identical by construction (no production change)** (AC: 1)
  - [x] Re-read the two field-aware sites and confirm the scalar branch is inert: `engine.run_training` only adds `field_extras`/`spec_hash` *inside* `if hasattr(holdout_model, "reproducibility_diagnostics")` (engine.py:148) — a scalar trainer has no such hook, so the lineage dict keeps exactly the pre-ROM keys; `services.reproduce` sets `is_field = bool(parent.basis_hash)` (services.py:95) and, when False, takes `diff_for_verdict = max_abs_diff` with all field fields left at their scalar-identical defaults.
  - [x] Confirm `_verdict` (services.py:137) is unchanged and shared by both paths. Confirm the only `assert_reproducible` edge that changed vs. pre-ROM is the **NaN-safe** `not (diff <= tol)` (services.py:247): for a *finite* scalar diff it is identical to the old `diff > tol`; on a (pathological, practically-unreachable) NaN scalar diff it now raises, which *matches* the `diverged` verdict `_verdict` already returns for NaN — a verdict-consistency fix, not a byte regression of real scalar runs. **Document this one intentional edge** in Completion Notes; it is not a violation of the byte-identity contract (which is about real, finite scalar runs).
  - [x] If (and only if) any genuine scalar byte-divergence surfaces, fix it minimally in place and record it; otherwise make **no** production edit.

- [x] **Task 2 — Regression test: scalar lineage carries no field keys** (AC: 1)
  - [x] New `tests/test_reproduce_scalar_identity.py` (flat tests/, one file per concern). Train a deterministic **polynomial** scalar run via the existing `project` fixture + `poly_config("polynomial")`.
  - [x] Assert on the **raw** `run.lineage` dict (not the defaulting accessors, which mask absence by returning `""`/`[]`/`{}`): `"basis_hash" not in run.lineage`, and likewise `"holdout_coefficients"`, `"reduction"`, `"spec_hash"` are absent. This pins "the field path is inert for scalar runs" — a future change that *unconditionally* added a field key (e.g. always writing `spec_hash`) fails here.
  - [x] Assert the pre-ROM scalar key set is intact: `prediction_hash`, `holdout_fingerprint`, `reproduction_tolerance`, `config_hash` are present.

- [x] **Task 3 — Regression test: scalar fingerprint bytes == pre-ROM algorithm** (AC: 1)
  - [x] Re-derive the out-of-sample holdout exactly as the pre-ROM engine does: `select_columns` → `train_test_split(x, y, test_size=split.test_fraction, random_state=split.seed, shuffle=True)` → fit a **fresh holdout model on `x_train`** → `np.ascontiguousarray(model.predict(x_test), dtype=np.float64)[:512]`. (Mirror the `test_fingerprint_is_out_of_sample` setup in `tests/test_reproduce.py`, but compare against the HOLDOUT model, not the final model.)
  - [x] Assert `recomputed.tobytes() == np.ascontiguousarray(run.holdout_fingerprint, dtype=np.float64).tobytes()` **and** `sha256(recomputed.tobytes()).hexdigest() == run.lineage["prediction_hash"]`. This is the pre-ROM fingerprint algorithm recomputed in-process; equality pins byte-identity without a fragile golden constant. Polynomial is closed-form (deterministic, bit-stable run-to-run even on Apple Accelerate — see `test_reproduce_is_bit_identical`, which carries **no** BLAS guard), so this needs **no** `requires_reproducible_blas` marker.
  - [x] **Shape note:** `holdout_fingerprint` is the **2-D** `(n_test, 1)` array `holdout_test_pred` (polynomial `predict` returns `(n, 1)` via `_as_2d`; the engine stores it `.tolist()` as a list-of-lists), so the JSON round-trip preserves the 2-D shape. Keep both sides 2-D and C-contiguous — do **not** flatten `run.holdout_fingerprint` to a vector, or the `.tobytes()` comparison breaks even though the values match.

- [x] **Task 4 — Regression test: scalar `ReproduceResult` is scalar-shaped** (AC: 1)
  - [x] `result = reproduce(project, run.run_id)` on the scalar run; assert `result.field_relative_diff is None`, `result.basis_hash_match is None`, `result.basis_regauged is False`, `result.coefficient_diagnostic is None`, and `result.verdict == "reproduced"` with `result.max_abs_diff == 0.0`. (For the deterministic polynomial the verdict is driven by `bit_identical` — the `prediction_hash` matches, so `_verdict` returns `"reproduced"` at services.py:149 *before* the `max_abs_diff <= tolerance` branch; `max_abs_diff` is `0.0` because the recomputed holdout bytes are identical. Do **not** infer the verdict came from the tolerance comparison.) Assert `assert_reproducible(result)` does not raise. This pins that `reproduce()` never routes a scalar run through the field metric.
  - [x] (Teeth, scalar-flavoured) Construct a scalar `ReproduceResult` with `field_relative_diff=None`, `max_abs_diff=1.0`, `verdict="diverged"` and assert `assert_reproducible` raises `ReproducibilityError` **via the `max_abs_diff` branch** (not the field branch) — proving the scalar raise path is unchanged. (This complements, does not duplicate, `test_reproduce.py::test_assert_reproducible_raises_on_numeric_divergence`.)

- [x] **Task 5 — Pin the pre-ROM scalar suite as the inherited baseline** (AC: 1, 2)
  - [x] Do **not** copy `tests/test_reproduce.py` assertions — it *is* the byte-identity baseline and already passes verbatim (R2.3 AC5). Instead, add a short module docstring + a top-of-file comment in the new test file naming `tests/test_reproduce.py` as the inherited pre-ROM contract this file guards, so the relationship is explicit for a future reader. The CI job (Task 6) runs both files together under pinned threads.

- [x] **Task 6 — CI: run the regression test under pinned threads** (AC: 2)
  - [x] Extend the `reproducibility (pinned threads)` job target in `.github/workflows/ci.yml` (`.github/workflows/ci.yml:72`) to also run `tests/test_reproduce_scalar_identity.py` — it currently runs `tests/test_reproduce.py tests/test_canonicalize.py tests/test_reproduce_field.py`. Threads are pinned at the job level (`OMP/OPENBLAS/MKL_NUM_THREADS=1`), so no per-test fixture is needed (consistent with R2.3; do not add a `@pytest.mark.slow` fixture).

- [x] **Task 7 — Gates green**
  - [x] `ruff check src tests`, `mypy src` (test-only story ⇒ src unchanged; still run it), `lint-imports` (3 contracts unchanged — no production edits expected), full `pytest` (0 failed; smt/onnxruntime/BLAS skips expected locally). Run the targeted file `pytest -q tests/test_reproduce_scalar_identity.py` and the regression set `pytest -q tests/test_reproduce.py tests/test_reproduce_field.py tests/test_canonicalize.py`.

### Review Findings

_Code review 2026-06-09 (Blind Hunter · Edge Case Hunter · Acceptance Auditor). Auditor verdict: **ACCEPT** — both ACs satisfied (raw-lineage field-key absence; recomputed-fingerprint-byte equality + `sha256 == prediction_hash`; scalar-shaped `ReproduceResult`; scalar-branch teeth; CI pin in the pinned-threads job). The "zero production change" claim is contract-consistent — R2.3 made the scalar path byte-identical by construction. The Edge Case Hunter verified every recomputation/import/shape/JSON-round-trip claim against source (split mirrors engine.py:102–104; `_FINGERPRINT_MAX_ROWS` importable; `hyperparameters == {"degree": 2}` matches the engine factory; the divergence message is exactly "max abs prediction"). All 8 Blind Hunter "fragility" findings were refuted by that source verification (the blind layer ran with no project context)._

- [x] [Review][Patch] Pin the documented NaN-safe `assert_reproducible` deviation with a scalar test [tests/test_reproduce_scalar_identity.py] — the story's one intentional deviation (R2.3's `not (diff <= tol)`, services.py:247) was documented in prose but unpinned; a future revert to `diff > tol` would not have been caught. **Fixed 2026-06-09:** added `test_scalar_nan_diff_raises_consistently_with_verdict` — pins both that `_verdict` classifies a NaN scalar diff as `diverged` (true pre- and post-ROM) and that `assert_reproducible` raises via the `max abs prediction` branch on a NaN-`max_abs_diff` scalar `ReproduceResult` (`field_relative_diff=None`), making the deviation tamper-evident. Regression set 43 passed / 1 skipped; ruff clean.
- [x] [Review][Defer] Over-tolerance scalar verdict not pinned end-to-end through `reproduce()` [tests/test_reproduce_scalar_identity.py] — deferred: AC1 names "verdict"; the happy-path `reproduced` is pinned end-to-end and the divergent verdict via a hand-built object, but no real scalar run is driven to over-tolerance `diverged`. Well-backstopped — `_verdict` is unit-tested across tolerance bands (test_reproduce.py:134–140) and `data_changed` end-to-end (test_reproduce.py:51–66); a real polynomial run reproduces bit-identically by design, so a genuine over-tolerance scalar run is awkward to construct. Low marginal value.
- [x] [Review][Defer] `metrics.json` / config-snapshot byte-shape not asserted [tests/test_reproduce_scalar_identity.py] — deferred: not an AC requirement (AC1 enumerates lineage + fingerprint + `ReproduceResult` only); the broader run-dir shape is backstopped by `test_reproduce.py` and the full suite. Out of R2.4's scoped surface.

_Dismissed (9): all 8 Blind Hunter findings — `train_test_split` re-derivation "fragile" (it deliberately mirrors the engine, Edge-verified vs engine.py:102–104; this is the intended no-golden-literal recomputation pin), `ndim==2` "breaks 1-D scalar runs" (project [MUST]: `predict` is always 2-D `(n,1)` via `_as_2d`; the test passes, confirming 2-D — the assert is itself a useful shape pin), `tobytes()`/`max_abs_diff==0.0` "BLAS-flaky" (polynomial is closed-form OLS, byte-stable on Accelerate; `test_reproduce_is_bit_identical` asserts the same unguarded and runs in pinned-threads CI), `_FINGERPRINT_MAX_ROWS` "truncation-order mismatch" (Edge-verified exact mirror + imports the real constant), `field_relative_diff is None` "dead weight" (intentional documentation that the fixture is scalar-shaped; the meaningful check is the `match=` message), `parent/reproduction=None` + `data_hash_match=True` "unrealistic fixtures" (the established house pattern in test_reproduce.py:107–118, isolating the abs-diff branch). Edge Case Hunter's "working tree not clean" [LOW] — based on a stale git snapshot; the current tree is clean except R2.4, and `reduced_field.py`/`canonicalize.py` are committed at HEAD (7e34f87/bcfbc5c)._

## Dev Notes

### The central realization (read this first — it shrinks the story)
**R2.4 is a pinning story, not an implementation story.** R2.3 was built so the scalar path is byte-identical *by construction*, and that is already verified two ways today:
1. `engine.run_training` adds field lineage **only** inside `if hasattr(holdout_model, "reproducibility_diagnostics")` (engine.py:148). A scalar trainer (`polynomial`/`kriging`/`rbf`) has no such hook, so `field_extras = {}`, the `spec_hash` line is skipped, and the persisted `lineage` dict has exactly the pre-ROM keys.
2. `services.reproduce` sets `is_field = bool(parent.basis_hash)` (services.py:95). For a scalar run that is `False`, so `diff_for_verdict = max_abs_diff` and every additive `ReproduceResult` field keeps its scalar-identical default (`None`/`None`/`False`/`None`). `_verdict` (services.py:137) is shared and unchanged.

So the work is to **pin** this with an explicit, named regression gate so a *future* edit to the field path cannot silently perturb scalar bytes. Expect **zero** production diff. If you find yourself editing `services.py`/`engine.py`, stop and re-read this — unless you've found a real divergence (then fix it minimally and document it; that discovery is the story succeeding).

### What "byte-identical to pre-ROM" means concretely (the contract to pin)
- **Lineage shape:** the scalar `lineage.json` has the pre-ROM key set and **none** of `{basis_hash, holdout_coefficients, reduction, spec_hash}`. (Assert on `run.lineage` directly — the `run.py` accessors default-mask absence: `basis_hash → ""`, `holdout_coefficients → []`, `reduction → {}`, `spec_hash → ""`, run.py:139–152.)
- **Fingerprint bytes:** `holdout_fingerprint` = `np.ascontiguousarray(holdout_test_pred, float64)[:512]` and `prediction_hash = sha256(that.tobytes())` — exactly `engine._prediction_fingerprint` (engine.py:136–139, 235–243). Pin by recomputing in-process, not by a golden string.
- **Verdict/result shape:** `ReproduceResult` scalar-shaped; `assert_reproducible` judges by `max_abs_diff` (services.py:240–255).

### The one intentional, verdict-consistent edge (call it out, don't "fix" it)
R2.3's review made `assert_reproducible` NaN-safe: `if not (diff <= tol):` (services.py:247). For a **finite** scalar diff this is identical to the pre-ROM `diff > tol`. The only behavioural change is on a **NaN** scalar diff: pre-ROM it slipped through `nan > tol == False` as a *silent pass* even though `_verdict` already returned `diverged` (a latent inconsistency); now it raises, matching the verdict. A NaN diff is practically unreachable for a real finite scalar run, and the change makes `assert_reproducible` *consistent with the verdict it claims to enforce*. **Document this in Completion Notes as a known, intentional, verdict-consistency edge — it is not a byte regression of the scalar contract** (which governs real, finite scalar runs). Do not revert it.

### The frozen contract this story guards
`_bmad-output/implementation-artifacts/spec-field-rom-fingerprint.md` (`<frozen-after-approval>`):
- **Decision #6:** "v1 ships one verdict only … For a scalar run (identity basis) it collapses to today's byte behaviour — proven separately by R2.4's regression guard." R2.4 *is* that separate proof.
- **Boundaries — Always:** "The scalar path stays **byte-identical** (identity basis ⇒ field = predictions; R2.4)."
- **I/O & Edge-Case Matrix:** the row "Scalar run (no reduction) | identity basis | **byte-identical to today** | R2.4 regression guard" is this story's acceptance row.

### Source-tree components to touch (current state → what changes → what to preserve)
- **NEW `tests/test_reproduce_scalar_identity.py`** — the regression gate (Tasks 2–5). Flat `tests/`, one file per concern.
- **UPDATE `.github/workflows/ci.yml`** — extend the `reproducibility (pinned threads)` job target only (ci.yml:72). Nothing else.
- **NO production change expected** to `src/surroforge/services.py`, `engine.py`, `core/run.py`, `trainers/reduced_field.py`, `cli/main.py`. *Preserve* the scalar branch and all its verdict outcomes/messages byte-for-byte (the whole point).

### Import-linter / architecture constraints (hard)
- Test-only story ⇒ the 3 import-linter contracts are untouched by construction. Re-run `lint-imports` anyway as a gate (Task 7). No new production imports; `engine` must still import neither `services` nor `reduced_field`.
- Tests may import from anywhere (they are not in the dependency graph), but prefer the public surface (`reproduce`, `assert_reproducible`, `run_training`, `parse_config`) and `tests.conftest` helpers, mirroring `tests/test_reproduce.py`.

### Testing standards
- pytest, flat `tests/`, deterministic data via the existing `project` fixture + `make_samples`; inner method = **polynomial** (closed-form, deterministic, no smt) so the asserts run on the local py3.14 venv with **no** BLAS guard. Reuse `poly_config("polynomial")` and the `select_columns`/`train_test_split` recomputation pattern already in `tests/test_reproduce.py::test_fingerprint_is_out_of_sample`.
- **[MUST] negative test with teeth** (project-context Testing): R2.4's teeth are *scalar-flavoured* — a divergent scalar `ReproduceResult` (`verdict="diverged"`, finite `max_abs_diff` over tolerance) must raise `ReproducibilityError` **through the `max_abs_diff` branch**, proving the scalar raise path is unchanged (Task 4). (The field-divergence teeth live in R2.3's `test_reproduce_field.py`.)
- **BLAS reality:** the polynomial fingerprint is closed-form and bit-stable run-to-run even on Apple Accelerate (`test_reproduce_is_bit_identical` carries no guard), so the byte-equality asserts need **no** `requires_reproducible_blas` marker. Do **not** add a golden literal hash (cross-platform-fragile). CI runs the file under pinned threads via the job env.

### Project Structure Notes
- The regression gate is a pure test + one CI-target line — no new module, no new production code, consistent with how Phase-1 pinned its boundary/profiling gates (`tests/test_architecture.py`, the `profile` CI job). The "pin an invariant with a dedicated gate" shape is the established house pattern (project-context: enforced architectural and quality gates).
- Note: `_bmad-output/project-context.md` still tags the reproducibility fix / CI / conformance as `(planned)` — **stale**; all shipped. Trust the source (`services._verdict`, `engine.run_training`, `.github/workflows/ci.yml`).

### Previous-story intelligence (R2.3, R2.2, R2.1)
- **R2.3** (done) delivered the field-primary verdict and was *explicitly designed to keep the scalar path byte-identical* (AC5). Its `tests/test_reproduce_field.py::test_scalar_run_records_no_field_diagnostics` and `test_scalar_run_uses_absolute_metric_not_field` already touch this — R2.4 deepens them into a **byte-level** pin (raw-lineage absence + recomputed-fingerprint-bytes equality) and adds the CI gate. Do not duplicate the R2.3 assertions; pin the *bytes*.
- **R2.3 review deferrals** (in `deferred-work.md`): the external-plugin `reproducibility_diagnostics` hook hardening and the stale frozen-spec tail prose — both out of R2.4 scope.
- **R2.2** (done) shipped `core/canonicalize.py`; irrelevant to the scalar path (no reduction ⇒ no canonicalization runs). Mentioned only to confirm R2.4 needs none of it.
- **R2.1 spike** proved Decision #6 empirically (a scalar/identity basis collapses to the byte path); R2.4 lifts that to a permanent CI gate.

### References
- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md#Story R2.4] — acceptance criteria (FR-R4, NFR-R1, NFR2).
- [Source: _bmad-output/implementation-artifacts/spec-field-rom-fingerprint.md#Contract Decisions (#6), #Boundaries & Constraints (Always), #I/O & Edge-Case Matrix (Scalar run row)] — the frozen contract R2.4 guards.
- [Source: src/surroforge/engine.py#run_training] — field branch gated on the duck-typed hook (engine.py:148); `holdout_fingerprint`/`prediction_hash` algorithm (engine.py:136–139, 235–243); `spec_hash` written only for field runs (engine.py:204–208).
- [Source: src/surroforge/services.py#reproduce] — `is_field = bool(parent.basis_hash)` discriminator (services.py:95); scalar `max_abs_diff` path + scalar-identical `ReproduceResult` defaults; NaN-safe `assert_reproducible` (services.py:247).
- [Source: src/surroforge/core/run.py] — defaulting field accessors (run.py:134–152) mask absence ⇒ assert on raw `run.lineage` for the lineage-shape pin.
- [Source: tests/test_reproduce.py] — the inherited pre-ROM scalar baseline (passes verbatim, R2.3 AC5); `test_fingerprint_is_out_of_sample` is the recomputation pattern to mirror.
- [Source: tests/test_reproduce_field.py#test_scalar_run_records_no_field_diagnostics] — R2.3's scalar-no-field-lineage check that R2.4 deepens to byte level.
- [Source: .github/workflows/ci.yml:72] — the `reproducibility (pinned threads)` job target to extend.
- [Source: _bmad-output/project-context.md#Reproducibility, #Testing] — named-tolerance + negative-test-with-teeth [MUST]s (its `(planned)` tags are stale).

## Dev Agent Record

### Agent Model Used

claude-opus-4-8[1m] (Opus 4.8, 1M context)

### Debug Log References

- Targeted: `pytest -q tests/test_reproduce_scalar_identity.py` → **4 passed**.
- Full suite: `pytest` → **217 passed, 3 skipped** (was 213 before R2.4 → exactly +4 new tests; the 3 skips are the pre-existing/expected onnxruntime, BLAS bit-repro guard, and lint-imports-PATH artifact).
- Pinned-threads regression set (CI job mirror): `OMP/OPENBLAS/MKL_NUM_THREADS=1 pytest tests/test_reproduce.py tests/test_canonicalize.py tests/test_reproduce_field.py tests/test_reproduce_scalar_identity.py` → **42 passed, 1 skipped**.
- Gates: `ruff check src tests` clean · `mypy src` clean (41 files) · `lint-imports` **3/3 kept** (no production edit ⇒ contracts untouched; re-run confirms `engine` still imports neither `services` nor `reduced_field`).

### Completion Notes List

- **Task 1 — confirmed byte-identical by construction; ZERO production change.** Re-verified both field-aware sites: `engine.run_training` writes `field_extras`/`spec_hash` only inside `if hasattr(holdout_model, "reproducibility_diagnostics")` (engine.py:148), and a scalar trainer has no such hook → the scalar `lineage` keeps exactly the pre-ROM keys; `services.reproduce` sets `is_field = bool(parent.basis_hash)` (services.py:95), which is `False` for a scalar run (`lineage.get("basis_hash", "")` → `""`), so `diff_for_verdict = max_abs_diff` and every additive `ReproduceResult` field keeps its scalar-identical default. No genuine divergence surfaced ⇒ **no `src/` edit made** (the story's expectation held).
- **One intentional, verdict-consistent edge (documented, not reverted):** R2.3's review made `assert_reproducible` NaN-safe via `if not (diff <= tol)` (services.py:247). For a *finite* scalar diff this is identical to the old `diff > tol`. Only a NaN scalar diff changes behaviour: it now raises, *matching* the `diverged` verdict `_verdict` already returns for NaN — a verdict-consistency fix on a practically-unreachable edge, not a byte regression of real (finite) scalar runs. The byte-identity contract governs real scalar runs; this is sound.
- **Tasks 2–5 — the regression guard** (`tests/test_reproduce_scalar_identity.py`, new, 4 tests):
  - *Lineage shape (T2):* asserts on the **raw** `run.lineage` dict (the `run.py` accessors default-mask absence) that `basis_hash`/`holdout_coefficients`/`reduction`/`spec_hash` are absent and the pre-ROM keys (`prediction_hash`/`holdout_fingerprint`/`reproduction_tolerance`/`config_hash`) are present.
  - *Fingerprint bytes (T3):* recomputes the pre-ROM algorithm in-process — same `train_test_split(test_size, random_state=seed, shuffle=True)`, the HOLDOUT model fit on `x_train` (via `get_trainer(method)(**hyperparameters)`, mirroring the engine factory), `np.ascontiguousarray(..., float64)[:512]` — and asserts `.tobytes()` equality + `sha256 == lineage["prediction_hash"]`. No golden literal hash (cross-platform-fragile); no BLAS guard needed (polynomial closed-form, bit-stable). Kept both sides 2-D `(n_test, 1)`.
  - *Result shape (T4):* scalar `ReproduceResult` has `field_relative_diff/basis_hash_match/coefficient_diagnostic` `None` and `basis_regauged False`; verdict `reproduced` (driven by `bit_identical`, not the tolerance branch — noted inline) with `max_abs_diff == 0.0`; `assert_reproducible` does not raise.
  - *Teeth (T4):* a divergent scalar `ReproduceResult` (`max_abs_diff=1.0`, `field_relative_diff=None`, `verdict="diverged"`) raises `ReproducibilityError` with `match="max abs prediction"` — proving the raise took the **scalar** branch, not the field branch.
  - *Baseline pin (T5):* module docstring names `tests/test_reproduce.py` as the inherited pre-ROM scalar contract this file guards (it passes verbatim — R2.3 AC5); this file deepens it to the byte level rather than duplicating it.
- **Task 6 — CI:** extended the `reproducibility (pinned threads)` job target (`.github/workflows/ci.yml`) to also run `tests/test_reproduce_scalar_identity.py`; threads are pinned at the job level, so no `@pytest.mark.slow` fixture (consistent with R2.3). Verified locally under the same pinned-thread env (42 passed, 1 skipped).
- **Scope held:** no probe coverage / global energy-norm (R2.5); no #72/#73; no scalar-tolerance change; no `reduced_field` import; no `src/` edit. A pure test + one-line CI story, as scoped.

### File List

- `tests/test_reproduce_scalar_identity.py` (new — 5 scalar byte-identity regression tests; the 5th, `test_scalar_nan_diff_raises_consistently_with_verdict`, added in code review to pin the documented NaN-safe deviation)
- `.github/workflows/ci.yml` (modified — pinned-threads `reproducibility` job also runs the new scalar-identity test)

### Change Log

- 2026-06-09: Story created (create-story) — scalar byte-identity regression guard for the field-primary fingerprint. Scoped as a test + CI gate with zero expected production change (R2.3 made the scalar path byte-identical by construction); pins lineage-shape absence of field keys, recomputed-fingerprint-byte equality, and scalar-shaped `ReproduceResult`, run under pinned threads in CI. Status → ready-for-dev.
- 2026-06-09: Validated (validate-create-story, independent fresh-context) → ACCEPT; applied 2 non-blocking clarifications (T4 verdict-driver = `bit_identical`; T3 `holdout_fingerprint` is 2-D `(n_test, 1)`).
- 2026-06-09: Implemented R2.4 (dev-story) — new `tests/test_reproduce_scalar_identity.py` (4 tests) pinning scalar byte-identity three ways (raw-lineage field-key absence; recomputed-fingerprint-byte equality with `sha256 == prediction_hash`; scalar-shaped `ReproduceResult` + max_abs_diff-branch teeth); CI `reproducibility` job extended. **Zero production change** — the scalar path was byte-identical by construction (R2.3); confirmed, no `src/` edit needed. Gates: 217 passed / 3 skipped; ruff/mypy clean; import-linter 3/3 kept. Status → review.
- 2026-06-09: Code review (Blind · Edge · Auditor) — Auditor **ACCEPT** (both ACs, zero-production-change contract-consistent, Edge verified every recomputation/import/shape claim vs source). Applied 1 [Patch]: added `test_scalar_nan_diff_raises_consistently_with_verdict` pinning the documented NaN-safe deviation (now 5 tests; regression set 43 passed / 1 skipped). 2 [Defer] recorded (over-tolerance scalar verdict end-to-end; metrics/config-snapshot shape — both backstopped by `test_reproduce.py`); 9 dismissed (8 blind findings refuted by source verification; 1 stale-git-snapshot). Status → done.
