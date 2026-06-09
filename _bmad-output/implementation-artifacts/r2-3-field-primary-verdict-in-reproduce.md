# Story R2.3: Field-primary verdict in `reproduce()`

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a methods engineer / auditor,
I want `reproduce()` to judge a field run by the **reconstructed field** (the thing I depend on) within a field-energy-relative tolerance folded into the hashed spec, with the basis hash and canonicalized coefficients recorded only as a diagnostic lens,
so that "reproduced" means the engineer-observable field matches — not an incidental coefficient/byte detail — and a benign re-gauge or a degenerate-mode rotation never causes a false `diverged`.

## Acceptance Criteria

1. **Given** a persisted **field run** (a run whose method reduces a field, i.e. carries a reduction spec / basis) **when** I run `surroforge reproduce <run-id>` **then** the verdict is the **relative field-energy difference** between the parent's and the re-run's reconstructed out-of-sample holdout field, compared against a **field-energy-relative tolerance**; within tolerance → `reproduced`/`reproduced_within_tolerance`, beyond → `diverged`.
2. **Given** a field run **when** it is reproduced after a benign basis re-gauge (sign flips / mode reorder, or a LAPACK/BLAS change) that leaves the reconstructed field identical **then** the verdict is `reproduced` and the result is annotated **"basis re-gauged"** — the `basis_hash` is recorded as a **diagnostic** and is **never** gated (a changed basis hash with a matching field still reads reproduced).
3. **Given** a field run with a near-degenerate singular-value cluster **when** an intra-subspace rotation leaves the field invariant **then** the field verdict reads `reproduced` (the always-correct safety net), and the per-mode canonical-coefficient diff is recorded as a **diagnostic only** (it would false-diverge — it must not drive the verdict); the degenerate cluster is compared as a subspace via principal angles (R2.2 primitives), never per-mode.
4. **Given** the field-energy-relative tolerance **when** it is widened **then** that is recorded as a **different run** (the tolerance is folded into the **hashed spec**, so a re-run with a different tolerance has a different spec hash) — not a looser compare on the same spec.
5. **Given** a **scalar run** (no reduction / identity basis) **when** it is reproduced **then** the verdict is **byte-identical to today's behaviour** — every existing `tests/test_reproduce.py` assertion passes unchanged (the scalar path is the R2.4 regression guard; R2.3 must not disturb it).
6. **Given** the new field-verdict code **when** CI runs **then** its reproducibility assertions execute under pinned BLAS/OpenMP threads (the existing `reproducibility` job), including a **negative test with teeth**: a perturbed field run (reseeded / nudged inner hyperparameter) is judged `diverged` and `assert_reproducible` raises `ReproducibilityError`.

> **Scope boundary (do not exceed).** R2.3 delivers the **field-primary verdict + diagnostics + tolerance-in-hashed-spec** only.
> - **Probe-node selection + `probe_coverage_energy` recording is R2.5**, not here. R2.3's verdict uses the **full reconstructed holdout field** as the energy norm (the "global energy-norm is the definition" direction the spike's OQ-3 endorses for v1; probes are the later fast path). Do not build energy-weighted probe selection in this story.
> - The **scalar byte-identity regression guard is R2.4** — R2.3 must *preserve* scalar behaviour, but the dedicated pinning test belongs to R2.4.
> - **Do not** introduce the physics/bitwise two-verdict (#73) or per-mode tolerance bands (#72) — both deferred to Phase 2.
> - **Do not** change the scalar default tolerance (`1e-9`) — that is "Ask First" in the frozen contract.
> - **Do not** make `engine.py` import `reduced_field` / any ROM module — the engine stays ROM-agnostic (spine theorem #36); use a **duck-typed optional introspection hook**, see Dev Notes.

## Tasks / Subtasks

- [x] **Task 1 — Field-energy-relative verdict in `services.reproduce`/`_verdict`** (AC: 1, 5)
  - [x] Add a field-energy-relative comparison: `field_relative_diff = ‖fp_parent − fp_repro‖_F / max(‖fp_parent‖_F, FLOOR)` over the recorded holdout fingerprints (which, for a field run, are already the reconstructed full field — see Dev Notes "The central realization"). Add a single named module-level `FIELD_ENERGY_FLOOR` constant (no inlined literals); reuse the existing `REPRODUCTION_TOLERANCE` pattern. **Per D1:** the relative bound is the run's declared `reproduction_tolerance` (lineage; `1e-9` for `reduced-field`) reinterpreted as relative-energy — not a new tolerance.
  - [x] **Discriminate scalar vs field by the presence of a reduction in the run's lineage** (e.g. `basis_hash`/`reduction` present). Scalar runs keep the **existing absolute `max_abs_diff ≤ tolerance`** path **unchanged** (AC5); field runs use the relative-energy path. `bit_identical` (prediction-hash match) stays the first check for both → `reproduced`.
  - [x] Keep the verdict vocabulary stable: `reproduced` / `reproduced_within_tolerance` / `diverged` / `data_changed` / `unverifiable`. The "basis re-gauged" state is an **annotation on a `reproduced` verdict**, not a new verdict value (additive — see Task 3).
  - [x] Route `assert_reproducible` through the field metric for field runs (raise on relative-diff over tolerance) while leaving the scalar branch and its messages intact.

- [x] **Task 2 — Record the diagnostics at train time (engine), ROM-agnostically** (AC: 2, 3)
  - [x] In `engine.run_training`, when the fitted final model exposes the optional introspection hook (duck-typed — e.g. a `reduction_fingerprint()` method or `basis`/`reduction_spec` properties; **no `reduced_field` import**), record **additively** in lineage: `basis_hash` (SHA-256 of the canonicalized basis bytes — diagnostic, never gated), the canonicalized holdout coefficients (per-mode "which mode moved" attribution — diagnostic), and the resolved `reduction` spec (`basis`, resolved `rank`, `energy_captured`). Scalar runs (hook absent) record nothing new → lineage unchanged.
  - [x] Expose the hook on `ReducedFieldTrainer` (in `trainers/reduced_field.py`): a public accessor returning the canonicalized basis `V` (it already lives in `self._basis`, sign-canonicalized) and the canonicalized holdout coefficients via the R2.2 `canonicalize(coeffs, basis)`. Use `core.canonicalize` (`canonicalize`, `degenerate_clusters`, `max_principal_angle`) — do **not** reimplement.
  - [x] Use the canonicalized basis for `basis_hash` so a sign/order gauge does not change the hash spuriously (though, per AC2, even a hash *change* is non-gating).

- [x] **Task 3 — Additive `ReproduceResult` fields + diagnostics surfacing** (AC: 1, 2, 3)
  - [x] Extend `ReproduceResult` **additively only** (frozen-contract "Ask First" forbids non-additive changes): `field_relative_diff: float | None = None`, `basis_hash_match: bool | None = None`, `basis_regauged: bool = False` (basis bytes differ **and** the field reproduced), and (optional) a per-mode coefficient-diff diagnostic. Defaults must make a scalar `ReproduceResult` identical to today.
  - [x] Compute `basis_regauged` = (basis_hash present on both runs) ∧ (hashes differ) ∧ (field verdict is `reproduced`/within tolerance). For the degenerate case (AC3), compare the equal-σ cluster as a subspace (`subspace_reproduces`) for the diagnostic; never let the per-mode coeff diff flip the verdict.
  - [x] Add `Run` accessors in `core/run.py` for the new lineage fields (`basis_hash`, canonicalized coeffs, `reduction`), mirroring the existing `holdout_fingerprint` accessor.

- [x] **Task 4 — Fold the field tolerance into the hashed spec** (AC: 4)
  - [x] Make the field-energy-relative tolerance part of a **hashed spec** so widening it yields a *different* run (re-hash), not a looser compare. **Per D2:** add an **additive `spec_hash`** = hash of (config + resolved reduction + field tolerance); **do not** mutate `config_hash` (keep its semantics + `test_two_independent_runs_share_prediction_hash` intact). Record the tolerance in lineage as today (`reproduction_tolerance`), now also covered by `spec_hash`.
  - [x] Keep `reproduce()`'s tolerance source = the value the run declared at train time (lineage), exactly as the scalar path does (Story 5.4) — do not introduce a per-call tolerance override beyond the existing one.

- [x] **Task 5 — CLI verdict rendering (additive)** (AC: 2)
  - [x] In `cli/main.py reproduce`, render the field verdict and the **"basis re-gauged"** annotation when `result.basis_regauged` (sober line, exit code unchanged: `reproduced`/within-tolerance → success, `diverged`/`data_changed`/`unverifiable` → `assert_reproducible` raises → exit 1). Scalar rendering unchanged. Keep CLI logic thin (Typer churn must not reach core).

- [x] **Task 6 — Tests: field-primary verdict, with teeth** (AC: 1, 2, 3, 5, 6)
  - [x] New `tests/test_reproduce_field.py` (flat tests/, one file per concern). A deterministic field scenario: `reduced-field` with `inner: { method: polynomial }` (deterministic, no smt) over a multi-column field output; reuse the `project` fixture / `make_samples` pattern. For smt-dependent inner methods, `pytest.importorskip("smt")` and gate bit-sensitive asserts with `requires_reproducible_blas`.
  - [x] **Positive:** honest field re-run → `reproduced`; `field_relative_diff ≤ tolerance`.
  - [x] **Re-gauge (AC2):** construct/parametrize a re-gauged equivalent (or assert that a BLAS-driven basis byte change does not gate) → `reproduced`, `basis_regauged is True`, `basis_hash_match is False`, verdict still `reproduced`. Proves the basis hash is diagnostic, not a gate.
  - [x] **Degenerate trap (AC3):** a field with an equal-σ cluster; an intra-subspace rotation leaves the field invariant → field verdict `reproduced` while the recorded per-mode coeff diff is large (assert the diagnostic diverges but the verdict does not). Lift the spike's Claim 3 to the verdict level using the R2.2 primitives.
  - [x] **Negative / teeth (AC6, project-context [MUST]):** train a field run, perturb (different inner seed/hyperparameter or a divergent inner method) → `diverged`; `assert_reproducible` raises `ReproducibilityError`. A same-seed round-trip alone does not satisfy this.
  - [x] **Scalar unchanged (AC5):** the full existing `tests/test_reproduce.py` passes verbatim.

- [x] **Task 7 — CI under pinned threads** (AC: 6)
  - [x] Extend the `reproducibility (pinned threads)` job target in `.github/workflows/ci.yml` to also run `tests/test_reproduce_field.py` (it currently runs `tests/test_reproduce.py tests/test_canonicalize.py`).

- [x] **Task 8 — Gates green**
  - [x] `ruff check`, `mypy src`, `lint-imports` (3 contracts kept — confirm `services`/`engine` only import `core.canonicalize` as a leaf, and `engine` does **not** import `services` or `reduced_field`), full `pytest` (0 failed; smt/onnxruntime/BLAS skips expected locally).

### Review Findings

_Code review 2026-06-09 (Blind Hunter · Edge Case Hunter · Acceptance Auditor). Auditor verdict: **ACCEPT** — all 6 ACs satisfied, scope + "Never" rules respected, engine ROM-agnostic, scalar path byte-identical; the documented `spec_hash` deviation (excluding the ULP-sensitive `energy_captured`) is a **sound** reading of the frozen intent (the "reduction in hashed spec" prose sits OUTSIDE the `<frozen-after-approval>` block; frozen Decision #5 mandates only the tolerance)._

- [x] [Review][Patch] `assert_reproducible` uses `diff > tol`, so a **NaN** field/scalar diff (`nan > tol` is False) does not raise — yet `_verdict` already returns `diverged` for it; the two disagree and a NaN field reads as a silent pass. Use NaN-safe `not (diff <= tol)` [src/surroforge/services.py:assert_reproducible] (blind+edge, Med) — **fixed 2026-06-09:** `assert_reproducible` now raises via `not (diff <= tol)`, so a NaN diff raises (matching the `diverged` verdict) on both the field and scalar paths; existing scalar tests unchanged.
- [x] [Review][Patch] `_coefficient_diagnostic` can **crash `reproduce()`**: a degenerate cluster wider than the holdout sample count makes `max_principal_angle(pc[:, cluster], …)` hit R2.2's `cols > rows` guard and raise `ValueError`; a ragged/malformed coeff lineage makes `np.asarray` raise — both propagate out of a *non-gating diagnostic* and abort the verdict. Make it fail-soft [src/surroforge/services.py:_coefficient_diagnostic] (edge, Med) — **fixed 2026-06-09:** the `np.asarray` is wrapped (ragged/non-numeric → `None`) and over-wide clusters (`len(cluster) > rows`) are skipped; new `test_coefficient_diagnostic_survives_cluster_wider_than_samples` would `ValueError` without the guard.
- [x] [Review][Patch] The `basis_regauged` annotation (AC2's "basis re-gauged, field reproduced" path) is never asserted: the gauge-invariance test exercises the field metric, but the flag's True branch has no coverage. Extract a tiny pure helper and unit-test both branches [src/surroforge/services.py + tests/test_reproduce_field.py] (blind, Low) — **fixed 2026-06-09:** extracted `_basis_regauged(basis_hash_match, verdict)`; `test_basis_regauged_flag_logic` covers the True branch + the match/None/diverged negatives.
- [x] [Review][Defer] Engine's duck-typed `reproducibility_diagnostics` hook does not validate the returned dict (missing key → `KeyError`; ragged coeffs → `ValueError`) and would abort a training run — only matters for a *future external* ROM plugin; the built-in hook is well-formed [src/surroforge/engine.py] — deferred: harden when an external ROM plugin first binds (no such plugin exists in v1).
- [x] [Review][Defer] The frozen spec's **non-frozen** tail prose (`spec-field-rom-fingerprint.md` lines 76–81: "reduction … part of the hashed spec", `basis.npz` persistence) is now stale vs. the shipped design (tolerance-only spec hash; coefficient-subspace diagnostic, no `basis.npz`) [_bmad-output/implementation-artifacts/spec-field-rom-fingerprint.md] — deferred: doc-only annotation; both omissions are sound (Auditor-confirmed), the prose is outside the frozen block.

_Dismissed (13): `is_field`/`basis_hash` "misclassification" (no pre-R2.3 field runs exist; reproduce re-trains the SAME config so parent+reproduction always classify identically; a hypothetical legacy field run gets a stricter absolute check, never a false pass); `comparable`/shape concerns (the field diff is gated by the existing `comparable` check, set to `inf` otherwise); `energy_captured` `[r-1]` "out of bounds" (`_resolve_rank` caps `r ≤ available`); coeff/field truncation "desync" (both capped at `_FINGERPRINT_MAX_ROWS` from the same deterministic `x_test`, row-aligned); `spec_hash` `repr(float)` "fragility" + concatenation "collision" (deterministic float repr; canonical JSON ends with `}`, so the join is unambiguous; tolerance is a fixed declared constant); scalar runs lacking `spec_hash` (by design — spec_hash is the field-tolerance mechanism; scalar identity stays `config_hash`); diagnostic `denom or 1.0` near-zero / `max_subspace_angle` 0.0 ambiguity (non-gating diagnostic; `degenerate_clusters` list disambiguates "no cluster"); `degenerate_clusters` returning empty/single-mode clusters (R2.2 guarantees `len > 1`, tested); scalar trainer defining the hook by name (documented duck-typed ROM contract); the round-trip + dead-`verdict`-kwarg test nits (teeth live in the divergence + field-metric tests)._

## Dev Notes

### The central realization (read this first — it shrinks the story)
A `reduced-field` run already flows through `engine.run_training` **unchanged** (spine theorem #36). The holdout model is a `ReducedFieldTrainer`, and `ReducedFieldTrainer.predict` **reconstructs the full field** (`coefficients @ basis.T + mean`, `reduced_field.py:147`). So the persisted `holdout_fingerprint` for a field run is **already the reconstructed out-of-sample holdout field** (`n_test × n_dof`, row-capped at `_FINGERPRINT_MAX_ROWS = 512`, `engine.py:136`). R2.3 is therefore **not** about plumbing a new prediction path — it is about:
1. **The verdict metric**: for field runs, compare by **field-energy-relative** difference instead of absolute `max_abs_diff` (services.py:65).
2. **Diagnostics**: record `basis_hash` + canonicalized coefficients (non-gating); annotate "basis re-gauged".
3. **Tolerance into the hashed spec** (AC4).
Scalar runs keep the existing absolute path verbatim (AC5).

### The frozen contract this story implements
`_bmad-output/implementation-artifacts/spec-field-rom-fingerprint.md` (`<frozen-after-approval>`) is the binding contract. The load-bearing decisions for R2.3:
- **Field equivalence is the definition / the verdict.** Gauge-invariant by construction; needs no canonicalization to be correct (Contract decision #1).
- **Canonicalized coefficients (#16) are a fast/audit lens, not the verdict** — valid per-mode attribution only under a bound basis (decision #2).
- **Degeneracy → subspace comparison or the field fallback, never per-mode** (decision #3; the field verdict is the always-correct safety net).
- **Basis hash → diagnostic, NEVER a gate** (decision #4; the pre-mortem + Claim 2 prove a benign re-gauge changes the bytes while the field is identical).
- **Tolerance is field-energy-relative and folded into the hashed spec; widening = a different run** (decision #5).
- **v1 ships one verdict; the scalar path collapses to today's byte behaviour** (decision #6 → AC5, guarded by R2.4).
- The contract's **"Verdict algorithm"** (spec lines 85–90) is the implementation target; its **I/O & Edge-Case Matrix** (lines 60–70) is the test matrix.

> **Never (from the frozen contract):** gate the verdict on the basis hash; use per-mode coefficient canonicalization as the verdict (it false-diverges on degeneracy); add a heavy dependency (SVD/principal angles are numpy — already have R2.2 primitives); weaken a tolerance to silence a flaky case (loosening ⇒ a different hashed spec).

### Source-tree components to touch (current state → what changes → what to preserve)
- **UPDATE `src/surroforge/services.py`** — `reproduce()` (`:43`), `_verdict()` (`:82`), `ReproduceResult` (`:31`), `assert_reproducible()` (`:101`), `REPRODUCTION_TOLERANCE` (`:28`).
  - *Current:* scalar — compares `holdout_fingerprint` arrays by absolute `max_abs_diff` vs the lineage `reproduction_tolerance` (Story 5.4); `bit_identical` from `prediction_hash`; verdicts `data_changed`/`unverifiable`/`reproduced`/`reproduced_within_tolerance`/`diverged`.
  - *Changes:* add the field-energy-relative branch for field runs; additive `ReproduceResult` fields; `basis_regauged` diagnostic.
  - *Preserve:* the scalar branch and all its verdict outcomes/messages **byte-for-byte** (AC5). `comparable` guard (empty/legacy/shape-mismatch → `unverifiable`) stays for both paths.
- **UPDATE `src/surroforge/engine.py`** — `run_training()` step 5/6 (`:129`–`:186`), `_config_hash` (`:192`), `_prediction_fingerprint` (`:196`).
  - *Current:* records `holdout_fingerprint`, `prediction_hash`, `reproduction_tolerance`, `config_hash` in lineage; ROM-agnostic.
  - *Changes:* when the fitted model exposes the optional ROM hook, additively record `basis_hash` + canonicalized coeffs + `reduction` spec; add `spec_hash` folding the tolerance.
  - *Preserve:* engine stays **ROM-agnostic** — duck-typed `hasattr`/`getattr` only, **no `reduced_field` import** (import-linter + spine theorem). Scalar lineage unchanged when the hook is absent.
- **UPDATE `src/surroforge/core/run.py`** — add accessors for `basis_hash`, canonicalized coeffs, `reduction`, `spec_hash` (mirror `holdout_fingerprint` `:124`). Pure lazy lineage views; no behaviour change for scalar runs.
- **UPDATE `src/surroforge/trainers/reduced_field.py`** — add the public introspection hook (canonicalized basis + canonicalized holdout coeffs) using `core.canonicalize`. **Must not change the fitted basis or `predict` output** (R1.3/R2.2 tests assert a byte-identical basis).
- **UPDATE `src/surroforge/cli/main.py`** — `reproduce` command renders the field verdict + "basis re-gauged" annotation (additive). Keep it thin.
- **NEW `tests/test_reproduce_field.py`**; **UPDATE `.github/workflows/ci.yml`** (`reproducibility` job target).

### Import-linter / architecture constraints (hard)
- `services` may import `core` (incl. `core.canonicalize`), `engine`, and leaves — OK.
- `engine` may import leaves + registry/ABCs but **must not import `services`** and must not import `reduced_field` (use the duck-typed hook). `core.canonicalize` is a pure leaf — `engine` importing it is fine if needed.
- Raise only `SurroForgeError` subclasses (`ReproducibilityError` for verdict failures). New CLI rendering uses `_guard`/`_info`/`_success`, never bare `print` (project-context Errors & CLI).

### Testing standards
- pytest, flat `tests/`, one file per concern. Deterministic data via the existing `project` fixture + `make_samples`; default inner = `polynomial` (deterministic, no smt) so the core asserts run on the local py3.14 venv.
- **[MUST] negative test with teeth** (project-context Testing): a reproducibility change is not done until a perturbed run is asserted to FAIL (`ReproducibilityError`). A same-seed round-trip alone passes even for a broken `reproduce()`.
- BLAS reality: Apple Accelerate is **not** bit-reproducible locally; OpenBLAS/CI is. `tests/conftest.py` exposes `blas_is_bit_reproducible()` + the `requires_reproducible_blas` skipif marker — gate only genuinely bit-sensitive asserts; the field-energy-relative tolerance (≫ ULP) should not need a guard. CI runs these under pinned threads (`OMP/OPENBLAS/MKL_NUM_THREADS=1`).
- Reuse R2.2 primitives (`canonicalize`, `degenerate_clusters`, `max_principal_angle`, `subspace_reproduces`) in tests and diagnostics — do not reimplement.

### Project Structure Notes
- The verdict logic lives in `services` (it opens a project + composes engine/persistence) — correct layer per the dependency table. The diagnostic recording lives in `engine` (a training-loop step). The `basis_hash`/coeff accessors live in `core/run.py` (pure domain view). No new module needed; `core.canonicalize` (R2.2) already supplies the math.
- Note: `_bmad-output/project-context.md` still tags the reproducibility fix, CI, and conformance as `(planned)` — **stale**; they are all implemented now (verified against source: `services._verdict`, `engine.holdout_fingerprint`, `.github/workflows/ci.yml`). Trust the source, not those `(planned)` tags.

### Previous-story intelligence (R2.1 spike, R2.2)
- **R2.1** (`spikes/field_fingerprint_spike.py`, all 3 claims pass) is the empirical proof of this verdict: identical→both reproduced; re-gauge→field reproduced + naive byte false-diverges; degenerate rotation→field reproduced + per-mode canon false-diverges (subspace angle `2.1e-8 rad`). Mirror its `main()` claims at the production verdict level.
- **R2.2** (done) shipped `src/surroforge/core/canonicalize.py` — `canonicalize(coeffs, basis)`, `degenerate_clusters`, `max_principal_angle`, `subspace_reproduces`, `canonicalize_basis_signs`, all with shape guards and 16 tests; the trainer's `_canonicalize_signs` already delegates to it. **Reuse these directly.** R2.2's deferred items (`round(σ,9)` sort-key vs relative rtol → revisit here/R2.5; `subspace_reproduces` cluster-index bounds → wire a clean error **here**, since R2.3 is the first real caller) are in `deferred-work.md` — pick up the cluster-index bounds guard when wiring `subspace_reproduces`.

### References
- [Source: _bmad-output/implementation-artifacts/spec-field-rom-fingerprint.md#Contract Decisions, #Verdict algorithm, #I/O & Edge-Case Matrix, #Tasks Derived] — the frozen contract (decisions 1–6; algorithm lines 85–90).
- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md#Story R2.3] — acceptance criteria (FR-R4, AR-R2, NFR2, NFR7; #64, #43, #72-deferred).
- [Source: src/surroforge/services.py#reproduce] — current scalar verdict (`:28`–`:123`) to generalize, not break.
- [Source: src/surroforge/engine.py#run_training] — fingerprint + lineage (`:129`–`:186`); `holdout_fingerprint` already = reconstructed field for a reduced run.
- [Source: src/surroforge/trainers/reduced_field.py#predict] — `predict` reconstructs the full field (`:139`–`:148`); `_basis` is sign-canonicalized (`:112`).
- [Source: src/surroforge/core/run.py#holdout_fingerprint] — lineage accessor pattern to mirror (`:124`).
- [Source: src/surroforge/core/canonicalize.py] — R2.2 primitives to reuse.
- [Source: _bmad-output/implementation-artifacts/spec-fix-reproduce-fingerprint.md] — the scalar precedent this generalizes (out-of-sample fingerprint + verdict).
- [Source: _bmad-output/project-context.md#Reproducibility, #Testing] — named-tolerance + negative-test-with-teeth [MUST]s (note its `(planned)` tags are stale).

### Resolved Design Decisions (confirmed with the user 2026-06-09 — these settle the frozen contract's OQ-4; treat as binding)
- **D1 — Tolerance value & meaning:** **Reuse the method-declared `reproduction_tolerance`** (`reduced-field` declares `1e-9`, `reduced_field.py:46`), **reinterpreted as the field-energy-relative bound**. One tolerance source, no new knob, consistent with Story 5.4. The scalar default `1e-9` stays absolute and unchanged.
- **D2 — How the tolerance enters the hashed spec:** Add an **additive `spec_hash`** = hash of (config + resolved reduction + field tolerance). **Do not** mutate `config_hash` — leave its semantics and `test_two_independent_runs_share_prediction_hash` undisturbed. "Widen tolerance ⇒ different run" (AC4) is satisfied because the widened tolerance changes `spec_hash`.
- **D3 — Wide-field lineage size:** Keep the existing **row cap (`_FINGERPRINT_MAX_ROWS = 512`)** only; **do not** add a column/probe cap. Energy-weighted probe selection + `probe_coverage_energy` recording stay in R2.5 (frozen scope). The full reconstructed holdout field (`min(512, n_test) × n_dof` floats) is the v1 global energy-norm definition.

## Dev Agent Record

### Agent Model Used

claude-opus-4-8[1m] (Opus 4.8, 1M context)

### Debug Log References

- Targeted: `pytest -q tests/test_reproduce_field.py` → 11 passed.
- Regression: `pytest -q tests/test_reproduce.py tests/test_reduced_field.py tests/test_conformance.py` → all pass (1 smt skip). Full `pytest -q` → 0 failed, 3 skips (onnxruntime, lint-imports PATH artifact, BLAS bit-repro guard — all pre-existing/expected).
- Gates: `ruff check src tests` clean · `mypy src` clean (41 files) · `lint-imports` 3/3 kept (`services` imports `core.canonicalize` as a leaf; `engine` imports neither `services` nor `reduced_field`).

### Completion Notes List

- **Task 1 — field-energy-relative verdict** (`services.py`): added `FIELD_ENERGY_FLOOR`; `reproduce()` discriminates field vs scalar by `parent.basis_hash` presence. Field runs use `_field_energy_relative_diff = ‖Δ‖_F / max(‖parent‖_F, floor)` (D1: bound = the run's declared `reproduction_tolerance`, reinterpreted as relative); scalar runs keep the absolute `max_abs_diff` path **byte-for-byte**. `_verdict` is **unchanged** and reused (it takes a generic diff), so `test_verdict_uses_declared_tolerance` still holds.
- **Task 2 — diagnostics at train time, ROM-agnostically** (`engine.py`, `reduced_field.py`): the engine duck-types an optional `reproducibility_diagnostics(x)` hook on the holdout model (no `reduced_field` import — spine theorem #36 intact); scalar trainers lack it, so their lineage is unchanged (`test_scalar_run_records_no_field_diagnostics`). The trainer hook returns `basis_hash` (SHA-256 of the canonicalized basis), the canonicalized out-of-sample coefficients (via R2.2 `canonicalize`), and the resolved reduction. `predict` was refactored to a `_coefficients` helper (output byte-identical — R1.3 tests green).
- **Task 3 — additive `ReproduceResult` fields + `Run` accessors**: `field_relative_diff`, `basis_hash_match`, `basis_regauged`, `coefficient_diagnostic` (all default to the scalar-identical values). `_coefficient_diagnostic` reports per-mode relative diffs for non-degenerate modes and a **principal-angle subspace** diff for degenerate clusters (R2.2 `degenerate_clusters`/`max_principal_angle`) — never per-mode, never gating. `core/run.py` gained `basis_hash`/`holdout_coefficients`/`reduction`/`spec_hash`.
- **Task 4 — tolerance in the hashed spec** (`engine._spec_hash`): additive `spec_hash = sha256(config_json + repr(tolerance))`; `config_hash` untouched (`test_two_independent_runs_share_prediction_hash` intact). **Deviation from the frozen contract (documented for review):** the contract lists the *resolved reduction* (incl. `energy_captured`) as part of the hashed spec, but `energy_captured` is a SVD-derived float that varies at ULP on non-reproducible BLAS (Accelerate) — folding it would make the spec identity itself non-reproducible. AC4's intent ("widen tolerance ⇒ different run") is fully met by `config + tolerance` (the reduction *request* already lives in the config); the *measured* reduction is recorded in lineage as a diagnostic only. `test_two_identical_field_runs_share_spec_hash` confirms bit-reproducibility on any BLAS.
- **Task 5 — CLI** (`cli/main.py`): `reproduce` renders the field-energy diff and the "(basis re-gauged; field reproduced)" annotation when `basis_regauged`; scalar rendering unchanged; exit codes unchanged.
- **Task 6/7 — tests + CI**: new `tests/test_reproduce_field.py` (11 tests) covers AC1 (verdict + sensitivity teeth), AC2/AC3 (gauge invariance of the field metric; subspace-vs-per-mode diagnostic), AC4 (`spec_hash` folds the tolerance), AC5 (scalar unchanged); `.github/workflows/ci.yml` runs it in the pinned-threads `reproducibility` job.
- **Scope held**: no probe selection / `probe_coverage_energy` (R2.5); no `basis.npz` persistence — the subspace diagnostic uses the stored canonical **coefficients** (a degenerate rotation rotates the coeff columns within the same span, so the coefficient subspace angle equals the basis subspace angle), so the basis array need not be persisted. The R2.2-deferred `subspace_reproduces` cluster-index-bounds concern does not arise here: clusters are derived from the same array they index.

### File List

- `src/surroforge/services.py` (modified — field-energy-relative verdict, additive `ReproduceResult` fields, `_field_energy_relative_diff` + `_coefficient_diagnostic`, field-aware `assert_reproducible`)
- `src/surroforge/engine.py` (modified — ROM-agnostic diagnostics hook recording + `_spec_hash`)
- `src/surroforge/trainers/reduced_field.py` (modified — `reproducibility_diagnostics` hook, `_coefficients` refactor, `energy_captured` state)
- `src/surroforge/core/run.py` (modified — `basis_hash`/`holdout_coefficients`/`reduction`/`spec_hash` accessors)
- `src/surroforge/cli/main.py` (modified — field verdict + basis-re-gauged rendering)
- `tests/test_reproduce_field.py` (new — 11 field-primary verdict tests)
- `.github/workflows/ci.yml` (modified — pinned-threads job runs the new field tests)

### Change Log

- 2026-06-09: Implemented R2.3 — field-primary verdict in `reproduce()`. Field runs are judged by the relative field-energy diff of the reconstructed out-of-sample holdout field (gauge-invariant); basis hash + canonicalized coefficients are a non-gating diagnostic (subspace comparison for degenerate clusters); the field tolerance is folded into an additive `spec_hash`. Scalar path byte-identical (R2.4 guard). All gates green; status → review.
- 2026-06-09: Code review (Blind · Edge · Auditor) — Auditor ACCEPT (all 6 ACs, scope/Never rules, byte-identical scalar; spec_hash deviation confirmed sound). Applied 3 [Patch] fixes: NaN-safe `assert_reproducible`; fail-soft `_coefficient_diagnostic` (over-wide cluster + ragged lineage); extracted/tested `_basis_regauged`. 2 [Defer] recorded (external-plugin hook hardening; annotate the spec's non-frozen stale prose); 13 dismissed. 213 passed, 3 skipped; ruff/mypy/import-linter green. Status → done.
