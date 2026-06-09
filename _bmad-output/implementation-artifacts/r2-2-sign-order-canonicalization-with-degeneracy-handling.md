# Story R2.2: Sign/order canonicalization with degeneracy handling

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a methods engineer,
I want the modal basis canonicalized (sign + order) before any hashing or comparison, with near-degenerate singular-value clusters compared as a subspace rather than per-mode,
so that SVD gauge ambiguity never causes a false `diverged` verdict.

## Acceptance Criteria

1. **Given** a POD basis (possibly with near-equal singular values) **when** it is canonicalized **then** mode **sign** (max-abs component positive) and **order** (σ descending, tie-broken by peak-node index) are fixed deterministically.
2. **Given** a basis with a near-equal singular-value cluster **when** the canonicalized basis is compared to a re-gauged equivalent **then** equal-σ clusters are detected (relative tolerance) and compared as a **subspace via principal angles** (`angle_tol ≈ 1e-6 rad`), never per-mode.
3. **Given** the same physics re-imported with a re-gauged basis (sign flips + mode reorder) **then** canonicalization yields the same basis (no false diverge); **and** an intra-subspace rotation on a degenerate cluster is correctly judged reproduced by the subspace/principal-angle test even though per-mode canonicalization would false-diverge.
4. **Given** the new code **when** CI runs **then** its reproducibility assertions execute under pinned BLAS/OpenMP threads (the existing `reproducibility` job).

> **Scope boundary (do not exceed):** R2.2 delivers the **pure-numeric canonicalization + degeneracy primitives and their tests only**. Wiring these into `services.reproduce()`/`_verdict` is **R2.3**; recording `probe_coverage_energy` / global energy-norm is **R2.5**; the scalar byte-identity guard is **R2.4**. Do not touch `services.py` or `engine.py` in this story.

## Tasks / Subtasks

- [x] **Task 1 — Create the shared canonicalization module** (AC: 1, 2, 3)
  - [x] New pure-numeric leaf module `src/surroforge/core/canonicalize.py` (numpy only — **no other SurroForge imports**, no scipy, no new dependency).
  - [x] Port `canonicalize(coeffs, basis) -> (coeffs_canon, basis_canon)` from the spike (`spikes/field_fingerprint_spike.py:99`): flip each mode so its basis max-abs entry is positive (apply the same sign to the matching coefficient column), then order by σ descending (σ = coefficient column norm), tie-broken by peak-node index `argmax(|V[:,k]|)`. Apply the identical sign+permutation gauge to **both** `coeffs` and `basis` so the reconstructed field is invariant.
  - [x] Port `degenerate_clusters(coeffs, rtol=1e-6) -> (clusters, sigma)` (spike `:82`): group mode indices whose column-norm σ is ~equal; return only clusters with `len > 1` (as `list[list[int]]`) plus the σ vector.
  - [x] Port `max_principal_angle(a, b) -> float` (spike `:146`): largest principal angle (radians) between column-spans via QR + SVD of `Qaᵀ·Qb`, clipped to `[-1, 1]` before `arccos`.
  - [x] Add `subspace_reproduces(basis_a, basis_b, cluster, *, angle_tol=1e-6) -> bool` wrapping `max_principal_angle` on the cluster columns (spike `subspace_verdict` `:156`). Document the `angle_tol` choice: a microradian is far below any physical subspace rotation but above QR/SVD conditioning noise (~1e-8 rad).
  - [x] Type hints + numpy docstrings; module docstring cites `spec-field-rom-fingerprint.md` and the originating spike.

- [x] **Task 2 — Reuse, don't duplicate: refactor the trainer's sign step** (AC: 1) — *regression-critical*
  - [x] `src/surroforge/trainers/reduced_field.py` already has `_canonicalize_signs(basis)` (`:157`). Extract the sign-fixing logic into the new module (e.g. `canonicalize_basis_signs(basis) -> basis`) and have `_canonicalize_signs` delegate to it — one implementation, not two.
  - [x] **Must not change `ReducedFieldTrainer`'s fitted basis.** The trainer's basis is already σ-ordered from the SVD, so it must continue to apply **sign-only** canonicalization (not the full sign+order `canonicalize`), or the existing r1-3 tests that assert an exact, deterministic, sign-canonical basis (`tests/test_reduced_field.py::test_basis_is_deterministic_and_sign_canonical`, `test_rank_target_selects_exact_rank`, reconstruction tests) will break. Keep the trainer's public behaviour byte-identical; the full `canonicalize` (sign+order) is for the fingerprint path (R2.3), not the trainer.
  - [x] Run the full `tests/test_reduced_field.py` and `tests/test_conformance.py[reduced-field]` to confirm no regression.

- [x] **Task 3 — Tests: port the three spike claims as real unit tests** (AC: 1, 2, 3)
  - [x] New `tests/test_canonicalize.py`. Build a synthetic scenario mirroring the spike: orthonormal basis `V` (n_dof × r), coefficient frame scaled to a spectrum with a **degenerate pair** (e.g. `[10,7,5,3,3,1]`), seeded `np.random.default_rng`.
  - [x] **Claim 1 (sign/order determinism):** canonicalizing twice is identical; each canonicalized mode's basis max-abs entry is positive; modes are σ-descending.
  - [x] **Claim 2 (re-gauge invariance):** a signed-permutation gauge `G` (`V@G`, `C@G`) canonicalizes to the **same** `(C, V)` (within ~1e-12); the reconstructed field is unchanged; a naive byte comparison of the raw re-gauged basis would differ (assert it does, to prove the canonicalization earns its keep).
  - [x] **Claim 3 (degeneracy trap):** an intra-subspace rotation (e.g. 35°) on the equal-σ pair leaves the reconstructed field invariant; `degenerate_clusters` detects the pair; `subspace_reproduces`/`max_principal_angle` returns reproduced (angle ≤ 1e-6); **and** assert that per-mode canonical-coefficient comparison would false-diverge here (documents why subspace handling is load-bearing).
  - [x] Determinism is robust by construction (1e-6 tol), but if any assert is BLAS-sensitive, gate it with the existing `requires_reproducible_blas` marker from `tests/conftest.py` (Accelerate is not bit-reproducible locally; OpenBLAS/CI is).

- [x] **Task 4 — CI: cover the new tests under pinned threads** (AC: 4)
  - [x] The `reproducibility (pinned threads)` job in `.github/workflows/ci.yml` runs `pytest -q tests/test_reproduce.py` with `OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS = 1`. Extend its pytest target to also run `tests/test_canonicalize.py` (e.g. `pytest -q tests/test_reproduce.py tests/test_canonicalize.py`).

- [x] **Task 5 — Gates green**
  - [x] `ruff check`, `mypy src`, `lint-imports` (3 contracts kept — confirm `core/canonicalize.py` adds no forbidden import), full `pytest` (0 failed; smt/onnxruntime/BLAS skips are expected locally).

### Review Findings

_Code review 2026-06-07 (Blind Hunter · Edge Case Hunter · Acceptance Auditor). Auditor verdict: all 4 ACs satisfied, scope boundary + "Never" rules respected, trainer basis byte-identical (verified over 200 random matrices)._

- [x] [Review][Patch] `max_principal_angle` understates the angle for mismatched / over-wide (cols > rows) column counts — reduced QR truncates dimensions, so `subspace_reproduces` can return a false "reproduced"; also `.min()` crashes on an empty cluster [src/surroforge/core/canonicalize.py:max_principal_angle] (blind+edge, High) — **fixed 2026-06-07:** guards added — equal column counts, equal ambient dimension, `k ≤ n_dof` (over-wide raises `ValueError` instead of silently understating), and a 0-column subspace returns angle `0.0` (vacuously identical, no `.min()` crash).
- [x] [Review][Patch] `canonicalize` silently broadcasts/truncates when `coeffs` and `basis` have mismatched column counts instead of raising a clear error [src/surroforge/core/canonicalize.py:canonicalize] (edge, Med) — **fixed 2026-06-07:** raises `ValueError` on non-2-D input or unequal mode (column) counts before applying the shared gauge.
- [x] [Review][Patch] Add tests: a 3+ mode degenerate cluster compared via the subspace path, and shape-mismatch guards that raise [tests/test_canonicalize.py] (auditor, Low) — **fixed 2026-06-07:** added a 3-fold degenerate cluster rotated by an arbitrary 3×3 orthogonal block (subspace reproduces, per-mode diverges), an empty-cluster vacuous-truth case, and four guard tests (`canonicalize` mode mismatch; `max_principal_angle` column / ambient / over-wide mismatches). 9 → 15 tests.
- [x] [Review][Defer] `degenerate_clusters` single-linkage chaining (each member compared to `cur[-1]`) can merge a σ-span wider than `rtol` [src/surroforge/core/canonicalize.py:degenerate_clusters] — deferred: byte-identical to the frozen spike contract; refining linkage is a contract renegotiation, not an R2.2 fix
- [x] [Review][Defer] `degenerate_clusters` `max(σ,1.0)` floor turns the relative tolerance absolute for σ<1, over-clustering small singular values [src/surroforge/core/canonicalize.py:degenerate_clusters] — deferred: byte-identical to the frozen spike; a future clustering refinement

_Dismissed (3): per-mode order instability **within** a degenerate cluster (by design — degenerate clusters are compared as a subspace, never per-mode; this is the premise of R2.2); NaN/Inf in `_mode_signs` (out of contract — SVD of finite field data is finite); two self-dismissed non-bugs from the Blind Hunter (sign/permutation consistency between coeffs & basis is correct)._

### Review Findings — Re-review (2026-06-07)

_Re-review after the three round-1 [Patch] fixes landed (Blind Hunter · Edge Case Hunter · Acceptance Auditor). Auditor verdict: all 4 ACs still satisfied, scope + "Never" rules respected, trainer basis byte-identical (re-verified over 500 random matrices), and the three round-1 fixes are real, complete, and regression-free._

- [x] [Review][Patch] No negative test for `subspace_reproduces`/`max_principal_angle` — every assertion checks "same subspace → small angle"; a silent always-~0 bug (e.g. a transpose error in `qaᵀ·qb`) would pass the whole suite. Add a distinct-subspaces case asserting a large principal angle and `subspace_reproduces` False [tests/test_canonicalize.py] (blind, Low — test robustness) — **fixed 2026-06-07:** added `test_distinct_subspaces_do_not_reproduce` (orthogonal basis-column subspaces → angle ≈ π/2, `subspace_reproduces` False). 15 → 16 tests; suite green.
- [x] [Review][Defer] `round(σ, 9)` absolute sort-key vs the relative `rtol` cluster metric — the two use different notions of "equal σ"; for σ differing by <5e-10 but not clustered, per-mode order can flip [src/surroforge/core/canonicalize.py:canonicalize] — deferred: byte-identical to the frozen spike; the `round(,9)` quantizer intentionally absorbs BLAS noise in the ordering and the field-primary verdict (R2.3) is the safety net. Revisit with R2.5 (global energy-norm).
- [x] [Review][Defer] `subspace_reproduces` does not bounds-check `cluster` against the basis column count — an out-of-range index raises a raw `IndexError` instead of a clear error [src/surroforge/core/canonicalize.py:subspace_reproduces] — deferred to R2.3: in-contract the cluster always comes from `degenerate_clusters` of the same-rank coeffs; the clean-error guard belongs with the R2.3 wiring that establishes the caller relationship.

_Re-raised from round 1, already deferred (no action): `degenerate_clusters` single-linkage chaining; `max(σ,1.0)` tolerance floor for σ<1._

_Dismissed (12): intra-cluster order instability (by design — the premise of R2.2); `_mode_signs` zero / sign-tied peak, rank-deficient `max_principal_angle`, NaN/Inf, 1-D inputs, negative/NaN tolerances (all out of contract — retained POD modes are finite, unit-norm, mutually orthonormal); coeffs/basis "row-count mismatch" (false positive — rows are samples vs DOFs; only the shared column axis is meaningful and it is validated); σ-tie ordering test + `rel_diff>1e-3` bound (would assert intentional instability / a rotation-dependent magnitude); `ANGLE_TOL` noise claim "unverified" (the passing <1e-6 asserts already bound it; auditor measured 2.1e-8 rad); CI `array_equal` "flakiness" (it is same-input idempotence within one process, not cross-run BLAS reproduction); docstring clip-rationale wording._

## Dev Notes

### The frozen contract this story implements
`_bmad-output/implementation-artifacts/spec-field-rom-fingerprint.md` is the **frozen** (`<frozen-after-approval>`) reproduce() contract from the R2.1 spike. R2.2 productionizes its primitives. Key decisions binding this story:
- **Canonicalization runs before any coefficient comparison.** Sign by max-abs entry, order by σ desc, tie-break by peak-node index. (Contract decision #2.)
- **Degeneracy is handled by subspace comparison, never per-mode** — per-mode canonicalization **false-diverges** on the (common, symmetric) degenerate case. Detect equal-σ clusters by relative tolerance; compare the subspace via principal angles, `angle_tol ≈ 1e-6 rad`. The field verdict is the always-correct safety net (that's R2.3). (Contract decision #3.)
- **Never** gate any verdict on the basis hash, and **never** use per-mode coefficient canonicalization as a verdict. (These are R2.3 concerns but the primitives here must not encourage them.)

### Reference implementations (port faithfully, then add types/docstrings)
All four functions exist and are proven in `spikes/field_fingerprint_spike.py`:
- `canonicalize(C, V)` — `:99`
- `degenerate_clusters(C, rtol=1e-6)` — `:82`
- `max_principal_angle(A, B)` — `:146`
- `subspace_verdict(parent, child, cluster, angle_tol=1e-6)` — `:156` (becomes `subspace_reproduces`)
- `rel_diff(a, b)` — `:116` — a relative-norm helper; you may port it too if a test needs it, but the *field* relative-diff verdict lives in R2.3, not here.

The spike's `main()` prints PASS for all three claims (`python spikes/field_fingerprint_spike.py` → exit 0). Your unit tests are the production version of those three claims.

### Source tree components to touch
- **NEW** `src/surroforge/core/canonicalize.py` — pure-numeric leaf (numpy only). `core/` currently holds `config.py, dataset.py, project.py, run.py`; this fits as shared pure logic. It will be imported by `trainers/reduced_field.py` (now) and `services.reproduce` (R2.3) — both are allowed to import `core` under the import-linter contracts.
- **UPDATE** `src/surroforge/trainers/reduced_field.py` — `_canonicalize_signs` (`:157`) delegates to the new module (DRY). Current behaviour: flips each mode so its max-abs entry is positive; the basis is otherwise σ-ordered from `np.linalg.svd(...)`. **Preserve this exact output** (see Task 2 regression note).
- **NEW** `tests/test_canonicalize.py`.
- **UPDATE** `.github/workflows/ci.yml` — `reproducibility` job pytest target.

### Testing standards
- pytest; deterministic synthetic data via seeded `np.random.default_rng` (no `Math.random`-style ambient seeds). Match the existing test style in `tests/test_reduced_field.py`.
- The repo's BLAS-reproducibility reality: numpy uses Apple Accelerate locally (not bit-reproducible), OpenBLAS on Linux CI (bit-reproducible). `tests/conftest.py` exposes `blas_is_bit_reproducible()` and the `requires_reproducible_blas` skipif marker — reuse them if (and only if) an assertion is genuinely bit-sensitive. Principal-angle / canonicalization tolerances (1e-6 / 1e-12) are well above ULP noise, so most asserts should pass on every BLAS without a guard.

### Project Structure Notes
- Module placement `src/surroforge/core/canonicalize.py` keeps it a pure leaf consumable by both `trainers/` and `services` without crossing the import-linter boundaries (`core/engine/services` must never import `cli/api`; `errors` is the only enforced pure leaf — `canonicalize` importing nothing internal is consistent with that spirit).
- No new dependency: principal angles use `numpy.linalg.qr` + `numpy.linalg.svd`; no scipy needed (contract: "Never add a new heavy dependency").

### References
- [Source: _bmad-output/implementation-artifacts/spec-field-rom-fingerprint.md#Contract Decisions] — sign/order rule, degeneracy-by-subspace, basis-hash-is-diagnostic; "Tasks Derived → R2.2".
- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md#Story R2.2] — acceptance criteria (lines 180–194).
- [Source: spikes/field_fingerprint_spike.py] — reference implementations + the 3 proven claims (lines 82, 99, 146, 156).
- [Source: src/surroforge/trainers/reduced_field.py#_canonicalize_signs] — existing sign canonicalization to reuse (line 157).
- [Source: .github/workflows/ci.yml#reproducibility] — pinned-threads job (`OMP/OPENBLAS/MKL=1`, runs `pytest -q tests/test_reproduce.py`).
- [Source: _bmad-output/planning-artifacts/prd-field-rom-v1.md#ADR-ROM-001] — fingerprint design (field-primary; basis hash diagnostic).

## Dev Agent Record

### Agent Model Used

claude-opus-4-8[1m] (Opus 4.8, 1M context)

### Debug Log References

- Targeted: `pytest tests/test_canonicalize.py tests/test_reduced_field.py` → 22 passed.
- Full regression: `pytest -q` → 0 failed, 3 skipped (lint-imports PATH artifact, onnxruntime no cp314 wheel, the BLAS bit-repro guard on local Accelerate — all pre-existing/expected).
- Gates: `ruff check` clean · `mypy src` clean (41 files) · `lint-imports` 3/3 kept.

### Completion Notes List

- **Task 1** — `src/surroforge/core/canonicalize.py` (new, pure-numpy leaf): ported `canonicalize(coeffs, basis)`, `degenerate_clusters(coeffs, rtol)`, `max_principal_angle(a, b)`, `subspace_reproduces(...)` from the R2.1 spike, plus a shared `canonicalize_basis_signs` and a private `_mode_signs` so the sign gauge has one implementation. `CLUSTER_RTOL`/`ANGLE_TOL = 1e-6` module constants; added an empty-input guard to `degenerate_clusters`.
- **Task 2** (reuse, no duplication) — `trainers/reduced_field.py::_canonicalize_signs` now delegates to `canonicalize_basis_signs`; the trainer keeps **sign-only** canonicalization on its already-σ-ordered basis, so its fitted basis is byte-identical (the full sign+order `canonicalize` is reserved for the R2.3 fingerprint path). Verified: `tests/test_reduced_field.py` + conformance[reduced-field] still green.
- **Task 3** — `tests/test_canonicalize.py` (new): the three spike claims as unit tests (deterministic sign/order form; re-gauge canonicalizes identically while a naive byte compare false-diverges; degenerate-cluster rotation → subspace reproduces but per-mode canonicalization false-diverges), plus field-invariance and empty/no-degeneracy edge cases. Tolerances (1e-12 / 1e-6 rad) sit above ULP/QR noise, so no BLAS guard was needed.
- **Task 4** — `.github/workflows/ci.yml`: the `reproducibility (pinned threads)` job now runs `pytest -q tests/test_reproduce.py tests/test_canonicalize.py` (AC 4).
- **Scope held**: no change to `services.py`/`engine.py` — the field-primary verdict wiring is R2.3.

### File List

- `src/surroforge/core/canonicalize.py` (new)
- `tests/test_canonicalize.py` (new)
- `src/surroforge/trainers/reduced_field.py` (modified — delegate sign gauge to the shared primitive)
- `.github/workflows/ci.yml` (modified — pinned-threads job also runs the canonicalization tests)

### Change Log

- 2026-06-07: Implemented R2.2 — pure-numeric sign/order canonicalization + degeneracy (principal-angle subspace) primitives in `core/canonicalize.py`, with the reducer's sign gauge refactored to delegate. Tests + CI pinned-threads coverage added. All gates green; status → review.
- 2026-06-07: Addressed code-review findings — resolved 3 [Patch] items (`max_principal_angle` shape/over-wide/empty guards; `canonicalize` mode-count validation; +6 hardening tests, 9 → 15). Trainer basis still byte-identical; ruff/mypy/import-linter (3/3)/pytest all green. Status → review (re-review).
- 2026-06-07: Re-review (Blind · Edge · Auditor) — Auditor re-confirmed all 4 ACs, scope/Never-rules, byte-identical trainer basis, and the 3 round-1 fixes. Applied 1 [Patch]: added `test_distinct_subspaces_do_not_reproduce` (negative principal-angle anchor; 15 → 16 tests). 2 new [Defer] recorded (round(σ,9) vs relative rtol → R2.5; `subspace_reproduces` cluster-index bounds → R2.3); 12 dismissed (out-of-contract / by-design). No unresolved High/Med. Status → done.
