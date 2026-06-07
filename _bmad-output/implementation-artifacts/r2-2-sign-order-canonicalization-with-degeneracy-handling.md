# Story R2.2: Sign/order canonicalization with degeneracy handling

Status: ready-for-dev

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

- [ ] **Task 1 — Create the shared canonicalization module** (AC: 1, 2, 3)
  - [ ] New pure-numeric leaf module `src/surroforge/core/canonicalize.py` (numpy only — **no other SurroForge imports**, no scipy, no new dependency).
  - [ ] Port `canonicalize(coeffs, basis) -> (coeffs_canon, basis_canon)` from the spike (`spikes/field_fingerprint_spike.py:99`): flip each mode so its basis max-abs entry is positive (apply the same sign to the matching coefficient column), then order by σ descending (σ = coefficient column norm), tie-broken by peak-node index `argmax(|V[:,k]|)`. Apply the identical sign+permutation gauge to **both** `coeffs` and `basis` so the reconstructed field is invariant.
  - [ ] Port `degenerate_clusters(coeffs, rtol=1e-6) -> (clusters, sigma)` (spike `:82`): group mode indices whose column-norm σ is ~equal; return only clusters with `len > 1` (as `list[list[int]]`) plus the σ vector.
  - [ ] Port `max_principal_angle(a, b) -> float` (spike `:146`): largest principal angle (radians) between column-spans via QR + SVD of `Qaᵀ·Qb`, clipped to `[-1, 1]` before `arccos`.
  - [ ] Add `subspace_reproduces(basis_a, basis_b, cluster, *, angle_tol=1e-6) -> bool` wrapping `max_principal_angle` on the cluster columns (spike `subspace_verdict` `:156`). Document the `angle_tol` choice: a microradian is far below any physical subspace rotation but above QR/SVD conditioning noise (~1e-8 rad).
  - [ ] Type hints + numpy docstrings; module docstring cites `spec-field-rom-fingerprint.md` and the originating spike.

- [ ] **Task 2 — Reuse, don't duplicate: refactor the trainer's sign step** (AC: 1) — *regression-critical*
  - [ ] `src/surroforge/trainers/reduced_field.py` already has `_canonicalize_signs(basis)` (`:157`). Extract the sign-fixing logic into the new module (e.g. `canonicalize_basis_signs(basis) -> basis`) and have `_canonicalize_signs` delegate to it — one implementation, not two.
  - [ ] **Must not change `ReducedFieldTrainer`'s fitted basis.** The trainer's basis is already σ-ordered from the SVD, so it must continue to apply **sign-only** canonicalization (not the full sign+order `canonicalize`), or the existing r1-3 tests that assert an exact, deterministic, sign-canonical basis (`tests/test_reduced_field.py::test_basis_is_deterministic_and_sign_canonical`, `test_rank_target_selects_exact_rank`, reconstruction tests) will break. Keep the trainer's public behaviour byte-identical; the full `canonicalize` (sign+order) is for the fingerprint path (R2.3), not the trainer.
  - [ ] Run the full `tests/test_reduced_field.py` and `tests/test_conformance.py[reduced-field]` to confirm no regression.

- [ ] **Task 3 — Tests: port the three spike claims as real unit tests** (AC: 1, 2, 3)
  - [ ] New `tests/test_canonicalize.py`. Build a synthetic scenario mirroring the spike: orthonormal basis `V` (n_dof × r), coefficient frame scaled to a spectrum with a **degenerate pair** (e.g. `[10,7,5,3,3,1]`), seeded `np.random.default_rng`.
  - [ ] **Claim 1 (sign/order determinism):** canonicalizing twice is identical; each canonicalized mode's basis max-abs entry is positive; modes are σ-descending.
  - [ ] **Claim 2 (re-gauge invariance):** a signed-permutation gauge `G` (`V@G`, `C@G`) canonicalizes to the **same** `(C, V)` (within ~1e-12); the reconstructed field is unchanged; a naive byte comparison of the raw re-gauged basis would differ (assert it does, to prove the canonicalization earns its keep).
  - [ ] **Claim 3 (degeneracy trap):** an intra-subspace rotation (e.g. 35°) on the equal-σ pair leaves the reconstructed field invariant; `degenerate_clusters` detects the pair; `subspace_reproduces`/`max_principal_angle` returns reproduced (angle ≤ 1e-6); **and** assert that per-mode canonical-coefficient comparison would false-diverge here (documents why subspace handling is load-bearing).
  - [ ] Determinism is robust by construction (1e-6 tol), but if any assert is BLAS-sensitive, gate it with the existing `requires_reproducible_blas` marker from `tests/conftest.py` (Accelerate is not bit-reproducible locally; OpenBLAS/CI is).

- [ ] **Task 4 — CI: cover the new tests under pinned threads** (AC: 4)
  - [ ] The `reproducibility (pinned threads)` job in `.github/workflows/ci.yml` runs `pytest -q tests/test_reproduce.py` with `OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS = 1`. Extend its pytest target to also run `tests/test_canonicalize.py` (e.g. `pytest -q tests/test_reproduce.py tests/test_canonicalize.py`).

- [ ] **Task 5 — Gates green**
  - [ ] `ruff check`, `mypy src`, `lint-imports` (3 contracts kept — confirm `core/canonicalize.py` adds no forbidden import), full `pytest` (0 failed; smt/onnxruntime/BLAS skips are expected locally).

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

### Debug Log References

### Completion Notes List

### File List
