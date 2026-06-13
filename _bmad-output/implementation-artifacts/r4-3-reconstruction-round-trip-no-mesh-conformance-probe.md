---
baseline_commit: 77fea4ece1a12db7f2f8d4a86f47fb548f560172
context:
  - _bmad-output/planning-artifacts/epics-field-rom-v1.md
  - _bmad-output/planning-artifacts/prd-field-rom-v1.md
  - _bmad-output/project-context.md
---

# Story R4.3: Reconstruction round-trip + no-mesh conformance probe

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a plugin author (and the Core maintainer self-certifying the built-in reducer),
I want the shipped conformance suite to verify my reducer is a faithful, mesh-free projector,
So that a third-party/commercial ROM can self-certify before claiming conformance — and the
`predict_coefficients` hook the engine/services already duck-type finally has a contract test.

## Acceptance Criteria

1. **Reconstruction round-trip check — the substance AC (faithful projector).**
   `check_trainer` gains a `reconstruction_round_trip` check. For a **reduced-field
   trainer** (detected by a callable `predict_coefficients` attribute — see Dev Notes
   "Reducer detection"), the check, on its **own multi-DOF field fixture** (NOT the scalar
   `_make_data`; see Dev Notes "Field fixture"):
   - fits a fresh trainer instance with the suite's `hyper`,
   - obtains the basis + mean from `trainer.reduction_artifacts()`
     (`basis` shape `(n_dof, r)`, `mean` shape `(1, n_dof)`),
   - computes coefficients `c = trainer.predict_coefficients(x_test)` and asserts
     `c.shape == (x_test.shape[0], r)` where `r == basis.shape[1]` — this is the
     `predict_coefficients` shape contract (folds in the R4.1 deferred item, see References),
   - reconstructs `field = c @ basis.T + mean`, projects back `c_rt = (field - mean) @ basis`,
     and asserts `max|c_rt - c| <= tol` where `tol = max(tolerance, declared
     reproduction_tolerance)` (the same tolerance-honouring rule as the `deterministic`
     check, Story 5.4) — proving the basis columns are orthonormal (`Bᵀ·B = I_r`), i.e. a
     faithful projector with no sign/normalization/scaling bug.
   - **A trainer exposing `predict_coefficients` but NOT `reduction_artifacts` FAILS** this
     check with a clear "incomplete reducer contract" detail (the two hooks are one contract).

   **[GOTCHA] Teeth (vacant-pin discipline, project-context):** the round-trip must FAIL for
   a reducer whose basis is non-orthonormal. The negative test (AC 5) uses a deliberately
   broken reducer that scales its stored basis (e.g. returns `2·B` from
   `reduction_artifacts`, or skips sign/norm canonicalization) so `Bᵀ·B ≠ I`; assert the
   `reconstruction_round_trip` result is FAIL. A round-trip that merely checks
   `c.shape` (presence-without-structure) or that passes for `2·B` (value degeneracy) is
   vacuous. The discriminating pin is the numeric `max|c_rt - c|` bound on a basis where a
   correct and a broken implementation differ at rendered precision.

2. **No-mesh check — DOF-permutation equivariance (mesh-free is a tested invariant, #53).**
   `check_trainer` gains a `mesh_free` check. For a reduced-field trainer, on the same field
   fixture:
   - fit a fresh trainer on `(x, y)`, predict the field `F = trainer.predict(x_test)` (shape
     `(n_test, n_dof)`),
   - choose a **fixed non-identity column permutation** `perm` of the `n_dof` DOFs (seeded,
     deterministic — see Dev Notes), fit a **second fresh trainer** on `(x, y[:, perm])`, and
     predict `F_perm = trainer2.predict(x_test)`,
   - assert `max|F_perm - F[:, perm]| <= tol` (same `tol` band as AC 1).
   The reconstructed **field** is gauge-invariant (sign/order of modes cancels in
   `Σ c_k·mode_k`), so equivariance holds in exact arithmetic and within the BLAS-noise band
   otherwise. A reducer that reads coordinates/connectivity (any geometry-dependent step)
   breaks equivariance and FAILS. Appears in the `ConformanceReport`.

   **[GOTCHA] Teeth:** the negative test (AC 5) uses a fake reducer that injects a
   DOF-index-dependent term (e.g. adds `column_index` to each DOF, or weights modes by a
   fixed coordinate vector) so it is NOT permutation-equivariant; assert `mesh_free` is FAIL.
   A signature-only check ("does `fit` take a `coords=` kwarg?") is explicitly **insufficient**
   (gameable, no teeth) and must NOT be the implementation — see Dev Notes "No-mesh: why
   behavioural".

3. **Non-reducer trainers: both checks are not-applicable and PASS (vacuously).**
   A trainer without a callable `predict_coefficients` (every scalar trainer: `polynomial`,
   `kriging`, `rbf`, and arbitrary third-party scalar trainers) records BOTH new checks as
   **PASS** with detail containing `"not a reduced-field trainer"` (e.g. "not a reduced-field
   trainer (no predict_coefficients) — not applicable"). `report.passed` stays `True` and
   `raise_for_status()` does not raise for these trainers.
   Teeth: assert the detail string names the not-applicable reason (discriminates a silent
   pass that forgot to run the check) AND that `report.passed is True` for a scalar built-in.

4. **Reuses existing `record()`/tolerance plumbing; report shape and public API unchanged.**
   - Both checks are registered via the existing `record(name, fn)` mechanism and appear as
     `CheckResult`s in `ConformanceReport.results`; failures surface through
     `failures`/`summary()`/`raise_for_status()` exactly like the other checks.
   - `check_trainer`'s **signature is unchanged** (no new required parameter); the field
     fixture is built internally. `conformance/__init__.py` exports are **unchanged** (no new
     public symbol — the checks are internal functions inside `check_trainer`).
   - The tolerance-band rule reuses the `deterministic` check's pattern: honour a declared
     `reproduction_tolerance`, never tighten below the suite floor `tolerance`.

5. **Core runs it on the built-in reducer; negative tests give it teeth; R4.1 gap closed.**
   In `tests/test_conformance.py`:
   - `EXPECTED_CHECKS` is extended with `reconstruction_round_trip` and `mesh_free` (now the
     full set the suite always records, for every trainer);
     `test_suite_runs_every_contract_check` therefore asserts all trainers record both.
   - `test_builtin_trainer_is_conformant` still holds: `ReducedFieldTrainer` PASSES both new
     checks **for real**; the scalar built-ins pass them as not-applicable. (The existing
     Kriging/Accelerate `deterministic`-only-failure escape hatch is untouched.)
   - New negative tests: a `_NonOrthonormalReducer` (basis scaled / un-canonicalized →
     `reconstruction_round_trip` FAIL) and a `_GeometryDependentReducer`
     (DOF-index-dependent → `mesh_free` FAIL). Each asserts the SPECIFIC check name is in
     `report.failures` and `report.passed is False`.
   - A pin that `predict_coefficients` returns `(n, r)` (the R4.1 deferred "no conformance
     backing" item) — satisfied by the AC 1 shape assertion exercised on the real builtin.

6. **Diff discipline.** Production diff confined to:
   - `src/surroforge/conformance/trainer.py` (two new internal check functions + their
     `record(...)` calls; a small field-fixture helper).
   Tests in `tests/test_conformance.py`. **Untouched:** `trainers/` (incl. `reduced_field.py`
   — it ALREADY exposes `predict_coefficients` + `reduction_artifacts`; read-only here),
   `engine.py`, `services.py`, `cli/`, `core/`, `validators/`, `persistence/`, `plugins/`,
   `conformance/__init__.py`, `pyproject.toml`. No `BaseTrainer` contract change; no new
   lineage fields; the `[rom]` packaging extra is **Story R4.4**, not here.

## Tasks / Subtasks

- [x] **Task 1 — reducer detection + field fixture helper (AC 1, 2, 3)**
  - [x] In `conformance/trainer.py`, add a module-private helper `_make_field_data(seed)`
    returning `train_test_split` of `(x, field)` where `field` is a small, deterministic,
    well-separated POD field: e.g. seed-`default_rng`, `n_dof = 8`, 3 QR-orthonormal spatial
    modes, ~40 samples, 2 inputs, smooth amplitudes (mirror the `_field_data` pattern in
    `tests/test_rom_inspect.py`). The fixture MUST yield `1 < r < n_dof` under the default
    energy threshold AND a rank that is stable under column permutation (well-separated mode
    energies — verify in a test fixture-sanity assert, same discipline as R4.2).
  - [x] Add `_is_reducer(trainer_or_instance)` → bool: `callable(getattr(obj,
    "predict_coefficients", None))`. (Works on the class; methods resolve.) This is THE
    reduced-field detection criterion; document it as the duck-typed ROM sub-contract.
  - [x] Decide the not-applicable record shape: when `not _is_reducer(trainer_cls)`, both
    new check fns return the string `"not a reduced-field trainer (no predict_coefficients)
    — not applicable"` (a returned detail string = PASS via `record`).

- [x] **Task 2 — `reconstruction_round_trip` check (AC 1)**
  - [x] Implement `_reconstruction_round_trip()`:
    1. If not a reducer → return the not-applicable detail.
    2. Build field data; fit a fresh `trainer_cls(**hyper)` on `(x_train, field_train)`.
    3. `c = t.predict_coefficients(x_test)`; require `reduction_artifacts` callable else
       `raise AssertionError("exposes predict_coefficients but not reduction_artifacts —
       incomplete reducer contract")`.
    4. `arts = t.reduction_artifacts()`; `basis = np.asarray(arts["basis"], float)`,
       `mean = np.asarray(arts["mean"], float)`; `r = basis.shape[1]`.
    5. Assert `c.shape == (x_test.shape[0], r)` (predict_coefficients shape contract).
    6. `field = c @ basis.T + mean`; `c_rt = (field - mean) @ basis`;
       `diff = float(np.max(np.abs(c_rt - c)))`; band `tol = max(tolerance, declared)`
       (reuse the `deterministic` check's `declared` resolution); assert `diff <= tol`.
    7. Return e.g. `f"projector round-trips within {diff:.2e} (r={r})"`.
  - [x] `record("reconstruction_round_trip", _reconstruction_round_trip)` — placed AFTER
    `declares_capabilities` (end of the list), so existing check ordering is undisturbed.

- [x] **Task 3 — `mesh_free` check (AC 2)**
  - [x] Implement `_mesh_free()`:
    1. If not a reducer → not-applicable detail.
    2. Build field data; fit `t1 = trainer_cls(**hyper)` on `(x_train, field_train)`;
       `F = np.asarray(t1.predict(x_test), float)` shape `(n_test, n_dof)`.
    3. Seeded fixed non-identity `perm = np.random.default_rng(seed).permutation(n_dof)`
       (guard: if `n_dof < 2` the check is trivially N/A — but the fixture guarantees
       `n_dof = 8`). Fit `t2 = trainer_cls(**hyper)` on `(x_train, field_train[:, perm])`;
       `F_perm = np.asarray(t2.predict(x_test), float)`.
    4. `diff = float(np.max(np.abs(F_perm - F[:, perm])))`; `tol = max(tolerance, declared)`;
       assert `diff <= tol`.
    5. Return e.g. `f"DOF-permutation equivariant within {diff:.2e}"`.
  - [x] `record("mesh_free", _mesh_free)` after `reconstruction_round_trip`.

- [x] **Task 4 — tests (AC 1–5)**
  - [x] `tests/test_conformance.py`: extend `EXPECTED_CHECKS` with the two names.
  - [x] Add a `_NonOrthonormalReducer(ReducedFieldTrainer)` (or a minimal hand-rolled reducer)
    whose `reduction_artifacts` returns a non-orthonormal basis (e.g. `basis * 2.0`) while
    keeping `predict_coefficients` consistent — assert `"reconstruction_round_trip"` ∈
    `{r.name for r in report.failures}` and `report.passed is False`. **Verify the trap:**
    confirm an unscaled (correct) basis makes the SAME assertion PASS (names the wrong impl it
    catches — vacant-pin rule (b) value-degeneracy).
  - [x] Add a `_GeometryDependentReducer` whose `predict`/reconstruction adds a fixed
    DOF-index ramp (`+ np.arange(n_dof)`), breaking permutation equivariance — assert
    `"mesh_free"` ∈ failures and `report.passed is False`. Confirm the base
    `ReducedFieldTrainer` passes `mesh_free` (the trap discriminates geometry-dependence, not
    all reducers).
  - [x] Add `test_scalar_trainer_rom_checks_not_applicable`: `check_trainer(PolynomialTrainer)`
    → both new checks PASS, detail contains `"not a reduced-field trainer"`, `report.passed
    is True`.
  - [x] `test_builtin_trainer_is_conformant` / `test_suite_runs_every_contract_check` already
    parametrize over BUILTINS incl. `reduced-field`; confirm green with the extended set.
  - [x] **Task 5 — gates (AC 6)**
  - [x] Full suite green; `ruff check src tests`, `mypy src`, `lint-imports` 3/3.
    Diff = AC 6 surfaces only.

### Review Findings (code-review, 2026-06-13)

Three adversarial layers (Blind Hunter, Edge Case Hunter, Acceptance Auditor). The Edge
Case Hunter verified every claim against the live code; several Blind-Hunter "CRITICAL"
items were disproved by execution (permutation is non-identity `[2 4 3 6 5 0 1 7]`; rank
stable at 3; round-trip 2.22e-15 — single-fit, bit-stable). One genuine soundness hole
remains, plus a cheap hardening.

- [x] [Review][Patch] **NaN/inf bypasses both new checks → silent PASS.** Both end with
  `diff = float(np.max(np.abs(...)))` then `if diff > tol`. `float("nan") > tol` is `False`
  (verified), so a non-finite basis/coeffs (round-trip) or field (mesh_free) — exactly the
  corruption these checks target — PASSES with detail "…within nan". Asymmetric with the rest
  of the suite, which guards finiteness (`_predict_2d`, `_fingerprint_participation`'s
  `np.all(np.isfinite)`). Fix: raise `AssertionError` when `diff` is non-finite, in both
  `_reconstruction_round_trip` and `_mesh_free`.
  [src/surroforge/conformance/trainer.py:_reconstruction_round_trip, _mesh_free]
- [x] [Review][Patch] **`mesh_free` permutation is not guarded against the identity for a
  non-default `seed`.** Default `seed=0` gives a non-identity perm (verified), but
  `check_trainer(seed=…)` is public API; an identity draw makes the check vacuously pass.
  Fix: fall back to a reversed permutation if `perm == arange(n_dof)`.
  [src/surroforge/conformance/trainer.py:_mesh_free]
- [x] [Review][Defer] **Round-trip tests basis orthonormality only — `mean` cancels and
  `predict()` is not cross-checked.** `recovered = (coeffs@Bᵀ+mean−mean)@B`, so a wrong
  `mean` (or a `predict()` that disagrees with `predict_coefficients()·Bᵀ+mean`) passes. AC 1
  scopes the check to `Bᵀ·B = I`, which it does correctly; cross-checking predict/mean is a
  deliberate suite-strengthening (needs its own negative fixture and would re-scope the
  geometry-dependent negative test). Deferred. [src/surroforge/conformance/trainer.py]
- [x] [Review][Defer] **Declared `reproduction_tolerance` has no upper bound.** A reducer
  declaring a huge tolerance passes both checks trivially (grades its own exam). Pre-existing
  and consistent with `_deterministic`/`declares_capabilities` (finite & ≥0 enforced, no
  ceiling); an upper-bound policy is a suite-wide decision, not R4.3-specific. Deferred.
  [src/surroforge/conformance/trainer.py:_resolve_tolerance]

Dismissed as noise (with reason): round-trip `1e-9` "flaky" — verified 2.22e-15, single-fit
bit-stable (not subject to the two-fit BLAS divergence that `mesh_free` has); `_make_field_data`
rank-3 "unasserted/unstable" — verified stable spectrum (16.2/10.0/2.1/0 → r=3, permutation-
invariant); "not applicable reported as PASS" — mandated by AC 3/AC 4 (`EXPECTED_CHECKS` is an
exact-equality set, so the checks must always record, and non-reducers must not fail);
`np.ascontiguousarray` "cargo-cult" — harmless and consistent with the suite's existing style;
`test_reduced_field.py` change — a justified, disclosed necessary regression fix (the
geometry/Accelerate escape), pre-authorized by Dev Notes; `test_builtin_trainer_is_conformant`
escape "not widened" — verified correct (Kriging there is scalar → `mesh_free` is N/A).

## Dev Notes

### Reducer detection (the duck-typed ROM sub-contract)

A **reduced-field trainer** is detected by a callable `predict_coefficients` attribute —
nothing more. This is already how `services` duck-types the field-prediction path (Story
R4.1) and how the engine duck-types `reproducibility_diagnostics`/`reduction_artifacts`
(R2.3/R4.1) — the spine theorem #36 means there is no `is_reducer` flag or base class to key
on. The round-trip check additionally requires `reduction_artifacts()` (to read the basis);
its absence on a `predict_coefficients`-exposing trainer is a conformance FAIL ("incomplete
reducer contract"), not a skip — the two hooks are one contract.

[Source: src/surroforge/trainers/reduced_field.py:159 `predict_coefficients`, :174
`reduction_artifacts`; src/surroforge/services.py predict path duck-types
`predict_coefficients`.]

### Field fixture (why the ROM checks build their own data)

`check_trainer._make_data` produces a **single-output** target (`y` shape `(n, 1)`) so SMT
methods stay in their MVP envelope. A 1-DOF "field" is degenerate (rank 1, nothing to
permute), so the round-trip and no-mesh checks are meaningless on it. The two ROM checks
therefore build their own small multi-DOF field via `_make_field_data(seed)` and fit a fresh
trainer — leaving the scalar checks and their data path **byte-identical** (the
"scalar path unchanged" discipline, PRD-ROM §5/NFR). Mirror the proven fixture in
`tests/test_rom_inspect.py:_field_data` (seed-`default_rng`, QR-orthonormal modes, smooth
polynomial amplitudes) but keep `n_dof` small (≈8) so two extra fits stay sub-second.

**[GOTCHA] Rank stability under permutation.** Use well-separated mode energies so the energy
threshold resolves the SAME rank for `y` and `y[:, perm]`; a near-degenerate spectrum could
resolve a different rank after permutation (BLAS noise at the energy cut), making `F[:, perm]`
and `F_perm` different shapes/ranks. The synthetic 3-mode field with decaying amplitudes
(e.g. 5.0/2.0/1.8 like test_rom_inspect) is comfortably separated.

### No-mesh: why behavioural (permutation equivariance), not signature introspection

"The reducer reads no coordinates/connectivity" (#53) is an **invariant about behaviour**, and
the only faithful test is behavioural: a mesh-free reducer treats DOFs as exchangeable opaque
features, so permuting the snapshot columns permutes the reconstructed field identically
(`F_perm == F[:, perm]`). In exact arithmetic this is exact: permuting columns permutes the
POD spatial modes (rows of the basis) while leaving the modal **coefficients** `c =
fluctuations @ basis` invariant (`P` cancels: `(YP)(PᵀV) = YV`), so the inner surrogates see
identical targets and `F_perm = c @ (PᵀV)ᵀ + Pᵀμ = P(c Vᵀ + μ)`. Any geometry-dependent step
(reading coordinates, a connectivity-weighted norm) would break this. A
signature/introspection check ("does `fit` accept a `coords` kwarg?") has **no teeth** — a
plugin can read geometry from anywhere — so it must not be the implementation.

### Honouring declared tolerance (reuse the `deterministic` pattern)

Both checks compare floats across independent SVD fits, so they must honour a declared
`reproduction_tolerance` exactly like `_deterministic` (trainer.py:202-208): `declared =
getattr(trainer_cls, "reproduction_tolerance", tolerance); tol = max(tolerance,
float(declared))` (guard `bool`/non-numeric). `ReducedFieldTrainer.reproduction_tolerance ==
1e-9`. The round-trip itself (pure orthonormality) is typically ~1e-14, well inside
`DEFAULT_TOLERANCE = 1e-9`; the permutation check re-runs SVD, so the 1e-9 band matters there.
The same BLAS-bit-reproducibility caveat as the rest of the suite applies (Apple Accelerate;
guarded in `tests/conftest.py` via `blas_is_bit_reproducible` — see the existing Kriging
escape in `test_builtin_trainer_is_conformant`). If a permutation flake ever appears on
Accelerate for `reduced-field`, extend that same escape hatch rather than loosening the band.

### Existing assets and integration points (verified, 2026-06-13)

- `src/surroforge/conformance/trainer.py` — `check_trainer` (line 90), the `record()` closure
  (line 115), `_make_data` (line 70), `_deterministic`'s declared-tolerance resolution
  (lines 202-208), and the `record(...)` call list (lines 298-307). Append the two new
  `record(...)` calls at the end (after `declares_capabilities`).
- `src/surroforge/conformance/__init__.py` — exports `CheckResult`, `ConformanceReport`,
  `check_trainer`, `ConformanceError`. **Do not add exports** (new checks are internal).
- `tests/test_conformance.py:30` `BUILTINS` (incl. `reduced-field`), `:39` `EXPECTED_CHECKS`,
  `:53` `test_builtin_trainer_is_conformant` (with the Accelerate/Kriging escape at :56-63),
  `:68` `test_suite_runs_every_contract_check`, and the `_OneDimTrainer`/`_BareErrorTrainer`
  family (:77+) — the pattern to follow for the two new deliberately-broken reducers.
- `src/surroforge/trainers/reduced_field.py` — `predict_coefficients` (:159, returns
  `(n, r)`), `reduction_artifacts` (:174, returns read-only `basis (n_dof,r)` / `mean (1,n_dof)`
  / `singular_values` views), `predict` (:153, `c @ basis.T + mean`). **Read-only** in this
  story. NB: `reduction_artifacts` returns **read-only views** (`writeable=False`) — the check
  only reads them (`@`, `np.asarray` copies on compute), so no mutation issue; do not write
  through them.
- `src/surroforge/errors.py` — `ConformanceError` (already raised by `raise_for_status`); no
  new error type needed.

### Constraints that bite

- **[MUST] Framework-agnostic suite, no pytest dependency in `conformance/`** — the checks
  live in `src/surroforge/conformance/trainer.py` and use only numpy + stdlib (the file
  already does). The negative-test trainers live in `tests/`, not in the suite.
- **[MUST] `check_trainer` never raises on a failed check** — failures are collected via the
  `record()` try/except. Raise `AssertionError` *inside* a check fn to signal FAIL (the
  established idiom); never let a check fn raise outside `record`.
- **[MUST] Non-reducers must not fail the new checks** — they record PASS (not-applicable).
  Otherwise every scalar trainer (and every existing conformance test) breaks.
- **[CONV] `EXPECTED_CHECKS` is an exact-equality set** (`test_suite_runs_every_contract_check`
  asserts `==`), so the two new checks MUST be recorded for EVERY trainer (reducer or not) —
  the not-applicable branch is what keeps that invariant true.
- **[GOTCHA] Holdout vs. final model does NOT apply here** — the conformance suite drives the
  trainer directly (its own fit), not the engine's holdout/final split; there is no lineage,
  no reproduce verdict. The R4.2/R2.3 holdout caveat is irrelevant to this story.
- **Vacant-pin discipline (project-context [GOTCHA]):** (a) negative tests assert the SPECIFIC
  failing check name (not just `report.passed is False` — which any other broken check would
  also satisfy); (b) the round-trip pin uses a basis where correct vs. broken differ at
  rendered precision (scaled basis), not a shape-only check; (c) the not-applicable pin
  asserts the reason string, not mere presence.

### Scope boundaries (what R4.3 does NOT do)

- No `[rom]` packaging extra / lazy-import install hint — that is **Story R4.4**.
- No new public conformance API, no `check_reducer` entry point — the two checks fold into the
  existing `check_trainer`/`ConformanceReport`.
- No node-subsampled "probe" reconstruction — v1's "probes" are the full reconstructed field
  (R2.5); the round-trip is the full-field projector identity.
- No mesh readers (meshio/pyvista) — deferred to Phase 2 (PRD-ROM §4.10); mesh-freeness is
  precisely why none are needed.
- No change to `reduced_field.py` — it already satisfies the contract being tested.

### Previous-story intelligence (R4.2, code-reviewed 2026-06-13)

- **Vacant-pin teeth are the recurring review focus.** R4.2's review confirmed all three trap
  classes (aliasing / value-degeneracy / presence-without-structure) must be actively
  discriminated. Here: assert the specific check NAME in `failures`, use a scaled basis (not a
  shape check) for the round-trip, and assert the not-applicable reason string.
- **Treat duck-typed disk/contract data defensively, but** — unlike R4.2's lineage reads — the
  conformance suite's `record()` already converts any exception into a clean FAIL, so an
  `AssertionError` inside a check is the correct, sufficient failure channel (no `ProjectError`
  wrapping here).
- **`ruff`/`mypy`/`lint-imports` are real gates** (R4.2 shipped a `review` status with a
  failing `ruff` F401). Run all three before marking done; no unused imports.
- Commit-at-done discipline: this story lands as ONE commit. Repo has no git identity —
  commit with `git -c user.name="Georg Babij" -c user.email="g.babij@gmail.com"` (see the
  R4.2 commit `77fea4e`).

### Project Structure Notes

- Production diff: ONE file (`src/surroforge/conformance/trainer.py`).
- Test diff: `tests/test_conformance.py` (extend EXPECTED_CHECKS, add 2 broken reducers + a
  not-applicable test).
- Tracking: this story file + `sprint-status.yaml`.

### References

- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md#Story R4.3 (lines 325-339)] —
  AC wording: `coeffs→field→coeffs` within `DEFAULT_TOLERANCE`; no-mesh invariant; reuse
  `record()`/tolerance plumbing; appears in `ConformanceReport`. Also AR-R5 (line 52), FR-R8
  (line 35).
- [Source: _bmad-output/planning-artifacts/prd-field-rom-v1.md §4.9 (line 162-164)] —
  Konformität: reconstruction probe + no-mesh check, "Mesh-frei ist eine getestete Invariante,
  kein Feature (#53)"; §6 table (line 194): `conformance/trainer.py` is the only production
  surface.
- [Source: src/surroforge/conformance/trainer.py] — `check_trainer`, `record`, `_deterministic`
  tolerance pattern, the `record(...)` registration list.
- [Source: src/surroforge/trainers/reduced_field.py] — `predict_coefficients`,
  `reduction_artifacts`, `predict` (the reducer contract under test).
- [Source: tests/test_rom_inspect.py:_field_data] — the field-fixture pattern to mirror.
- [Source: deferred-work.md "code review round 2 of r4-1" — "Duck-typed `predict_coefficients`
  has no conformance backing … R4.3's reconstruction round-trip conformance probe is the
  natural home — add a `predict_coefficients` shape/identity check there."] — closed by AC 1.
- [Source: _bmad-output/project-context.md] — vacant-pin [GOTCHA], sober-tone [CONV], the
  conformance suite ships as library code under `src/surroforge/conformance/` (not `tests/`),
  dependency direction, `SurroForgeError`-only discipline.

## Dev Agent Record

### Agent Model Used

claude-opus-4-8[1m]

### Debug Log References

- ruff B009 ("replace getattr with attribute access") fired on a 2-arg
  `getattr(t, "predict_coefficients")`; switched to the 3-arg form + a `callable()` guard
  (mirrors the existing `reduction_artifacts` pattern), which also narrows the type for mypy.
- mypy "None not callable" on the 3-arg `getattr(..., None)` until the combined
  `if not (callable(...) and callable(...))` guard narrowed both hooks.
- `test_reduced_field.py::test_conformance_with_smt_inner[kriging]` failed on Apple
  Accelerate: `mesh_free` re-fits the trainer (a second SVD/Kriging-MLE), which is not
  bit-reproducible on Accelerate — the same class as the existing `deterministic` escape.
  Extended that test's allowed-failure set to `{"deterministic", "mesh_free"}` (CI/OpenBLAS
  is bit-reproducible, so both pass there). `reconstruction_round_trip` is single-fit and
  stays bit-stable, so it is NOT in the escape set.

### Completion Notes List

- **Task 1:** Added `_make_field_data(seed)` (8-DOF, 3 QR-orthonormal modes, decreasing
  amplitudes → stable rank 3 < 8, permutation-invariant rank), `_is_reducer` (duck-types a
  callable `predict_coefficients`), and `_resolve_tolerance` (declared-tolerance band, mirrors
  the `deterministic` rule). Scalar `_make_data` path untouched.
- **Task 2:** `reconstruction_round_trip` check — `coeffs→field→coeffs` via the basis from
  `reduction_artifacts`; asserts `predict_coefficients` shape `(n, r)` (closes the R4.1
  deferred item) and `max|c_rt − c| ≤ tol`. Real round-trip on the builtin: 2.22e-15.
- **Task 3:** `mesh_free` check — DOF-permutation equivariance of the reconstructed field
  (behavioural, not signature introspection). Real diff on the builtin: 7.55e-15.
- **Both checks** record via the existing `record()` plumbing after `declares_capabilities`;
  non-reducers record PASS with a "not a reduced-field trainer" detail (keeps `EXPECTED_CHECKS`
  exact-equality and `report.passed` true). `check_trainer` signature + `__init__.py` exports
  unchanged.
- **Task 4:** Tests — extended `EXPECTED_CHECKS`; `_NonOrthonormalReducer` (scaled basis →
  `reconstruction_round_trip` FAIL, `mesh_free` unaffected) and `_GeometryDependentReducer`
  (DOF-index ramp → `mesh_free` FAIL, round-trip unaffected), each pinning the SPECIFIC check
  name; a trap-guard that the correct builtin passes both; a scalar not-applicable test.
- **Task 5:** ruff/mypy/lint-imports(3/3) clean; full suite 326 passed, 3 skipped.

### File List

- src/surroforge/conformance/trainer.py (modified — `_make_field_data`, `_is_reducer`,
  `_resolve_tolerance` helpers + `reconstruction_round_trip` & `mesh_free` checks)
- tests/test_conformance.py (modified — EXPECTED_CHECKS + 4 new tests / 2 broken reducers)
- tests/test_reduced_field.py (modified — Accelerate escape set extended to include `mesh_free`)
- _bmad-output/implementation-artifacts/r4-3-reconstruction-round-trip-no-mesh-conformance-probe.md
- _bmad-output/implementation-artifacts/sprint-status.yaml

### Change Log

- 2026-06-14: code-review complete (3 adversarial layers). 2 patches applied: (P1) NaN/inf
  finiteness guard in both `reconstruction_round_trip` and `mesh_free` — a non-finite `diff`
  previously slipped through `nan > tol == False`; pinned by a new `_NaNBasisReducer` test.
  (P2) `mesh_free` falls back to a reversed permutation if the seeded draw is the identity
  (robust for any caller seed). 2 defers recorded (mean/predict not cross-checked; declared
  tolerance has no ceiling). 327 passed, 3 skipped; ruff/mypy/lint-imports clean. Status → done.
- 2026-06-13: dev-story complete. 3 code/test files changed (conformance/trainer.py,
  test_conformance.py, test_reduced_field.py). 326 passed, 3 skipped; ruff/mypy/lint-imports
  clean. Status → review.
- 2026-06-13: Story created (create-story). Key findings: the two ROM checks fold into the
  existing `check_trainer` via `record()`, gated on a callable `predict_coefficients`
  (the duck-typed reducer marker); non-reducers record both as not-applicable PASS to keep
  `EXPECTED_CHECKS` exact-equality and `report.passed` true. Round-trip = basis orthonormality
  (`Bᵀ·B=I`) via `coeffs→field→coeffs`; no-mesh = DOF-permutation equivariance of the
  reconstructed field (behavioural, real teeth — signature introspection rejected). ROM checks
  build their own multi-DOF field fixture (scalar `_make_data` path untouched). Closes the
  R4.1 `predict_coefficients`-shape deferred item. Status → ready-for-dev.
