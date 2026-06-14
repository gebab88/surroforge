---
baseline_commit: 3397ce2d25c7593ab35bd16c529491857a33d693
context:
  - _bmad-output/planning-artifacts/epics-field-rom-v1.md
  - _bmad-output/planning-artifacts/prd-field-rom-v1.md
  - _bmad-output/project-context.md
---

# Story R4.4: `surroforge[rom]` packaging extra

Status: done

## Story

As a user,
I want ROM to install as an optional extra without bloating the base install,
So that the Core stays lean and permissively licensed.

## Acceptance Criteria

**Given** a base `pip install surroforge`
**When** I `pip install 'surroforge[rom]'`
**Then** ROM becomes available with lazy imports (mirroring `[smt]`/`[onnx]`), adding **no new
heavy dependency** in v1 (SVD from scipy) and no GPL/LGPL dependency (NFR8)
**And** requesting `reduced-field` without the extra installed yields an actionable install
hint, not an `ImportError`.

## The v1 tension (and how it is resolved)

The two clauses of the AC pull in opposite directions, and PRD-ROM §4.10 reveals why:

> `surroforge[rom]`-Extra: numpy/scipy (bereits Core-Deps; v1 braucht **keine neue schwere
> Abhängigkeit** — SVD aus scipy). meshio/pyvista **deferred** auf Phase 2. Lazy-Imports,
> gleiches Muster wie `[smt]`/`[onnx]`.

ROM's reduction runs on the numpy/scipy linear-algebra stack, both already Core deps (numpy is
a hard base dependency; scipy is a base dependency AND a transitive one via scikit-learn). So
the `[rom]` extra adds **nothing** in v1, and the "install hint when the extra is missing" path
cannot fire in any real install — you cannot install a working SurroForge without scipy.

PRD-ROM §4.10 nonetheless instructs "Lazy-Imports, gleiches Muster wie `[smt]`/`[onnx]`". The
**Faithful-pattern** decision (lead, 2026-06-14) honours that instruction literally: ship the
lazy-guard *shape* now (tested via a simulated missing backend) so (a) the AC is satisfied to the
letter, (b) the canonical ROM install target `pip install 'surroforge[rom]'` exists and is
documented, and (c) the Phase-2 mesh readers (meshio/pyvista) have a wired home — they will join
this exact guard and populate the (currently empty) `[rom]` extra. The guard is transparently
commented and recorded in `deferred-work.md` as a v1-non-firing seam.

The `np.linalg.svd` numeric path is **not** touched: the field fingerprint is frozen on it (R2.x),
so the guard checks scipy's *presence* (the PRD-named SVD backend) without changing any number.

## Tasks / Subtasks

- [x] **Task 1 — declare the `[rom]` extra (AC clause 1).**
  - [x] `pyproject.toml` `[project.optional-dependencies]`: `rom = []` with a comment stating
    v1 adds no new heavy dependency (numpy/scipy already Core), that the extra is the documented
    install target + Phase-2 mesh-reader home, and that the trainer's lazy guard provides the
    install hint. `[all]` unchanged (rom contributes no package in v1; noted in its comment).
  - [x] `reduced-field` entry point stays registered unconditionally (mirrors kriging/rbf): the
    method is always discoverable; the backend check is at fit time.

- [x] **Task 2 — lazy-import guard + actionable hint (AC clause 2).**
  - [x] `src/surroforge/trainers/reduced_field.py`: module-level `_require_rom_support()` —
    `try: import scipy.linalg except ImportError: raise TrainerError("Reduced-field ROM requires
    the 'rom' extra. Install it with: pip install 'surroforge[rom]'")`. Mirrors `kriging.fit`'s
    `try: import smt` exactly (same message shape, `SurroForgeError`-only discipline).
  - [x] Called as the FIRST statement of `fit()` (before the lazy `get_trainer` import and the
    SVD), so the hint fires on `surroforge train` (the "requesting reduced-field" path) and the
    engine surfaces it as a clean error (errors-as-a-feature, Story 6.2). `predict` /
    `predict_coefficients` / `reduction_artifacts` on a loaded model are NOT guarded (the model
    is already fitted; numpy suffices — consistent with kriging guarding only `fit`).

- [x] **Task 3 — docs (AC clause 1, NFR8).**
  - [x] `README.md` + `docs/install.md`: add the `pip install 'surroforge[rom]'` line and a
    `rom` row in the extras table (`Unlocks: Field ROM (reduced-field, POD field reduction)`,
    `Packages: — (uses core numpy/scipy)`). The existing GPL/LGPL-free licensing note already
    covers NFR8 (numpy/scipy are BSD).

- [x] **Task 4 — test (AC clause 2, teeth).**
  - [x] `tests/test_reduced_field.py::test_missing_rom_backend_yields_install_hint`:
    `monkeypatch.setitem(sys.modules, "scipy", None)` makes the guard's `import scipy.linalg`
    raise; assert `t.fit(...)` raises `TrainerError` matching `surroforge\[rom\]` — proving an
    actionable hint, NOT a raw `ImportError`. (Closes the analogous gap the Story 6.3 deferred
    item flagged for the smt path — for the ROM path specifically.)

- [x] **Task 5 — gates (AC, project-context).**
  - [x] Full suite green; `ruff check src tests`, `mypy src`, `lint-imports` (3/3 contracts).

## Review Findings (post-hoc adversarial review, 2026-06-14)

Ran during the Epic R4 retrospective to restore the review rhythm R4.4 had skipped (it
landed on gates + self-review only). Three parallel adversarial layers — Blind Hunter
(correctness/robustness), Edge Case Hunter (branch/boundary), Acceptance Auditor (ACs +
diff discipline) — each verifying claims against the live runtime. **No Critical/Major/Minor
findings; all 7 AC checks PASS.** Two doc-only patches applied; the rest dismissed or deferred.

- [x] [Review][Patch] **Guard docstring undersold scipy as incidental.** Said the guard
  "checks scipy's presence without touching the numeric path", implying scipy is a token gate
  while the numpy SVD is the real work. The Auditor proved otherwise: with scipy absent the
  first raw failure is `import scipy.sparse` from the **sklearn polynomial inner** (reached at
  `reduced_field.py:159`), so scipy is a *genuine* fit-time requirement (every default inner
  pulls it in via scikit-learn/SMT). Refined the docstring to state this — the gate preempts a
  real `scipy.sparse` ImportError, and the numpy SVD path is separately frozen.
  [src/surroforge/trainers/reduced_field.py `_require_rom_support`]
- [x] [Review][Patch] **Test teeth depend on an invisible import-ordering precondition.** The
  test pins the guard only because surroforge+sklearn+scipy are imported at module load before
  `sys.modules["scipy"]=None`; without the guard the sklearn inner's `import scipy.sparse`
  raises a raw `ModuleNotFoundError` (≠ `TrainerError`), which is exactly what gives the test
  teeth. Added a docstring note making this explicit (both reviewers independently flagged it).
  Verified empirically by neutralizing the guard → test fails as designed.
  [tests/test_reduced_field.py::test_missing_rom_backend_yields_install_hint]
- [x] [Review][Defer] **scipy ships GPL/LGPL license *text* (bundled reference code) in its
  metadata.** The CI license job (`.github/workflows/ci.yml`) reads pip-licenses' top-level
  `License` *classifier* field, which reports scipy as `BSD License`, so NFR8 enforcement is
  unaffected today. But a future switch of the scanner to scan full license *text* would
  false-positive on scipy. Pre-existing (scipy is a base dep regardless of R4.4), surfaced by
  this audit; recorded in `deferred-work.md`. Revisit only if the license job's detection
  method changes.
- Dismissed (with reason): **guard masks a genuinely-broken scipy as the rom hint** (Blind
  Hunter Nit) — byte-identical to the ratified `[smt]`/`[onnx]` pattern in `kriging.fit`
  (`except ImportError → install hint`); scipy is a hard Core dep so unreachable; tightening it
  is a suite-wide policy for the whole guard family, not R4.4. **AC7 vacuous-firing** — confirmed
  an honest, fully documented trade-off (Faithful-pattern), not a blocker.

## Dev Notes

### Why gate on scipy (not numpy) and keep the extra empty

- **scipy** is the PRD-named SVD backend (§4.10 "SVD aus scipy") and the natural anchor of the
  scientific stack the Phase-2 mesh readers depend on, so it is the honest gate target. The v1
  reduction itself calls `np.linalg.svd` (frozen fingerprint, R2.x) — the guard checks scipy's
  presence without touching that path. Gating on numpy would be misleading: a missing numpy means
  the whole package is unimportable, not "the ROM extra is absent".
- **`rom = []`** rather than `rom = ["scipy>=1.10"]`: scipy is already a hard base dependency, so
  listing it in the extra would be a redundant declaration that falsely implies `[rom]` is what
  pulls scipy in. Empty is the truthful statement that v1 ROM needs nothing beyond base; the extra
  is a named install target + Phase-2 home. The trade-off (the guard cannot fire in a real v1
  install) is recorded in `deferred-work.md`.

### Scope boundaries (what R4.4 does NOT do)

- No meshio/pyvista, no mesh readers — deferred to Phase 2 (PRD-ROM §4.10). They are the first
  real packages the `[rom]` extra will carry, and they will gate through `_require_rom_support()`.
- No change to the numeric reduction (`np.linalg.svd`), the fingerprint, lineage, the engine,
  services, the conformance suite, or the registry — `reduced-field` already works on base.
- No new public API symbol.

### Constraints that bit

- **[MUST] `SurroForgeError`-only:** the guard raises `TrainerError` (a `SurroForgeError`), never
  lets the `ImportError` propagate — same discipline as kriging/polynomial-onnx.
- **[CONV] `ruff` F401:** `import scipy.linalg` is a presence-check side-effect import, so it
  carries `# noqa: F401` (the import is intentionally "unused").
- **[GOTCHA] sys.modules patch:** `sys.modules["scipy"] = None` makes `import scipy.linalg` raise
  `ImportError` even if `scipy.linalg` was cached earlier, because the import machinery re-resolves
  the parent `scipy` (which is `None`). monkeypatch auto-restores it.

## Dev Agent Record

### Agent Model Used

claude-opus-4-8[1m]

### Completion Notes List

- **Task 1:** `rom = []` extra declared with a comment documenting the no-new-dependency v1
  reality and the Phase-2 mesh-reader home. `reduced-field` entry point unchanged (always
  registered).
- **Task 2:** `_require_rom_support()` lazy guard added; called first in `fit()`. Raises the
  actionable `pip install 'surroforge[rom]'` `TrainerError`, mirroring `[smt]`/`[onnx]`. The
  `np.linalg.svd` path is unchanged (frozen fingerprint).
- **Task 3:** `[rom]` line + extras-table row added to `README.md` and `docs/install.md`.
- **Task 4:** `test_missing_rom_backend_yields_install_hint` simulates a missing scipy via
  `sys.modules` and asserts the actionable `TrainerError` (not `ImportError`).
- **Task 5:** 329 passed, 2 skipped (smt-inner cases skip on the local py3.13 venv);
  ruff/mypy/lint-imports(3/3) clean.

### File List

- pyproject.toml (modified — `rom = []` extra + comment)
- src/surroforge/trainers/reduced_field.py (modified — `_require_rom_support()` guard + fit() call)
- tests/test_reduced_field.py (modified — `import sys` + install-hint test)
- README.md (modified — `[rom]` install line)
- docs/install.md (modified — `[rom]` install line + extras-table row)
- _bmad-output/implementation-artifacts/r4-4-surroforge-rom-packaging-extra.md
- _bmad-output/implementation-artifacts/deferred-work.md (R4.4 note)
- _bmad-output/implementation-artifacts/sprint-status.yaml (r4-4 + epic-r4 → done)

### Change Log

- 2026-06-14: dev-story complete. Faithful-pattern decision (lead): ship the lazy-guard shape now,
  tested via a simulated missing backend, even though `[rom]` is empty in v1 and the guard cannot
  fire in a real install. 5 source/doc files + tracking. 329 passed, 2 skipped; ruff/mypy/
  lint-imports clean. Epic R4 complete. Status → done.
</content>
</invoke>
