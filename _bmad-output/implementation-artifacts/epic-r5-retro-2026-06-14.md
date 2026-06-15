# Epic R5 Retrospective — Solver-Native & Mesh Readers (Phase 2)

- **Date:** 2026-06-14
- **Epic:** R5 — Solver-Native & Mesh Readers (the first Phase-2 epic, on the reserved `[rom]` seam)
- **Stories:** r5-1 (OpenFOAM field ingestion — `surroforge.data.openfoam`, ASCII reader; commit
  `b60c7a0`), r5-2 (`surroforge data add-openfoam` CLI + `meshio` binary/mesh readers; commit
  `6b33bbd`) — both `done`
- **Facilitator:** Amelia (Developer) · **Participants:** Winston (Architect), Murat (Test
  Architect), Georg (Project Lead)
- **Previous retro:** epic-r4-retro-2026-06-14.md
- **Milestone:** Phase 2 is open and producing. The lead's release call (this retro): **cut a
  single `v0.2.0`** covering Phase-1 + Field ROM v1 (R1–R4) + Phase-2 R5.

## Epic Summary

| Metric | Value |
|---|---|
| Stories completed | 2/2 (100%) |
| Test-suite growth | 350 (post-R4) → 365 (r5-2 dev) → **371 collected** (365 passed, 6 skipped) post-review |
| Skips | 6 = 3 SMT (kriging/rbf) + 3 meshio (not installed locally) — both env-gated |
| Gates | green throughout (ruff, mypy 42 files, import-linter 3/3) |
| Reviews | r5-1: 3-layer, 1 decision + 11 patches, 5 dismissed · r5-2: 3-layer, 1 decision + **8 patches** (incl. **2 live-verified bugs**), 6 dismissed |
| Commits | 2 clean single-commit stories (`b60c7a0`, `6b33bbd`) — **committed, NOT yet pushed** |
| Production-diff discipline | r5-1: new `data/openfoam.py` + export + example + tests · r5-2: `cli/main.py` + `data/openfoam.py` + `pyproject` + tests; **engine/registry/trainers/config untouched** |
| Production incidents | 0 |

R5 delivered solver-native ingestion: a mesh-free ASCII OpenFOAM reader promoted into the package
(r5-1), then a `data add-openfoam` CLI and a `meshio`-backed binary/mesh reader that finally fills
the `surroforge[rom]` extra (r5-2). Every case sweep becomes the ordinary `(n_samples, n_dof)`
matrix the unchanged `reduced-field` pipeline already trains on.

## What Went Well

- **The spine theorem extends to ingestion, not just trainers.** A new input source shipped as a
  `data`-leaf reader + a thin CLI adapter + an optional extra — **zero** change to engine,
  registry, trainers, or config (AR-R1). The same bet that made ROM "one trainer" makes a new
  solver format "one reader".
- **Prototype-first de-risked the library.** The runnable example landed first and proved the
  parser end-to-end before it was promoted into `src/` (r5-1) — a clean spike→library path that
  kept the public reader honest.
- **Review cadence held conservatively — and paid for itself.** Both stories took the full 3-layer
  adversarial review even though r5-2 ("a thin CLI wrapper") qualified for the R4 risk-tier skip.
  The r5-2 review caught **two real bugs** — a dataset-clobber on name collision and integer
  case-name coercion — both **verified live** and both masked by happy-path fixtures.
- **The faithful-pattern seam was used exactly as designed.** r4-4 shipped `[rom]` *empty* (no dep
  was needed then); r5-2 filled it with `meshio` precisely when a genuine binary-reader need
  arrived, behind the same lazy-import + install-hint discipline as `[smt]`/`[onnx]`.

## Challenges & Friction

- **Happy-path fixtures hid genuine bugs.** r5-2's first tests used only the convenient layout
  (string case names, no pre-existing dataset). The clobber and the integer-case-name bug live in
  the *inconvenient* layout; only adversarial live-verification surfaced them. This is the
  vacant-pin lesson (R3) generalised one level: a test can be non-vacant yet still only exercise
  the easy input class.
- **A direction pivot left the v1 release uncut.** The R4 retro set "v1 release prep" as the next
  direction and deferred Phase 2; the lead instead opened Phase 2 (R5) to meet a real OpenFOAM
  need. Legitimate — but the release (R4 action item 3) was never executed.
- **R5 moved the release surface.** `[rom]` is no longer empty (it now carries `meshio`), so
  `RELEASING.md`'s "`surroforge[rom]` is empty in v1 by design" note and the v1 framing went stale
  mid-epic. Surfaced as the significant discovery below and resolved by the lead's `v0.2.0` call.

## Key Insights

1. **AR-R1 covers adapters, not just trainers.** Adding an input source or a CLI surface is a
   leaf-plus-adapter change; if a new reader reaches into the engine, that's the smell, not the
   spec. Worth defending as hard as the trainer seam.
2. **Fixture diversity is the next rung of the vacant-pin discipline.** Beyond "does this assertion
   catch the wrong wiring", ask "does this fixture exercise the *awkward* input" — integer IDs,
   name collisions, ragged data. The convenient layout passing proves little.
3. **The risk-tier review allowance should be spent rarely.** r5-2 was the archetypal "low-risk"
   story and still harbored two real bugs. Full review stays the default; skipping is the
   exception that must be argued, not the reflex.

## Previous Retro Follow-Through (R4 action items)

1. ✅ **Post-hoc r4-4 adversarial review** — completed in the R4 retro.
2. ✅ **Risk-tiered review, consciously chosen** — applied by *not* invoking the skip: both R5
   stories took the full 3-layer pass. The two r5-2 bugs retroactively justify the conservative
   default (codified in Action Item 5).
3. ⚠️ **v1 release preparation** — **not executed**; superseded by the Phase-2 pivot and now
   re-scoped to `v0.2.0` (Action Item 1). Carried forward, not dropped.
4. ✅ **Commit-at-done + create→dev→review pipeline** — held; two clean single-commit stories,
   each through the full pipeline.

## Significant Discovery → resolved by the lead

**Epic-update alarm: YES — for the release plan, not for any downstream epic.** R5 added a
dependency to `surroforge[rom]` (`meshio`), contradicting `RELEASING.md` ("[rom] empty in v1 by
design") and the v1 framing. **Lead decision:** cut a single **`v0.2.0`** that includes Phase-1 +
Field ROM v1 (R1–R4) + Phase-2 R5, and update the release artifacts to the non-empty-`[rom]`
reality (rather than tag a pre-R5 v0.1.0 first). No downstream plan is invalidated.

## Action Items

1. **[Critical path → release] Cut `v0.2.0` (Phase-1 + R1–R4 + R5).** Bump `pyproject` version
   `0.1.0`→`0.2.0`; move `CHANGELOG [Unreleased]` into a `## [0.2.0]` section; correct
   `RELEASING.md` (the `[rom]` extra now carries `meshio` — it is no longer empty); confirm a
   full-matrix CI green and the PyPI-readiness checks. Owner: dev/lead.
2. **[CI/test] Exercise the `meshio` path in CI.** The 3 `read_mesh_field` happy/tensor tests skip
   locally (meshio absent); ensure the CI `[all]`/`[rom]` job installs `meshio` so the success and
   component-validation paths actually run — mirroring how SMT is exercised in CI but skipped
   locally. Owner: dev.
3. **[Standing] Fixture diversity in reader/parser tests.** Any new reader/parser test must include
   at least one *adversarial-layout* case (integer IDs, name collision, ragged input), not only the
   convenient layout. Extends the R3 vacant-pin rule; add to project-context Testing. Owner: dev.
4. **[Outward-facing] Push `b60c7a0` + `6b33bbd` to `origin/main`** (R1–R4 were pushed; R5 is local
   only). Lead's call on timing. Owner: lead.
5. **[Standing] Full 3-layer review stays the default.** The R4 risk-tier skip remains available but
   is the argued exception, not the reflex — R5 evidence: a "thin wrapper" carried two real bugs.

## Next Direction: cut v0.2.0; further Phase-2 is optional

**There is no Epic R6 defined.** R5 closes the currently-planned Phase-2 slice (OpenFOAM ingestion).
The natural next gate is the **`v0.2.0` release** (Action Item 1). Beyond that, Phase-2 has open
roadmap (more mesh/result formats, commercial ROM surfaces on the same seam) but nothing is
committed — define a new epic when a concrete need arrives, the way R5 itself was opened.

## Readiness Assessment

| Dimension | Status |
|---|---|
| Tests & quality | ✅ 371 collected, 0 failures (365 passed, 6 env-gated skips); gates clean (ruff, mypy, import-linter 3/3) |
| Stories | ✅ 2/2 done; epic-r5 stories complete |
| Committed | ✅ both stories committed (`b60c7a0`, `6b33bbd`) — ⚠️ **not pushed to origin/main** (Action Item 4) |
| meshio coverage | ⚠️ happy/tensor paths skip locally (meshio absent); CI `[all]`/`[rom]` must install it (Action Item 2) |
| Incidents / blockers | ✅ none |
| Review cadence | ✅ full 3-layer on both stories |
| Significant-discovery / epic-update alarm | ✅ raised (release surface) and resolved → `v0.2.0` incl. R5 |

**Verdict: Epic R5 is complete.** Phase 2 is producing on the spine-theorem seam with zero core
disruption. The project's next gate is cutting **`v0.2.0`** (Phase-1 + Field ROM v1 + Phase-2 R5).

## Next Steps

1. Execute Action Item 1: prepare and cut `v0.2.0` (version + changelog + corrected `RELEASING.md`
   + full-matrix CI confirmation).
2. Action Item 2: wire `meshio` into the CI `[all]`/`[rom]` job so the binary-reader path is covered.
3. Action Item 4: push the two R5 commits to `origin/main`.
4. Apply Action Item 3 (fixture-diversity rule) to all future reader/parser tests.
