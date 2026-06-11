# Epic R3 Retrospective — Field Validation & Report

- **Date:** 2026-06-11
- **Epic:** R3 — Field Validation & Report (Field ROM v1)
- **Stories:** r3-1 (field metrics), r3-2 (two-source split), r3-3 (honest-basis CV) — all `done`
- **Facilitator:** Amelia (Developer) · **Participants:** Winston (Architect), Murat (Test Architect), Georg (Project Lead)
- **Previous retro:** epic-r1-retro-2026-06-10.md

## Epic Summary

| Metric | Value |
|---|---|
| Stories completed | 3/3 (100%) |
| Test-suite growth | 269 (post-R1) → 286 (r3-1) → 296 (r3-2) → 299 (r3-3) |
| Gates | green throughout (ruff, mypy, import-linter 3/3) |
| Reviews | 3 adversarial reviews (Blind · Edge · Auditor): 11 patches, 2 defers, 37 dismissals (~60 raw findings) |
| validate-create-story | 3/3 — every story had Critical/Major findings caught **before** dev |
| Production incidents | 0 |
| Production-diff discipline | r3-1: 2 files; r3-2: 2 files; r3-3: **1 file** (`reporters/markdown.py`); `cross_validation.py` byte-identical per epic AC |
| Commits | `2dfd3b4` (R3.1), `701aa94` (R3.2); **R3.3 + this retro pending commit** |

R3 made field error legible: `energy_l2`/`truncation_error` as plain registered metrics,
the report's two-source split (truncation floor vs. surrogate error — "add modes or add
data"), and honest-basis CV pinned with a leaky comparison plus a per-fold basis-variance
report line. The entire feature set added **zero new persisted fields** — everything is
derived at render time from existing records.

## What Went Well

- **The pipeline carries its third epic in a row.** Every `validate-create-story` caught
  vacant pins pre-dev (r3-1: seed-fragile fixture; r3-2: vacuous Jensen pin C1 — a
  Jensen-wrong implementation would have passed every test; r3-3: aliased std string +
  value-degenerate fixture). Every adversarial review found real patches after (r3-1:
  BLAS-flaky `<1e-12` + tautological Pythagoras assert; r3-2: asymmetric render guard;
  r3-3: presence-without-structure substring asserts). The r3-3 Acceptance Auditor
  returned **zero findings** — all ACs verified empirically.
- **"Pin, don't build" is a working story form** (second instance after R1.4's proof
  story): honest CV already existed structurally (per-fold fresh factory, r1-3); r3-3
  pinned it with a test-only leaky CV (optimism ratio 1.082, 2× threshold headroom) and
  surfaced the per-fold basis spread instead of building new plumbing.
- **Render-time derivation over persistence.** Pythagoras + Parseval turn two persisted
  metrics into the full two-source view; the basis-variance line labels and decomposes a
  number the table already carried. Cheap features, zero migration surface.
- **The R1-retro design question resolved cleanly:** `truncation_error` fits the pure
  `(y_true, y_pred)` callable shape via subspace-from-predictions — no special handling,
  exactly as the epic AC demanded.

## Challenges & Friction

- **Vacant pins are the epic's defining anti-pattern: 5 occurrences across all 3 stories.**
  We catch them reliably — but always in validation or review, never at writing time.
  Now codified as a `project-context.md` Testing `[GOTCHA]` with the three trap classes:
  (a) aliasing, (b) value degeneracy, (c) presence-without-structure (action item 2,
  executed in this retro).
- **The BLAS tax continues** (Apple Accelerate): every numeric story pays margin/headroom
  thought-cost. The house pattern is now routine (engine-measured literals as comments,
  `pytest.approx(rel=1e-3)`, derived-gap thresholds 1e-6 never 1e-12, 2× headroom on
  optimism margins) — solved by pattern, not free.
- **Planning artifacts age:** PRD §6's seam table says "reporters Unverändert", yet §4.6
  mandates report content. Ratified in r3-2 (the seam row describes plumbing, not
  templates) and cited in r3-3 — resolved, but a reminder that seam tables and feature
  mandates drift apart.
- **Defer ledger grew moderately:** r3-1 `engine.py` DataError rewrap of CV `ValueError`s
  (pre-existing; LinAlgError subclasses ValueError) and r3-2 `two_source_split` overflow on
  FIELD_ENERGY_FLOOR-dominated relative errors (degenerate zero-energy input). Neither
  blocks R4.

## Previous Retro Follow-Through (R1 action items)

1. ✅ **Commit R1.4 + retro artifacts** — landed (`b0b6f16`, `c5131c4`) before R3 began.
2. ✅ **Frozen-spec cleanup** — done in the R1 retro itself.
3. ✅ **Holdout-vs-final gotcha in project-context** — done; no recurrence in R3.
4. ✅ **Pipeline + story-commit discipline** — r3-1 and r3-2 landed as clean single
   commits; r3-3 reached `done` during this session and its commit is queued before any
   new work. Three epics running without a shared-file entanglement recurrence.

## Key Insights

1. **Design pins with teeth at writing time, not review time.** A discriminating assertion
   must name the wrong implementation it catches and use a fixture where that
   implementation actually renders differently. The three vacancy trap classes (aliasing /
   value degeneracy / presence-without-structure) are now a project-context Testing rule.
2. **"Pin, don't build" generalizes:** when a guarantee already falls out of existing
   contracts, the story's deliverable is a regression-proof demonstration (leaky
   comparison, proof-of-absence diff), not new code.
3. **Prefer render-time derivation over new persistence** when the math allows it
   (Pythagoras/Parseval): no migration, no compat surface, and degradation paths stay
   trivially testable.
4. **Keep the deliberately-wrong implementation in the test module only** — a leaky CV
   path in src/, even "for comparison", would be an attractive nuisance.

## Action Items

1. **[Critical path] Commit R3.3 + retro artifacts** before any further work (standing
   rule #2). Owner: Georg/dev. Status: queued next.
2. **[Process/Doc] Vacant-pin taxonomy → project-context.md Testing `[GOTCHA]`.**
   Owner: dev. **Done in this retro** (aliasing / value degeneracy /
   presence-without-structure, with the prescribed counter-moves).
3. **[Design note, carried into r4-1 create-story] `basis.npz` disambiguation:** R4.1's
   content-addressed `basis.npz` is a **storage/prediction artifact**; the R2 spec marked
   `basis.npz` SUPERSEDED **for the fingerprint**. The story must state explicitly that
   the new file is not a fingerprint gate (field-primary verdict unchanged). Owner:
   create-story for r4-1.
4. **[Team agreement, continued] Pipeline (create → validate → dev → adversarial review)
   + commit-at-done discipline** — confirmed effective three epics running.

## Next Epic Preview: R4 — Ingestion, Query, Conformance & Packaging

**Dependencies on R1–R3 — all satisfied:**
- r4-1 (coefficients by default, `--field`, lazy storage): trainer + reduction config +
  field metrics all in place; needs the basis.npz disambiguation note (action item 3).
- r4-2 (`rom inspect`): `lineage["reduction"]` (rank, energy_captured) + probe coverage
  (R2.5) recorded; the story must check whether per-mode energy contribution needs an
  extra lineage field (story-level question, not an epic update).
- r4-3 (round-trip + no-mesh conformance): conformance suite (5-3) + `record()`/tolerance
  plumbing exist; `ReducedFieldTrainer` passes the suite since r1-3.
- r4-4 (`[rom]` extra): lazy-import pattern established by `[smt]`/`[onnx]`.

**Sequencing (Georg): default confirmed — r4-1 first.** OQ-2 (field schema) and OQ-3
(probe selection) get resolved inside r4-1's create-story as needed.

**No significant-discovery alarm.** Nothing from R3 invalidates the R4 plan.

## Readiness Assessment

| Dimension | Status |
|---|---|
| Tests & quality | ✅ 299 collected, exit 0, 0 failures; gates clean (ruff, mypy, import-linter 3/3) |
| Stories | ✅ 3/3 done; epic-r3 `done` in sprint-status |
| Committed | ⚠️ R3.3 + retro artifacts pending commit (action item 1, queued) |
| Incidents / blockers | ✅ none (Georg confirmed: nothing unstable, no carried blockers) |
| Stakeholder | n/a (single-lead project; lead ratified all decisions inline) |
| Significant-discovery / epic-update alarm | ✅ none |

**Verdict: Epic R3 is complete.** Field ROM v1 now has its foundation (R1), critical path
(R2), and validation/report layer (R3); R4 can start after the R3.3 commit.

## Next Steps

1. Commit R3.3 + retro artifacts (action item 1).
2. `create-story` for **r4-1** (predict coefficients by default; reconstruct on `--field`),
   carrying the basis.npz disambiguation note — then the validate → dev → review pipeline.
3. Commit each story at done (standing agreement).
