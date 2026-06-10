# Epic R1 Retrospective — Reduction Engine & Trainer

- **Date:** 2026-06-10
- **Epic:** R1 — Reduction Engine & Trainer (Field ROM v1 foundation)
- **Stories:** r1-3 (delivered ad hoc, pre-tracking), r1-1, r1-2, r1-4 — all `done`
- **Facilitator:** Amelia (Developer) · **Participants:** Winston (Architect), Murat (Test Architect), Georg (Project Lead)
- **Previous retro:** epic-r2-retro-2026-06-10.md (R2 finished before R1 was back-filled)

## Epic Summary

| Metric | Value |
|---|---|
| Stories completed | 4/4 (100%) — unusual order: R1.3 first (the spine), then all of Epic R2, then R1.1 → R1.2 → R1.4 |
| Test-suite growth | 228 (post-R2) → 241 (r1-1) → 264 (r1-2) → 269 (r1-4, + review-hardened assertions) |
| Gates | green throughout (ruff, mypy, import-linter 3/3, pinned-threads reproduce family) |
| Reviews | 3 adversarial reviews (Blind · Edge · Auditor) → 2 High bugs caught pre-merge, 10 patches, 5 defers, 18 dismissals |
| Production incidents | 0 |
| Commits | `f706da8` (R1.1), `dc65113` (R1.2); **R1.4 + this retro pending commit** |

Epic question: does spine theorem #36 hold — ROM as *one trainer* on the existing stack?
**Answer: yes, with one deliberate, bounded compromise.** R1.1 was a single-choke-point change
(`select_columns`, zero engine change); R1.4 proved the theorem *as a diff* (the absence of
source changes was the deliverable); only R1.2 placed structured config blocks in core — a
ratified lead decision contained to `core/config.py`, bridged back to the trainer via the
`hyperparameters` dict and a duck-typed `resolved_config_overrides()` hook.

## What Went Well

- **The review pipeline pays measurably.** Every story's adversarial review surfaced at least
  one *empirically reproduced* finding: R1.1 — bracketed-index columns (`disp[0]`, `stress[xx]`)
  misparsed as globs (High; near-universal naming convention); R1.2 — adaptive write-back read
  the FINAL model's rank while the fingerprint comes from the HOLDOUT model → false `diverged`
  on a straddling spectrum (High); R1.4 — the 0.9 r² floor sat below the fixture's single-mode
  energy share (0.945), so a rank collapse would have passed.
- **`validate-create-story` catches gate-breakers pre-dev:** r1-2's `mode="before"` validator
  trap (`rank: true` would have coerced to 1 under the `int | Literal` union) and r1-4's
  originally non-discriminating Task-1 assertion (default-polynomial fallback produced
  identical metrics/shape).
- **"Proof story" is a working story format (R1.4):** the deliverable was the *absence* of
  changes, with an explicit STOP instruction if any source edit seemed needed. Diff discipline
  as acceptance criterion worked exactly as designed.
- **Empirical fixture verification is now house standard:** straddling-spectrum seed search
  (R1.2), energy-margin probes (0.943–0.945 vs. the 0.99 floor, R1.4), and teeth checks before
  trusting a discriminating assertion.

## Challenges & Friction

- **Holdout vs. final model is a recurring trap.** R1.2's High bug lived exactly there.
  Documented residual: the persisted final artifact may hold a different mode count than the
  snapshot rank (diagnostic, not verdict). → Action item 3 (project-context gotcha, done).
- **The BLAS tax continues** (Apple Accelerate): every numeric story pays guard/carve-out
  thought-cost. Solved by pattern, not free. Newly deferred: `blas_is_bit_reproducible` is
  optimistic for *unknown* backends (MKL gap) — deferred-work.md.
- **Legacy compat deliberately broken** (r1-3-era persisted snapshots no longer re-parse) —
  ratified pre-release decision with a documented migration path (deferred-work.md).
- **A hardening cluster keeps growing in deferred-work.md:** external-plugin ROM-hook
  robustness (R2.3/R2.5/R1.2) + `reduction`/`inner` name shadowing — all gated on "first
  external plugin binds".

## Previous Retro Follow-Through (R2 action items)

1. ✅ **Pipeline (spike/validate/dev/adversarial-review)** — applied to all three tracked R1
   stories, with demonstrable catches (see above). Keep.
2. ✅ **Commit each story at `done` before the next** — r1-1 and r1-2 landed as clean single
   commits; no shared-file entanglement recurred. (r1-4 reached `done` during this retro
   session and is committed next — compliant, no next story started.)
3. ❌ → ✅ **Frozen-spec drift cleanup** — was NOT done during R1; **executed in this retro**:
   `spec-field-rom-fingerprint.md`'s "proposed shape" section is now explicitly marked
   SUPERSEDED (no `basis.npz`, no `probe_nodes`; global energy-norm is the verdict), and the
   frontmatter status reads `implemented`. Frozen block untouched.

## Key Insights

1. A "proof story" — where the deliverable is a pinned *absence* of changes — is a legitimate,
   cheap way to convert an architectural claim (AR-R1 "zero downstream touchpoints") into a
   regression-guarded fact.
2. Verify empirically that a fixture exercises the intended branch before pinning assertions
   on it (straddling spectrum, energy margins, teeth checks). An assertion that cannot fail
   under the bug it claims to catch is decoration.
3. Engine introspection hooks must be queried on the holdout model (fingerprint-defining),
   never the final model. Now a project-context `[GOTCHA]`.
4. Defaults are non-discriminating test inputs: a parametrized case whose config equals the
   fallback default proves nothing about the plumbing (R1.4's `degree: 2` → `degree: 3`).

## Action Items

1. **[Critical path] Commit R1.4 (+ retro artifacts)** before any further work (retro rule #2).
   Owner: Georg/dev. Status: queued next.
2. **[Doc, carried from R2] Frozen-spec cleanup** — Owner: dev. **Done in this retro** (see
   follow-through #3).
3. **[Process] Holdout-vs-final gotcha in project-context** — Owner: dev. **Done in this
   retro** (`_bmad-output/project-context.md`, Reproducibility section).
4. **[Team agreement, continued] Keep the pipeline + story-commit discipline** — confirmed
   effective two epics running.

## Next Epic Preview: R3 — Field Validation & Report

**Dependencies on R1/R2 — all satisfied:**
- r3-1 (field metrics): `AVAILABLE_METRICS` registry exists (`validators/metrics.py:29`) with
  the unknown-metric error path.
- r3-2 (two-source error split): `energy_captured` already recorded in lineage (R2.5).
- r3-3 (honest-basis CV): per-fold fresh-factory (r1-3) + complete `model.reduction` config
  (r1-2) both in place.

**No significant-discovery alarm.** Nothing from R1 invalidates the R3 plan.
**One design question for r3-1's create-story** (story-level, not an epic update):
`truncation_error` does not fit the two-argument `(y_true, y_pred)` metric callable shape —
it needs basis context. The story must decide how it enters `AVAILABLE_METRICS` without new
plumbing (AC: "no special handling").

## Readiness Assessment

| Dimension | Status |
|---|---|
| Tests & quality | ✅ 269 passed / 3 expected skips; pinned-threads family green; gates clean |
| Stories | ✅ 4/4 done; epic-r1 `done` in sprint-status |
| Committed | ⚠️ R1.4 + retro artifacts pending commit (action item 1, queued) |
| Incidents / blockers | ✅ none |
| Stakeholder | n/a (single-lead project; lead ratified all decisions inline) |
| Significant-discovery / epic-update alarm | ✅ none |

**Verdict: Epic R1 is complete.** Field ROM v1's foundation (R1) and critical path (R2) are
both done; R3 can start after the R1.4 commit.

## Next Steps

1. Commit R1.4 + retro artifacts (action item 1).
2. `create-story` for **r3-1** (field metrics as registered callables), then the
   validate → dev → review pipeline.
3. Commit each story at done (standing agreement).
