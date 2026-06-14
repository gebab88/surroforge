# Epic R4 Retrospective — Ingestion, Query, Conformance & Packaging

- **Date:** 2026-06-14
- **Epic:** R4 — Ingestion, Query, Conformance & Packaging (Field ROM v1, final epic)
- **Stories:** r4-1 (predict coefficients by default; reconstruct on `--field`), r4-2 (`rom inspect`), r4-3 (reconstruction round-trip + no-mesh conformance probe), r4-4 (`surroforge[rom]` packaging extra) — all `done`
- **Facilitator:** Amelia (Developer) · **Participants:** Winston (Architect), Murat (Test Architect), Georg (Project Lead)
- **Previous retro:** epic-r3-retro-2026-06-11.md
- **Milestone:** with R4, **Field ROM v1 (R1–R4) is feature-complete.** There is no R5 — the next direction is v1 release preparation.

## Epic Summary

| Metric | Value |
|---|---|
| Stories completed | 4/4 (100%) |
| Test-suite growth | 299 (post-R3) → ~327 (r4-3) → **331 collected** (329 passed, 2 skipped) post-R4 |
| Gates | green throughout (ruff, mypy, import-linter 3/3) |
| Reviews | r4-1: 3 layers, 24→17 findings, **13 patches** · r4-2: patches + 4 defers (2 rounds) · r4-3: 3 layers, 2 patches, 2 defers · r4-4: **no review at done — remediated post-hoc in this retro** (3 layers, 2 doc patches, 1 defer, 0 Critical/Major/Minor) |
| Commits | 4 clean single-commit stories (`f898ce8`, `77fea4e`, `3397ce2`, `06be679`), **all pushed — origin/main synced** |
| Production-diff discipline | r4-1: 2 files · r4-2: 2 files · r4-3: **1 file** (`conformance/trainer.py`) · r4-4: 2 files (`pyproject.toml` + `reduced_field.py`); numeric SVD / fingerprint / lineage untouched |
| Production incidents | 0 |

R4 delivered the entire user-facing ROM surface — coefficients-by-default prediction with
`--field` reconstruction and lazy content-addressed storage (r4-1), `rom inspect` for the
energy spectrum / rank / coverage (r4-2), the `coeffs→field→coeffs` round-trip + no-mesh
DOF-permutation conformance probes (r4-3), and the `surroforge[rom]` install seam (r4-4) — all
while the engine, registry, exporters, `compare`, and the conformance core kept seeing a normal
`(n, n_dof)` trainer. The final packaging story needed **zero** change to numerics, engine, or
registry.

## What Went Well

- **The spine theorem cashed out in full.** R4 is the proof at epic scale: the whole feature
  shipped as *one trainer + a fingerprint evolution*, and the last story (r4-4) added the
  packaging seam with no core change at all. The brainstorm's organizing bet (#36 — basis hidden
  inside the trainer) held end to end.
- **"Pin / declare, don't build" kept generalizing.** r4-3 pinned the duck-typed
  `predict_coefficients` contract with teeth (the `_NonOrthonormalReducer` /
  `_GeometryDependentReducer` negatives), closing an r4-1 defer rather than adding plumbing; r4-4
  *declared* an empty extra + a lazy guard instead of inventing a dependency.
- **The honesty culture held under a contradictory AC.** r4-4's two AC clauses conflicted in v1
  (literal install hint vs. "no new heavy dependency"). Rather than ship theatrical gating, the
  tension was surfaced to the lead, an explicit Faithful-pattern decision was made, and the
  "guard cannot fire in a real install" reality was recorded in `deferred-work.md`.
- **Adversarial review strengthened the design, not just defended it.** The post-hoc r4-4 audit
  discovered the guard is *more* substantive than its own docstring claimed: scipy is a genuine
  fit-time requirement (the sklearn/SMT inner surrogate imports `scipy.sparse`), so the gate
  preempts a real `ImportError`, not a hypothetical one. The narrative was corrected to match.

## Challenges & Friction

- **r4-4 skipped the adversarial review at `done`.** The three prior R4 stories each had a
  3-layer review before commit; r4-4 landed on gates + self-review only. Defensible by risk
  (packaging + a tested guard, no numeric/contract/engine change) — but it broke the epic's
  review cadence. **Remediated in this retro**: a post-hoc 3-layer review found 0
  Critical/Major/Minor, 2 doc-accuracy patches, 1 latent-license defer. The lesson is about
  *consciousness of the decision*, not the outcome (see Action Item 2).
- **A generically-worded AC drifted from the PRD's own reality.** r4-4's AC mandated an install
  hint "mirroring [smt]/[onnx]" while PRD-ROM §4.10 simultaneously states v1 adds no new
  dependency. The two only reconcile by reading the hint as a forward-compat seam. Cost a design
  decision, not code — but a reminder that ACs and the PRD body can disagree.
- **Holdout-vs-final model display nuance** (r4-2 defers): `energy_captured` (holdout-model
  lineage) vs. the `cumulative`/`[retained]` spectrum (final-model npz) can differ slightly in
  `rom inspect`. Cosmetic, mandated by the reproduce-verdict-consistency spec; a Phase-2 display
  polish.
- **Defer ledger grew but stayed disciplined.** Every R4 defer carries an explicit revisit
  trigger — almost all "when the first external ROM plugin binds" or Phase-2 (mesh readers,
  store GC/refcount, display polish, the license-scan-method note). None blocks a v1 release.

## Previous Retro Follow-Through (R3 action items)

1. ✅ **Commit R3.3 + retro before new work** — honored; R4 started from a clean, committed base.
2. ✅ **Vacant-pin taxonomy → project-context Testing `[GOTCHA]`** — applied throughout R4:
   r4-3's negative reducers name the *specific* failing check; r4-1/r4-2 pins discriminate
   (content-address framing, malformed-lineage guards). The taxonomy did its job.
3. ✅ **basis.npz disambiguation carried into r4-1** — implemented exactly as flagged:
   `reduction_artifacts()` is a storage/prediction artifact, explicitly **not** a fingerprint
   gate (its docstring says so); the field-primary verdict is unchanged.
4. ⚠️ **Pipeline (create → validate → dev → adversarial review) + commit-at-done** — held for
   r4-1/r4-2/r4-3 (each reviewed, single commit). **r4-4 deviated on the review layer** — now
   remediated, and codified as Action Item 2.

## Key Insights

1. **The spine theorem is validated at feature scale.** A non-trivial capability (reduced-order
   field models, reproducibility and all) shipped without a single special-case in the engine,
   registry, exporters, or conformance core. Architecture bets of this kind are worth making
   early and defending hard.
2. **When an AC and the PRD conflict, decide explicitly and record the trade-off where the next
   reader will find it** (here: `deferred-work.md` + the in-code docstring + the story's "v1
   tension" section). Satisfying the letter of an AC blindly would have produced dishonest code.
3. **The review rhythm is a safeguard, not a formality.** Skipping it on a low-risk story was
   defensible, but the post-hoc pass still found two real doc-accuracy issues — including one
   that *improved* the design's self-description. Even low-risk stories earn one adversarial
   pass; the rule should be a conscious risk-tier decision, not silent drift.
4. **Adversarial review can be generative.** The r4-4 audit didn't just fail to find bugs — it
   upgraded the rationale (scipy is genuinely required via the inner), turning a guard that read
   as symbolic into a substantive one. Use review to sharpen the story, not only to defend it.

## Action Items

1. **[Done in this retro] Post-hoc adversarial review of r4-4.** 3 layers (Blind / Edge /
   Auditor), all 7 ACs PASS, 0 Critical/Major/Minor; 2 doc-only patches applied (guard docstring
   accuracy + test-teeth comment), 1 defer recorded (scipy license-text scan-method). Review
   cadence restored. Owner: dev. **Complete.**
2. **[Process / Team agreement] Risk-tiered review, consciously chosen.** Risk-low stories
   (packaging, docs, pure-config, pin-only) *may* skip the full 3-layer adversarial pass — but
   only as an **explicit, recorded decision** at create-story or done time, never as drift. A
   one-line "review tier: light (reason)" note in the story doc suffices. Owner: dev/lead.
3. **[Critical path → next direction] v1 release preparation** (lead's call: tag/release v1).
   Field ROM v1 + the Phase-1 MVP are feature-complete and clean. Prepare a release candidate:
   confirm/settle the version (`pyproject` `0.1.0`), write a changelog / release notes covering
   Phase-1 + Field ROM v1, a final full-matrix CI green confirmation, and a PyPI-readiness check
   (metadata, `[rom]`/`[all]` extras, license classifiers, `Development Status` trove). Owner:
   dev/lead. Status: queued as the immediate next work.
4. **[Standing] Commit-at-done + the create→validate→dev→review pipeline** — confirmed effective
   across four epics; the r4-4 review-layer deviation is the only break, now remediated and
   codified (Action Item 2).

## Next Direction: Field ROM v1 is complete — prepare the v1 release

**There is no Epic R5.** The Field ROM v1 plan was R1–R4, and all four epics are `done`:

| Epic | Scope | Status |
|---|---|---|
| R1 | Reduction Engine & Trainer | ✅ done |
| R2 | Field Reproducibility (Fingerprint) | ✅ done |
| R3 | Field Validation & Report | ✅ done |
| R4 | Ingestion, Query, Conformance & Packaging | ✅ done |

Combined with the Phase-1 MVP (Epics 1–6, feature-complete), the project is at a natural
**release milestone**. The lead's chosen next direction is to treat this as a v1 release
candidate (Action Item 3) rather than open Phase-2 feature work (commercial ROM surfaces + mesh
readers on the reserved seam), which remains the longer-term roadmap.

**No significant-discovery alarm.** Nothing in R4 invalidates a downstream plan; the post-hoc
audit surfaced only two doc-accuracy patches and one latent license-scan note — no architectural
surprise.

## Readiness Assessment

| Dimension | Status |
|---|---|
| Tests & quality | ✅ 331 collected, 0 failures; gates clean (ruff, mypy, import-linter 3/3) |
| Stories | ✅ 4/4 done; epic-r4 `done` in sprint-status |
| Committed & pushed | ✅ all 4 stories on `origin/main`; review patches + this retro pending one commit |
| Incidents / blockers | ✅ none |
| Review cadence | ✅ restored (r4-4 post-hoc review done in this retro) |
| Stakeholder | n/a (single-lead project; lead ratified decisions inline) |
| Significant-discovery / epic-update alarm | ✅ none |

**Verdict: Epic R4 is complete, and Field ROM v1 (R1–R4) is feature-complete.** The project is
ready to move into v1 release preparation.

## Next Steps

1. Commit the r4-4 post-hoc review patches + this retrospective (standing commit-at-done rule).
2. Begin **v1 release preparation** (Action Item 3): version + changelog/release notes +
   full-matrix CI confirmation + PyPI-readiness check.
3. Defer Phase-2 feature work (commercial ROM surfaces, mesh readers) until after the v1 release
   is cut.
</content>
