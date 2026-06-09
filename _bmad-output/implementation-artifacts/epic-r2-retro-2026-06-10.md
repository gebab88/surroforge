# Epic R2 Retrospective — Field Reproducibility (Fingerprint)

- **Date:** 2026-06-10
- **Epic:** R2 — Field Reproducibility (Fingerprint), the Field ROM v1 critical path
- **Stories:** r2-1 (spike), r2-2, r2-3, r2-4, r2-5 — all `done`, committed to `main`
- **Facilitator:** Amelia (Developer) · **Participants:** Winston (Architect), Murat (Test Architect), John (PM), Georg (Project Lead)
- **Previous retro:** none (first Field ROM retrospective)

## Epic Summary

| Metric | Value |
|---|---|
| Stories completed | 5/5 (100%) |
| Shape | 1 de-risk spike (r2-1) + 4 implementation stories |
| Production incidents | 0 |
| Gates | green throughout — 228 tests, ruff/mypy clean, import-linter 3/3 |
| Review | every story adversarially reviewed (Blind · Edge · Auditor); r2-5 also validate-create-story'd |
| Commits | `791609b` (R2.4), `5bb0566` (R2.5); R2.1–R2.3 in prior commits |

R2 evolved `reproduce()` to a **field-primary verdict** (the reconstructed field is the definition; basis hash + canonicalized coefficients are a diagnostic lens, never a gate) while keeping the **scalar path byte-identical**. It was the highest-leverage, highest-risk slice of Field ROM v1.

## What Went Well

- **Spike-first discipline (r2-1) paid off.** The spike proved the three load-bearing assumptions empirically *before* the contract froze — and it **changed the design**: the basis hash was demoted from a "key guard" to a pure diagnostic. Freezing without it would have locked in a broken contract.
- **The definition flip.** "The field *is* the verdict; coefficients/basis are diagnostic" made the verdict gauge-invariant by construction — no false-diverge on a benign re-gauge. This was the conceptual unlock (First-Principles, from the spike).
- **Spine theorem #36 held.** ROM shipped as *one trainer + a fingerprint evolution*. The engine stayed ROM-agnostic via a duck-typed hook (`engine` imports neither `services` nor `reduced_field`); import-linter 3/3 across all five stories. The scalar path stayed byte-identical *by construction* (R2.3 design), pinned by R2.4 with **zero production change**.
- **Reviews have teeth.** The adversarial review caught real, end-to-end-reproduced bugs (R2.5: the constant-field "0% verified" warning on a perfectly-reproduced run; the warning firing on `unverifiable`/`data_changed`). `validate-create-story` caught two gate-breakers in R2.5 *before* dev (the `float | None` mypy trap; `pytest.warns(None)` removed in pytest 9).
- **"Diagnostic, never a gate" is now a reusable pattern** — basis hash, then `probe_coverage_energy`. Advisory signals that never fail the build.

## Challenges & Friction

- **The frozen spec drifted from the as-built reality.** The (non-frozen) "proposed shape" section of `spec-field-rom-fingerprint.md` proposed `probe_nodes` and a persisted `basis.npz`; R2.3/R2.5 shipped the global energy-norm over the full field instead. Reconciliation notes had to be added *outside* the frozen block in both R2.3 and R2.5. A "proposed shape" section sitting next to a `<frozen-after-approval>` block but not itself frozen invites confusion.
- **BLAS non-reproducibility (Apple Accelerate) is a recurring tax.** Every story had to reason about ULP sensitivity (basis hash run-to-run, `energy_captured`, coverage). Margins / `pytest.approx` instead of bit-equality is now the house pattern — solved, but it costs thought on every new numeric quantity.
- **R2.4 + R2.5 were entangled in two shared files** (`ci.yml` line, the scalar-identity test) because both were built before anything was committed; clean separated commits needed temp-edit surgery.
- **Degeneracy edges deferred (R2.2):** single-linkage σ-chain chaining and the `max(σ, 1.0)` tolerance floor — documented in `deferred-work.md`, revisit if a real field produces a problematic σ-chain.
- **An external-plugin ROM-hook hardening cluster is forming** in `deferred-work.md` (R2.3 dict-shape validation + R2.5 `KeyError` for a hook omitting `probe_coverage_energy`).

## Key Insights

1. De-risk the highest-risk contract with a runnable spike before freezing it — the spike *should* be allowed to change the design (it did).
2. Make the engineer-observable quantity the verdict; everything else is diagnostic. Gauge-invariance falls out for free.
3. Advisory-not-a-gate is the right shape for signals that inform but must never break reproducibility (basis hash, probe coverage).
4. On non-reproducible BLAS, pin behaviour by margins/recomputation, never by golden literals or bit-equality.

## Action Items (committed)

1. **[Process] Keep the spike-first + validate-create-story + adversarial-review pipeline** as a standing team agreement. Owner: Georg / dev. Evidence it works: caught two gate-breakers pre-dev and real bugs pre-merge in R2.5. Success: applied to every non-trivial story going forward.
2. **[Process] Commit each story at "done" before starting the next.** Owner: Georg. Avoids the R2.4/R2.5 shared-file entanglement (which needed temp-edit surgery to split). Success: no future story pair shares uncommitted edits.
3. **[Documentation] Fix the frozen-spec drift.** Clean up the stale "proposed shape" tail of `spec-field-rom-fingerprint.md` — move it out or mark it clearly superseded (beyond the per-story reconciliation notes already added in R2.3/R2.5). Owner: dev/tech-writer. Success: a future reader cannot mistake `probe_nodes`/`basis.npz` for the binding contract.

## Tracked (not a committed action item)

- **External-plugin ROM-hook hardening cluster** (`deferred-work.md`: R2.3 dict-shape validation + R2.5 `KeyError`). Cannot fire in v1 (the built-in `ReducedFieldTrainer` always returns well-formed diagnostics). Harden when the first external ROM plugin binds — a Phase-2 item.

## Next Epic Preparation

**Decision (Georg): complete Epic R1 before starting R3.**

R1 is the foundation R3 builds on (more than R2, which is solid and blocks nothing). Remaining R1 work:
- **r1-1** — ingest a field as a snapshot matrix (deterministic glob expansion of the output column-group; ragged-group `DataError`).
- **r1-2** — `model.reduction` block parsing/validation + `rank: adaptive` resolved-rank snapshot write-back.
- **r1-4** — built-in methods pass conformance unmodified as inner surrogates.

Then **R3 — Field Validation & Report**:
- r3-1 field metrics as registered `AVAILABLE_METRICS` callables (`energy_l2`, `truncation_error`).
- r3-2 two-source error split in the report (truncation floor vs. surrogate error) — `energy_captured` already recorded (R2.5).
- r3-3 honest-basis CV — partly falls out of r1-3's per-fold fresh-factory; benefits from a complete r1-2.

## Readiness Assessment

| Dimension | Status |
|---|---|
| Tests & quality | ✅ 228 passed / 2 skipped; reproduce family green under pinned threads |
| Gates | ✅ ruff, mypy, import-linter 3/3 |
| Committed | ✅ R2.4 (`791609b`), R2.5 (`5bb0566`) on `main`; working tree clean |
| Incidents / blockers | ✅ none |
| Significant-discovery / epic-update alarm | ✅ none — the frozen-spec drift is doc hygiene, not an R3-blocking architectural change |

**Verdict: Epic R2 is complete and production-ready.** No epic-update review required before the next epic. The only forward adjustment is the sequencing choice (finish R1 before R3).

## Next Steps

1. Execute Action Item 3 (frozen-spec cleanup) — small, safe doc fix.
2. `create-story` for **r1-1** (or r1-2 if preferred), then the validate → dev → review pipeline (Action Item 1).
3. Commit each story at done (Action Item 2).
4. After R1 is complete, begin R3 with r3-1.
