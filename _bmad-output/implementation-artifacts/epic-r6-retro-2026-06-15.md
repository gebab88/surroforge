# Epic R6 Retrospective — Advanced Surrogate Methods (Phase 2)

- **Date:** 2026-06-15
- **Epic:** R6 — Advanced Surrogate Methods (a second Phase-2 track, parallel to R5; the first
  "advanced method" beyond the polynomial/kriging/rbf trio)
- **Stories:** r6-1 (neural-network `mlp` trainer — `surroforge.trainers.mlp`, scikit-learn
  MLPRegressor; commit `06925c2`) + the example follow-up `examples/06_mlp_surrogate.py` (commit
  `f5d616c`) — story `done`
- **Facilitator:** Amelia (Developer) · **Participants:** Winston (Architect), Murat (Test
  Architect), Georg (Project Lead)
- **Previous retro:** epic-r5-retro-2026-06-14.md
- **Milestone:** SurroForge has its **first neural network** — and it cost zero new dependencies,
  kept every reproducibility/export guarantee, and touched no core code.

## Epic Summary

| Metric | Value |
|---|---|
| Stories completed | 1/1 (100%) |
| Test-suite growth | 371 collected (post-R5) → **408 collected** (401 passed, 7 skipped) |
| Skips | 7, all env-gated (meshio absent locally + `@requires_reproducible_blas` probes skipped on macOS/Accelerate + the slow repro marker) |
| Gates | green throughout (ruff, mypy **43** files, import-linter 3/3) |
| Review | 3-layer intended; **Blind Hunter ran as a subagent**, Edge/Auditor done **inline by the lead** after a transient subagent session-limit. **6 patches applied, 3 dismissed.** |
| Commits | 2 (`06925c2` trainer + tests; `f5d616c` example) — **committed AND pushed to `origin/main`** |
| Production-diff discipline | new `trainers/mlp.py` + `tests/test_mlp.py` + 2 registration lines (`pyproject`, `registry._BUILTIN`) + conformance `BUILTINS` + CHANGELOG + example; **engine/config/services/cli/other-trainers untouched** (AR-R1) |
| Production incidents | 0 |
| New capability | `model.method: mlp` (+ `model.inner.method: mlp` POD-NN); the **2nd ONNX-capable method** after polynomial |
| New dependency | **none** (scikit-learn is already Core, BSD → NFR8 holds) |

R6 added the first NN surrogate as a built-in trainer backed by `sklearn.neural_network.
MLPRegressor` (input `StandardScaler`, `solver="lbfgs"` default, native multi-output). It declares
an honest reproduction tolerance for its iterative optimiser and exports to ONNX via skl2onnx.

## What Went Well

- **The spine theorem held a third time — now for a method, after R1 (trainer) and R5
  (ingestion).** A genuine neural network shipped as a `trainers/` leaf + two registration lines;
  the engine's `trainer_cls(**hyperparameters)` factory and `ModelConfig(extra="allow")` carried it
  with **zero** engine/config/services/cli edits (AR-R1). The diff is almost entirely new file +
  tests.
- **The backend decision protected the project's identity.** Choosing scikit-learn's MLP over
  PyTorch gave a *real* NN with **no new dependency**, kept Core permissive (NFR8), and stayed
  inside the sklearn-style `random_state` reproducibility model. The Golden Path rule
  (framework/third-party plugins are entry-points-only, never `_BUILTIN`) made the decision almost
  automatic: PyTorch couldn't be "in Core" anyway, so the heavyweight track was correctly deferred
  to the external plugin seam.
- **Reproducibility was handled honestly, not waved away.** The first stochastic method forced the
  questions the closed-form/SMT methods never did: declare a *positive* `reproduction_tolerance`
  (1e-6, looser than SMT's 1e-9, with documented rationale), default `lbfgs` (measured bit-identical
  same-seed in-process, vs `~2.3e-2` on a seed change), and pin BLAS in the same-seed test.
- **Fixture diversity was applied by default, not retrofitted.** The R5 standing rule landed in this
  epic's first test pass: multi-output, unknown/invalid hyperparameters, the diff-seed *divergence*
  negative case, and the reduced-field-inner e2e — not just the tidy single-output fit.
- **The review caught real contract gaps.** Even on a small leaf change, the adversarial pass found
  6 genuine hardenings (raw sklearn errors leaking instead of `TrainerError`; a `to_onnx` that
  trusted the caller's feature count; a half-fitted-model window). All applied and test-pinned.

## Challenges & Friction

- **The 3-layer review degraded to 1 subagent layer + 2 inline.** A transient subagent
  session-limit killed the Edge-Case and Acceptance reviewers; the lead performed both inline with
  full code+project context. Adequate here (small, well-understood change; strong test coverage),
  but it was a *fallback*, not the designed parallel-independent review — worth flagging so it isn't
  normalised for higher-risk work.
- **`reproduction_tolerance` is absolute, and that's a latent edge for future methods.** An absolute
  1e-6 max-abs-diff band is fine for mlp on O(1) targets, but a future method with large-magnitude
  outputs could find it mis-scaled. Noted, not a problem this epic (and it intersects the existing
  "Read First" fingerprint defect in project-context).
- **A seam fact only became visible with the first stochastic method:** the run `seed` is threaded
  into `train_test_split`/`KFold` but **not** into the trainer. The closed-form/SMT methods never
  exposed it; mlp had to carry its **own** `random_state`. Correct as-is, but undocumented until now.

## Key Insights

1. **AR-R1 now has three independent proofs** (trainer, ingestion adapter, method). "If a new
   method/reader/adapter reaches into the engine, that's the smell, not the spec" is no longer a
   hopeful claim — it is the observed pattern across R1, R5, R6.
2. **A reproducibility-first tool shapes which NN you adopt, not just whether.** The honest path is
   a deterministic-by-default, framework-free MLP with a declared tolerance — *not* the most
   powerful framework. The heavyweight, stochastic options remain valid but belong on the external
   plugin seam, behind their own (looser) declared guarantees.
3. **"Small leaf change" still earns a real review.** Six genuine contract gaps in a ~200-line
   trainer echoes R5's "thin wrapper carried two bugs": the full review stays the default; the
   degradation to inline was a tooling fallback, not a decision to review less.

## Previous Retro Follow-Through (R5 action items)

1. ✅ **Cut `v0.2.0`** — done before this epic (`5fd7a95` "Release prep v0.2.0"); R6 work now sits in
   `CHANGELOG [Unreleased]` for the next bump.
2. ✅ **Exercise the `meshio` path in CI** — `meshio` is in the `[all]` extra and CI installs
   `.[all,dev]`; believed covered (CI YAML not re-verified this epic — see Action Item 5).
3. ✅ **Fixture diversity in reader/parser tests** — applied directly to R6's trainer tests
   (multi-output, invalid hyperparameters, diff-seed divergence, inner e2e); the rule is already in
   project-context Testing.
4. ✅ **Push R5 commits to `origin/main`** — done (R5 is on origin; this retro confirms R6 is too).
5. ⚠️ **Full 3-layer review stays the default** — *intent honoured, execution degraded*: the full
   3-layer was launched, but only the Blind layer ran as a subagent (session-limit); Edge/Auditor
   were inline. Carried into Action Item 2.

## Significant Discovery → epic-update alarm

**No alarm.** There is **no Epic R7 defined**, so nothing downstream to invalidate. Two latent
observations were recorded as insights (absolute reproduction tolerance; run-seed not threaded to
trainers) — both are forward-looking notes, not changes that break an existing plan.

## Action Items

1. **[Doc] Record the seeding seam in project-context.** Add a one-liner under Reproducibility:
   *trainers own their `random_state`; the run `seed` is threaded into split/CV but not into the
   trainer.* Relevant to every future stochastic method and the external NN-plugin track. Owner: dev.
2. **[Process] Subagent-review fallback is explicit, not default.** When subagents are unavailable,
   inline Edge/Auditor review is an acceptable fallback for low-risk leaf changes (as here); for
   higher-risk work, retry the parallel independent reviewers when the limit resets. Owner: lead/dev.
3. **[Forward / optional] Heavyweight NN = a new epic on the external seam.** If PyTorch/DeepONet/FNO
   is ever wanted (large, strongly-nonlinear fields), open a dedicated epic for an *entry-points-only*
   plugin distribution — do not add it to Core. Owner: lead (when a concrete need arrives).
4. **[Release] Fold `mlp` into the next version bump.** It is in `CHANGELOG [Unreleased]`; cut it
   into a `v0.3.0` (or fold into the next release) with a `Development Status`/matrix check. Owner: lead.
5. **[CI verify] Confirm the matrix runs the new paths green.** The mlp tests + example 06 must pass
   on Linux/OpenBLAS where the `@requires_reproducible_blas` same-seed test actually executes (it
   skips on local Accelerate) and where the ONNX round-trip runs under `[all]`. Owner: dev.

## Next Direction: no Epic R7 yet — release or extend on demand

R6 closes the first "advanced methods" slice (one NN). Like R5, **nothing further is committed.**
The natural next gate is a **release including `mlp`** (Action Item 4). Beyond that the Phase-2
roadmap stays open — more methods, or the external heavyweight-NN track (Action Item 3) — and a new
epic should be opened when a concrete need arrives, exactly the way R5 and R6 themselves were.

## Readiness Assessment

| Dimension | Status |
|---|---|
| Tests & quality | ✅ 408 collected, 0 failures (401 passed, 7 env-gated skips); gates clean (ruff, mypy 43, import-linter 3/3) |
| Stories | ✅ 1/1 done; epic-r6 complete |
| Committed & pushed | ✅ both commits on `origin/main` (`06925c2`, `f5d616c`) |
| Review cadence | ⚠️ 3-layer intended; ran as 1 subagent (Blind) + 2 inline (Edge/Auditor) after a session-limit — adequate for this low-risk leaf change (Action Item 2) |
| BLAS-pinned repro | ⚠️ same-seed mlp test skips on local Accelerate; must run on OpenBLAS/CI (Action Item 5) |
| Incidents / blockers | ✅ none |
| Significant-discovery / epic-update alarm | ✅ none (no downstream epic to invalidate) |

**Verdict: Epic R6 is complete.** The spine theorem now has a third independent proof, SurroForge
has a genuine neural network with zero new dependencies and all guarantees intact, and the work is
shipped to `origin/main`. The next gate is a release that includes `mlp`.

## Next Steps

1. Action Item 4: fold `mlp` into the next release (`v0.3.0` or the next bump).
2. Action Item 1: add the seeding-seam note to project-context Reproducibility.
3. Action Item 5: confirm the CI matrix runs the new mlp tests + example 06 green (esp. the
   Accelerate-skipped BLAS-pinned probe on Linux/OpenBLAS).
4. Open a new epic only when a concrete need arrives (more methods, or the external heavyweight-NN
   track — Action Item 3).
