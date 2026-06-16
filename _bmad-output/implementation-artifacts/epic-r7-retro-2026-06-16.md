# Epic R7 Retrospective — PyTorch Surrogates (external-plugin distribution)

- **Date:** 2026-06-16
- **Epic:** R7 — PyTorch Surrogates (the first heavyweight-framework track, delivered *off* Core as a
  separate `surroforge-torch` distribution; the first real exercise of the plugin seam)
- **Stories:** r7-1 (the `surroforge-torch` plugin — `TorchMLPTrainer` / `torch-mlp`; commit
  `ad0ca0e`) + `examples/08_torch_mlp.py` (same commit) — story `done`
- **Facilitator:** Amelia (Developer) · **Participants:** Winston (Architect), Murat (Test
  Architect), Georg (Project Lead)
- **Previous retro:** epic-r6-retro-2026-06-15.md
- **Milestone:** SurroForge now has **PyTorch** — switchable with the built-in `mlp` via one config
  line — while Core stays torch-free and every reproducibility/export/conformance guarantee holds.
  This is the **first real distribution on the plugin seam**, validating AR-R1 end-to-end.

## Epic Summary

| Metric | Value |
|---|---|
| Stories completed | 1/1 (100%) — r7-1 |
| New tests | **+29** (plugin suite `plugins/surroforge-torch/tests`); Core unchanged at **402 passed / 7 skipped** |
| Gates | ruff + mypy clean on the package; Core green throughout |
| Review | **Full 3-layer adversarial review run as a workflow** (Blind / Edge-Case / Acceptance), each finding **re-verified by running code**. **8 findings, 7 applied, 1 dismissed.** No degradation (directly applying R6 Action Item 2). |
| Commits | 1 (`ad0ca0e`) — committed; **not yet pushed** to `origin/main` |
| Production-diff discipline | entire new `plugins/surroforge-torch/` package + `examples/08_torch_mlp.py` + docs (configuration, examples) + CHANGELOG + BMad artifacts; **zero `src/surroforge/**`, Core `pyproject.toml`, or `_BUILTIN` change** (AR-R1 + Golden Path, confirmed by a `git status` scope audit) |
| Production incidents | 0 |
| New capability | `model.method: torch-mlp` (+ `model.inner.method: torch-mlp` POD-NN); switchable with `mlp` by a one-line `method:` edit (mirrored hyperparameter surface) |
| New dependency | **torch (BSD-3)** — confined to the **opt-in plugin**, never Core; NFR8 holds |

R7 ships a real PyTorch MLP (`TorchMLPTrainer`) as a separate package that registers `torch-mlp`
through the `surroforge.trainers` entry point only. It is deterministic by construction (own integer
`random_state`, seeded weight init under a saved/restored torch RNG, full-batch training; honest
float32 `reproduction_tolerance = 1e-4` + a negative divergence test), exports to ONNX (torch's legacy
exporter, standardisation baked into the graph), passes the shipped `check_trainer` suite (all 12),
and works as a `reduced-field` inner (POD-NN). `examples/08_torch_mlp.py` demonstrates the
`mlp` ↔ `torch-mlp` switch (mlp R²=0.9993 vs torch-mlp R²=0.9992 from the same config block) and
degrades to `mlp`-only when the plugin is absent (CI-safe).

## What Went Well

- **The plugin seam works — proven, not just documented.** This is the first real external
  distribution to register a method via the entry-point seam. The AR-R1 / Golden-Path claim ("a
  heavyweight or closed-source method ships as an ordinary entry point, never `_BUILTIN`") now has a
  concrete proof: PyTorch was integrated with **zero Core change** (scope-audited).
- **Exact execution of R6 Action Item 3.** R6 pre-scoped "heavyweight NN = a new epic on the external
  seam — do not add it to Core." R7 is precisely that epic, delivered exactly as specified.
- **R6 Action Item 2 applied and upgraded.** The full 3-layer adversarial review ran as a workflow
  with no degradation (the session limit had reset), and was strengthened with **per-finding
  adversarial verification by running code** — every claimed bug confirmed or refuted empirically.
- **Determinism solved honestly for a stochastic framework.** Same discipline as `mlp`: the trainer
  carries its own `random_state`, weights are seeded under a saved/restored torch RNG (global RNG
  untouched), training is full-batch (no shuffling). Measured same-seed bit-identical; a declared,
  honest float32 band (`1e-4`) with a negative divergence test that has teeth.
- **The "switch" is genuinely one line.** `torch-mlp` mirrors `mlp`'s eight-hyperparameter surface,
  so the same YAML block trains either backend; example 08 shows comparable held-out R² for both.

## Challenges & Friction

- **Empirical numbers leaked from a throwaway probe into the docs.** An early standalone probe showed
  a different-seed divergence of "≈ 1.0 / four orders"; the actual *test regime* (standardised inputs,
  a deeper net) diverges ~1e-2 (~100×, two orders). The review caught it; comments/story corrected.
  Lesson: quote magnitudes from the test fixtures, not from scratch experiments.
- **A subtle cross-solver semantic inconsistency.** `alpha` meant a 2× different L2 across `lbfgs`
  (explicit penalty) vs `adam`/`sgd` (`weight_decay`), and handled biases inconsistently. Fixed
  (0.5 factor on the lbfgs penalty + weight-only param groups) so `alpha` is consistent and matches
  scikit-learn — found only by the review, not by tests.
- **Latent maintenance debt: the legacy ONNX exporter is deprecated.** torch 2.9+ defaults to the
  dynamo exporter; we pin `dynamo=False` for the dependency-light legacy path. A future torch may
  remove it, forcing a migration to the dynamo exporter + an `onnxscript` dependency.
- **CI does not cover the plugin.** The main matrix installs `.[all,dev]` (no torch) and collects only
  `tests/`, so the whole `surroforge-torch` package is currently untested in CI — the same CI theme as
  R6 Action Item 5, now larger (a full package, not one test).
- **Release/publish path for a separate distribution is undefined.** The plugin is its own package
  (`surroforge-torch` v0.1.0); whether it ships to PyPI as a sibling or stays install-from-source is
  an open product decision.

## Key Insights

1. **The plugin seam is now proven, not aspirational.** AR-R1's extensibility payoff — a heavyweight
   or commercial method as an ordinary external entry point — has its first real-distribution proof.
   "If a new method reaches into the engine, that's the smell, not the spec" extends to *plugins*: a
   conformant `BaseTrainer` in a separate package flows through unchanged Core.
2. **Backend interchangeability is a deliberate design property, not an automatic one.** "Switch with
   one config line" works *because* `torch-mlp` mirrors `mlp`'s hyperparameter surface. Interchange is
   designed in, by mirroring the surface, not granted by the seam.
3. **A framework can keep ALL the project's guarantees if they are imposed at the plugin boundary.**
   Determinism, ONNX export, and conformance are contract-level (`BaseTrainer` / `check_trainer`), not
   Core-level — so a heavyweight, stochastic backend can satisfy them behind its own declared band.
4. **The full adversarial review earns its keep even on a "clean" new package.** Seven real fixes,
   including a semantic inconsistency (`alpha`) and a documentation-honesty bug (inverted test-teeth
   labeling), that the green test suite did not surface.

## Previous Retro Follow-Through (R6 action items)

1. ✅ **[Doc] Seeding seam in project-context** — applied; `torch-mlp` relied on it directly (carries
   its own `random_state`; the run seed is not threaded into trainers).
2. ✅ **[Process] Full 3-layer review for non-trivial work** — the full 3-layer ran as a workflow with
   no degradation, plus per-finding code-verification. Exactly the "retry the parallel independent
   reviewers when the limit resets" intent.
3. ✅✅ **[Forward] Heavyweight NN = a new epic on the external seam** — **Epic R7 IS this action
   item**, executed exactly as scoped (entry-points-only distribution; not in Core).
4. ⏳ **[Release] Fold `mlp` into the next version bump** — still pending; R7 enlarges the release
   surface (now `mlp` *and* the `surroforge-torch` plugin). Carried as R7 Action Item 5.
5. ⚠️ **[CI verify] Confirm the matrix runs the new paths** — partially; and R7 surfaces a larger CI
   gap (the plugin is not covered by CI at all). Carried as R7 Action Item 1.

## Significant Discovery → epic-update alarm

**No alarm.** There is no Epic R8 defined, so nothing downstream to invalidate. Two forward-looking
observations were recorded as action items (CI does not cover the plugin; the release/publish path for
a separate distribution is undefined) — both are preparation notes, not changes that break a plan.

## Action Items

1. **[CI] Add an optional plugin CI job.** A separate job that installs torch + the plugin and runs
   `plugins/surroforge-torch/tests` + `check_trainer` — kept **off** the main matrix so torch does not
   bloat every job. Owner: dev. (Extends R6 Action Item 5.)
2. **[Release] Decide the `surroforge-torch` distribution/publish strategy.** PyPI as a separate
   package vs. install-from-source / monorepo-only; fold into the release plan alongside `mlp`.
   Owner: lead.
3. **[Maintenance] Track the torch ONNX-exporter deprecation.** When torch removes the legacy
   (`dynamo=False`) path, migrate to the dynamo exporter and add `onnxscript` to the plugin's `[onnx]`
   extra. Owner: dev (when torch forces it).
4. **[Forward / optional] Further heavyweight methods extend the same plugin, never Core.** DeepONet /
   FNO, if ever wanted, ship in `surroforge-torch` (or a sibling package) on the same seam. Owner: lead
   (on demand).
5. **[Release / carry] R6 Action Item 4 still open.** Cut a release that includes `mlp` and the
   `surroforge-torch` plugin (with a `Development Status` / matrix check). Owner: lead.

## Next Direction: no Epic R8 yet — release or extend on demand

R7 closes the first external-plugin slice (PyTorch). As with R5/R6, **nothing further is committed.**
The natural next gate is a **release** that includes `mlp` (R6 AI4) and decides how the
`surroforge-torch` plugin is published (R7 AI2). Beyond that the Phase-2 roadmap stays open — more
methods, or further heavyweight surrogates on the same seam (R7 AI4) — to be opened as a new epic when
a concrete need arrives, exactly as R5/R6/R7 were.

## Readiness Assessment

| Dimension | Status |
|---|---|
| Tests & quality | ✅ plugin suite 29 passed; Core 402 passed / 7 skipped (no regression); ruff + mypy clean on the package; `check_trainer` all 12 pass (no BLAS carve-out needed) |
| Stories | ✅ 1/1 done; epic-r7 complete |
| Committed & pushed | ⚠️ committed (`ad0ca0e`); **not yet pushed** to `origin/main` (lead's call) |
| Scope (AR-R1 / Golden Path) | ✅ zero Core source change; `torch-mlp` registered via entry point only (audited) |
| Review cadence | ✅ full 3-layer adversarial review + per-finding code-verification (R6 AI2 honoured, no degradation) |
| CI coverage of the plugin | ⚠️ not covered (torch absent in CI; plugin tests not collected) — Action Item 1 |
| Release / publish path | ⚠️ undefined for the separate distribution — Action Item 2 |
| Incidents / blockers | ✅ none |
| Significant-discovery / epic-update alarm | ✅ none (no downstream epic to invalidate) |

**Verdict: Epic R7 is complete.** The plugin seam has its first real proof, PyTorch is integrated and
switchable with `mlp` via one config line, every guarantee holds at the plugin boundary, and Core is
provably untouched. The open items are forward-looking (push, a plugin CI job, the release/publish
decision) — none blocks "done".

## Next Steps

1. Push `ad0ca0e` to `origin/main` (lead's call).
2. Action Item 1: add the optional plugin CI job (torch + `check_trainer`, off the main matrix).
3. Action Item 2 + R6 AI4: cut a release including `mlp` + decide how `surroforge-torch` is published.
4. Open a new epic only when a concrete need arrives (more methods, or further heavyweight surrogates
   on the same seam — Action Item 4).
