---
baseline_commit: 5fd7a95ba478cc844fdc2b22617a0ab5f7dd66f9
---

# Story R6.1: Neural-network surrogate — built-in `mlp` trainer (scikit-learn MLPRegressor)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->
<!-- Phase-2 / Epic R6, story 1. First "advanced method" beyond the polynomial/kriging/rbf trio. -->

## Story

As a methods engineer,
I want a neural-network surrogate method (`model.method: mlp`) that trains a multi-layer perceptron
on my DOE table, ranks/compares against the existing methods, and exports to ONNX,
so that I can fit smooth nonlinear response surfaces with a genuine NN **without** the project
taking on a heavy deep-learning framework or losing its reproducibility and export guarantees.

## Context & Rationale

SurroForge ships three methods today — `polynomial` (scikit-learn, ONNX-capable), `kriging` and
`rbf` (SMT, pickle-only) — plus the `reduced-field` reducer. There is **no neural network**. This
story adds the first one as a first-class built-in.

**Backend decision (lead, this story): scikit-learn `MLPRegressor`, not PyTorch.** A multi-layer
perceptron *is* a neural network; `MLPRegressor` is a full MLP, and scikit-learn is **already a
Core dependency** — so this adds a real NN with **zero new dependencies**, stays permissive (BSD →
NFR8 holds), and plugs straight into the project's reproducibility model (sklearn-style integer
`random_state`). PyTorch was rejected for Core: it is heavy, SGD-stochastic (weakens the
determinism invariant), and the Golden Path is explicit that **framework/third-party plugins use
entry points only, never `_BUILTIN`** — i.e. PyTorch would have to be a *separate distribution*,
not "integrated into SurroForge." A PyTorch/DeepONet/FNO track remains available later on exactly
that external-plugin seam for the large, strongly-nonlinear field regime.

It rides the **spine theorem (AR-R1)**: the engine builds a trainer with
`trainer_cls(**config.model.hyperparameters)` and `ModelConfig` has `extra="allow"`, so a new
method is a **pure leaf + registry change** — `trainers/mlp.py` + the two registration sites. The
engine, `core/config.py`, `services.py`, the CLI, export, compare, predict, report, reproduce, and
conformance paths stay **unchanged**: downstream sees an ordinary `BaseTrainer`. Because any
`BaseTrainer` is automatically usable as a `reduced-field` inner, `mlp` also becomes a **POD-NN**
option (`model.inner.method: mlp`) for free.

Two domain choices, both deliberate and documented (verified during story prep):

- **`solver="lbfgs"` default** (scikit-learn's default is `adam`). The surrogate regime is
  small-data (a DOE of dozens–hundreds of points); the scikit-learn docs themselves recommend
  `lbfgs` there, and it is far more reproducible: a same-`random_state` lbfgs refit was measured
  **bit-identical** (max diff `0.0`), where a different seed diverged (`~2.3e-2`). This is the
  right default for a reproducibility-first tool.
- **input `StandardScaler`** wrapped in a `Pipeline` (NNs need scaled inputs; deterministic;
  skl2onnx converts the pipeline cleanly).

## Acceptance Criteria

**AC1 — `mlp` is a built-in method, resolvable with no engine/config change.**
**Given** a config with `model.method: mlp`
**When** the run is built
**Then** `get_trainer("mlp")` resolves `MLPTrainer`, `mlp` appears in `available_trainers()`, and it
is registered in **both** `pyproject.toml` `[project.entry-points."surroforge.trainers"]` **and**
the `_BUILTIN` table in `plugins/registry.py` (the built-in convention)
**And** there is **no** change to `engine.py`, `core/config.py`, `services.py`, `cli/`, `plugins/`
(beyond the one `_BUILTIN` line), or the other trainers (AR-R1; the engine's
`trainer_cls(**hyperparameters)` factory already passes hyperparameters through).

**AC2 — BaseTrainer contract + conformance, including native multi-output.**
**Given** `MLPTrainer`
**When** it fits and predicts
**Then** `fit(x, y)` accepts 2-D arrays, `predict(x)` always returns a 2-D `(n, n_outputs)` array
(single-output is `(n, 1)`, never squeezed), multi-output `y` is supported natively (one network,
`n_outputs` columns), the trainer is **picklable**, and `check_trainer(MLPTrainer)` passes
**unmodified** — `MLPTrainer` is added to the conformance `BUILTINS` in `tests/test_conformance.py`
**And** a single-output fit raises **no** scikit-learn `DataConversionWarning` (1-D `y` is passed
to the estimator when `n_outputs == 1`; predictions are re-shaped back to 2-D).

**AC3 — Hyperparameters: curated, validated, surrogate-appropriate defaults.**
**Given** the constructor
**When** I pass hyperparameters
**Then** a curated supported set is accepted (`hidden_layer_sizes`, `activation`, `solver`, `alpha`,
`max_iter`, `tol`, `learning_rate_init`, `random_state`) and an **unknown** hyperparameter raises a
`TrainerError` that **lists the supported names** (mirroring `kriging`/`polynomial`)
**And** the defaults suit the surrogate regime: `solver="lbfgs"`, an input `StandardScaler`, a fixed
`random_state` (deterministic by default), and a `max_iter` large enough that the bundled tests
converge; `__init__` stays cheap and free of a top-level heavy import (the estimator import lives
inside `fit`, per the optional-dependency `[GOTCHA]`, even though sklearn is Core).

**AC4 — Reproducibility: declares a tolerance; has a negative test.**
**Given** the method's iterative (non-convex) optimiser
**When** capabilities are declared
**Then** `reproduction_tolerance` is a small, **honest** NN-appropriate band (positive, not `0.0` —
the optimiser is iterative and platform/BLAS-sensitive), recorded in lineage and honoured by
`reproduce()`/the conformance determinism probe (which already honours the declared tolerance)
**And** tests pin both directions: a same-`random_state` refit reproduces **within** the declared
tolerance, and changing `random_state` makes predictions diverge **beyond** it (the negative case
with teeth — `[MUST]` a numerics change isn't done without it).

**AC5 — ONNX export (becomes the second ONNX-capable method).**
**Given** a fitted `MLPTrainer`
**When** I export it
**Then** `export_formats == {"pickle", "onnx"}`, `to_onnx(n_features)` returns a valid ONNX model
via `skl2onnx` (the `StandardScaler` + `MLPRegressor` pipeline; verified to convert for single- and
multi-output, agreeing with the sklearn prediction to ~1e-6), and a missing `onnx` extra raises an
`ExportError` with a `pip install 'surroforge[onnx]'` hint (mirroring `polynomial`)
**And** ONNX export is exercised by a test that `importorskip`s the `onnx` stack and asserts the
ONNX-runtime output matches `predict` within a float32 tolerance.

**AC6 — Works unchanged as a `reduced-field` inner (POD-NN), and `data` stays clean.**
**Given** a `reduced-field` config with `model.inner.method: mlp`
**When** the run trains end-to-end on a low-rank field
**Then** it fits one MLP per mode through the **unchanged** `ReducedFieldTrainer` and reconstructs
the field (R² above a sensible bar on a smooth low-rank fixture) — no ROM-specific change to either
trainer
**And** `lint-imports` stays green: `trainers/mlp.py` imports only `errors` + `trainers.base`
(+ lazy sklearn inside methods); no `cli`/`api`/`services`/`engine` import; **no new dependency** is
added to `pyproject.toml` (sklearn is already Core — the `[smt]`/`[onnx]`/`[rom]` extras are
untouched except that `mlp`'s ONNX rides the existing `[onnx]` extra).

## Tasks / Subtasks

- [x] **Task 1 — `MLPTrainer`** (AC1, AC2, AC3)
  - [x] New `src/surroforge/trainers/mlp.py`: `class MLPTrainer(BaseTrainer)`, `name = "mlp"`,
        `from __future__ import annotations` + PRD-citing docstring.
  - [x] Curated `_SUPPORTED` set; `__init__` validates unknowns → `TrainerError` listing supported;
        cheap, import-free (`super().__init__(...)` so lineage captures the hyperparameters).
  - [x] `_build()` returns `make_pipeline(StandardScaler(), MLPRegressor(... defaults ...))` with
        the estimator import **inside** the method.
  - [x] `fit`: `_as_2d` both; ravel `y` to 1-D when `n_outputs == 1` (no DataConversionWarning);
        `predict`: `_as_2d(self._model.predict(...))`; "not fitted" → `TrainerError`.
- [x] **Task 2 — capabilities** (AC4, AC5)
  - [x] `reproduction_tolerance = 1e-6` (documented NN-appropriate band); `export_formats =
        frozenset({"pickle", "onnx"})`.
  - [x] `to_onnx(n_features)` mirroring `polynomial` (skl2onnx + `FloatTensorType`; missing extra →
        `ExportError` hint).
- [x] **Task 3 — registration** (AC1)
  - [x] `pyproject.toml`: added `mlp = "surroforge.trainers.mlp:MLPTrainer"` to the trainers
        entry-point group.
  - [x] `plugins/registry.py`: added the same `mlp` line to `_BUILTIN[TRAINER_GROUP]`.
- [x] **Task 4 — tests** (AC2, AC3, AC4, AC5, AC6)
  - [x] `tests/test_mlp.py`: fit/predict 2-D shape (single + multi-output), unknown-hyperparameter
        rejection (names supported), no-DataConversionWarning on single-output, same-seed
        reproduce-within-tolerance (BLAS-pinned) + diff-seed diverge-beyond (negative test), ONNX
        round-trip (`importorskip`, single + multi), not-fitted error, reduced-field-`inner: mlp`
        e2e (R² bar). + review-added: invalid activation/solver, non-int random_state, bad numeric
        hyperparameters, ONNX feature-count-mismatch guard.
  - [x] `tests/test_conformance.py`: added `MLPTrainer` to `BUILTINS`.
- [x] **Task 5 — gates & docs** (all ACs)
  - [x] `ruff check src tests`, `mypy src` (43 files), `lint-imports` (3 kept), `pytest` all green
        (400 passed, 7 skipped).
  - [x] `CHANGELOG.md` `[Unreleased]`: added the `mlp` method entry.

### Review Findings (code-review 2026-06-15)

3-layer adversarial review intended (Blind / Edge / Auditor). **Blind Hunter** ran and produced
substantive findings; the Edge-Case and Acceptance layers hit a transient subagent session limit
and were performed inline by the lead with full code + project context. Verdict: ship-able — scope
clean (AR-R1 respected: only the trainer leaf + the two registration sites + tests + changelog
changed; engine/config/services/cli untouched), licensing clean (no new dep). 6 findings applied,
3 dismissed with reasons.

**Patches applied (test-pinned):**

- [x] [Review][Patch] `random_state` accepted non-ints (`None`/`RandomState`/float) → silent
      non-determinism under a tolerance/lineage that assume a fixed seed. Reject non-int (bool
      excluded) → `TrainerError`. [trainers/mlp.py:__init__] (blind)
- [x] [Review][Patch] A typo'd `activation`/`solver` raised a **raw sklearn ValueError** deep in
      `fit` (violates the no-bare-error `[MUST]`). Validate both against `_ACTIVATIONS`/`_SOLVERS`
      at construction → `TrainerError` listing valid choices. [trainers/mlp.py:__init__] (blind)
- [x] [Review][Patch] Negative/zero `alpha`/`tol`/`learning_rate_init` leaked raw sklearn errors at
      fit. Added positivity guards → `TrainerError`. [trainers/mlp.py:__init__] (blind)
- [x] [Review][Patch] `to_onnx` trusted the caller's `n_features`; a wrong value builds a graph
      that **silently mispredicts**. Guard against the fitted `n_features_in_` → `ExportError`.
      [trainers/mlp.py:to_onnx] (blind)
- [x] [Review][Patch] `to_onnx` only translated `ImportError`; a skl2onnx conversion failure would
      escape as a raw error. Wrap `convert_sklearn` → `ExportError`. [trainers/mlp.py:to_onnx] (blind)
- [x] [Review][Patch] `fit` assigned `self._model` **before** `.fit()`; a failed fit left a
      half-built pipeline that `predict` would treat as fitted (raw `NotFittedError`). Build into a
      local, assign only on success. [trainers/mlp.py:fit] (blind)
- [x] [Review][Patch] Same-seed reproduce test wasn't BLAS-pinned; Accelerate's ULP drift amplifies
      through lbfgs (the Kriging lesson). Mark `@requires_reproducible_blas`. [tests/test_mlp.py] (edge/lead)

**Dismissed (3):** withdrawn shape finding (the hunter self-verified `_as_2d` keeps `predict` 2-D
for single- and multi-output — no bug); `reproduction_tolerance=1e-6` "too tight" (it is honest and
cleanly separates a same-seed refit `0.0` on reproducible BLAS from a diff-seed `~2.3e-2`, three
orders away — and the BLAS guard covers Accelerate); x/y row-count mismatch leaks a raw sklearn
ValueError (clear message; matches every other built-in trainer; pre-existing project-wide, out of
scope here).

## Dev Notes

### What to reuse (do NOT reinvent)

- **`polynomial.py` is the template** — it is the other sklearn-backed, ONNX-capable trainer.
  Copy its shape: `_build()` returning a `make_pipeline(...)`, lazy estimator import inside the
  method, `to_onnx` via `skl2onnx`/`FloatTensorType` with the `ExportError` install hint, the
  "Model is not fitted." guard. [Source: src/surroforge/trainers/polynomial.py]
- **`kriging.py`/`rbf.py` are the template for the iterative-method capabilities** — `_SUPPORTED`
  hyperparameter set, the unknown-hyperparameter `TrainerError`, and a **positive**
  `reproduction_tolerance` (they declare `1e-9`). [Source: src/surroforge/trainers/kriging.py]
- **`BaseTrainer`** supplies `_as_2d`, `supports_export`, the `export_formats`/
  `reproduction_tolerance` class attributes, and the picklability contract. `predict` MUST return
  2-D. [Source: src/surroforge/trainers/base.py]
- **Registry**: built-ins live in `_BUILTIN[TRAINER_GROUP]` **and** the entry points; the engine
  resolves via `get_trainer`. Do not resolve by name anywhere else.
  [Source: src/surroforge/plugins/registry.py]

### Architecture compliance (project-context.md — non-negotiable)

- **Layer:** `trainers/` is a leaf — import only `errors` + `trainers.base`; lazy sklearn import
  inside `fit`/`to_onnx`. No `cli`/`api`/`services`/`engine` import. `lint-imports` enforces this.
- **Optional-dependency `[GOTCHA]`:** keep `__init__` import-free even though sklearn is Core, so
  the registry can list the method cheaply (matches `polynomial`). The ONNX backend (`skl2onnx`)
  IS optional → its import stays inside `to_onnx` with the `[onnx]` install hint.
- **Errors `[MUST]`:** raise `TrainerError`/`ExportError` (never bare `ValueError`); messages must
  state the fix (list supported hyperparameters; name the extra).
- **Reproducibility `[CONV]`/`[MUST]`:** seeding is sklearn-style integer `random_state`. The run
  `seed` is threaded into `train_test_split`/`KFold` by the engine but **not** into the trainer, so
  the MLP carries its **own** `random_state` hyperparameter with a fixed default (deterministic out
  of the box; override in YAML). A numerics method is not done without a **negative** test (perturb
  → assert divergence beyond the declared tolerance).
- **Plugin Golden Path `[MUST]`:** `name` lowercase == entry-point key == `model.method`; built-in
  ⇒ both registration sites; **third-party/commercial ⇒ entry points only** (this is the rule that
  keeps PyTorch out of Core). The trainer must be picklable (it is the persisted artifact).
- **Licensing `[MUST]`:** no new dependency — scikit-learn is Core (BSD). Nothing GPL/LGPL.

### Domain notes (verified during story prep)

- skl2onnx converts the `StandardScaler` + `MLPRegressor` pipeline for **both** single- and
  multi-output (max |onnx − sklearn| ≈ 6.7e-7 multi / 1.2e-6 single, float32 boundary).
- `lbfgs` + fixed `random_state`: same-seed refit max diff `0.0`; different seed `≈ 2.3e-2`. Hence
  `solver="lbfgs"` default and a small-but-positive `reproduction_tolerance` (honest: lbfgs is not
  guaranteed bit-identical across BLAS thread counts / platforms, and the surface is non-convex).
- Single-output `MLPRegressor` warns `DataConversionWarning` on a column-vector `y`; pass 1-D `y`
  when `n_outputs == 1` to keep output clean (project tone: no noise), reshape on predict.

### Testing — anti-vacant-pin / fixture diversity (project-context.md)

- **Fixture diversity `[GOTCHA]` (R5 retro):** include the awkward classes, not just the tidy one —
  **multi-output** `y`, **unknown-hyperparameter** rejection, the **different-seed divergence**
  negative case, and the **reduced-field inner** e2e. A happy-path-only suite is the exact trap R5
  flagged.
- **Vacant pins `[GOTCHA]`:** for the reproduce/diverge assertions, name the wrong wiring each
  catches (e.g. a trainer that ignored `random_state` would pass same-seed but must **fail**
  diff-seed). Pick a fixture where right and wrong wiring differ at the asserted precision; prefer
  exact/`np.allclose`-with-named-tolerance over substring checks.
- ONNX/SMT-dependent tests `importorskip` their backend; the `mlp` core path needs neither beyond
  Core sklearn.

### Scope boundary

R6.1 = the `mlp` trainer + registration + capabilities (tolerance, ONNX) + tests + changelog (+
optional docs/example). It does **not** add PyTorch, GPU, target scaling, a CLI flag, output ONNX
for `reduced-field` (still pickle-only there), or any engine/config change. If a change reaches
into `engine`/`config`/`services`, stop — this is leaf + registry only.

### Project Structure Notes

- Add: `src/surroforge/trainers/mlp.py`, `tests/test_mlp.py`.
- Touch: `pyproject.toml` (entry point), `src/surroforge/plugins/registry.py` (`_BUILTIN` line),
  `tests/test_conformance.py` (`BUILTINS`), `CHANGELOG.md` (+ optional `docs/`, `examples/`).
- No change to `engine.py`, `core/config.py`, `services.py`, `cli/`, the other trainers (AR-R1).

### References

- [Source: src/surroforge/trainers/polynomial.py] — sklearn + ONNX template.
- [Source: src/surroforge/trainers/kriging.py, rbf.py] — `_SUPPORTED` + positive tolerance template.
- [Source: src/surroforge/trainers/base.py] — the contract (`_as_2d`, 2-D predict, export/tolerance).
- [Source: src/surroforge/plugins/registry.py] — `_BUILTIN` + `get_trainer`.
- [Source: src/surroforge/engine.py:92-99] — `get_trainer(method)` + `trainer_cls(**hyperparameters)`.
- [Source: src/surroforge/core/config.py:146-169] — `ModelConfig` `extra="allow"` passes
  hyperparameters through (why no config change is needed).
- [Source: tests/test_conformance.py:30-35] — `BUILTINS` list to extend.
- [Source: _bmad-output/project-context.md] — layers, optional-dependency + error discipline,
  reproducibility (negative-test `[MUST]`), vacant-pin + fixture-diversity taxonomy, licensing.

## Dev Agent Record

### Agent Model Used

claude-opus-4-8[1m] (Claude Opus 4.8, 1M context)

### Debug Log References

- Verified during prep: skl2onnx converts the StandardScaler + MLPRegressor pipeline for single-
  and multi-output (max |onnx − sklearn| ≈ 1.2e-6 / 6.7e-7, float32 boundary); lbfgs + fixed
  random_state same-seed refit max diff `0.0`, diff-seed `≈ 2.3e-2`.
- No lint fixes needed (ruff/mypy clean on first run).

### Completion Notes List

- **Trainer:** `surroforge.trainers.mlp.MLPTrainer` — a genuine MLP via `sklearn.neural_network.
  MLPRegressor` (input `StandardScaler`, `solver="lbfgs"` default for the small-data DOE regime,
  native multi-output). **Zero new dependency** (sklearn is Core, BSD → NFR8). Import-free
  `__init__`; estimator import inside `_build`/`to_onnx` (optional-dependency `[GOTCHA]`).
- **Capabilities:** `export_formats = {pickle, onnx}` → the **second ONNX-capable method** after
  polynomial; `reproduction_tolerance = 1e-6` (honest band for the iterative, non-convex optimiser —
  looser than the SMT methods' 1e-9; a changed seed moves predictions ~1e-2, three orders above).
- **Spine theorem (AR-R1):** pure leaf + registry change. The engine's `trainer_cls(**hyper)`
  factory + `ModelConfig(extra="allow")` carry it with **no** engine/config/services/cli edit.
  Works unchanged as a `reduced-field` inner (POD-NN: `model.inner.method: mlp`), proven e2e.
- **Reproducibility:** the run seed is not threaded into trainers, so `mlp` carries its own fixed
  `random_state` (deterministic by default; override in YAML). Negative test pins divergence on a
  changed seed; the same-seed positive test is `@requires_reproducible_blas` (Accelerate skips,
  OpenBLAS/CI asserts).
- **Review hardening:** 6 patches from the adversarial pass applied (random_state/activation/solver/
  numeric validation, to_onnx feature-count + conversion-failure guards, build-then-assign in fit).
- Validation: **400 passed, 7 skipped**; ruff / mypy(43) / lint-imports(3) green.

### File List

- `src/surroforge/trainers/mlp.py` (new) — `MLPTrainer` (fit/predict/to_onnx + hyperparameter
  validation).
- `src/surroforge/plugins/registry.py` (modified) — `mlp` added to `_BUILTIN[TRAINER_GROUP]`.
- `pyproject.toml` (modified) — `mlp` trainers entry point.
- `tests/test_mlp.py` (new) — full suite (contract, multi-output, validation, reproducibility,
  ONNX, reduced-field inner e2e).
- `tests/test_conformance.py` (modified) — `MLPTrainer` added to `BUILTINS`.
- `CHANGELOG.md` (modified) — `[Unreleased]` `mlp` method entry.

## Change Log

| Date | Change |
|---|---|
| 2026-06-15 | Story drafted (R6.1, Phase-2 Epic R6). Backend decision recorded: scikit-learn MLPRegressor (not PyTorch) — real NN, zero new deps, permissive, reproducibility-compatible. Status → ready-for-dev. |
| 2026-06-15 | R6.1 implemented: built-in `mlp` trainer (sklearn MLPRegressor, StandardScaler + lbfgs, native multi-output, ONNX via skl2onnx). Registered in both sites; added to conformance BUILTINS; CHANGELOG entry. Suite 392 passed / 5 skipped before review. Status → review. |
| 2026-06-15 | Code review (Blind ran; Edge/Auditor inline after a subagent session-limit): 6 patches applied (hyperparameter validation, to_onnx guards, build-then-assign, BLAS-pinned repro test), 3 dismissed. Suite 400 passed / 7 skipped; all gates green. Status → done. |
