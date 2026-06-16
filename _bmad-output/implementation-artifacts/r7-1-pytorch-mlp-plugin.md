---
baseline_commit: 19a1783fde06c968a6551ea867759e3ac84b4ac8
---

# Story R7.1: PyTorch neural-network surrogate — the `surroforge-torch` external plugin (`torch-mlp`)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->
<!-- Phase-2 / Epic R7, story 1. First heavyweight-framework method; first REAL exercise of the
     external entry-points-only plugin seam (a separate distribution, never _BUILTIN). -->

## Story

As a methods engineer,
I want a PyTorch-backed neural-network surrogate (`model.method: torch-mlp`) that I switch to from
the built-in scikit-learn `mlp` by changing one config line and installing one package,
so that I can reach for a real deep-learning framework on strongly-nonlinear problems **without**
SurroForge Core taking on a heavy dependency or losing its reproducibility and export guarantees.

## Context & Rationale

R6.1 added the first neural network — `mlp`, scikit-learn's `MLPRegressor` — as a **Core built-in**,
deliberately choosing it over PyTorch: PyTorch is heavy (hundreds of MB), and the Golden Path is
explicit that **framework/third-party plugins use entry points only, never `_BUILTIN`** — a separate
distribution, not integrated into Core. The R6.1 story (lines 30–34) and the R6 retrospective
(Action Item 3) both pre-scoped a PyTorch track as exactly this: **a dedicated epic for an
entry-points-only plugin distribution — do not add it to Core.** This story is that epic's first
deliverable.

**Placement decision (lead, this story): a separate `surroforge-torch` package inside the monorepo
(`plugins/surroforge-torch/`), registered through the `surroforge.trainers` entry point ONLY** — not
in Core's `pyproject.toml`, not in `_BUILTIN`. The registry already supports this: `_resolve`
(`plugins/registry.py:81-100`) lets a single external entry point provide a method, and the built-in
table is only a fallback for source checkouts. PyTorch is BSD-3 → **NFR8 (no GPL/LGPL) holds**; the
blocker was never licensing but weight + stochasticity, both of which an external, opt-in package
resolves. Installing the plugin (`pip install ./plugins/surroforge-torch`) makes `torch-mlp`
resolvable; Core stays torch-free. **This is also the first time the documented plugin seam is
exercised by a real shipped package** — it validates the extensibility claim end-to-end.

**The switch is the `method:` field.** A user moves between backends by changing
`model.method: mlp` ↔ `model.method: torch-mlp` (and installing the plugin once). To make that
switch a near-zero-churn edit, `torch-mlp` **mirrors `mlp`'s exact hyperparameter surface**
(`hidden_layer_sizes`, `activation`, `solver`, `alpha`, `max_iter`, `tol`, `learning_rate_init`,
`random_state`) — the same YAML block trains either backend.

It rides the **spine theorem (AR-R1)** from the outside: the engine builds any trainer via
`trainer_cls(**config.model.hyperparameters)` and `ModelConfig` is `extra="allow"`, so a plugin that
subclasses `BaseTrainer` flows through the unchanged engine/config/services/CLI/export/compare/
reproduce/conformance paths — **and there is no change to Core at all** (only docs/CHANGELOG note the
plugin's existence). Because any `BaseTrainer` is automatically a `reduced-field` inner, `torch-mlp`
is a POD-NN option for free.

Domain choices (verified during story prep against torch 2.12, CPU, Python 3.13):

- **Determinism by construction.** Weights are initialised under a saved/seeded/restored torch RNG
  (`torch.manual_seed(random_state)` bracketed by `get_rng_state`/`set_rng_state`, so the user's
  global torch RNG is left untouched), and training is **full-batch** (no `DataLoader` shuffling), so
  the only stochastic input is the weight init — which the seed pins. Measured: two same-seed fits
  are **bit-identical (max|Δ| = 0.0)** even with `np.random` perturbed between them (the conformance
  `deterministic` probe), while a different seed diverges by **~1e-2** — ~100× (two orders) above the
  band, in the standardised test regime.
- **`reproduction_tolerance = 1e-4`** — looser than `mlp`'s `1e-6` because the network trains in
  **float32** and cross-platform/thread float32 reductions are not bit-stable; honest, and still
  ~two orders below a real (different-seed / changed-data) divergence.
- **`solver="lbfgs"` default** (full-batch, `strong_wolfe` line search) — mirrors `mlp`'s default and
  suits the small-data DOE regime; `adam`/`sgd` are available (epoch loop with a `tol` early-stop).
- **Standardisation is baked INTO the exported graph.** Input/output mean-std are stored as module
  buffers and applied inside `forward`, so the ONNX graph is self-contained and its output matches
  `predict` (rather than expecting pre-normalised inputs).

## Acceptance Criteria

**AC1 — `torch-mlp` resolves through the entry-point seam ONLY; zero Core change.**
**Given** the `surroforge-torch` package is installed (`pip install ./plugins/surroforge-torch`)
**When** the registry resolves the method
**Then** `get_trainer("torch-mlp")` returns `TorchMLPTrainer`, `torch-mlp` appears in
`available_trainers()`, and it is registered **only** via the package's
`[project.entry-points."surroforge.trainers"]` — it is **NOT** added to Core's `pyproject.toml` or to
`_BUILTIN` in `plugins/registry.py`
**And** **no** Core source file changes (the engine's `trainer_cls(**hyperparameters)` factory +
`ModelConfig(extra="allow")` carry it unchanged); only `docs/`+`CHANGELOG.md` note the plugin.

**AC2 — BaseTrainer contract + the shipped conformance suite, including multi-output.**
**Given** `TorchMLPTrainer`
**When** it fits and predicts
**Then** `fit(x, y)` accepts 2-D arrays, `predict(x)` always returns a 2-D `(n, n_outputs)` array
(single-output is `(n, 1)`, never squeezed), multi-output `y` is supported natively (one network,
`n_outputs` columns), the **fitted** trainer is picklable and the unpickled model predicts
**bit-identically**, and `check_trainer(TorchMLPTrainer)` passes **unmodified** (all 12 checks) — run
from the plugin's own test, since Core's `tests/test_conformance.py` BUILTINS stays unchanged.

**AC3 — Hyperparameters mirror `mlp`, validated with `TrainerError`.**
**Given** the constructor
**When** I pass hyperparameters
**Then** the **same curated set as `mlp`** is accepted (`hidden_layer_sizes`, `activation`, `solver`,
`alpha`, `max_iter`, `tol`, `learning_rate_init`, `random_state`); an **unknown** hyperparameter
raises a `TrainerError` listing the supported names; an invalid `activation`/`solver`, a non-int
`random_state` (bool excluded), and a bad numeric (`max_iter < 1`, `alpha < 0`, `tol <= 0`,
`learning_rate_init <= 0`) each raise a `TrainerError` — **never** a raw torch error deep in `fit`
**And** the defaults match `mlp`'s where shared (so switching backends is a one-line `method:` edit).

**AC4 — Reproducibility: own seed, honest tolerance, negative test with teeth.**
**Given** the iterative optimiser
**When** capabilities are declared and exercised
**Then** the trainer carries its **own** integer `random_state` (the engine does not thread the run
seed into trainers), declares `reproduction_tolerance = 1e-4` (positive, float32-honest), seeds torch
from it inside `fit` via save/seed/restore (leaving the global torch RNG untouched)
**And** tests pin both directions: a same-`random_state` refit reproduces **within** tolerance — the
case **with teeth** against a seed-ignoring / global-RNG-leaking impl (no BLAS guard needed; torch's
float32 band absorbs platform drift) — and a different `random_state` diverges **beyond** it (guarding
against a vacuous tolerance / a degenerate constant fit).

**AC5 — ONNX export via torch's own exporter; standardisation baked in.**
**Given** a fitted `TorchMLPTrainer`
**When** I export it
**Then** `export_formats == {"pickle", "onnx"}`, `to_onnx(n_features)` returns a valid ONNX model via
`torch.onnx.export(..., dynamo=False)` (the dependency-light legacy exporter), the graph **includes
the input/output standardisation** so `onnxruntime` output matches `predict` within a float32
tolerance (verified single- and multi-output), a wrong `n_features` raises an `ExportError` before
export, and a missing optional export dependency raises an `ExportError` with a
`pip install 'surroforge-torch[onnx]'` hint
**And** the round-trip test `importorskip`s `onnxruntime`.

**AC6 — Works unchanged as a `reduced-field` inner (POD-NN), and the package is self-contained.**
**Given** a `reduced-field` config with `model.inner.method: torch-mlp` (plugin installed)
**When** the run trains end-to-end on a smooth low-rank field
**Then** it fits one network per mode through the **unchanged** `ReducedFieldTrainer` and reconstructs
the field (R² above a sensible bar)
**And** the plugin is a self-contained distribution: its own `pyproject.toml` (name
`surroforge-torch`, deps `surroforge` + `torch`, an `[onnx]` extra), its own `README.md`, and its own
test suite; `ruff`/`mypy` are clean on the package; nothing GPL/LGPL is introduced (torch is BSD-3 →
NFR8).

## Tasks / Subtasks

- [x] **Task 1 — package skeleton + registration** (AC1, AC6)
  - [x] `plugins/surroforge-torch/pyproject.toml`: name `surroforge-torch`, `dependencies =
        ["surroforge", "torch>=2.0"]`, `[project.optional-dependencies] onnx = [...]`, and
        `[project.entry-points."surroforge.trainers"] torch-mlp = "surroforge_torch.trainer:TorchMLPTrainer"`.
  - [x] `src/surroforge_torch/__init__.py` (exports `TorchMLPTrainer`); `README.md` (install +
        the `mlp` → `torch-mlp` switch + reproducibility/ONNX notes).
  - [x] Confirm **no** Core file is touched for registration (entry point only).
- [x] **Task 2 — `TorchMLPTrainer`** (AC2, AC3)
  - [x] `class TorchMLPTrainer(BaseTrainer)`, `name = "torch-mlp"`; curated `_SUPPORTED` (mirrors
        `mlp`); `__init__` validates unknowns/enums/seed/numeric → `TrainerError`; `super().__init__(...)`.
  - [x] A module-level `_StandardizedMLP(nn.Module)` (buffers for input/output mean-std; `forward`
        normalises → net → denormalises) so it is picklable and ONNX-exportable.
  - [x] `fit`: `_as_2d` both; standardise; build under saved/seeded/restored torch RNG; full-batch
        optimise (`lbfgs` default; `adam`/`sgd` epoch loop with `tol` early-stop); assign `_model`
        only on success. `predict`: not-fitted → `TrainerError`; `eval()` + `no_grad`; `_as_2d`.
- [x] **Task 3 — capabilities** (AC4, AC5)
  - [x] `reproduction_tolerance = 1e-4`; `export_formats = frozenset({"pickle", "onnx"})`.
  - [x] `to_onnx(n_features)`: not-fitted + feature-count + missing-`onnx` (`find_spec`) guards →
        `ExportError` (the `onnx`-missing one names the `surroforge-torch[onnx]` extra);
        `torch.onnx.export(..., dynamo=False, dynamic_axes=...)`; other failures wrapped → `ExportError`.
- [x] **Task 4 — tests** (AC2, AC3, AC4, AC5, AC6)
  - [x] `tests/test_torch_mlp.py`: registration (after `pip install -e`), 2-D shape (single + multi),
        picklable bit-identical, learns nonlinear (R² bar), unknown/invalid-hyperparameter rejection,
        predict wrong-feature-count guard, same-seed-reproduces (unguarded — the 1e-4 band absorbs
        Accelerate drift) + different-seed-diverges, not-fitted, `check_trainer` passes (all 12), ONNX
        round-trip (`importorskip onnxruntime`, single + multi), ONNX wrong-feature-count +
        missing-`onnx` hint + before-fit, reduced-field `inner: torch-mlp` e2e.
- [x] **Task 5 — gates, docs & BMad** (all ACs)
  - [x] `pip install -e plugins/surroforge-torch`; `pytest plugins/surroforge-torch/tests`,
        `check_trainer` green; `ruff`/`mypy` clean on the package.
  - [x] `docs/reference/configuration.md`: name `torch-mlp` as the concrete plugin example;
        `CHANGELOG.md` `[Unreleased]`: add the `surroforge-torch` plugin.
  - [x] Epic R7 stub in `epics-field-rom-v1.md`; `sprint-status.yaml` entries.

### Review Findings (code-review 2026-06-15)

Full 3-layer adversarial review (Blind Hunter / Edge-Case Hunter / Acceptance Auditor) run as a
workflow, each finding **independently re-verified by running code** in the plugin venv. 8 findings;
7 confirmed (1 dismissed). The Acceptance Auditor confirmed the **scope is clean** (`git status`: no
`src/surroforge/**`, no Core `pyproject.toml`, no `_BUILTIN` change — only the new plugin dir +
docs/CHANGELOG/BMad), the **Golden Path** holds (`torch-mlp` registered via entry point only), and
**all 12 `check_trainer` checks pass** on Accelerate. All 7 confirmed findings were fixed (the four
cosmetic/nit ones too — each genuinely improved correctness or honesty).

**Patches applied (test-pinned):**

- [x] [Review][Patch] **b1** — an oversized `random_state` (≥ 2⁶⁴ or < −2⁶³) leaked a bare
      `ValueError` from `torch.manual_seed` deep in `fit` (violates the no-bare-error `[MUST]`).
      Added a 64-bit-range bound check in `__init__` → `TrainerError`. [trainer.py:__init__] (blind)
- [x] [Review][Patch] **b2** — `float(loss)` on a grad-tracking tensor emitted a `UserWarning` on
      every `adam`/`sgd` fit. Switched to `loss.item()` (identical value, no warning).
      [trainer.py:_optimise] (blind)
- [x] [Review][Patch] **b3** — `alpha` meant a 2× different L2 across solvers and inconsistently
      (in/ex)cluded biases. Aligned: `0.5` factor on the lbfgs penalty (gradient `alpha*w`) + adam/sgd
      `weight_decay` via param groups on **weights only** — so `alpha` is consistent (and matches
      sklearn). [trainer.py:_optimise/_l2_penalty] (blind)
- [x] [Review][Patch] **e4** — `predict()` with a wrong feature count raised a bare torch
      `RuntimeError` (and silently mispredicted a 1-D input). Added the same width guard `to_onnx`
      has → `TrainerError`. [trainer.py:predict] (edge)
- [x] [Review][Patch] **e5** — the legacy exporter serialises through `onnx` (not `onnxscript`); a
      missing `onnx` raised torch's `OnnxExporterError`, so the dead `ModuleNotFoundError` branch
      meant the `[onnx]` hint never fired. Replaced it with a proactive `find_spec("onnx")` guard →
      `ExportError` with the hint; fixed the test to exercise the real path; corrected the
      pyproject/README "torch alone" claim. [trainer.py:to_onnx, test, pyproject] (edge)
- [x] [Review][Patch] **repro-teeth** — the **labeling was inverted**: the different-seed (negative)
      test does NOT pin a seed-ignoring impl (an unseeded net still diverges); the **same-seed** test
      is the one with teeth. Corrected the test docstrings + story AC4/Dev-Notes accordingly. (auditor)
- [x] [Review][Patch] **doc-divergence** — "≈ 1.0 / four orders" came from a throwaway probe; the
      real test regime (standardised, deep net) diverges ~1e-2 (~100×, two orders). Corrected the
      code comments + story; the verdict (a real divergence still fails) is unchanged. (auditor)

**Dismissed (1):** `lr-unused-lbfgs` — `learning_rate_init` is inert under the default `lbfgs`.
Intentional, exact parity with scikit-learn's `MLPRegressor` (which also ignores it under lbfgs), and
already documented in the constructor docstring; adding a warning would *break* the
one-line-switch parity that is the package's whole point.

## Dev Notes

### What to reuse (do NOT reinvent)

- **`src/surroforge/trainers/mlp.py` is the template** — clone its `_SUPPORTED` set, the
  unknown/enum/seed/numeric validation with `TrainerError` messages, the not-fitted guard, the
  build-then-assign in `fit`, and the `export_formats`/`reproduction_tolerance` shape. `torch-mlp`
  intentionally mirrors `mlp`'s public surface so the two are interchangeable via `method:`.
  [Source: src/surroforge/trainers/mlp.py]
- **`BaseTrainer`** supplies `_as_2d`, `supports_export`, the class attributes, and the picklability
  contract; `predict` MUST return 2-D. [Source: src/surroforge/trainers/base.py]
- **Registry seam** — `_resolve` lets a single external entry point provide a method; a built-in is
  only a fallback. The plugin needs **only** an entry point. [Source: src/surroforge/plugins/registry.py:81-100]
- **Conformance** — `check_trainer` (12 checks). The `deterministic` and `picklable` checks compare
  predictions within the declared/`floor` tolerance; both are bit-identical for this design.
  [Source: src/surroforge/conformance/trainer.py]
- **`requires_reproducible_blas`** — Core's marker exists for tight bit-level bands (mlp's 1e-6);
  torch-mlp's looser 1e-4 float32 band absorbs Accelerate drift, so its same-seed test runs
  **unguarded** (it passes locally on Accelerate too). [Source: tests/conftest.py:12-38]

### Architecture compliance (project-context.md — non-negotiable)

- **Plugin Golden Path `[MUST]`:** `name` lowercase == entry-point key == `model.method`;
  **third-party/framework ⇒ entry points only, never `_BUILTIN`** — this story IS that rule. The
  trainer must be picklable (it is the persisted artifact); a torch `nn.Module` and its tensors are.
- **Spine theorem (AR-R1):** no Core change. If a change reaches into `engine`/`config`/`services`/
  `cli`/Core `pyproject.toml`/`_BUILTIN`, stop — the seam is supposed to carry this from outside.
- **Errors `[MUST]`:** raise `TrainerError`/`ExportError` (never a bare torch/Python error); messages
  must state the fix (list supported hyperparameters; name the `[onnx]` extra; give the feature count).
- **Reproducibility `[CONV]`/`[MUST]`:** the run `seed` is **not** threaded into the trainer, so
  `torch-mlp` carries its own integer `random_state` (deterministic default) and declares a positive
  `reproduction_tolerance`. A numerics method is not done without the **negative** divergence test.
- **Licensing `[MUST]`:** PyTorch is BSD-3 → NFR8 holds. The plugin is a separate distribution, so it
  does not change Core's dependency surface or license posture.
- **Optional-dependency note:** unlike a Core leaf, this package's *sole reason to exist* is torch, so
  a top-level `import torch` in the plugin module is correct — the registry only `ep.load()`s it when
  `torch-mlp` is actually resolved (listing names never imports it). The ONNX **inference** backend
  (`onnxruntime`) stays optional behind the package's `[onnx]` extra + an install hint.

### Domain notes (verified during story prep — torch 2.12, CPU, py3.13/arm64)

- **Determinism:** save `torch.get_rng_state()` → `torch.manual_seed(random_state)` → build the net →
  `torch.set_rng_state(state)`; train **full-batch** (no shuffling). Measured same-seed max|Δ| = 0.0
  with `np.random` perturbed between fits; different-seed max|Δ| ≈ 1e-2 (standardised test regime).
  Hence `reproduction_tolerance = 1e-4` is honest (float32) with a ~100× (two-order) divergence margin.
- **ONNX:** torch 2.12's `torch.onnx.export` defaults to the dynamo exporter (needs `onnxscript`);
  pass `dynamo=False` for the dependency-light legacy (TorchScript) exporter, which round-trips to
  `onnxruntime` within ~3e-8. Bake standardisation into the module (buffers in `forward`) so the graph
  matches `predict`.
- **Picklability:** an `nn.Module` (and its float32 tensors/buffers) pickles; the module class must be
  **module-level** (not nested) to be referenceable by pickle. The unpickled module predicts
  bit-identically in-process.

### Testing — anti-vacant-pin / fixture diversity (project-context.md)

- **Fixture diversity `[GOTCHA]` (R5 retro):** include the awkward classes — **multi-output**, the
  **different-seed divergence** negative case, **unknown/invalid-hyperparameter** rejection, the
  **reduced-field inner** e2e, the **ONNX wrong-feature-count** + **missing-backend** guards.
- **Vacant pins `[GOTCHA]`:** the **same-seed** test is what fails a trainer that ignores
  `random_state` (a same-seed refit must be bit-identical); the picklable test must fail a
  non-bit-identical round trip; prefer named tolerances over substrings.
- The same-seed test runs **unguarded** (the 1e-4 float32 band absorbs Accelerate drift); the ONNX
  round-trip `importorskip`s `onnxruntime`; the core fit/predict path needs only torch.

### Scope boundary

R7.1 = the `surroforge-torch` package: `TorchMLPTrainer` + entry-point registration + capabilities
(tolerance, ONNX) + its own tests + README + a Core docs/CHANGELOG note. It does **NOT** touch Core
source, add GPU/CUDA handling, mini-batch `DataLoader` training, target/early-stopping schedules
beyond a simple `tol`, DeepONet/FNO, a CI torch job (a separate decision — torch bloats every Core
job), or an `examples/` script (a post-story follow-up like 06/07). If a change reaches into Core,
stop — this is an external-plugin distribution only.

### Project Structure Notes

- Add (new package): `plugins/surroforge-torch/{pyproject.toml,README.md}`,
  `plugins/surroforge-torch/src/surroforge_torch/{__init__.py,trainer.py}`,
  `plugins/surroforge-torch/tests/test_torch_mlp.py`.
- Touch (Core, docs only): `docs/reference/configuration.md`, `CHANGELOG.md`.
- Touch (BMad): `_bmad-output/planning-artifacts/epics-field-rom-v1.md` (Epic R7 stub),
  `_bmad-output/implementation-artifacts/sprint-status.yaml`.
- **No** change to any `src/surroforge/**` file, Core `pyproject.toml`, or `_BUILTIN` (AR-R1 + Golden Path).

### References

- [Source: src/surroforge/trainers/mlp.py] — the surface/validation template `torch-mlp` mirrors.
- [Source: src/surroforge/trainers/base.py] — the contract (`_as_2d`, 2-D predict, export/tolerance).
- [Source: src/surroforge/plugins/registry.py:81-100] — `_resolve`: a single external entry point
  provides a method (why no Core change is needed).
- [Source: src/surroforge/conformance/trainer.py] — `check_trainer` (12 checks) the plugin must pass.
- [Source: src/surroforge/engine.py:92-99] — `trainer_cls(**hyperparameters)` factory.
- [Source: tests/conftest.py:12-38] — `requires_reproducible_blas` for the same-seed test.
- [Source: r6-1-neural-network-mlp-trainer.md:26-34] — the R6.1 backend decision deferring PyTorch
  to exactly this external seam.
- [Source: epic-r6-retro-2026-06-15.md, Action Item 3] — "heavyweight NN = a new epic on the
  external seam … do not add it to Core."

## Dev Agent Record

### Agent Model Used

claude-opus-4-8[1m] (Claude Opus 4.8, 1M context)

### Debug Log References

- Verified during prep (torch 2.12, CPU, py3.13/arm64): a same-seed refit is **bit-identical**
  (max|Δ| = 0.0) even with both `np.random` and the torch global RNG perturbed between fits; a
  different seed diverges by ≈ 1e-2 (~100× the 1e-4 band) in the standardised test regime. The legacy
  `torch.onnx.export(dynamo=False)` round-trips to `onnxruntime` within ~3e-8 (single- and multi-output).
- `check_trainer(TorchMLPTrainer)` passes **fully on Apple Accelerate** (`report.passed`), with no
  BLAS carve-out — unlike gradient-MLE kriging, the seeded full-batch fit is exactly reproducible,
  so the same-seed reproduce test runs unguarded (looser float32 band `1e-4`).
- torch 2.12's exporter emits a `DeprecationWarning` for the legacy path; suppressed inside
  `to_onnx` (deliberate, non-actionable choice) so user output stays clean — confirmed by running
  the ONNX tests under `-W error::DeprecationWarning`.
- mypy: two torch-stub-vs-runtime mismatches on `torch.onnx.export` (its stubs target the new dynamo
  signature; the legacy path accepts a single-tensor args + a file-like sink) resolved with the
  canonical `(dummy,)` args-tuple + one targeted `# type: ignore[arg-type]` on the `BytesIO` sink.

### Completion Notes List

- **Plugin (new distribution):** `surroforge-torch` (`plugins/surroforge-torch/`) — a real PyTorch
  MLP (`TorchMLPTrainer`) registered as `torch-mlp` via the `surroforge.trainers` entry point
  **only**; **no Core file changed** for registration (the registry resolves a single external
  entry point directly). First real exercise of the documented plugin seam.
- **The switch is `method:`** — `torch-mlp` mirrors `mlp`'s eight-hyperparameter surface, so a user
  flips backend with a one-line edit (`model.method: mlp` ↔ `torch-mlp`).
- **Determinism by construction:** own integer `random_state`; weights initialised under a
  saved/seeded/restored torch RNG (global RNG left untouched); full-batch training (no shuffling).
  `reproduction_tolerance = 1e-4` (honest float32; ~two orders below a real divergence). The
  **same-seed** reproduce test is the one with teeth (it fails a seed-ignoring impl); the different-seed
  test guards against a vacuous tolerance.
- **ONNX:** `export_formats = {pickle, onnx}`; `torch.onnx.export(dynamo=False)` with input/output
  standardisation baked into the module (buffers in `forward`) so the graph matches `predict`;
  not-fitted / wrong-feature-count / missing-`onnx` (`find_spec`) guards all raise `ExportError`.
- **Picklable:** holds a module-level `_StandardizedMLP(nn.Module)`; unpickled model predicts
  bit-identically. Passes the shipped `check_trainer` suite (all 12) and works as a `reduced-field`
  inner (POD-NN), proven e2e.
- **Validation:** plugin suite **29 passed**; Core **402 passed / 7 skipped** (no regression — no
  Core test pins the exact trainer set); `ruff`/`mypy` clean on the package. PyTorch is BSD-3 → NFR8.

### File List

- `plugins/surroforge-torch/pyproject.toml` (new) — package + `[onnx]`/`[dev]` extras + the
  `torch-mlp` `surroforge.trainers` entry point.
- `plugins/surroforge-torch/src/surroforge_torch/__init__.py` (new) — exports `TorchMLPTrainer`.
- `plugins/surroforge-torch/src/surroforge_torch/trainer.py` (new) — `TorchMLPTrainer` +
  `_StandardizedMLP` (fit/predict/to_onnx + hyperparameter validation).
- `plugins/surroforge-torch/tests/test_torch_mlp.py` (new) — full suite (contract, multi-output,
  validation, reproducibility ±, conformance, ONNX, reduced-field inner e2e).
- `plugins/surroforge-torch/README.md` (new) — install + the `mlp` → `torch-mlp` switch.
- `docs/reference/configuration.md` (modified) — `torch-mlp` named as the concrete plugin example.
- `CHANGELOG.md` (modified) — `[Unreleased]` `surroforge-torch` plugin entry.
- `_bmad-output/planning-artifacts/epics-field-rom-v1.md` (modified) — Epic R7 stub.
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (modified) — Epic R7 / R7.1 entries.

## Change Log

| Date | Change |
|---|---|
| 2026-06-15 | Story drafted (R7.1, Phase-2 Epic R7 — first external-plugin distribution). Placement decision recorded: separate `surroforge-torch` package, entry-points-only (never `_BUILTIN`); switch via `method: mlp` ↔ `torch-mlp`. Torch determinism + legacy-ONNX feasibility verified during prep. Status → ready-for-dev. |
| 2026-06-15 | R7.1 implemented: `surroforge-torch` plugin (`TorchMLPTrainer` on PyTorch — seeded/full-batch determinism, float32 `reproduction_tolerance=1e-4` + negative test, ONNX via torch's legacy exporter with standardisation baked in, POD-NN inner). Entry-points-only; **no Core change**. Plugin suite 28 passed; Core 402 passed / 7 skipped; ruff/mypy clean. Status → review. |
| 2026-06-15 | Code review (full 3-layer adversarial, each finding verified by running code): 8 findings, 7 confirmed + applied (random_state bound check, `loss.item()`, alpha-L2 alignment across solvers, predict feature-count guard, proactive `onnx` find_spec hint, corrected inverted repro-teeth labeling, corrected divergence magnitude), 1 dismissed (lr-inert-under-lbfgs — intended sklearn parity). Scope clean (no Core change), Golden Path + all 12 conformance checks confirmed. Plugin suite 29 passed; Core 402 passed / 7 skipped; ruff/mypy clean. Status → done. |
| 2026-06-15 | Follow-up (like 06/07 after R6.1): added `examples/08_torch_mlp.py` — the one-line `mlp` ↔ `torch-mlp` backend switch (same hyperparameter block; mlp R²=0.9993 vs torch-mlp R²=0.9992 on the oscillatory target). Gracefully degrades to mlp-only when the plugin is absent (verified CI-safe: exits 0 with torch uninstalled). README/docs/examples + CHANGELOG rows added; example smoke-tested. |
