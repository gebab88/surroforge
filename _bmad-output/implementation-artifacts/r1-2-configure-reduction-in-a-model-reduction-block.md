# Story R1.2: Configure reduction in a `model.reduction` block

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a methods engineer,
I want to declare the reduction (POD, an energy **or** rank target) in a structured `model.reduction` block beside a `model.inner` method block, validated at config load,
so that a reduced run is a single, versionable, Pydantic-validated source of truth — and `rank: adaptive` is resolved at fit and written back so the **snapshot is canonical**.

> **Architectural decision (made by the lead for this story).** R1.2 deliberately puts ROM-specific structure into the generic core config (`ModelConfig` gains structured `reduction`/`inner` blocks with load-time **`ConfigError`**), accepting a bounded compromise of the spine theorem (#36) in exchange for the literal AC: Pydantic validation at parse time. The engine and the `ReducedFieldTrainer` stay decoupled via two mechanisms below (the `hyperparameters` bridge + a duck-typed snapshot-override hook), so the compromise is contained to the config model. This was chosen over the spine-preserving alternative (trainer-only `TrainerError` validation) — see the Change Log.

## Acceptance Criteria

1. **Given** a config with `model.method: reduced-field`, a `model.reduction` block (`basis: pod`, and **exactly one** of `energy: <0–1>` or `rank: <int≥1 | adaptive>`), and a `model.inner` method block **when** the config is loaded **then** Pydantic validates it: `basis` defaults to/accepts only `pod`; `energy` must be in `(0, 1]`; `rank` is a positive int or the literal `adaptive`; the `inner` block is an ordinary method spec (`method: <name>` + method hyperparameters) **resolved through `get_trainer` at fit** (not at config load — config stays a leaf).
2. **Given** an invalid reduction **when** the config is loaded **then** a **`ConfigError`** is raised (naming the offending field) for: **both** `energy` and `rank` set; **neither** set; `basis` ≠ `pod`; `energy` outside `(0, 1]`; a non-positive / non-`adaptive` `rank`.
3. **Given** `reduction.rank: adaptive` **when** the run trains **then** the engine resolves the rank at fit time and **writes the resolved integer rank into the config snapshot** (`config.snapshot.yaml`), and `config_hash` (and, for the field run, `spec_hash`) is computed from that **resolved** snapshot — so the snapshot, not the input YAML, is canonical. `reproduce()` reads the resolved rank and re-trains deterministically.
4. **Given** a **scalar** run (no `reduction`/`inner` block) **then** nothing changes: the config snapshot, `config_hash`, and reproduce behaviour are **byte-for-byte identical** to pre-R1.2 (the R2.4 scalar-identity guard stays green). A field run with `energy` (not `adaptive`) is likewise unchanged (no write-back; existing R2.3 field tests stay green).
5. **Given** these new config blocks **then** the existing flow is preserved: the engine's method-agnostic `trainer_cls(**hyperparameters)` factory still receives `reduction`/`inner` as dict kwargs, and the `ReducedFieldTrainer` keeps its own `TrainerError` validation as defense-in-depth (the config layer is strict; the trainer remains independently valid for the conformance suite / Python API).

> **Scope boundary (do not exceed).**
> - **`exclude_none=True` in config serialization is mandatory and load-bearing** (see Task 2 / Dev Notes). Without it, adding `reduction`/`inner = None` fields makes every scalar config dump emit `reduction: null, inner: null`, changing `config_hash` and **breaking AC4 / the R2.4 guard**. This is the single highest-risk detail.
> - **Do NOT resolve `get_trainer` / import `plugins` at config-load time** — config stays a leaf (imports only `errors`). The inner method is resolved by the trainer at fit, as today.
> - **Do NOT remove the trainer's own reduction validation** — keep it (defense-in-depth; the conformance suite and API construct the trainer directly, bypassing config). Config validation is an *additional* strict layer, not a replacement.
> - **Do NOT change the field verdict, tolerance, `_field_energy_relative_diff`, or `assert_reproducible`** (R2.3/R2.5 frozen). The only hashing change is computing `config_hash`/`spec_hash` from the *resolved* config for `rank: adaptive` runs.
> - **Do NOT make `reduction` mandatory for `reduced-field` at the config level** (that would put method→requirement logic in the generic config). `reduction` is optional; if present it is validated; if absent the trainer defaults (`energy=0.999`), as today.
> - The write-back fires **only** for `rank: adaptive` (replace the literal with the resolved int). An `energy`-specified run leaves the config as-is (energy is the canonical spec; the resolved rank is already a lineage diagnostic via R2.3's `reduction.rank`).

## Tasks / Subtasks

- [x] **Task 1 — Structured `ReductionConfig` + `InnerConfig` with `ConfigError` validation** (AC: 1, 2)
  - [x] In `src/surroforge/core/config.py` add `ReductionConfig(BaseModel)`: `model_config = ConfigDict(extra="forbid")`; `basis: Literal["pod"] = "pod"`; `energy: float | None = Field(default=None, gt=0.0, le=1.0)`; `rank: int | Literal["adaptive"] | None = None`. Add a `@model_validator(mode="after")` raising `ValueError` when **both** or **neither** of `energy`/`rank` are set, and a `@field_validator("rank", mode="before")` rejecting a non-positive int and `bool`. **`mode="before"` is mandatory here:** under the `int | Literal["adaptive"]` union an *after* validator never sees `True` — Pydantic coerces `True → 1` first — so a `@field_validator("rank")` (default `mode="after"`) would let `rank: true` silently become `rank=1` and the AC2 / Task-4 `rank: true` test would FAIL. A `mode="before"` validator sees the raw `True` and can reject it (`isinstance(v, bool)`). Pydantic `ValidationError` is converted to `ConfigError` by the existing `parse_config` wrapper (config.py:131-132) — verify the messages name the field.
  - [x] Add `InnerConfig(BaseModel)`: `model_config = ConfigDict(extra="allow", protected_namespaces=())`; `method: str`; a `hyperparameters` property returning `dict(self.__pydantic_extra__ or {})` (mirrors `ModelConfig`). Do **not** resolve `get_trainer` here.
  - [x] On `ModelConfig` add `reduction: ReductionConfig | None = None` and `inner: InnerConfig | None = None` (keep `extra="allow"` for scalar-method hyperparameters like `degree`/`corr`).

- [x] **Task 2 — Preserve back-compat: `hyperparameters` bridge + `exclude_none` serialization** (AC: 4, 5)
  - [x] Update `ModelConfig.hyperparameters` to **synthesize the trainer kwargs** from the structured blocks so the engine's `trainer_cls(**hyper)` is unchanged: start from `dict(self.__pydantic_extra__ or {})`; if `self.reduction is not None` add `hyper["reduction"] = self.reduction.model_dump(exclude_none=True)`; if `self.inner is not None` add `hyper["inner"] = self.inner.model_dump()`. (For a scalar method both are `None` ⇒ `hyperparameters` is unchanged, e.g. `{"degree": 2}`.) The `ReducedFieldTrainer.__init__(reduction=None, inner=None, **extra)` then receives the same dicts it does today.
  - [x] **Make config serialization `exclude_none` so the new `None` fields never reach the wire (the R2.4-critical detail).** Change `ExperimentConfig.canonical_dict()` to `self.model_dump(mode="json", exclude_none=True)` and `to_yaml()` to dump `self.model_dump(mode="json", exclude_none=True)`. Pre-R1.2 configs had **no** `None` fields, so `exclude_none` is a no-op for them ⇒ a scalar config dumps and hashes **byte-identically** to today (no `reduction: null` / `inner: null` keys); within a field run's `reduction`, the unset `energy`/`rank` is dropped, matching r1-3's extra-dict shape.
  - [x] Update the engine's snapshot write (engine.py:177 — `write_yaml(run_dir / CONFIG_SNAPSHOT, config.model_dump(mode="json"))`) to use the same `exclude_none` dump (route it through `canonical_config.canonical_dict()` or `model_dump(mode="json", exclude_none=True)`) so the persisted snapshot matches what is hashed. **Two separate call sites** must both get the flag: `canonical_dict()` (drives the hash) **and** the snapshot write at engine.py:177 (drives the file). They are independent today — if only one is changed, the persisted snapshot desyncs from the hashed bytes and the `Run.config` re-parse round-trip differs.

- [x] **Task 3 — `rank: adaptive` resolved-rank write-back to the canonical snapshot** (AC: 3, 4)
  - [x] In `src/surroforge/trainers/reduced_field.py` add a duck-typed hook `resolved_config_overrides(self) -> dict | None`: return `{"model": {"reduction": {"rank": self.rank}}}` **only** when the input rank was the literal `"adaptive"` (track this — e.g. `self._rank == "adaptive"`) and the model is fitted; otherwise return `None`. Keep `__init__`'s existing `TrainerError` validation (defense-in-depth).
  - [x] In `src/surroforge/engine.py`, after the final model is fitted and **before** persisting, compute a `canonical_config`: if `hasattr(final_model, "resolved_config_overrides")` and it returns a non-empty dict, **deep-merge** that dict into `config.model_dump(mode="json", exclude_none=True)` and `parse_config(...)` the result; otherwise `canonical_config = config`. Add a small generic `_deep_merge(base, overrides)` helper (the engine stays ROM-agnostic — it merges an opaque dict, it does not name `reduction.rank`). Use `canonical_config` for the snapshot write, `_config_hash`, and `_spec_hash`.
  - [x] **Scalar / energy-field runs:** no hook (scalar) or hook returns `None` (energy field) ⇒ `canonical_config is config` ⇒ snapshot + hashes byte-identical to today. Only an `adaptive` field run gets the rewritten rank + resolved-config hash.

- [x] **Task 4 — Tests** (AC: 1–5)
  - [x] **`tests/test_config.py` (extend):** valid `reduction` blocks parse (`{basis: pod, energy: 0.999}`, `{basis: pod, rank: 12}`, `{rank: adaptive}`); `ConfigError` for both-set, neither-set, `basis: svd`, `energy: 1.5`, `energy: 0`, `rank: 0`, `rank: -1`, `rank: true`; `inner` parses as `{method, **hyper}`; `hyperparameters` synthesizes `reduction`/`inner` dicts; a **scalar** config's `hyperparameters` and `canonical_json()` are unchanged (assert no `reduction`/`inner` key, and that `canonical_json` of a polynomial config equals the pre-R1.2 string — pin via recomputation, not a golden literal).
  - [x] **`tests/test_reduced_field.py` or a new `tests/test_reduction_config.py`:** an `energy` field run trains unchanged; a `rank: 3` field run trains with exactly rank 3; a **`rank: adaptive`** run trains, and `run.config_dict["model"]["reduction"]["rank"]` is a concrete **int** (not `"adaptive"`), and `run.reduction["rank"]` (lineage) equals it.
  - [x] **`rank: adaptive` reproduce determinism:** an adaptive run reproduces (`reproduce()` verdict is a reproduce; `assert_reproducible` does not raise) — the snapshot's resolved rank makes re-training deterministic. Assert the reproduction's snapshot rank equals the parent's.
  - [x] **AC4 regression (teeth):** run `tests/test_reproduce_scalar_identity.py` (the R2.4 byte-identity guard) and `tests/test_reproduce_field.py` (R2.3) — both must stay green. **But note: the R2.4 guard does NOT pin the scalar `config_hash` *value*** — it only asserts `config_hash` is *present* in lineage and pins the prediction-fingerprint bytes. A botched `exclude_none` (emitting `reduction: null`) would change `config_hash` and **still pass R2.4**. So this story's new assertion is the **only** net for that failure mode — do not skip it: assert a scalar polynomial run's `canonical_json()` contains no `reduction`/`inner` key (and, ideally, equals a recomputed pre-R1.2 shape — by recomputation, not a golden literal).

- [x] **Task 5 — Gates green**
  - [x] `ruff check src tests`, `mypy src` (the `int | Literal["adaptive"] | None` union + model validators are typed), `lint-imports` (config must **not** gain an import of `plugins`/`engine`; it stays a leaf importing only `errors` + pydantic/yaml — re-run to confirm the 3 contracts). Full `pytest` (0 failed). Explicitly run the pinned-threads reproduce family (`test_reproduce.py test_canonicalize.py test_reproduce_field.py test_reproduce_scalar_identity.py test_probe_coverage.py`) + the new tests under `OMP/OPENBLAS/MKL_NUM_THREADS=1` to mirror CI.

### Review Findings

- [x] [Review][Decision] (RESOLVED: accept — pre-release, no shipped users; snapshots are canonical from R1.2 on; documented in deferred-work.md) Pre-R1.2 field-run compat break (legacy snapshots) — pre-R1.2-legal shapes `reduction: {basis: pod}` / `reduction: {}` / `inner: {degree: 2}` (no method) no longer parse: `Run.config` re-parses old snapshots through the new strict models and raises `ConfigError`, breaking reproduce/predict/compare/report on r1-3-era persisted runs; re-parsed old field snapshots also hash differently (`basis: pod` default injected; `energy: 1` → `1.0`). The *input-YAML* side is AC1/AC2-mandated (neither-set MUST raise; `method` is part of the inner spec) — the open question is only legacy persisted runs. Options: (a) accept the break (pre-release, no shipped users; document) or (b) add a lenient legacy-snapshot re-parse/migration.
- [x] [Review][Patch] Adaptive write-back reads the FINAL model's rank, but the fingerprint comes from the HOLDOUT model — when the energy threshold straddles a mode (verified empirically: holdout rank 6 vs final rank 7), `reproduce()` re-trains the holdout at the snapshot's rank and falsely diverges (4.9e-1 vs 1e-9), and lineage `reduction.rank` (holdout) contradicts the snapshot rank. Fix: query `resolved_config_overrides()` on the holdout model (the model that defines the fingerprint and the R2.3 lineage diagnostics) + a straddling-spectrum regression test. [src/surroforge/engine.py:170]
- [x] [Review][Patch] Explicit-`null` hyperparameter extras are silently dropped from snapshot + `config_hash` while the parent trainer was constructed WITH them (`hyperparameters` keeps None extras) — train/reproduce mismatch; the `canonical_dict` docstring's "no pre-R1.2 config has a None-valued field" overclaims (true for declared fields, false for extras). Fix: reject None-valued extras in `ModelConfig`/`InnerConfig` with a clear ConfigError; correct the docstring. [src/surroforge/core/config.py]
- [x] [Review][Patch] `energy: true` silently coerces to `energy=1.0` — the same lax-bool coercion the story explicitly guarded on `rank` is unguarded on `energy`. Fix: `mode="before"` bool rejection on energy. [src/surroforge/core/config.py]
- [x] [Review][Patch] The AC4 scalar-serialization test is partially circular (`dumped == _drop_nones(model_dump)` is near-tautological). Fix: pin the exact expected `model` section by construction. [tests/test_config.py]
- [x] [Review][Patch] Duplicated prefix in validator messages — renders as `model.reduction: Value error, reduction: specify exactly one…`. Fix: drop the hardcoded `reduction:` prefix (the Pydantic loc already names the field). [src/surroforge/core/config.py]
- [x] [Review][Defer] Engine ROM-hook robustness for external plugins — a non-dict hook return crashes `_deep_merge` with `AttributeError`; a failing post-fit re-parse (`ConfigError` at `<resolved config>`) discards the completed training run; an override touching `experiment`/`method` would desync lineage (read from the pre-merge config). Built-in hook is well-formed; folds into the standing R2.3/R2.5 "external-plugin ROM-hook input validation" deferral. [src/surroforge/engine.py:170] — deferred, pre-existing deferral family
- [x] [Review][Defer] Declared `reduction`/`inner` fields shadow same-named hyperparameters of third-party trainers (entry-point seam) — such configs/snapshots now fail strict parsing. No external plugin is bound; revisit with the seam-hardening deferral. [src/surroforge/core/config.py:125] — deferred, pre-existing seam policy

## Dev Notes

### The central realization (read this first)
**This story changes the core config model and touches the reproducibility hash — it is the highest-risk R1 story.** Two mechanisms keep the blast radius contained:
1. **The `hyperparameters` bridge** re-synthesizes `reduction`/`inner` as dict kwargs, so the engine's method-agnostic `trainer_cls(**hyperparameters)` factory and the `ReducedFieldTrainer.__init__` signature are **both unchanged**. The structured config is purely a *front-end* validation layer.
2. **`exclude_none=True` serialization** keeps scalar configs (and energy-field configs) serializing/hashing exactly as before, so AC4 and the R2.4 guard hold.

### Why `exclude_none` is load-bearing (do not skip)
Adding declared fields `reduction: ReductionConfig | None = None` and `inner: InnerConfig | None = None` to `ModelConfig` means a **plain `model_dump()` emits `"reduction": null, "inner": null` for every config**, including scalar polynomial/kriging/rbf runs. That would change `config_hash` for every scalar run and break the R2.4 byte-identity contract. `model_dump(exclude_none=True)` drops them; since no pre-R1.2 config model had any `None`-valued field, this is a **no-op for existing configs** → byte-identical. Apply it consistently to `canonical_dict()` (drives `config_hash`/`spec_hash` via `canonical_json()`) **and** the engine snapshot write (engine.py:177), so the persisted snapshot equals the hashed bytes.

### The `rank: adaptive` write-back + hash flow (AC3)
- Today (r1-3): `_resolve_rank` already resolves `"adaptive"` via the energy threshold at fit, and the resolved rank is recorded in `lineage.reduction.rank` (a diagnostic, via R2.3's `reproducibility_diagnostics`). What's missing is the **config snapshot** carrying the resolved int and the **hash** reflecting it.
- New flow: trainer exposes `resolved_config_overrides()` (returns the rank override only for adaptive); the engine deep-merges it into the config dict, re-`parse_config`s to a `canonical_config`, and computes the snapshot + `config_hash` + `spec_hash` from that. The engine merges an **opaque** dict (it never names `reduction.rank`), so it stays ROM-agnostic.
- **Reproduce determinism:** the snapshot now says `rank: 7`; `reproduce()` reads it, the trainer gets `rank=7` (not adaptive) → resolves to 7 trivially → re-trains identically; its snapshot/`config_hash` match the parent's (both rank 7). Note `reproduce()` does **not** gate on `config_hash` (it gates on `data_hash` + the field/scalar fingerprint), so even the metadata is now consistent rather than divergent.
- **Risk guardrails:** scalar runs (no hook) and energy-field runs (hook → `None`) keep `canonical_config is config` ⇒ unchanged bytes. Only adaptive field runs change. The R2.4 guard + R2.3 field tests are the regression net (Task 4/5).

### Validation lives in two layers (intentional)
The config validator (`ConfigError`, load time, strict) and the trainer's `__init__` validator (`TrainerError`, resolution time, self-contained) **both** enforce the reduction rules. This duplication is deliberate: the config gives early, precise UX errors; the trainer stays independently valid for the conformance suite (`check_trainer(ReducedFieldTrainer, hyperparameters={"inner": {...}})` constructs it with **no** reduction → trainer defaults `energy=0.999`) and the Python API. Keep both; they agree on the rules.

### Source-tree components to touch
- **UPDATE `src/surroforge/core/config.py`** — `ReductionConfig`, `InnerConfig`; `ModelConfig.reduction`/`.inner` fields + `hyperparameters` synthesis; `ExperimentConfig.canonical_dict()`/`to_yaml()` → `exclude_none=True`. Keep config a leaf (no `plugins`/`engine` import).
- **UPDATE `src/surroforge/engine.py`** — `canonical_config` via the duck-typed `resolved_config_overrides()` hook + a generic `_deep_merge`; route snapshot write + `_config_hash` + `_spec_hash` through `canonical_config`. The field branch (R2.3) and scalar path (R2.4) are otherwise untouched.
- **UPDATE `src/surroforge/trainers/reduced_field.py`** — track whether input rank was `"adaptive"`; add `resolved_config_overrides()`. Keep `__init__` `TrainerError` validation and `_resolve_rank`/`rank`/`_resolved_reduction` as-is.
- **TESTS** — extend `tests/test_config.py`; add reduction-config + adaptive-rank + reproduce-determinism tests; run the R2.4/R2.3 regression families.

### Current behaviour to preserve (read before editing)
- `ModelConfig` (config.py:50-63): `extra="allow"`, `hyperparameters` returns `__pydantic_extra__`. r1-3's reduced-field configs pass `reduction`/`inner` as **extra** today; after R1.2 they are declared fields but `hyperparameters` must still surface them as dicts so `trainer_cls(**hyper)` is unchanged.
- `ExperimentConfig.canonical_json()` (config.py:106-107) → `config_hash` (engine.py:219-220) and `spec_hash` (engine.py:223-236, field runs). Both must stay byte-identical for scalar + energy-field runs.
- `ReducedFieldTrainer.__init__` (reduced_field.py:49-90): validates `basis`/`energy`/`rank`/unknown keys with `TrainerError`, defaults `energy=0.999` when neither given, stores `self._rank` as int/"adaptive"/None. `_resolve_rank` (reduced_field.py:129-141) handles `"adaptive"` via energy. `rank` property (reduced_field.py:161-164) returns the fitted rank.
- The R2.4 guard `tests/test_reproduce_scalar_identity.py` and R2.3 `tests/test_reproduce_field.py` define the byte-identity / field-verdict contracts this story must not regress.

### Import-linter / architecture constraints (hard)
- **Config must stay a leaf** — adding `get_trainer` resolution at config load would create a `config → plugins` edge. Do NOT. The inner method resolves in the trainer at fit (as today). **Note: `lint-imports` does NOT machine-enforce this** — the 3 contracts forbid only cli/api imports, `engine→services`, and `errors`-leaf; none forbids `config→plugins`. So a green `lint-imports` is *not* proof you stayed a leaf — keep the discipline by review/intent, not by trusting the gate.
- The engine's `_deep_merge` is a pure helper (no new imports). The `resolved_config_overrides` hook is duck-typed (engine does not import `reduced_field`).

### Testing standards
- pytest, flat `tests/`. **[MUST] negative tests with teeth:** the `ConfigError` cases (both/neither/bad-basis/bad-energy/bad-rank) via `pytest.raises(ConfigError, match=…)`; the AC4 scalar `config_hash`-unchanged assertion (a regression in `exclude_none` fails here). **BLAS:** the SVD rank resolution is deterministic for the polynomial inner under pinned threads; assert the resolved rank is a specific int only for a controlled low-rank fixture (else assert `>= 1` / determinism), never a golden hash. Run the reproduce family under pinned threads (Task 5).

### Previous-story intelligence
- **R1.1** (done) added output globs at the `select_columns` choke point — orthogonal to R1.2 (config-model layer). **R1.3** (done) shipped the trainer with reduction/inner as hyperparameters + `TrainerError` validation; R1.2 wraps a strict config layer around that and adds the adaptive write-back. **R2.3/R2.4/R2.5** (done) own the field verdict + scalar byte-identity + coverage — R1.2 must keep all green; the `config_hash`/`spec_hash` flow is the shared seam.
- **Process (Epic R2 retro):** commit this story at `done` (after review) before starting r1-4.

### Project Structure Notes
- This is the one R1 story that consciously trades a slice of the spine theorem for the literal AC (Pydantic + `ConfigError`); the `hyperparameters` bridge + duck-typed write-back hook keep the engine and trainer decoupled so the compromise is contained to `core/config.py`. The `(planned)` tags in `project-context.md` for the reduction block / `rank: adaptive` write-back are this story.

### References
- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md#Story R1.2] — acceptance criteria (FR-R2, AR-R6; #29, #34).
- [Source: _bmad-output/planning-artifacts/prd-field-rom-v1.md §1.5] — "`model.reduction:`-Konfigblock (#29) mit Rang- oder Energieziel, Rang als aufgelöst im Snapshot (#34)."
- [Source: src/surroforge/core/config.py] — `ModelConfig` (50-63), `ExperimentConfig.canonical_dict/json/to_yaml` (102-110), `parse_config` ValidationError→ConfigError wrapper (125-132).
- [Source: src/surroforge/engine.py] — snapshot write (177), `_config_hash` (219-220), `_spec_hash` (223-236, field-only), factory (95-97).
- [Source: src/surroforge/trainers/reduced_field.py] — `__init__` TrainerError validation (49-90), `_resolve_rank` (129-141), `rank` (161-164), `_resolved_reduction` (196-198).
- [Source: tests/test_reproduce_scalar_identity.py] — R2.4 byte-identity guard (regression net for AC4).
- [Source: tests/test_reproduce_field.py] — R2.3 field tests (energy-field runs must stay green).
- [Source: tests/test_reduced_field.py] — existing reduced-field e2e (energy config) + conformance (`check_trainer` with no reduction → trainer default).

## Dev Agent Record

### Agent Model Used

claude-fable-5 (Claude Code)

### Debug Log References

- Red phase confirmed twice: 12 new config tests failed pre-implementation (`ModelConfig` had
  no `reduction` attribute); 2 engine tests failed pre-implementation (snapshot carried
  `rank: null` from the un-flagged dump; adaptive snapshot still said `"adaptive"`).
- Full suite: 259 passed, 2 skipped (smt-dependent, expected on the local py3.14 venv).
- Pinned-threads family (`OMP/OPENBLAS/MKL_NUM_THREADS=1`): test_reproduce, test_canonicalize,
  test_reproduce_field, test_reproduce_scalar_identity, test_probe_coverage, test_config,
  test_reduction_config → 82 passed, 1 skipped.
- Gates: `ruff check src tests` clean (one E501 in a new test docstring, fixed),
  `mypy src` clean (41 files), `lint-imports` 3 contracts kept. Config-leaf discipline
  verified by inspection: `core/config.py`'s only internal import remains `surroforge.errors`.

### Completion Notes List

- **Task 1:** `ReductionConfig` (extra=forbid; `basis: Literal["pod"]`; `energy` via
  `Field(gt=0, le=1)`; `rank: int | Literal["adaptive"] | None`) with the
  `@field_validator("rank", mode="before")` (rejects bool + non-positive int on the RAW
  value — the validated `rank: true` trap) and an after-`model_validator` for
  exactly-one-of. `InnerConfig` (extra=allow, `method: str`, `hyperparameters` property).
  Pydantic errors flow through the existing `parse_config` wrapper and name the field
  (asserted via `pytest.raises(ConfigError, match=<field>)` for all 8 invalid cases).
- **Task 2:** `ModelConfig.hyperparameters` synthesizes `reduction` (model_dump
  exclude_none) and `inner` (model_dump) so `trainer_cls(**hyper)` receives the same dicts
  as r1-3's extra-dict form. `canonical_dict()`/`to_yaml()` now dump `exclude_none=True`
  (a verified no-op for every pre-R1.2 config); the engine snapshot write routes through
  `canonical_config.canonical_dict()` so the persisted file equals the hashed bytes.
- **Task 3:** `ReducedFieldTrainer.resolved_config_overrides()` returns
  `{"model": {"reduction": {"rank": <int>}}}` only for a fitted adaptive run, else `None`.
  Engine computes `canonical_config` via the duck-typed hook + a generic `_deep_merge` +
  `parse_config` re-validate, and uses it for the snapshot write, `_config_hash`, and
  `_spec_hash`. Scalar runs (no hook) and energy/explicit-rank runs (hook → None) keep
  `canonical_config is config` ⇒ byte-identical bytes (R2.3/R2.4 regression nets green).
- **Task 4:** test_config.py +12 (valid blocks, 8 named ConfigError cases incl. `rank: true`,
  inner parse, hyperparameters bridge, AC4 scalar-serialization pin via recomputed
  None-free dump — no golden literal). New tests/test_reduction_config.py +4: energy run
  unchanged (snapshot spec + hash equals input-config hash), explicit rank trains at rank 3,
  adaptive write-back is a concrete int matching lineage with `config_hash` recomputed from
  the resolved snapshot, adaptive reproduce determinism (verdict + reproduction snapshot
  rank + config_hash equal the parent's).
- **Task 5:** all gates green (see Debug Log).
- Net behaviour delta beyond the ACs: a config that *omits* `basis` now snapshots/hashes
  the normalized `basis: pod` default (structured field vs former opaque extra) — accepted
  canonicalization; all existing tests spell `basis: pod` explicitly and stay green.

### File List

- src/surroforge/core/config.py (modified)
- src/surroforge/engine.py (modified)
- src/surroforge/trainers/reduced_field.py (modified)
- tests/test_config.py (modified)
- tests/test_reduction_config.py (new)
- _bmad-output/implementation-artifacts/sprint-status.yaml (modified)
- _bmad-output/implementation-artifacts/r1-2-configure-reduction-in-a-model-reduction-block.md (modified)
- _bmad-output/implementation-artifacts/deferred-work.md (modified, review deferrals)

### Change Log

- 2026-06-10: Story created (create-story). **Architectural decision (lead):** chose the literal-AC path — structured `model.reduction`/`model.inner` Pydantic blocks in core config with load-time `ConfigError` — over the spine-preserving trainer-only-`TrainerError` alternative, accepting a bounded spine-theorem (#36) compromise contained to `core/config.py`. Design keeps the engine + trainer decoupled via the `hyperparameters` dict bridge and a duck-typed `resolved_config_overrides()` write-back hook; `exclude_none=True` serialization preserves scalar/energy-field byte-identity (R2.4) and field `config_hash` stability (R2.3). `rank: adaptive` resolves at fit and is written back to the canonical snapshot. Status → ready-for-dev.
- 2026-06-10: Code review (adversarial, 3 fresh-context layers: Blind Hunter / Edge Case Hunter / Acceptance Auditor). Auditor: AC1–AC5 satisfied, every scope-boundary constraint honored. Triage: 1 decision, 5 patches (all applied), 2 deferred, 4 dismissed. **Patched:** (High, Edge-verified) adaptive write-back read the FINAL model's rank while the fingerprint comes from the HOLDOUT model — a straddling energy threshold (holdout 3 vs full-data 4) made `reproduce()` falsely diverge and lineage contradict the snapshot; fixed by querying `resolved_config_overrides()` on the holdout model + a straddling-spectrum regression test (documented residual: the persisted final artifact may hold a different mode count than the snapshot rank — diagnostic, not verdict). (High/Med) explicit-`null` hyperparameter extras were silently dropped from snapshot/hash while the trainer received them — now rejected with a named `ConfigError` (`_reject_null_extras` on `ModelConfig`/`InnerConfig`); docstring overclaim corrected. (Med) `energy: true` no longer coerces to 1.0 (before-validator, mirrors the rank bool guard). (Med) AC4 scalar test strengthened to pin the exact model section by construction. (Low) duplicated `reduction:` prefix dropped from validator messages. **Decision (lead):** pre-R1.2 field-run snapshot compat break ACCEPTED (pre-release; documented in deferred-work.md). **Deferred:** external-plugin hook robustness (folds into R2.3/R2.5 family); `reduction`/`inner` shadowing third-party hyperparameter names (seam hardening). Post-patch: 264 passed / 2 skipped, ruff/mypy/lint-imports green, pinned-threads family 87 passed. Status → done.
- 2026-06-10: Implemented (dev-story). All 5 tasks complete; 16 new tests (12 config + 4 engine e2e); full suite 259 passed / 2 skipped; ruff/mypy/lint-imports green; pinned-threads reproduce family green. R2.4 scalar byte-identity and R2.3 field tests unregressed. Status → review.
- 2026-06-10: Validated (validate-create-story, independent fresh-context) → ACCEPT-WITH-FIXES. Verified the two linchpins empirically: `exclude_none=True` is a true no-op for every pre-R1.2 config (no model has a `None`-valued field), and the structured-field serialization is byte-identical to r1-3's extra-dict form for energy-field configs (`config_hash`/`spec_hash` stable). Also confirmed: the `hyperparameters` bridge feeds the trainer cleanly, `reproduce()` does not gate on `config_hash`, the adaptive write-back round-trips, mypy-clean, no other consumer reads `reduction`/`inner` from `__pydantic_extra__`. Applied fixes: **(critical)** the `rank` validator must be `@field_validator("rank", mode="before")` — under the `int | Literal` union Pydantic coerces `True→1` before an after-validator, so `rank: true` would slip through and the AC2 test would fail; corrected the line-citation cluster (snapshot write engine.py:177; `_config_hash` 219-220; `_spec_hash` 223-236; `_resolved_reduction` 196-198); flagged that `lint-imports` does NOT enforce the config-leaf constraint (no contract forbids `config→plugins`); flagged that the R2.4 guard does NOT pin the scalar `config_hash` value, so the story's new AC4 assertion is the only net for an `exclude_none` regression.
