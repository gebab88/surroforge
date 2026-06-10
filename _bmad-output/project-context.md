---
project_name: 'SurroForge'
user_name: 'Georg'
date: '2026-06-04'
sections_completed: ['technology_stack', 'critical_rules']
existing_patterns_found: 9
status: 'complete'
optimized_for_llm: true
---

# Project Context for AI Agents

_Critical, unobvious rules for writing code in this project. Verified against source on 2026-06-04._

**How to read this file.** Each rule carries a tier tag:
- `[MUST]` — violating it breaks the build, a contract, or a guarantee.
- `[GOTCHA]` — subtle; the obvious choice is wrong here.
- `[CONV]` — convention for consistency.

`(planned)` next to a thing means **it does not exist in the code yet** — treat it as work to do, never as existing behavior. Verify against source before relying on it.

---

## ⚠️ Read First

1. **[MUST] Known critical defect — reproducibility fingerprint.** `engine.py:_prediction_fingerprint()` hashes predictions on the **training inputs** and compares by **SHA bit-identity**. This is wrong: interpolating methods (kriging) near-reproduce training points, so a *diverged* model can hash-match → `reproduce()` reports success when it failed. The fix `(planned)`: fingerprint a recorded **held-out + grid** sample, compare with a **named numeric tolerance** (not exact equality), and have `reproduce()` **emit a verdict** (raise `ReproducibilityError` on divergence). Do **not** "fix" by loosening a check.
2. **`(planned)` features — DO NOT assume these exist:** CI (`.github/workflows/`), `src/surroforge/conformance/`, the held-out+grid fingerprint + tolerance constant, `reproduce()`-emits-verdict, a `BaseTrainer` reproducibility-characteristics field, a data-driven exporter-capability query (today it's just `supports_onnx: bool`), the registry `gate.check` call site + plugin `api_version`, config `schema_version`, pickle provenance warning, MkDocs `docs/`.

---

## Technology Stack & Versions

| Area | Choice | Note |
|---|---|---|
| Language / build | Python ≥3.10, `hatchling`, `src/` layout | PyPI `surroforge` |
| Core deps (permissive-only) | numpy ≥1.24, scipy ≥1.10, pandas ≥2.0, scikit-learn ≥1.3, pyyaml ≥6.0, pydantic ≥2.5, typer ≥0.12, jinja2 ≥3.1 | **[MUST]** no GPL/LGPL deps in Core (NFR8) |
| Optional backends (extras) | `[smt]` (kriging/rbf), `[onnx]`, `[formats]` (pyarrow/h5py) | **never** core deps |
| Dev | pytest ≥8, ruff ≥0.4 (line 100; E/F/I/UP/B/W), mypy ≥1.9 (py310) | |

Pydantic is **v2 API**. Typer is **pre-1.0** — keep CLI logic thin so its churn can't reach the core.

## Module & Language Conventions

- `[CONV]` Every module opens with `from __future__ import annotations` then a one-line docstring citing the PRD section (e.g. `(PRD 4.6)`).
- `[CONV]` `PascalCase` classes; `snake_case` functions/vars; leading `_` for module-private. Full type hints (mypy py310); arrays `np.ndarray`, paths `pathlib.Path`.

## Dependency Direction (import-verified — hard constraint)

| Layer | May import | Must NOT import |
|---|---|---|
| `cli`, `api` | anything below | each other is N/A; nothing imports *them* |
| `services` | `engine` + all leaves | `cli`, `api` |
| `engine` | leaves + registry/ABCs | **`services`** (no back-edge), `cli`, `api` |
| `core`, `data`, … (leaves) | `errors`, narrowly each other | `cli`, `api`, `services`, `engine` |
| `errors` | nothing internal (leaf) | everything |

- `[MUST]` The three invariants that matter: (a) nothing imports `cli`/`api`; (b) `engine` never imports `services`; (c) `errors.py` stays a leaf.
- `[CONV]` Where new code goes: calls `plugins.get_*` or opens a project → `services`; runs a training-loop step → `engine`; pure domain object → `core`.

## Plugin Authoring — Golden Path

To add a trainer (exporters/reporters are analogous):
1. Subclass `BaseTrainer` in `trainers/<name>.py`. Set `name` (**lowercase**, == entry-point key == `model.method` value) and `supports_onnx: bool`.
2. `[GOTCHA]` Keep `__init__` cheap and **import-free** (no top-level backend import). Validate hyperparameters in `__init__`, rejecting unknowns with `TrainerError` listing supported ones.
3. Implement `fit(x, y)` and `predict(x)` on 2-D arrays via inherited `_as_2d`. `[MUST]` `predict` always returns **2-D `(n, n_outputs)`** — even single-output is `(n, 1)`; never squeeze to `(n,)`.
4. `[MUST]` The trainer must be **picklable** (it is the persisted artifact). Override `to_onnx` only if `supports_onnx = True`.
5. Register: built-ins go in **both** `pyproject.toml` entry points **and** the `_BUILTIN` table in `plugins/registry.py`. **[MUST] Third-party/commercial plugins use entry points ONLY** — never `_BUILTIN`. Installed entry points take precedence over `_BUILTIN`.

- `[MUST]` `plugins/registry.py` is the **only** place doing dynamic name→class resolution (`entry_points`/`import_module`). Never resolve plugins by name elsewhere.

## Reproducibility (headline invariant)

- `[MUST]` Lineage capture (config snapshot, SHA-256 data hash, `seed`, versions) is automatic and **no-opt-out**. Don't add a skip flag.
- `[CONV]` Seeding is **sklearn-style integer `random_state`**, not `np.random.default_rng`: the run `seed` (config default `42`) is threaded as `train_test_split(random_state=seed)` and `KFold(..., random_state=seed if shuffle else None)`. Don't introduce global `np.random.seed`.
- See the **Read First** defect. The fix `(planned)` requires a **single named tolerance constant** (relative for magnitudes, absolute floor near zero) owned in one place — `[GOTCHA]` do not inline `np.allclose` defaults per call site.
- `[GOTCHA]` Reproducibility is BLAS/threading-sensitive. The `(planned)` repro test must **pin threads** (`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS=1`) in its fixture **and** assert the verdict — never trade thread-count against tolerance to make a flaky test pass.
- `[GOTCHA]` **Holdout vs. final model (Epic R1 retro):** the fingerprint and all reproducibility diagnostics come from the **holdout** model; the persisted artifact is the **final** model. Any new engine introspection hook (e.g. `resolved_config_overrides`, `reproducibility_diagnostics`) must be queried on the **holdout** model — reading the final model falsely diverges `reproduce()` when the two resolve differently (R1.2's straddling-spectrum High bug).

## Errors & CLI Output

- `[MUST]` Raise a `SurroForgeError` subclass for domain failures (`ProjectError`, `ConfigError`, `DataError`, `TrainerError`, `PluginError`, `ExportError`, `RunNotFoundError`, `ReproducibilityError`) — never bare `ValueError`/`Exception`. New category → new subclass in `errors.py`. Messages must be actionable (state the fix).
- `[MUST]` New CLI commands use the `_guard` decorator (clean `Error: …`; traceback only under `--verbose`). Output via `_info`/quiet-aware helpers — never bare `print`. Tone: precise, sober, English, no emojis, no marketing.

## Optional-Dependency Discipline

- `[GOTCHA]` **Never import an optional backend at module top** (no top-level `smt`, `torch`, `skl2onnx`, `pyarrow`, `h5py`). Import inside the method, so the registry can list methods without the backend installed.
- `[MUST]` Translate a missing-backend `ImportError` into a domain error + install hint. Core extras use `pip install 'surroforge[onnx]'`; **commercial plugins are separate distributions** (`pip install surroforge-professional`) — hint the distribution, not an extra.

## Methods, Export & Data

- `[MUST]` ONNX export works **only for scikit-learn methods (`polynomial`)**. SMT methods (`kriging`, `rbf`) are **pickle-only** — don't attempt ONNX; raise `ExportError` (decided per Spike 1; holds at SMT 2.14).
- `[GOTCHA]` `data/hashing.py:hash_dataframe` is **column- AND row-order-sensitive** (it iterates `frame.columns` in order and hashes row bytes). Reordering columns or rows changes the hash — normalize order upstream if that's not intended.
- `[CONV]` Run ids are sequential, zero-padded `run-001` (`persistence.format_run_id`).
- `[MUST]` Persist files via `persistence.write_json` / `write_yaml` (atomic: tmp + `os.replace`). Never hand-roll `open(...,'w')` for run/project state.
- `[GOTCHA]` Pickle loading is an arbitrary-code-execution surface; prefer ONNX as the interchange format where supported. Provenance warning is `(planned)`.

## Testing

- `[CONV]` Flat `tests/`, one file per concern, plus `test_scenarios.py` for the three end-to-end UX scenarios. Keep new tests in this shape.
- `[MUST]` **A reproducibility- or plugin-numerics change is not done until it has a negative test:** train, then perturb (reseed / change thread count / nudge a hyperparameter) and assert `reproduce()` **FAILS** (`ReproducibilityError`). A same-seed round-trip test passes even for a broken `reproduce()` — the negative case is the one with teeth.
- `(planned)` The plugin conformance suite ships as library code in `src/surroforge/conformance/` (so out-of-tree plugins can import it), **not** under `tests/`. Mark the slow/threading-pinned repro test `@pytest.mark.slow`.

## No-Network & Licensing

- `[MUST]` Core makes **no outbound network calls** (except an opt-in update check); no telemetry without opt-in (NFR5). Don't add network deps to Core.
- `[MUST]` Core deps stay permissively licensed (Apache/MIT/BSD); a GPL/LGPL dep breaks the Open-Core model (NFR8).

---

## Maintenance

- **Agents:** read this before writing code; when unsure, prefer the more restrictive option; treat `(planned)` as work-to-do, not behavior.
- **Humans:** keep it lean and gotcha-focused; when a `(planned)` item ships, delete its `(planned)` tags and move its rule to active; drop rules that become obvious. Full rationale lives in `_bmad-output/planning-artifacts/architecture.md`.

_Last updated: 2026-06-04._
