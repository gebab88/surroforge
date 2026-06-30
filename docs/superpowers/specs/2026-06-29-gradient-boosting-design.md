# Gradient Boosting Trainer Design

## Goal

Add a gradient-boosting surrogate method to SurroForge Core, rounding out the
Phase 2 "extended methods" (alongside the existing `mlp`). It is a Core builtin —
not a plugin — because it is backed by scikit-learn, which is already a Core
dependency (BSD → NFR8), so it adds **no new dependency**. The method mirrors
`mlp` in structure and reproducibility model.

> **Revised during implementation (2026-06-29):** the originally-planned ONNX export
> was dropped — `skl2onnx`'s `TreeEnsembleRegressor` converter emits numpy bool/uint32
> attributes that the installed `onnx>=1.21` rejects, so tree-ensemble ONNX export is
> unavailable in this stack (`mlp`/`polynomial` still export to ONNX — only tree
> ensembles are affected). `gradient-boosting` is therefore **pickle-only**; ONNX can be
> re-added as a follow-up once the `skl2onnx`/`onnx` stack converts tree ensembles cleanly.
> The sections below are kept as written; treat every "pickle + ONNX" / `to_onnx`
> statement as superseded by pickle-only.

## User-Facing Configuration

One new method `gradient-boosting`, selected like any other:

```yaml
model:
  method: gradient-boosting
  learning_rate: 0.1
  max_iter: 100
  max_leaf_nodes: 31
  max_depth: null
  min_samples_leaf: 20
  l2_regularization: 0.0
  random_state: 0
```

All hyperparameters are optional. It also works as a `reduced-field` inner:

```yaml
model:
  method: reduced-field
  reduction: { basis: pod, rank: 4 }
  inner: { method: gradient-boosting, learning_rate: 0.05, max_iter: 200 }
```

## Architecture

`GradientBoostingTrainer(BaseTrainer)` in `src/surroforge/trainers/gradient_boosting.py`,
registered as `gradient-boosting` in the `_BUILTIN[TRAINER_GROUP]` table in
`src/surroforge/plugins/registry.py`. Structure follows `src/surroforge/trainers/mlp.py`.

The model is `sklearn.ensemble.HistGradientBoostingRegressor`:

- **No input scaler.** Tree ensembles are scale-invariant, so — unlike `mlp`'s
  `StandardScaler` pipeline — the estimator is used directly. This keeps the model
  and the ONNX graph simpler.
- **Multi-output via per-target wrapper.** `HistGradientBoostingRegressor` is
  single-output. In `fit(x, y)`: if `y` has one column, fit the bare regressor on
  the 1-D target (as `mlp` does, to avoid the `DataConversionWarning`); if `y` has
  more than one column, fit `sklearn.multioutput.MultiOutputRegressor(...)` — one
  independent booster per target column. The fitted object (bare or wrapped) is the
  persisted artifact and the unit ONNX export operates on.
- **`early_stopping=False`** is set explicitly: no internal validation split, so the
  fit is fully deterministic for the small-DOE regime.
- `fit` assigns `self._model` only after a successful fit (no half-built model);
  `predict` re-widens output to 2-D via `_as_2d`.

## Hyperparameters

Validated at construction (a typo or out-of-range value raises `TrainerError`, not a
raw scikit-learn error deep in `fit`), then passed to `HistGradientBoostingRegressor`:

| Name | Default | Validation |
| --- | --- | --- |
| `learning_rate` | `0.1` | `> 0` |
| `max_iter` | `100` | `>= 1` |
| `max_leaf_nodes` | `31` | `>= 2` or `None` |
| `max_depth` | `None` | `>= 1` or `None` |
| `min_samples_leaf` | `20` | `>= 1` |
| `l2_regularization` | `0.0` | `>= 0` |
| `random_state` | `0` | an `int` (not `None`/`RandomState`; a fixed seed is what makes the method reproducible) |

Unknown hyperparameters raise `TrainerError` listing the supported set. `max_bins`
and other rarely-tuned knobs stay at scikit-learn defaults (YAGNI).

## Reproducibility

A tree ensemble with a fixed integer `random_state` and `early_stopping=False` is
deterministic in-process. Like `mlp`, the method declares an honest
`reproduction_tolerance = 1e-6` rather than bit-for-bit determinism, to absorb any
thread-/platform-level float differences in histogram summation. A genuine seed or
data drift moves predictions orders of magnitude above this band, so a real
divergence still fails the reproduction verdict.

## Export

`export_formats = frozenset({"pickle", "onnx"})`.

`to_onnx(n_features)` converts `self._model` (the bare `HistGradientBoostingRegressor`
or the `MultiOutputRegressor` wrapper) via `skl2onnx.convert_sklearn` with
`initial_types=[("input", FloatTensorType([None, n_features]))]`. It guards that
`n_features` matches the fitted width (raising `ExportError` otherwise) and maps a
missing skl2onnx import to an `ExportError` pointing at the `surroforge[onnx]` extra —
identical to the `mlp` ONNX path. skl2onnx supports both `HistGradientBoostingRegressor`
and `MultiOutputRegressor`.

## Scope

In scope (v1): the `gradient-boosting` Core method (single + multi-output), the seven
hyperparameters above, pickle + ONNX export, registration in `_BUILTIN`, passing the
`check_trainer` conformance suite, usability as a `reduced-field` inner, a rich method
docstring, a CHANGELOG entry, and a mention in the docs method list if one exists.

Out of scope (v1): XGBoost/LightGBM backends (would be heavyweight-dependency plugins
on the existing trainer seam, like `surroforge-torch`); a classic `GradientBoostingRegressor`
variant; early stopping and `max_bins` tuning; a new example script.

## Validation / Testing

Tests under `tests/` (mirroring the existing trainer tests):

- **Construction/validation:** unknown hyperparameter raises `TrainerError`; each
  range check (`learning_rate > 0`, `max_iter >= 1`, `max_leaf_nodes >= 2`/`None`,
  `max_depth >= 1`/`None`, `min_samples_leaf >= 1`, `l2_regularization >= 0`,
  `random_state` must be an int).
- **fit/predict:** correct output shapes for single-output and multi-output; the
  multi-output path uses `MultiOutputRegressor`.
- **Determinism:** two fits with the same `random_state` produce identical predictions.
- **Behavior:** boosting achieves a high R² on a known nonlinear target where a linear
  model does not — verifying it learns, not just that shapes line up.
- **ONNX:** round-trip export for single- and multi-output, predictions match the
  scikit-learn model within a float32 tolerance; `n_features` mismatch raises
  `ExportError`.
- **Registration & conformance:** `available_trainers()` includes `gradient-boosting`,
  `get_trainer("gradient-boosting")` returns the class, and `check_trainer(GradientBoostingTrainer)`
  passes.
