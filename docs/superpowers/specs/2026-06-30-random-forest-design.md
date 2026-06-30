# Random Forest Trainer Design

## Goal

Add a `random-forest` surrogate method to SurroForge Core, completing the Phase 2
"extended methods" (GP via `kriging`, NN via `mlp`, boosting via `gradient-boosting`
are already present). It is a Core builtin backed by scikit-learn's
`RandomForestRegressor`, so it adds **no new dependency** (scikit-learn is already
Core, BSD → NFR8). The method mirrors `gradient-boosting`/`mlp` in structure and
reproducibility model.

## User-Facing Configuration

One new method `random-forest`, selected like any other:

```yaml
model:
  method: random-forest
  n_estimators: 100
  max_depth: null
  max_leaf_nodes: null
  min_samples_leaf: 1
  max_features: 1.0
  random_state: 0
```

All hyperparameters are optional. It also works as a `reduced-field` inner:

```yaml
model:
  method: reduced-field
  reduction: { basis: pod, rank: 4 }
  inner: { method: random-forest, n_estimators: 200 }
```

## Architecture

`RandomForestTrainer(BaseTrainer)` in `src/surroforge/trainers/random_forest.py`,
registered as `random-forest` in the `_BUILTIN[TRAINER_GROUP]` table in
`src/surroforge/plugins/registry.py`. Structure follows
`src/surroforge/trainers/gradient_boosting.py`.

The model is `sklearn.ensemble.RandomForestRegressor`:

- **No input scaler.** Tree ensembles are scale-invariant, so the estimator is used
  directly (a simpler model).
- **Native multi-output.** Unlike `gradient-boosting` (which wraps a single-output
  booster in `MultiOutputRegressor`), `RandomForestRegressor` predicts multiple
  targets directly. `fit(x, y)` passes a 2-D `y` straight through; a single output
  passes a 1-D target (`y[:, 0]`) to avoid sklearn's `DataConversionWarning`. The
  fitted estimator is the persisted artifact and the unit ONNX export operates on.
- `fit` assigns `self._model` only after a successful fit; `predict` re-widens the
  output to 2-D via `_as_2d`.

## Hyperparameters

Validated at construction (a typo or out-of-range value raises `TrainerError`, not a
raw scikit-learn error deep in `fit`), then passed to `RandomForestRegressor`:

| Name | Default | Validation |
| --- | --- | --- |
| `n_estimators` | `100` | `>= 1` |
| `max_depth` | `None` | `>= 1` or `None` |
| `max_leaf_nodes` | `None` | `>= 2` or `None` |
| `min_samples_leaf` | `1` | `>= 1` |
| `max_features` | `1.0` | `"sqrt"`/`"log2"`/`None`, an `int >= 1`, or a `float` in `(0, 1]` |
| `random_state` | `0` | an `int` (not `None`/`bool`/`RandomState`; a fixed seed is what makes the method reproducible) |

Unknown hyperparameters raise `TrainerError` listing the supported set. `bootstrap`
stays at scikit-learn's default (`True`) and is not exposed (YAGNI), which keeps the
"seed drives the model via bootstrap" behaviour clean.

## Reproducibility

A random forest with a fixed integer `random_state` is deterministic (bootstrap
resampling and per-split feature subsampling are seeded). Like `mlp`/`gradient-boosting`,
the method declares an honest `reproduction_tolerance = 1e-6` rather than bit-for-bit,
to absorb any thread/platform float differences in tree averaging.

Unlike `gradient-boosting` (whose `HistGradientBoostingRegressor` is seed-invariant on
small data), **`random_state` genuinely changes a random forest** (verified: different
seeds move predictions by ~0.14). So the test suite includes both a same-seed
reproducibility test and a different-seed divergence test (the negative case with teeth,
mirroring `mlp`).

## Export

`export_formats = frozenset({"pickle", "onnx"})`.

ONNX export **works for random forests** in the current stack: `skl2onnx`'s
`RandomForestRegressor` converter emits clean integer attributes (unlike the
`HistGradientBoosting` converter, which `gradient-boosting` had to ship pickle-only).
Verified by round-trip: single- and multi-output predictions match the scikit-learn
model within ~1e-7.

`to_onnx(n_features)` converts `self._model` via `skl2onnx.convert_sklearn` with
`initial_types=[("input", FloatTensorType([None, n_features]))]`, guards that
`n_features` matches the fitted width (raising `ExportError` otherwise), and maps a
missing skl2onnx import to an `ExportError` pointing at the `surroforge[onnx]` extra —
identical to the `mlp` ONNX path.

Known benign detail: for multi-output, onnxruntime prints a C-level shape-metadata note
(the model declares output shape `{-1, 1}` while it returns all target columns). The
values are correct after reshaping to the reference shape — which is exactly what the
round-trip test asserts. It is not a Python warning and does not fail tests.

## Scope

In scope (v1): the `random-forest` Core method (single + native multi-output), the six
hyperparameters above, pickle + ONNX export, registration in `_BUILTIN`, passing the
`check_trainer` conformance suite, usability as a `reduced-field` inner, a rich method
docstring, a CHANGELOG entry, and a docs method-list entry.

Out of scope (v1): `bootstrap=False` / out-of-bag scoring, `n_jobs` exposure (parallelism
is a speed knob, not a modelling one), `ExtraTrees`/other forest variants, and a new
example script.

## Validation / Testing

Tests under `tests/` (mirroring `test_gradient_boosting.py`/`test_mlp.py`):

- **Construction/validation:** unknown hyperparameter raises `TrainerError`; each range
  check (`n_estimators >= 1`, `max_depth >= 1`/`None`, `max_leaf_nodes >= 2`/`None`,
  `min_samples_leaf >= 1`, `max_features` accepts the documented forms and rejects others,
  `random_state` must be an int).
- **fit/predict:** correct output shapes for single-output and native multi-output.
- **No `DataConversionWarning`** on a single-output fit (`warnings.simplefilter("error")`).
- **Reproducibility:** same `random_state` reproduces within tolerance; **different
  `random_state` diverges beyond tolerance** (the negative case with teeth).
- **Behavior:** high R² on a known nonlinear target where a linear model fails.
- **ONNX:** round-trip export for single- and multi-output, predictions match the
  scikit-learn model within a float32 tolerance; `n_features` mismatch raises
  `ExportError`; missing backend raises an actionable `ExportError`.
- **Registration & conformance:** `available_trainers()` includes `random-forest`,
  `get_trainer("random-forest")` returns the class, and `check_trainer(RandomForestTrainer)`
  passes.
- **Reduced-field inner:** `random-forest` reconstructs a smooth low-rank field as a
  `reduced-field` inner on a holdout.
