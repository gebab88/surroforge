# Configuration reference

A single YAML file is the complete and only specification of a training run. It is
validated with Pydantic; an invalid file fails with a precise, field-located error
(never a traceback). The same canonical form is hashed into the run's lineage.

## Full example

```yaml
experiment: beam-kriging          # required: a human-readable run name
data:                             # required
  source: beam                    # registered dataset name OR a file path
  inputs: [length, width, load]   # >= 1, unique
  outputs: [max_stress]           # >= 1, unique
  split:                          # optional (defaults shown)
    method: random                # only "random" in the MVP
    test_fraction: 0.2            # 0 < x < 1
    seed: 42
model:                            # required
  method: kriging                 # registered method name
  corr: squar_exp                 # any extra key = method-specific hyperparameter
validation:                       # required
  cross_validation: 5             # int shorthand for {folds: 5}; or a mapping
  metrics: [rmse, r2, mae]        # optional; defaults to [rmse, r2, mae]
```

## Top level

| Field | Type | Required | Notes |
|---|---|---|---|
| `experiment` | string | yes | Human-readable name for the run. |
| `data` | mapping | yes | Data source and column roles (below). |
| `model` | mapping | yes | Method and hyperparameters (below). |
| `validation` | mapping | yes | Cross-validation strategy and metrics (below). |

Unknown top-level keys are rejected.

## `data`

| Field | Type | Default | Notes |
|---|---|---|---|
| `source` | string | — (required) | A registered dataset name, or a file path (relative to the project root, or absolute). |
| `inputs` | list of strings | — (required) | Input column names; at least one; must be unique. |
| `outputs` | list of strings | — (required) | Output column names; at least one; must be unique. |
| `split` | mapping | see below | Train/test split. |

### `data.split`

| Field | Type | Default | Notes |
|---|---|---|---|
| `method` | string | `random` | Only `random` is supported in the MVP. |
| `test_fraction` | float | `0.2` | Held-out fraction; must satisfy `0 < x < 1`. |
| `seed` | int | `42` | Split (and CV) seed; recorded in lineage. |

## `model`

| Field | Type | Default | Notes |
|---|---|---|---|
| `method` | string | — (required) | A registered method: `kriging`, `rbf`, `polynomial`, or a plugin. |
| *(any other key)* | — | — | Collected as **method-specific hyperparameters** and passed straight to the trainer. |

Hyperparameters are not enumerated centrally — each method validates its own and
rejects unknown ones with a clear error. Examples:

- **polynomial:** `degree` (int ≥ 1, default 2), `include_bias` (bool, default true),
  `interaction_only` (bool, default false).
- **kriging:** `corr` (e.g. `squar_exp`, `abs_exp`), `poly` (`constant`/`linear`/`quadratic`),
  `theta0` (list of floats), `nugget` (float).
- **rbf:** `d0` (float), `poly_degree` (int: -1, 0, or 1), `reg` (float).

## `validation`

| Field | Type | Default | Notes |
|---|---|---|---|
| `cross_validation` | int **or** mapping | — (required) | An int `k` is shorthand for `{folds: k}`. |
| `metrics` | list of strings | `[rmse, r2, mae]` | Metrics computed for train, test, and each CV fold. |

### `validation.cross_validation` (mapping form)

| Field | Type | Default | Notes |
|---|---|---|---|
| `folds` | int | `5` | Number of k-fold splits; must be ≥ 2. |
| `shuffle` | bool | `true` | Shuffle before splitting (uses `data.split.seed`). |

## Errors

A configuration problem raises `ConfigError` with each offending field located, e.g.:

```
Invalid configuration in configs/kriging.yaml:
  - data.inputs: List should have at least 1 item after validation
  - data.split.test_fraction: Input should be less than 1
```
