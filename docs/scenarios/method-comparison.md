# Scenario: Method comparison

**Goal:** train several methods on the same data and pick the best by a unified,
out-of-sample metric — in one command.

## Train the candidates

Write one config per method, differing only in `model`. The data and validation
blocks stay identical so the comparison is fair:

```yaml title="configs/kriging.yaml"
experiment: beam-kriging
data: &data
  source: beam
  inputs: [length, width, load]
  outputs: [max_stress]
validation: &validation
  cross_validation: 5
  metrics: [rmse, r2, mae]
model:
  method: kriging
```

```yaml title="configs/polynomial.yaml"
experiment: beam-polynomial
data:
  source: beam
  inputs: [length, width, load]
  outputs: [max_stress]
validation:
  cross_validation: 5
  metrics: [rmse, r2, mae]
model:
  method: polynomial
  degree: 2
```

```bash
surroforge train --config configs/kriging.yaml      # -> run-001
surroforge train --config configs/polynomial.yaml   # -> run-002
```

## Compare

```bash
surroforge compare --runs run-001,run-002 --metric rmse
surroforge compare --runs run-001,run-002 --format html -o comparison.html   # Professional plugin
```

With the Professional plugin installed, `--format html` writes a self-contained
interactive comparison report (`reports/comparison.html` by default) instead of the
terminal tables.

This prints a table of each run's **test** metrics side by side and names the best
run by the chosen metric:

```
  run         method        rmse         r2         mae
  run-001     kriging       3.214      0.987       2.51
  run-002     polynomial    9.880      0.901       7.44
  Best by test rmse: run-001
```

- `--metric` selects the ranking metric (default `rmse`). For `rmse`/`mae` lower is
  better; for `r2` higher is better — SurroForge knows the direction.
- The comparison is over **test** (out-of-sample) metrics, never training fit.

## Notes

- Methods are commensurable because they share the same data, split, and metric
  set. Keep those identical across the configs you compare.
- Add as many runs as you like to `--runs` (comma-separated).
- The same operation is available from Python via `surroforge.services.compare_runs`
  — see the [Python API](../reference/python-api.md).
