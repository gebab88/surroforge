# Extended Run Comparisons (Core) Design — Cycle 1

## Goal

Turn SurroForge's minimal run comparison (PRD 4.1; today a flat test-metric table + a
single best run) into an **extended comparison** in Core: a richer metric table
(train/test/CV mean±std, ranking with deltas, best-per-metric), a
**config/hyperparameter diff** (what actually differs between runs), and a
**lineage/reproducibility comparison** (which runs share data). Output is an enriched
terminal view plus a structured, JSON-serializable result returned by the Python API.

This is **Cycle 1 of two**. Cycle 2 (a separate spec) adds a Professional HTML comparison
report with interactive cross-run charts that consumes this cycle's structured result.

## Scope of the larger feature

The PRD 9.2 item "Erweiterte Run-Vergleiche" decomposes into two subsystems, built in
sequence:

- **Cycle 1 (this spec, Core / Apache):** the comparison *data model* + CLI.
- **Cycle 2 (later, Professional plugin):** an HTML comparison report + interactive
  cross-run charts (uPlot), consuming Cycle 1's structured result.

This spec covers Cycle 1 only.

## User-Facing Behaviour

Selection is unchanged — the explicit `--runs` list:

```
surroforge compare --runs run-001,run-002,run-003 --metric rmse
```

`--metric` chooses the ranking metric (default `rmse`). The command prints four blocks:
a ranking table (full detail for the chosen metric), an all-test-metrics table
(best-per-metric marked), a configuration-differences table, and a lineage block — plus
any warnings and the best-run line. The Python API gains the structured result via
`services.compare_runs(...)` / `surroforge.comparison.build_comparison(...)`.

No new dependency (NFR8): terminal tables use the existing manual column formatting and
`_info`/`_success` prints.

## Architecture

A dedicated Core module holds the pure comparison logic; `services` loads runs and
delegates; the CLI renders.

- `src/surroforge/comparison.py` (**new**) — `build_comparison(runs, *, metric="rmse") -> dict`,
  a pure, isolated-testable function. The ranking-direction table `_LOWER_IS_BETTER`
  and the best-selection logic **move here** from `services.py` (comparison is their new
  home; this avoids duplication and any import cycle, since `comparison.py` imports
  nothing from `services`).
- `src/surroforge/services.py` (**modify**) — `compare_runs(project, run_ids, *, metric="rmse")`
  loads the runs (`project.run(rid)`) and delegates to `build_comparison`, returning its
  dict. The old `_select_best`/`_LOWER_IS_BETTER` are removed from `services` (moved).
- `src/surroforge/cli/main.py` (**modify**) — the `compare` command renders the four
  blocks from the result via small helper functions.

The config diff is derived generically from `run.config.canonical_dict()` (already used
for the run snapshot), so it covers method, hyperparameters, data source, split,
validation, and reduction without hand-listing fields.

## Structured Result — `build_comparison(runs, *, metric="rmse") -> dict`

JSON-serializable (Cycle 2 consumes it):

```python
{
    "metric": "rmse",                       # ranking metric
    "metric_names": ["rmse", "r2", "mae"],  # union of all runs' validation.metrics, stable order
    "runs": [
        {
            "run_id": "run-003",
            "method": "random-forest",
            "experiment": "beam-study",
            "created_at": "2026-06-30T…",
            "metrics": {                    # one entry per name in metric_names
                "rmse": {"train": 0.0091, "test": 0.0123, "cv_mean": 0.0130, "cv_std": 0.0021},
                # a metric a run lacks -> {"train": None, "test": None, "cv_mean": None, "cv_std": None}
            },
            "rank": 1,                      # rank by chosen metric's test value; None if metric absent
            "delta": 0.0,                   # gap to the best test value, always >= 0 (0 = best); None if unranked
            "lineage": {
                "seed": 7, "data_hash": "1a2b…", "config_hash": "4d5e…", "parent_run": None,
            },
        },
        # … one per input run, in input order
    ],
    "best": "run-003",                      # best run_id by chosen metric's test value, or None
    "best_per_metric": {"rmse": "run-003", "r2": "run-001", "mae": "run-003"},
    "config_diff": {                        # differing canonical-config leaf paths only; {} if identical
        "model.method": {"run-001": "mlp", "run-003": "random-forest"},
        "model.hyperparameters.n_estimators": {"run-001": "<absent>", "run-003": 300},
        "data.split.seed": {"run-001": 42, "run-003": 7},
    },
    "lineage_summary": {
        "same_data": False,                 # do all runs share one data_hash?
        "data_hash_groups": {"1a2b…": ["run-001", "run-002"], "9f8e…": ["run-003"]},
    },
    "warnings": ["Runs use different datasets (data_hash differs); …"],
}
```

### Metrics, ranking, best

- `metric_names` is the **union** of every run's `config.validation.metrics`, in stable
  first-seen order. A run missing a metric gets `None` values for it (`"—"` in the table).
- Per run, `metrics[m]` reads `train`/`test` from `run.metrics["train"|"test"]` and
  `cv_mean`/`cv_std` from `run.metrics["cross_validation"][m]["mean"|"std"]`; any missing
  piece is `None`.
- **Ranking** is by the chosen metric's `test` value, using `_LOWER_IS_BETTER`
  (`{"rmse": True, "mae": True, "r2": False}`; unknown metric defaults to lower-is-better).
  `rank` 1 is best. Runs whose chosen-metric `test` is `None` get `rank=None` (unranked)
  and a warning.
- `delta` is the gap from the best `test` value in the "worse" direction (always `>= 0`,
  `0` for the best); `None` for unranked runs.
- `best` is the best run by the chosen metric; `best_per_metric[m]` is the best run for
  each metric that at least one run has (same direction rule). Metrics no run has are
  omitted from `best_per_metric`.

### Config diff

`config_diff` is a deep-diff over each run's `run.config.canonical_dict()`. A leaf path
(dotted, e.g. `model.hyperparameters.n_estimators`) is included **only** when the runs do
not all agree on it. A run lacking the key contributes the sentinel string `"<absent>"`.
Identical configs across all runs → `config_diff == {}`. Values are taken verbatim from
the canonical dict (already JSON-friendly scalars / small structures).

### Lineage summary

`lineage_summary.same_data` is `True` iff every run shares one `data_hash`.
`data_hash_groups` maps each distinct `data_hash` to the run-ids that share it (the
apples-to-apples check). Per-run `config_hash` stays in `runs[].lineage` (expected to
differ when comparing — that is what `config_diff` explains).

### Warnings (strings in `result["warnings"]`)

`build_comparison` raises **only** on an empty run list (`DataError`). All heterogeneity
is a non-fatal warning:

- chosen metric absent from some runs → names the unranked run-ids;
- chosen metric absent from **all** runs → `best=None`, all `rank=None`, one warning;
- runs validated on different metric sets (their `validation.metrics` differ);
- `same_data` is `False` (runs use different datasets — comparison may not be apples-to-apples);
- more than one distinct `experiment` among the runs;
- runs have different input or output column sets.

## Terminal Rendering (CLI `compare`)

Rendered from the result by small helpers in `cli/main.py`, using manual column
formatting + `_info`/`_success` (no new dependency). Four blocks, then warnings + best:

**a) Ranking table** — full detail for the chosen metric:

```
Ranking by test rmse (3 runs)
  rank  run        method           train      test       cv(mean±std)      Δtest
  1     run-003    random-forest    0.0091     0.0123     0.0130±0.0021     —
  2     run-001    mlp              0.0150     0.0210     0.0225±0.0030     +0.0087
```

Unranked runs (chosen metric absent) are listed after ranked ones with `rank`/values as `—`.

**b) All test metrics** — breadth across metrics, best-per-metric marked with `*`:

```
Test metrics
  run        rmse       r2         mae
  run-003    0.0123*    0.991*     0.0098*
  run-001    0.0210     0.985      0.0150
```

**c) Configuration differences** — differing leaves only, else a one-liner:

```
Configuration differences
  field                                 run-003          run-001
  model.method                          random-forest    mlp
  model.hyperparameters.n_estimators    300              —
```

(`<absent>` renders as `—`.) If `config_diff` is empty: `Configurations are identical.`

**d) Lineage** + summary line, then the best line:

```
Lineage
  run        seed  data_hash  config_hash  created
  run-003    7     1a2b3c…    4d5e6f…      2026-06-30T…
  run-001    42    9f8e7d…    0c1d2e…      2026-06-29T…
Warning: runs use different datasets (data_hash differs); comparisons may not be apples-to-apples.
Best by test rmse: run-003 (0.0123)
```

Warnings print as sober `Warning:` lines. Hashes are abbreviated for display; the full
values stay in the structured result.

## Error Handling

- Empty `--runs` / empty list → `DataError` (unchanged).
- A single run → valid: `config_diff == {}`, `rank == 1`, `delta == 0.0`, no deltas to others.
- Unknown / absent ranking metric → handled via warnings above, never an exception.
- Missing metric pieces (e.g. a run without CV for some metric) → `None`, rendered `—`.

## Validation / Testing

**`tests/test_comparison.py` (new)** — `build_comparison`:

- result shape for 2–3 real runs (fields present, `metric_names` union, `metrics` block);
- ranking + `delta` correct for a lower-is-better (`rmse`) and higher-is-better (`r2`)
  metric; `best` and `best_per_metric` correct;
- `config_diff`: only differing paths; identical configs → `{}`; `<absent>` when a key is
  missing from one run;
- `lineage_summary`: `same_data` and `data_hash_groups` for same-data vs different-data runs;
- warnings: differing metric sets, ranking metric missing from some runs (those unranked),
  different datasets, more than one experiment;
- a single run; empty list → `DataError`.

**`tests/test_cli.py` (extend / update)** — the existing `test_compare_cli` is updated to
the new output: the `compare` command prints the four blocks, the best line, and warning
lines when runs differ.

Determinism: the same runs yield the same result.

## Scope

**In (Cycle 1, Core):** `comparison.py` with `build_comparison`; `services.compare_runs`
delegating to it; enriched CLI `compare` output (four blocks + warnings + best); config
diff via `canonical_dict()`; ranking/deltas/best-per-metric; lineage summary; warnings;
tests; CHANGELOG entry.

**Out (Cycle 2 / YAGNI):** the Professional HTML comparison report + interactive cross-run
charts (separate cycle); Markdown file output; `--all` / `--experiment` selectors;
terminal plots; cross-project comparison; statistical significance testing.
