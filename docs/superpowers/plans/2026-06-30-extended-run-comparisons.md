# Extended Run Comparisons (Core, Cycle 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the minimal `compare` (flat test-metric table + single best) into an extended Core comparison: richer metric table, config/hyperparameter diff, lineage comparison — as a structured result plus an enriched terminal view.

**Architecture:** A pure `surroforge.comparison.build_comparison(runs, *, metric)` builds a JSON-serializable result dict (metrics/ranking/deltas/best-per-metric, config diff via `canonical_dict()`, lineage summary, warnings). `services.compare_runs` loads runs and delegates; the CLI renders four blocks. The ranking-direction table moves into `comparison.py`.

**Tech Stack:** Python 3.10+, existing deps only (no new dependency). Terminal output uses the existing manual column formatting + `_info`/`_success`.

## Global Constraints

- **No new dependency** (NFR8): terminal tables use manual f-string formatting and the existing `_info`/`_success` prints.
- **`build_comparison` raises only on an empty run list** (`DataError`); every heterogeneity (differing metrics/data/experiment/columns, missing ranking metric) is a non-fatal **warning** string in `result["warnings"]`.
- **Config diff is generic**: a deep-diff over each run's `run.config.canonical_dict()`, reporting only differing dotted leaf paths; a run lacking a key contributes the sentinel string `"<absent>"`. Hyperparameters live **flat under `model`** in the canonical dict (e.g. `model.n_estimators`, `model.degree`), not under `model.hyperparameters`.
- **Ranking** is by the chosen metric's `test` value via `_LOWER_IS_BETTER = {"rmse": True, "mae": True, "r2": False}` (unknown metric → lower-is-better). `rank` 1 = best; runs missing the metric are `rank=None`/`delta=None`. `delta = abs(test - best_test)` (always ≥ 0; 0 for best).
- **Structured result is JSON-serializable** (Cycle 2 consumes it) — only built-in scalars/lists/dicts; no objects.
- **Lint:** `ruff check src tests` must pass (line-length 100 in this tree per the repo config).

---

### Task 1: `comparison.py` — `build_comparison` + tests

**Files:**
- Create: `src/surroforge/comparison.py`
- Test: `tests/test_comparison.py`

**Interfaces:**
- Consumes: `Run` accessors `run.run_id`, `run.method`, `run.created_at`, `run.seed`, `run.data_hash`, `run.config_hash`, `run.parent_run`, `run.metrics` (dict with `train`/`test`/`cross_validation`), `run.config` (`.experiment`, `.model.method`, `.data.inputs`, `.data.outputs`, `.validation.metrics`, `.canonical_dict()`); `surroforge.errors.DataError`.
- Produces: `build_comparison(runs: list[Run], *, metric: str = "rmse") -> dict` with keys `metric`, `metric_names`, `runs`, `best`, `best_per_metric`, `config_diff`, `lineage_summary`, `warnings` (exact shape in the spec). Also exports the module-level `_LOWER_IS_BETTER`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_comparison.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import pytest

from surroforge.comparison import build_comparison
from surroforge.core.config import parse_config
from surroforge.errors import DataError


def _cfg(*, method="polynomial", degree=2, n_estimators=None, metrics=("rmse", "r2", "mae"),
         experiment="exp", inputs=("a", "b"), outputs=("y",), seed=42):
    model: dict = {"method": method}
    if method == "polynomial":
        model["degree"] = degree
    if n_estimators is not None:
        model["n_estimators"] = n_estimators
    return parse_config(
        {
            "experiment": experiment,
            "data": {
                "source": "s", "inputs": list(inputs), "outputs": list(outputs),
                "split": {"test_fraction": 0.2, "seed": seed},
            },
            "model": model,
            "validation": {"cross_validation": 5, "metrics": list(metrics)},
        }
    )


def _metrics(*, names=("rmse", "r2", "mae"), test=None):
    test = test or {"rmse": 0.02, "r2": 0.98, "mae": 0.01}
    train = {"rmse": 0.01, "r2": 0.99, "mae": 0.008}
    cv = {"rmse": (0.02, 0.002), "r2": (0.98, 0.001), "mae": (0.01, 0.001)}
    return {
        "train": {n: train[n] for n in names if n in train},
        "test": {n: test[n] for n in names if n in test},
        "cross_validation": {
            "folds": 5,
            **{n: {"mean": cv[n][0], "std": cv[n][1]} for n in names if n in cv},
        },
    }


def _run(run_id, *, config=None, metrics=None, data_hash="d0", config_hash="c0",
         created_at="2026-06-30T00:00:00Z", seed=42, parent_run=None):
    config = config if config is not None else _cfg()
    return SimpleNamespace(
        run_id=run_id, method=config.model.method, config=config,
        metrics=metrics if metrics is not None else _metrics(),
        created_at=created_at, seed=seed, data_hash=data_hash,
        config_hash=config_hash, parent_run=parent_run,
    )


def test_result_shape():
    result = build_comparison([_run("run-001"), _run("run-002")])
    assert result["metric"] == "rmse"
    assert result["metric_names"] == ["rmse", "r2", "mae"]
    assert len(result["runs"]) == 2
    entry = result["runs"][0]
    assert set(entry) >= {"run_id", "method", "experiment", "created_at", "metrics",
                          "rank", "delta", "lineage"}
    assert set(entry["metrics"]["rmse"]) == {"train", "test", "cv_mean", "cv_std"}
    assert set(entry["lineage"]) == {"seed", "data_hash", "config_hash", "parent_run"}


def test_ranking_and_delta_lower_is_better():
    a = _run("A", metrics=_metrics(test={"rmse": 0.02, "r2": 0.98, "mae": 0.01}))
    b = _run("B", metrics=_metrics(test={"rmse": 0.01, "r2": 0.95, "mae": 0.02}))
    result = build_comparison([a, b], metric="rmse")
    by_id = {e["run_id"]: e for e in result["runs"]}
    assert result["best"] == "B"
    assert by_id["B"]["rank"] == 1 and by_id["B"]["delta"] == 0.0
    assert by_id["A"]["rank"] == 2
    assert by_id["A"]["delta"] == pytest.approx(0.01)


def test_ranking_higher_is_better_metric():
    a = _run("A", metrics=_metrics(test={"rmse": 0.02, "r2": 0.98, "mae": 0.01}))
    b = _run("B", metrics=_metrics(test={"rmse": 0.01, "r2": 0.95, "mae": 0.02}))
    result = build_comparison([a, b], metric="r2")
    assert result["best"] == "A"  # higher r2 is better


def test_best_per_metric():
    a = _run("A", metrics=_metrics(test={"rmse": 0.02, "r2": 0.99, "mae": 0.05}))
    b = _run("B", metrics=_metrics(test={"rmse": 0.01, "r2": 0.90, "mae": 0.01}))
    result = build_comparison([a, b])
    assert result["best_per_metric"] == {"rmse": "B", "r2": "A", "mae": "B"}


def test_config_diff_only_differing_paths():
    a = _run("A", config=_cfg(degree=2))
    b = _run("B", config=_cfg(degree=3))
    diff = build_comparison([a, b])["config_diff"]
    assert diff["model.degree"] == {"A": 2, "B": 3}
    assert "model.method" not in diff  # both polynomial


def test_config_diff_identical_is_empty():
    a = _run("A", config=_cfg(degree=2))
    b = _run("B", config=_cfg(degree=2))
    assert build_comparison([a, b])["config_diff"] == {}


def test_config_diff_absent_key_sentinel():
    a = _run("A", config=_cfg(method="random-forest", n_estimators=300))
    b = _run("B", config=_cfg(method="random-forest", n_estimators=None))
    diff = build_comparison([a, b])["config_diff"]
    assert diff["model.n_estimators"] == {"A": 300, "B": "<absent>"}


def test_lineage_summary_same_and_different_data():
    same = build_comparison([_run("A", data_hash="h1"), _run("B", data_hash="h1")])
    assert same["lineage_summary"]["same_data"] is True
    diff = build_comparison([_run("A", data_hash="h1"), _run("B", data_hash="h2")])
    assert diff["lineage_summary"]["same_data"] is False
    assert diff["lineage_summary"]["data_hash_groups"] == {"h1": ["A"], "h2": ["B"]}


def test_warning_on_different_metric_sets_and_missing_ranking_metric():
    a = _run("A")  # rmse, r2, mae
    b = _run("B", config=_cfg(metrics=("rmse",)), metrics=_metrics(names=("rmse",)))
    result = build_comparison([a, b], metric="mae")
    text = " ".join(result["warnings"])
    assert "different metric sets" in text
    assert "mae" in text and "B" in text  # B unranked on mae
    by_id = {e["run_id"]: e for e in result["runs"]}
    assert by_id["B"]["rank"] is None
    assert by_id["B"]["metrics"]["mae"]["test"] is None


def test_warning_on_different_data_and_experiment():
    a = _run("A", config=_cfg(experiment="exp-1"), data_hash="h1")
    b = _run("B", config=_cfg(experiment="exp-2"), data_hash="h2")
    text = " ".join(build_comparison([a, b])["warnings"])
    assert "different datasets" in text
    assert "different experiments" in text


def test_single_run():
    result = build_comparison([_run("A")])
    assert result["best"] == "A"
    assert result["runs"][0]["rank"] == 1
    assert result["runs"][0]["delta"] == 0.0
    assert result["config_diff"] == {}


def test_empty_runs_raises():
    with pytest.raises(DataError):
        build_comparison([])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_comparison.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'surroforge.comparison'`.

- [ ] **Step 3: Implement `comparison.py`**

Create `src/surroforge/comparison.py`:

```python
"""Extended run comparison (PRD 4.1, 9.2).

``build_comparison`` turns a list of persisted Runs into a structured, JSON-serializable
comparison: a per-run metric table (train/test/CV mean±std), a ranking by a chosen metric
with deltas, best-per-metric, a config/hyperparameter diff (differing canonical-config
leaves only), a lineage summary (which runs share data), and human-readable warnings.

Pure and render-free — the CLI renders the result, and a later Professional HTML
comparison report consumes the same dict.
"""

from __future__ import annotations

from typing import Any

from surroforge.core.run import Run
from surroforge.errors import DataError

# Ranking direction by metric; unknown metrics default to lower-is-better. This module is
# the home of the ranking direction — services delegates here.
_LOWER_IS_BETTER = {"rmse": True, "mae": True, "r2": False}

_ABSENT = "<absent>"


def _lower_is_better(metric: str) -> bool:
    return _LOWER_IS_BETTER.get(metric, True)


def _metric_names(runs: list[Run]) -> list[str]:
    """Union of every run's validation metrics, in stable first-seen order."""
    names: list[str] = []
    for run in runs:
        for name in run.config.validation.metrics:
            if name not in names:
                names.append(name)
    return names


def _run_metrics(run: Run, metric_names: list[str]) -> dict[str, dict[str, Any]]:
    metrics = run.metrics if isinstance(run.metrics, dict) else {}
    train = metrics.get("train") if isinstance(metrics.get("train"), dict) else {}
    test = metrics.get("test") if isinstance(metrics.get("test"), dict) else {}
    cv = (
        metrics.get("cross_validation")
        if isinstance(metrics.get("cross_validation"), dict)
        else {}
    )
    out: dict[str, dict[str, Any]] = {}
    for name in metric_names:
        cvrec = cv.get(name) if isinstance(cv.get(name), dict) else {}
        out[name] = {
            "train": train.get(name),
            "test": test.get(name),
            "cv_mean": cvrec.get("mean"),
            "cv_std": cvrec.get("std"),
        }
    return out


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _config_diff(runs: list[Run]) -> dict[str, dict[str, Any]]:
    flats = {run.run_id: _flatten(run.config.canonical_dict()) for run in runs}
    all_paths = sorted({path for flat in flats.values() for path in flat})
    diff: dict[str, dict[str, Any]] = {}
    for path in all_paths:
        values = {rid: flat.get(path, _ABSENT) for rid, flat in flats.items()}
        first = next(iter(values.values()))
        if any(value != first for value in values.values()):
            diff[path] = values
    return diff


def _lineage_summary(runs: list[Run]) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    for run in runs:
        groups.setdefault(run.data_hash, []).append(run.run_id)
    return {"same_data": len(groups) == 1, "data_hash_groups": groups}


def _test_value(entry: dict[str, Any], metric: str) -> Any:
    record = entry["metrics"].get(metric)
    return record.get("test") if record else None


def _best_per_metric(entries: list[dict[str, Any]], metric_names: list[str]) -> dict[str, str]:
    best: dict[str, str] = {}
    for name in metric_names:
        candidates = [
            (e["run_id"], _test_value(e, name))
            for e in entries
            if _test_value(e, name) is not None
        ]
        if not candidates:
            continue
        pick = min if _lower_is_better(name) else max
        best[name] = pick(candidates, key=lambda pair: pair[1])[0]
    return best


def _warnings(
    runs: list[Run],
    entries: list[dict[str, Any]],
    metric: str,
    rankable: list[dict[str, Any]],
    same_data: bool,
) -> list[str]:
    warnings: list[str] = []
    if len({tuple(run.config.validation.metrics) for run in runs}) > 1:
        warnings.append(
            "Runs were validated on different metric sets; the table uses the union "
            "and shows '-' where a run lacks a metric."
        )
    unranked = [e["run_id"] for e in entries if _test_value(e, metric) is None]
    if not rankable:
        warnings.append(
            f"Ranking metric '{metric}' is not present in any run; runs are not ranked."
        )
    elif unranked:
        warnings.append(
            f"Ranking metric '{metric}' is missing for: {', '.join(unranked)}; "
            "they are left unranked."
        )
    if not same_data:
        warnings.append(
            "Runs use different datasets (data_hash differs); comparisons may not be "
            "apples-to-apples."
        )
    experiments = {run.config.experiment for run in runs}
    if len(experiments) > 1:
        warnings.append(
            "Comparing runs from different experiments "
            f"({', '.join(sorted(experiments))}); metrics may not be comparable."
        )
    inputs = {tuple(run.config.data.inputs) for run in runs}
    outputs = {tuple(run.config.data.outputs) for run in runs}
    if len(inputs) > 1 or len(outputs) > 1:
        warnings.append("Runs have different input or output columns.")
    return warnings


def build_comparison(runs: list[Run], *, metric: str = "rmse") -> dict[str, Any]:
    if not runs:
        raise DataError("Provide at least one run to compare.")

    metric_names = _metric_names(runs)

    entries: list[dict[str, Any]] = []
    for run in runs:
        entries.append(
            {
                "run_id": run.run_id,
                "method": run.method,
                "experiment": run.config.experiment,
                "created_at": run.created_at,
                "metrics": _run_metrics(run, metric_names),
                "rank": None,
                "delta": None,
                "lineage": {
                    "seed": run.seed,
                    "data_hash": run.data_hash,
                    "config_hash": run.config_hash,
                    "parent_run": run.parent_run,
                },
            }
        )

    # Rank by the chosen metric's test value (entries are shared dicts, so setting rank
    # on the rankable view fills it in on the returned entries too).
    rankable = [e for e in entries if _test_value(e, metric) is not None]
    rankable.sort(key=lambda e: _test_value(e, metric), reverse=not _lower_is_better(metric))
    best = rankable[0]["run_id"] if rankable else None
    best_test = _test_value(rankable[0], metric) if rankable else None
    for position, entry in enumerate(rankable, start=1):
        entry["rank"] = position
        entry["delta"] = abs(_test_value(entry, metric) - best_test)

    lineage_summary = _lineage_summary(runs)

    return {
        "metric": metric,
        "metric_names": metric_names,
        "runs": entries,
        "best": best,
        "best_per_metric": _best_per_metric(entries, metric_names),
        "config_diff": _config_diff(runs),
        "lineage_summary": lineage_summary,
        "warnings": _warnings(runs, entries, metric, rankable, lineage_summary["same_data"]),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_comparison.py -q`
Expected: PASS (12 tests).

- [ ] **Step 5: Lint**

Run: `ruff check src/surroforge/comparison.py tests/test_comparison.py`
Expected: "All checks passed!".

- [ ] **Step 6: Commit**

```bash
git add src/surroforge/comparison.py tests/test_comparison.py
git commit -m "feat(core): add build_comparison extended-comparison data builder"
```

---

### Task 2: Wire `services.compare_runs` + rewrite CLI `compare`

**Files:**
- Modify: `src/surroforge/services.py` (`compare_runs`; remove the moved `_select_best`/`_LOWER_IS_BETTER`)
- Modify: `src/surroforge/cli/main.py` (`compare` command + render helpers; remove now-unused `_metric_cols`)
- Test: `tests/test_cli.py` (update `test_compare_cli`)

**Interfaces:**
- Consumes: `build_comparison` from Task 1.
- Produces: `services.compare_runs(project, run_ids, *, metric="rmse") -> dict` returning the `build_comparison` result; an enriched `compare` CLI command.

- [ ] **Step 1: Update `test_compare_cli` (the failing test)**

In `tests/test_cli.py`, replace the body of `test_compare_cli` with:

```python
def test_compare_cli(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])
    result = runner.invoke(app, ["--config-dir", str(root), "compare", "--runs", "run-001,run-002"])
    assert result.exit_code == 0, result.output
    assert "Ranking by test rmse" in result.output
    assert "Test metrics" in result.output
    assert "Configurations are identical." in result.output  # two identical-config runs
    assert "Lineage" in result.output
    assert "run-001" in result.output
    assert "Best by test rmse" in result.output
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_compare_cli -q`
Expected: FAIL — the new block headers are not in the current output.

- [ ] **Step 3: Delegate `services.compare_runs` to `build_comparison`**

In `src/surroforge/services.py`:

1. Add the import near the top with the other `surroforge` imports:

```python
from surroforge.comparison import build_comparison
```

2. Replace the entire `compare_runs` function and the `_select_best` helper and the
`_LOWER_IS_BETTER` module constant with just this (delete `_select_best` and
`_LOWER_IS_BETTER` — they now live in `comparison.py`; first run
`grep -n "_select_best\|_LOWER_IS_BETTER" src/surroforge/services.py` and confirm
`compare_runs` is their only user):

```python
def compare_runs(project: Project, run_ids: list[str], *, metric: str = "rmse") -> dict[str, Any]:
    """Compare runs by an extended metric / config / lineage view (PRD 4.1).

    Loads the runs and delegates to :func:`surroforge.comparison.build_comparison`, whose
    structured result the CLI renders and the Professional HTML comparison report consumes.
    """
    if not run_ids:
        raise DataError("Provide at least one run id to compare.")
    runs = [project.run(rid) for rid in run_ids]
    return build_comparison(runs, metric=metric)
```

- [ ] **Step 4: Rewrite the CLI `compare` command + render helpers**

In `src/surroforge/cli/main.py`: (1) replace the `compare` command body with the
one-liner `_render_comparison(...)` call shown below; (2) **delete** the now-unused
`_metric_cols` helper; (3) **keep the existing `_fmt`** function exactly as-is — do NOT
redefine it; (4) add the new render helpers below (they are all module-level, so order
relative to `_fmt` does not matter — place them after the existing `_fmt`). The new code
does **not** include `_fmt` (it is reused from what is already in the file).

```python
@app.command()
@_guard
def compare(
    runs: str = typer.Option(..., "--runs", "-r", help="Comma-separated run ids."),
    metric: str = typer.Option("rmse", "--metric", "-m", help="Metric used to rank runs."),
) -> None:
    """Compare several training runs side by side (PRD 4.1)."""
    project = _project()
    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    _render_comparison(services.compare_runs(project, run_ids, metric=metric))


# NOTE: `_fmt` already exists in this file — reuse it, do not redefine it.


def _fmt_cv(mean, std) -> str:
    if not isinstance(mean, (int, float)):
        return "-"
    if isinstance(std, (int, float)):
        return f"{mean:.4g}±{std:.2g}"
    return f"{mean:.4g}"


def _abbrev(value, n: int = 8) -> str:
    text = str(value) if value is not None else "-"
    return f"{text[:n]}…" if len(text) > n else text


def _render_comparison(result) -> None:
    _render_ranking_table(result)
    _render_test_metrics(result)
    _render_config_diff(result)
    _render_lineage(result)
    for warning in result["warnings"]:
        _info(f"Warning: {warning}")
    if result["best"] is not None:
        best_test = _best_test_value(result)
        _success(f"Best by test {result['metric']}: {result['best']} ({_fmt(best_test)})")


def _best_test_value(result):
    for entry in result["runs"]:
        if entry["run_id"] == result["best"]:
            record = entry["metrics"].get(result["metric"], {})
            return record.get("test")
    return None


def _render_ranking_table(result) -> None:
    metric = result["metric"]
    _info(f"Ranking by test {metric} ({len(result['runs'])} runs)")
    _info(
        f"  {'rank':<5}{'run':<12}{'method':<16}{'train':>11}{'test':>11}"
        f"{'cv(mean±std)':>18}{'Δtest':>11}"
    )
    ranked = sorted((e for e in result["runs"] if e["rank"] is not None), key=lambda e: e["rank"])
    unranked = [e for e in result["runs"] if e["rank"] is None]
    for entry in ranked + unranked:
        record = entry["metrics"].get(metric, {})
        rank = str(entry["rank"]) if entry["rank"] is not None else "-"
        delta = "—" if entry["delta"] in (None, 0.0) else f"+{_fmt(entry['delta'])}"
        _info(
            f"  {rank:<5}{entry['run_id']:<12}{entry['method']:<16}"
            f"{_fmt(record.get('train')):>11}{_fmt(record.get('test')):>11}"
            f"{_fmt_cv(record.get('cv_mean'), record.get('cv_std')):>18}{delta:>11}"
        )


def _render_test_metrics(result) -> None:
    names = result["metric_names"]
    best = result["best_per_metric"]
    _info("Test metrics")
    _info(f"  {'run':<12}" + "".join(f"{m:>12}" for m in names))
    for entry in result["runs"]:
        cells = []
        for name in names:
            value = entry["metrics"].get(name, {}).get("test")
            text = _fmt(value)
            if best.get(name) == entry["run_id"] and isinstance(value, (int, float)):
                text += "*"
            cells.append(f"{text:>12}")
        _info(f"  {entry['run_id']:<12}" + "".join(cells))


def _render_config_diff(result) -> None:
    _info("Configuration differences")
    diff = result["config_diff"]
    if not diff:
        _info("  Configurations are identical.")
        return
    run_ids = [entry["run_id"] for entry in result["runs"]]
    _info(f"  {'field':<40}" + "".join(f"{rid:>16}" for rid in run_ids))
    for path in sorted(diff):
        values = diff[path]
        cells = "".join(f"{_fmt_diff(values.get(rid)):>16}" for rid in run_ids)
        _info(f"  {path:<40}{cells}")


def _fmt_diff(value) -> str:
    return "—" if value in (None, "<absent>") else str(value)


def _render_lineage(result) -> None:
    _info("Lineage")
    _info(f"  {'run':<12}{'seed':>6}  {'data_hash':<11}{'config_hash':<12}{'created':<22}")
    for entry in result["runs"]:
        lineage = entry["lineage"]
        _info(
            f"  {entry['run_id']:<12}{str(lineage['seed']):>6}  "
            f"{_abbrev(lineage['data_hash']):<11}{_abbrev(lineage['config_hash']):<12}"
            f"{str(entry['created_at'])[:21]:<22}"
        )
    summary = result["lineage_summary"]
    if summary["same_data"]:
        shared = next(iter(summary["data_hash_groups"]))
        _info(f"  All runs share the same data (data_hash {_abbrev(shared)}).")
```

- [ ] **Step 5: Run the updated CLI test + the comparison tests**

Run: `python -m pytest tests/test_cli.py::test_compare_cli tests/test_comparison.py -q`
Expected: PASS.

- [ ] **Step 6: Guard against regressions in services/CLI**

Run: `python -m pytest tests/test_cli.py tests/test_services.py -q 2>/dev/null || python -m pytest tests/test_cli.py -q`
Expected: PASS (if `tests/test_services.py` does not exist, the CLI suite alone must pass). Then `ruff check src tests`.
Expected: "All checks passed!".

- [ ] **Step 7: Commit**

```bash
git add src/surroforge/services.py src/surroforge/cli/main.py tests/test_cli.py
git commit -m "feat(core): render extended run comparison in the compare CLI"
```

---

### Task 3: CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add the CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]`, add this bullet to the existing
`### Added — Python API` section (append it after the last bullet in that subsection;
confirm the heading exists with `grep -n "### Added — Python API" CHANGELOG.md`):

```markdown
- **Extended run comparisons** — `surroforge compare` now prints a ranking table
  (train/test/CV mean±std with deltas to the best), an all-test-metrics table with the
  best run per metric marked, a configuration/hyperparameter **diff** (only the fields
  that differ between runs, derived from each run's canonical config), and a **lineage**
  block flagging whether the runs share the same data. Heterogeneous comparisons (different
  metric sets, datasets, experiments, or columns) emit warnings rather than failing. The
  structured result is available via `surroforge.comparison.build_comparison(runs, metric=...)`
  / `services.compare_runs(...)`. Run selection is unchanged (`--runs a,b,c`).
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: document extended run comparisons"
```

---

## Final Verification

```bash
python -m pytest tests/test_comparison.py tests/test_cli.py -q
ruff check src tests
```

Expected: tests pass, "All checks passed!". Then proceed to the whole-branch review and
`superpowers:finishing-a-development-branch`.
