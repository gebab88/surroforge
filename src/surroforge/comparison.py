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


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce an untrusted persisted value to a dict, or ``{}`` when it is not one."""
    return value if isinstance(value, dict) else {}


def _run_metrics(run: Run, metric_names: list[str]) -> dict[str, dict[str, Any]]:
    metrics = _as_dict(run.metrics)
    train = _as_dict(metrics.get("train"))
    test = _as_dict(metrics.get("test"))
    cv = _as_dict(metrics.get("cross_validation"))
    out: dict[str, dict[str, Any]] = {}
    for name in metric_names:
        cvrec = _as_dict(cv.get(name))
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
