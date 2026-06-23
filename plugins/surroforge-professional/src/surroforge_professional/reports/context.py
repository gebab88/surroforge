"""Context assembly for Professional HTML reports."""

from __future__ import annotations

from typing import Any

from surroforge.core.run import Run
from surroforge.errors import PluginError


def fmt(value: Any) -> str:
    """Compact numeric formatting used by scalar report tables."""
    try:
        return f"{float(value):.5g}"
    except (TypeError, ValueError):
        return str(value)


def _require_key(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise PluginError(f"HTML report requires {path}[{key!r}].")
    return mapping[key]


def _require_non_empty(mapping: dict[str, Any], key: str, path: str) -> str:
    value = _require_key(mapping, key, path)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise PluginError(f"HTML report requires non-empty {path}[{key!r}].")
    return text


def _require_dict(mapping: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    value = _require_key(mapping, key, path)
    if not isinstance(value, dict):
        raise PluginError(f"HTML report requires {path}[{key!r}] to be a mapping.")
    return value


def _metric_rows(metrics: dict[str, Any], metric_names: list[str]) -> list[dict[str, Any]]:
    train = _require_dict(metrics, "train", "metrics")
    test = _require_dict(metrics, "test", "metrics")
    cv = _require_dict(metrics, "cross_validation", "metrics")

    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        cv_metric = _require_dict(cv, metric, "metrics['cross_validation']")
        train_value = _require_key(train, metric, "metrics['train']")
        test_value = _require_key(test, metric, "metrics['test']")
        cv_mean = _require_key(cv_metric, "mean", f"metrics['cross_validation'][{metric!r}]")
        cv_std = _require_key(cv_metric, "std", f"metrics['cross_validation'][{metric!r}]")
        rows.append(
            {
                "name": metric,
                "train": train_value,
                "test": test_value,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "train_fmt": fmt(train_value),
                "test_fmt": fmt(test_value),
                "cv_mean_fmt": fmt(cv_mean),
                "cv_std_fmt": fmt(cv_std),
            }
        )
    return rows


def build_context(run: Run) -> dict[str, Any]:
    config = run.config
    metrics = run.metrics
    lineage = run.lineage
    if not isinstance(metrics, dict):
        raise PluginError("HTML report requires metrics to be a mapping.")
    if not isinstance(lineage, dict):
        raise PluginError("HTML report requires lineage to be a mapping.")

    metric_rows = _metric_rows(metrics, config.validation.metrics)
    cv = _require_dict(metrics, "cross_validation", "metrics")
    dependencies = lineage.get("dependencies", [])
    if not isinstance(dependencies, list):
        dependencies = [str(dependencies)]

    return {
        "run_id": run.run_id,
        "experiment": config.experiment,
        "method": config.model.method,
        "created_at": lineage.get("created_at", ""),
        "parent_run": lineage.get("parent_run"),
        "data_source": lineage.get("data_source", config.data.source),
        "data_hash": _require_non_empty(lineage, "data_hash", "lineage"),
        "config_hash": _require_non_empty(lineage, "config_hash", "lineage"),
        "prediction_hash": _require_non_empty(lineage, "prediction_hash", "lineage"),
        "n_samples": metrics.get("n_samples", "?"),
        "n_inputs": metrics.get("n_inputs", len(config.data.inputs)),
        "n_outputs": metrics.get("n_outputs", len(config.data.outputs)),
        "inputs": config.data.inputs,
        "outputs": config.data.outputs,
        "hyperparameters": [
            {"key": key, "value": value} for key, value in config.model.hyperparameters.items()
        ],
        "metric_rows": metric_rows,
        "cv_folds": cv.get("folds", config.validation.cross_validation.folds),
        "two_source": None,
        "config_yaml": config.to_yaml(),
        "python_version": lineage.get("python_version", ""),
        "surroforge_version": lineage.get("surroforge_version", ""),
        "dependencies": [str(dep) for dep in dependencies],
        "seed": run.seed,
        "test_fraction": config.data.split.test_fraction,
        "reproduce_command": f"surroforge reproduce {run.run_id}",
        "fmt": fmt,
    }
