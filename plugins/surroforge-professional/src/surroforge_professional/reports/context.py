"""Context assembly for Professional HTML reports."""

from __future__ import annotations

import math
from typing import Any

from surroforge.core.run import Run
from surroforge.errors import PluginError
from surroforge.validators.metrics import two_source_split


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
        if metric not in cv:
            raise PluginError(f"HTML report requires cross-validation metric {metric!r}.")
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


def _per_fold(metric_record: Any) -> list[Any] | None:
    if not isinstance(metric_record, dict):
        return None
    values = metric_record.get("per_fold")
    if values is None or isinstance(values, (str, bytes)):
        return None
    try:
        per_fold = list(values)
    except TypeError:
        return None
    return per_fold or None


def _component_stats(fold_splits: list[dict[str, float]], component: str) -> tuple[float, float]:
    values = [split[component] for split in fold_splits]
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _two_source_context(
    config: Any, metrics: dict[str, Any], lineage: dict[str, Any]
) -> dict[str, Any] | None:
    field_names = ("energy_l2", "truncation_error")
    if not set(field_names) <= set(config.validation.metrics):
        return None

    train = metrics.get("train")
    test = metrics.get("test")
    cv = metrics.get("cross_validation")
    if not all(isinstance(section, dict) for section in (train, test, cv)):
        return None
    if not all(metric in train and metric in test for metric in field_names):
        return None
    if not all(metric in cv for metric in field_names):
        return None

    energy_folds = _per_fold(cv["energy_l2"])
    truncation_folds = _per_fold(cv["truncation_error"])
    if (
        energy_folds is None
        or truncation_folds is None
        or len(energy_folds) != len(truncation_folds)
    ):
        return None

    try:
        train_split = two_source_split(train["energy_l2"], train["truncation_error"])
        test_split = two_source_split(test["energy_l2"], test["truncation_error"])
        fold_splits = [
            two_source_split(energy, truncation)
            for energy, truncation in zip(energy_folds, truncation_folds, strict=True)
        ]
        cv_mean: dict[str, float] = {}
        cv_std: dict[str, float] = {}
        for component in ("total", "truncation", "surrogate"):
            cv_mean[component], cv_std[component] = _component_stats(fold_splits, component)
    except (ArithmeticError, KeyError, TypeError, ValueError):
        return None

    rows = []
    for component, label in (
        ("total", "total field error"),
        ("truncation", "truncation floor"),
        ("surrogate", "surrogate error"),
    ):
        rows.append(
            {
                "label": label,
                "train": train_split[component],
                "test": test_split[component],
                "cv_mean": cv_mean[component],
                "cv_std": cv_std[component],
                "train_fmt": fmt(train_split[component]),
                "test_fmt": fmt(test_split[component]),
                "cv_mean_fmt": fmt(cv_mean[component]),
                "cv_std_fmt": fmt(cv_std[component]),
            }
        )

    reduction = lineage.get("reduction")
    if not isinstance(reduction, dict):
        reduction = {}
    energy_captured = reduction.get("energy_captured")

    return {
        "rows": rows,
        "reduction_rank": reduction.get("rank"),
        "energy_captured": energy_captured,
        "energy_captured_fmt": fmt(energy_captured) if energy_captured is not None else None,
        "truncation_per_fold": ", ".join(fmt(value) for value in truncation_folds),
    }


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
        "two_source": _two_source_context(config, metrics, lineage),
        "config_yaml": config.to_yaml(),
        "python_version": lineage.get("python_version", ""),
        "surroforge_version": lineage.get("surroforge_version", ""),
        "dependencies": [str(dep) for dep in dependencies],
        "seed": run.seed,
        "test_fraction": config.data.split.test_fraction,
        "reproduce_command": f"surroforge reproduce {run.run_id}",
        "fmt": fmt,
    }
