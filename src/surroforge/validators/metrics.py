"""Standard regression metrics (PRD 4.6: RMSE, R², MAE).

Metrics operate on 2-D arrays ``(n_samples, n_outputs)`` and average uniformly
across outputs, so single- and multi-output models share one code path.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from surroforge.errors import ConfigError


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average")))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred, multioutput="uniform_average"))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred, multioutput="uniform_average"))


AVAILABLE_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": _rmse,
    "mae": _mae,
    "r2": _r2,
}


def validate_metric_names(names: list[str]) -> None:
    unknown = [n for n in names if n not in AVAILABLE_METRICS]
    if unknown:
        raise ConfigError(
            f"Unknown metric(s): {unknown}. Available: {sorted(AVAILABLE_METRICS)}"
        )


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, names: list[str]
) -> dict[str, float]:
    """Compute the requested metrics for one set of predictions."""
    validate_metric_names(names)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {name: AVAILABLE_METRICS[name](y_true, y_pred) for name in names}
