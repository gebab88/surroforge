"""Standard regression metrics (PRD 4.6: RMSE, R², MAE).

Metrics operate on 2-D arrays ``(n_samples, n_outputs)`` and average uniformly
across outputs, so single- and multi-output models share one code path.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from surroforge.errors import ConfigError

# Floor for the field-energy-relative denominator, so a (near-)zero reference field
# does not divide by zero; far below any meaningful field energy (Story R2.3). Single
# source of truth shared with the reproduce() verdict norm — services imports it from
# here (Story R3.1).
FIELD_ENERGY_FLOOR = 1e-300


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average")))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred, multioutput="uniform_average"))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred, multioutput="uniform_average"))


def _check_field_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Enforce the contract sklearn's check_array gives the other metrics.

    The field metrics consume raw arrays, so without this a shape mismatch would
    silently broadcast to a plausible-but-wrong number, 1-D input would surface as
    a raw LinAlgError from the SVD, non-finite values would propagate as silent
    ``nan``, and empty arrays would score a perfect 0.0.
    """
    if y_true.ndim != 2 or y_true.shape != y_pred.shape:
        raise ValueError(
            f"Field metrics require matching 2-D arrays, got {y_true.shape} vs {y_pred.shape}."
        )
    if y_true.size == 0:
        raise ValueError("Field metrics require non-empty arrays.")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Field metrics require finite values; got NaN or inf.")


def _energy_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative field-energy error ``‖y_pred − y_true‖_F / max(‖y_true‖_F, floor)``.

    Deliberately a GLOBAL Frobenius norm over the whole ``(n, n_outputs)`` block —
    not sklearn's per-column ``uniform_average`` like rmse/mae/r2. The asymmetry is
    the point: this is the same norm the reproduce() verdict uses
    (``services._field_energy_relative_diff``, Story R2.3), so a run's field error
    and its reproducibility verdict speak one language (Story R3.1).
    """
    _check_field_inputs(y_true, y_pred)
    return float(np.linalg.norm(y_pred - y_true) / max(np.linalg.norm(y_true), FIELD_ENERGY_FLOOR))


def _truncation_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Orthogonal-residual fraction of ``y_true`` outside the predictions' subspace.

    The basis is recovered FROM the predictions: a reduced-field prediction is
    ``mean + c·Vᵀ``, so the centered prediction block lies exactly in ``span(V)``
    and an SVD recovers the inhabited subspace — no model access needed. The
    orthogonal residual of ``y_true`` against that affine space is the truncation
    floor; together with ``energy_l2`` it splits Pythagorean:
    ``energy_l2² = truncation_error² + (in-span surrogate error)²``.

    Semantics: this is the floor of the subspace the predictions ACTUALLY inhabit.
    If an inner surrogate predicts a constant for a retained mode, that mode's true
    variation counts as truncation — the model effectively does not use the mode,
    which is the honest answer. A full-rank or scalar model spans the full output
    space, so the metric is ≈0.

    This is a BATCH-level metric: the value depends on the evaluation block's
    empirical mean and SVD, not on a per-sample average — a CV fold's value is the
    floor measured on that fold's block, which is not algebraically a mean of
    per-sample contributions. Inputs are validated like the sklearn-backed entries
    (matching 2-D shapes, non-empty, finite) — clear ``ValueError`` otherwise.
    """
    _check_field_inputs(y_true, y_pred)
    mean = y_pred.mean(axis=0, keepdims=True)
    centered = y_pred - mean
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    if s.size and s[0] > 0.0:
        tol = s[0] * max(centered.shape) * np.finfo(float).eps  # matrix_rank default
        basis = vt[s > tol].T
    else:
        # All-zero predictions: empty basis, the metric is the full relative residual.
        basis = np.zeros((y_pred.shape[1], 0))
    resid = y_true - mean
    ortho = resid - (resid @ basis) @ basis.T
    return float(np.linalg.norm(ortho) / max(np.linalg.norm(y_true), FIELD_ENERGY_FLOOR))


def two_source_split(energy_l2: float, truncation_error: float) -> dict[str, float]:
    """Split the total field error into truncation floor and surrogate error (Story R3.2).

    The truncation residual is orthogonal to the in-span surrogate error (see
    ``_truncation_error``), so Pythagoras gives
    ``surrogate = sqrt(max(energy_l2² − truncation_error², 0))``. The clamp is
    load-bearing, not cosmetic: rounding noise can push the floor infinitesimally
    above the total (the derived gap amplifies 1-ulp noise on ~0.1-magnitude values
    to ~1e-8) — without it the sqrt returns nan. By Parseval (the POD basis is
    orthonormal) the surrogate component equals the coefficient-space error in the
    basis the predictions actually inhabit.

    Aggregation warning (Jensen): across CV folds, split per fold and then average
    the components — never feed aggregated means into this function.

    Not an ``AVAILABLE_METRICS`` entry: it derives from two metric values, it does
    not consume ``(y_true, y_pred)``.
    """
    surrogate = math.sqrt(max(energy_l2**2 - truncation_error**2, 0.0))
    return {
        "total": float(energy_l2),
        "truncation": float(truncation_error),
        "surrogate": surrogate,
    }


AVAILABLE_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": _rmse,
    "mae": _mae,
    "r2": _r2,
    "energy_l2": _energy_l2,
    "truncation_error": _truncation_error,
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
