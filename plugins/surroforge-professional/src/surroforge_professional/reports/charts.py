"""Chart-data assembly for the Professional interactive HTML report.

Pure and render-free: ``build_charts(run)`` turns a persisted :class:`Run` into the
numeric arrays the HTML template feeds to uPlot. It never raises on missing, old, or
malformed run data — a chart whose inputs are absent is simply omitted, and if nothing
is plottable the whole section is dropped (``build_charts`` returns ``None``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from surroforge.core.run import Run


def _as_2d(values: Any) -> np.ndarray | None:
    """Coerce a stored list (1-D or 2-D) into a finite 2-D float array, else ``None``."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def _output_labels(run: Run, n_cols: int) -> list[str]:
    outputs = [str(o) for o in run.config.data.outputs]
    if len(outputs) == n_cols:
        return outputs
    return [f"output {i + 1}" for i in range(n_cols)]


def _parity(run: Run) -> dict[str, Any] | None:
    # Field/ROM runs store a reconstructed field as the fingerprint and keep no field
    # actuals, so predicted-vs-actual is meaningless there — skip it.
    if run.reduction:
        return None
    actual = _as_2d(run.holdout_actual)
    predicted = _as_2d(run.holdout_fingerprint)
    if actual is None or predicted is None or actual.shape != predicted.shape:
        return None
    # Output-major (one inner list per output column) so the selector picks a column.
    return {"actual": actual.T.tolist(), "predicted": predicted.T.tolist()}


def _cv(run: Run) -> dict[str, Any] | None:
    metrics = run.metrics
    if not isinstance(metrics, dict):
        return None
    cv = metrics.get("cross_validation")
    if not isinstance(cv, dict):
        return None
    per_fold: dict[str, list[float]] = {}
    mean: dict[str, float] = {}
    n_folds = 0
    kept: list[str] = []
    for name in run.config.validation.metrics:
        rec = cv.get(name)
        if not isinstance(rec, dict):
            continue
        arr = _as_2d(rec.get("per_fold"))
        if arr is None or arr.shape[1] != 1:
            continue
        flat = arr.ravel().tolist()
        per_fold[name] = flat
        kept.append(name)
        n_folds = max(n_folds, len(flat))
        try:
            candidate = float(rec["mean"])
        except (TypeError, ValueError, KeyError):
            candidate = None
        if candidate is None or not math.isfinite(candidate):
            candidate = float(np.mean(flat))
        mean[name] = candidate
    if not kept:
        return None
    return {
        "metrics": kept,
        "folds": list(range(1, n_folds + 1)),
        "per_fold": per_fold,
        "mean": mean,
    }


def _metric_comparison(run: Run) -> dict[str, Any] | None:
    metrics = run.metrics
    if not isinstance(metrics, dict):
        return None
    train = metrics.get("train")
    test = metrics.get("test")
    cv = metrics.get("cross_validation")
    # Spelled as a conjunction (not all(...)) so type checkers narrow all three to dict.
    if not (isinstance(train, dict) and isinstance(test, dict) and isinstance(cv, dict)):
        return None
    kept: list[str] = []
    tr: list[float] = []
    te: list[float] = []
    cm: list[float] = []
    for name in run.config.validation.metrics:
        cvrec = cv.get(name)
        if name not in train or name not in test or not isinstance(cvrec, dict):
            continue
        try:
            tval = float(train[name])
            eval_ = float(test[name])
            mval = float(cvrec["mean"])
        except (TypeError, ValueError, KeyError):
            continue
        if not all(math.isfinite(v) for v in (tval, eval_, mval)):
            continue
        kept.append(name)
        tr.append(tval)
        te.append(eval_)
        cm.append(mval)
    if not kept:
        return None
    return {"metrics": kept, "train": tr, "test": te, "cv_mean": cm}


def build_charts(run: Run) -> dict[str, Any] | None:
    parity = _parity(run)
    cv = _cv(run)
    comparison = _metric_comparison(run)
    if parity is None and cv is None and comparison is None:
        return None
    charts: dict[str, Any] = {}
    if parity is not None:
        charts["parity"] = parity
        charts["outputs"] = _output_labels(run, len(parity["actual"]))
    if cv is not None:
        charts["cv"] = cv
    if comparison is not None:
        charts["metric_comparison"] = comparison
    return charts
