"""k-fold cross-validation (PRD 4.6).

A fresh, untrained trainer is built for every fold via a factory, so no fitted
state leaks between folds. Per-metric mean and standard deviation are reported.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.model_selection import KFold

from surroforge.validators.metrics import compute_metrics

if TYPE_CHECKING:
    from surroforge.trainers.base import BaseTrainer

TrainerFactory = Callable[[], "BaseTrainer"]


def cross_validate(
    factory: TrainerFactory,
    x: np.ndarray,
    y: np.ndarray,
    *,
    folds: int,
    shuffle: bool,
    seed: int,
    metrics: list[str],
) -> dict[str, Any]:
    """Run k-fold CV and aggregate metrics across folds.

    Returns ``{"folds": k, "<metric>": {"mean": ..., "std": ..., "per_fold": [...]}}``.
    """
    n_samples = x.shape[0]
    if folds > n_samples:
        raise ValueError(
            f"Cannot run {folds}-fold cross-validation on {n_samples} samples."
        )

    kfold = KFold(n_splits=folds, shuffle=shuffle, random_state=seed if shuffle else None)
    per_fold: dict[str, list[float]] = {name: [] for name in metrics}

    for train_idx, test_idx in kfold.split(x):
        trainer = factory()
        trainer.fit(x[train_idx], y[train_idx])
        preds = trainer.predict(x[test_idx])
        fold_metrics = compute_metrics(y[test_idx], preds, metrics)
        for name, value in fold_metrics.items():
            per_fold[name].append(value)

    summary: dict[str, Any] = {"folds": folds}
    for name, values in per_fold.items():
        arr = np.asarray(values, dtype=float)
        summary[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "per_fold": [float(v) for v in arr],
        }
    return summary
