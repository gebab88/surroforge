"""k-fold cross-validation module (PRD 4.6)."""

from __future__ import annotations

import numpy as np
import pytest

from surroforge.trainers.polynomial import PolynomialTrainer
from surroforge.validators.cross_validation import cross_validate


def _data(n=50, n_features=2):
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (n, n_features))
    y = (x**2).sum(axis=1, keepdims=True) + 0.5
    return x, y


def test_cross_validate_shape_of_summary():
    x, y = _data()
    summary = cross_validate(
        lambda: PolynomialTrainer(degree=2),
        x, y, folds=5, shuffle=True, seed=42, metrics=["rmse", "r2"],
    )
    assert summary["folds"] == 5
    for metric in ("rmse", "r2"):
        assert set(summary[metric]) == {"mean", "std", "per_fold"}
        assert len(summary[metric]["per_fold"]) == 5
    assert summary["rmse"]["mean"] >= 0.0


def test_cross_validate_uses_a_fresh_trainer_per_fold():
    x, y = _data()
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return PolynomialTrainer(degree=2)

    cross_validate(factory, x, y, folds=4, shuffle=True, seed=0, metrics=["rmse"])
    assert calls["n"] == 4


def test_cross_validate_is_deterministic():
    x, y = _data()

    def run():
        return cross_validate(
            lambda: PolynomialTrainer(degree=2),
            x, y, folds=5, shuffle=True, seed=7, metrics=["rmse"],
        )

    assert run()["rmse"]["per_fold"] == run()["rmse"]["per_fold"]


def test_cross_validate_rejects_too_many_folds():
    x, y = _data(n=4)
    with pytest.raises(ValueError):
        cross_validate(
            lambda: PolynomialTrainer(degree=1),
            x, y, folds=10, shuffle=True, seed=0, metrics=["rmse"],
        )
