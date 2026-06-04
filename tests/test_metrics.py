"""Regression metrics (PRD 4.6)."""

from __future__ import annotations

import numpy as np
import pytest

from surroforge.errors import ConfigError
from surroforge.validators.metrics import compute_metrics, validate_metric_names


def test_perfect_prediction():
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    m = compute_metrics(y, y.copy(), ["rmse", "r2", "mae"])
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_known_values():
    y_true = np.array([[0.0], [0.0], [0.0]])
    y_pred = np.array([[1.0], [-1.0], [2.0]])
    m = compute_metrics(y_true, y_pred, ["rmse", "mae"])
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 1 + 4) / 3))
    assert m["mae"] == pytest.approx((1 + 1 + 2) / 3)


def test_multioutput_averages():
    y_true = np.array([[1.0, 10.0], [2.0, 20.0]])
    m = compute_metrics(y_true, y_true.copy(), ["rmse"])
    assert m["rmse"] == pytest.approx(0.0)


def test_unknown_metric_rejected():
    with pytest.raises(ConfigError):
        validate_metric_names(["rmse", "bogus"])
