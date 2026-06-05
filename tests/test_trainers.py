"""Trainer-level behaviour for the three Core methods (PRD 4.6)."""

from __future__ import annotations

import numpy as np
import pytest

from surroforge.errors import ExportError, TrainerError
from surroforge.trainers.base import BaseTrainer
from surroforge.trainers.kriging import KrigingTrainer
from surroforge.trainers.polynomial import PolynomialTrainer
from surroforge.trainers.rbf import RBFTrainer


def _line(n=40, slope=3.0, intercept=1.5):
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (n, 1))
    y = slope * x + intercept
    return x, y


# --- Polynomial --------------------------------------------------------------

def test_polynomial_recovers_linear_relation():
    x, y = _line()
    model = PolynomialTrainer(degree=1)
    model.fit(x, y)
    preds = model.predict(x)
    assert preds.shape == (len(x), 1)
    assert np.allclose(preds, y, atol=1e-6)


def test_polynomial_accepts_1d_targets_and_returns_2d():
    x, y = _line()
    model = PolynomialTrainer(degree=1)
    model.fit(x, y.ravel())  # 1-D target
    assert model.predict(x).shape == (len(x), 1)


def test_polynomial_unknown_hyperparameter_raises():
    with pytest.raises(TrainerError):
        PolynomialTrainer(degree=2, bogus=1)


def test_polynomial_bad_degree_raises():
    with pytest.raises(TrainerError):
        PolynomialTrainer(degree=0)


def test_predict_before_fit_raises():
    with pytest.raises(TrainerError):
        PolynomialTrainer().predict(np.zeros((2, 1)))


# --- Kriging / RBF (SMT) -----------------------------------------------------

def test_kriging_fits_and_predicts():
    pytest.importorskip("smt")
    x, y = _line()
    model = KrigingTrainer()
    model.fit(x, y)
    assert model.predict(x).shape == (len(x), 1)


def test_rbf_fits_and_predicts():
    pytest.importorskip("smt")
    x, y = _line()
    model = RBFTrainer(d0=1.0)
    model.fit(x, y)
    assert model.predict(x).shape == (len(x), 1)


@pytest.mark.parametrize("cls", [KrigingTrainer, RBFTrainer])
def test_smt_methods_reject_multioutput(cls):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (30, 2))
    y = rng.uniform(0, 1, (30, 2))  # two outputs
    with pytest.raises(TrainerError):
        cls().fit(x, y)


@pytest.mark.parametrize("cls", [KrigingTrainer, RBFTrainer])
def test_smt_methods_reject_unknown_hyperparameter(cls):
    with pytest.raises(TrainerError):
        cls(nonsense=1)


def test_kriging_does_not_support_onnx():
    # Export capability is a static declaration (Story 5.4) — no fit/SMT needed.
    model = KrigingTrainer()
    assert model.supports_export("onnx") is False
    assert model.supports_export("pickle") is True
    with pytest.raises(ExportError):
        model.to_onnx(n_features=1)


def test_polynomial_declares_export_and_determinism():
    model = PolynomialTrainer()
    assert model.supports_export("pickle") is True
    assert model.supports_export("onnx") is True
    assert model.supports_export("nonsense") is False
    # Closed-form OLS is declared deterministic.
    assert model.reproduction_tolerance == 0.0


# --- Base contract -----------------------------------------------------------

def test_all_trainers_are_basetrainer():
    for cls in (PolynomialTrainer, KrigingTrainer, RBFTrainer):
        assert issubclass(cls, BaseTrainer)
