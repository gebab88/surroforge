"""GradientBoostingTrainer — HistGradientBoosting surrogate via scikit-learn.

Mirrors test_mlp.py's structure: registration, the BaseTrainer contract (shapes,
multi-output, picklability), nonlinear-fit behaviour, hyperparameter validation,
and same-seed reproducibility. There is deliberately NO different-seed divergence
test: HistGradientBoosting bins on the full data for n < 200_000 and has no row
subsampling, so random_state does not change predictions in the DOE regime.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.metrics import r2_score

from surroforge.errors import TrainerError
from surroforge.plugins import available_trainers, get_trainer
from surroforge.trainers.gradient_boosting import GradientBoostingTrainer


def _nonlinear_problem(n=240, seed=0):
    """A nonlinear 2-D regression target a boosted ensemble fits well."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, (n, 2))
    y = (np.sin(2.0 * x[:, 0]) + x[:, 1] ** 2 - 0.5 * x[:, 0] * x[:, 1]).reshape(-1, 1)
    return x, y


# --- registration -----------------------------------------------------------

def test_registered_as_builtin():
    assert "gradient-boosting" in available_trainers()
    assert get_trainer("gradient-boosting") is GradientBoostingTrainer


# --- BaseTrainer contract ----------------------------------------------------

def test_predict_returns_2d_single_output():
    x, y = _nonlinear_problem()
    t = GradientBoostingTrainer()
    t.fit(x[:200], y[:200])
    preds = t.predict(x[200:])
    assert preds.ndim == 2 and preds.shape == (40, 1)


def test_multi_output_via_wrapper():
    x, y1 = _nonlinear_problem()
    y = np.column_stack([y1[:, 0], x[:, 0] ** 2 - x[:, 1]])  # 2 outputs
    t = GradientBoostingTrainer()
    t.fit(x[:200], y[:200])
    preds = t.predict(x[200:])
    assert preds.shape == (40, 2)


def test_single_output_emits_no_dataconversion_warning():
    """Pass a 1-D target for a single output so sklearn does not warn."""
    x, y = _nonlinear_problem(n=80)
    t = GradientBoostingTrainer(max_iter=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        t.fit(x, y)


def test_picklable_round_trip():
    import pickle

    x, y = _nonlinear_problem(n=120)
    t = GradientBoostingTrainer(max_iter=50)
    t.fit(x, y)
    restored = pickle.loads(pickle.dumps(t))
    np.testing.assert_array_equal(restored.predict(x), t.predict(x))


def test_learns_nonlinear_function():
    x, y = _nonlinear_problem()
    t = GradientBoostingTrainer()
    t.fit(x[:200], y[:200])
    score = r2_score(y[200:], t.predict(x[200:]))
    assert score > 0.9, f"R²={score:.3f} — gradient boosting failed to fit a nonlinear target"


def test_predict_before_fit_raises():
    with pytest.raises(TrainerError, match="not fitted"):
        GradientBoostingTrainer().predict(np.zeros((2, 2)))


# --- hyperparameter validation ----------------------------------------------

def test_unknown_hyperparameter_rejected():
    with pytest.raises(TrainerError) as exc:
        GradientBoostingTrainer(hidden_layer_sizes=(8,))  # an mlp param, not a boosting one
    msg = str(exc.value)
    assert "hidden_layer_sizes" in msg
    assert "Supported:" in msg and "learning_rate" in msg


def test_known_hyperparameters_accepted():
    t = GradientBoostingTrainer(learning_rate=0.05, max_iter=200, max_depth=3, max_leaf_nodes=None)
    assert t.learning_rate == 0.05
    assert t.max_iter == 200
    assert t.max_depth == 3
    assert t.max_leaf_nodes is None


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_leaf_nodes": 1}, "max_leaf_nodes"),
        ({"max_depth": 0}, "max_depth"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"l2_regularization": -1.0}, "l2_regularization"),
    ],
)
def test_bad_numeric_hyperparameters_rejected(kwargs, field):
    with pytest.raises(TrainerError, match=field):
        GradientBoostingTrainer(**kwargs)


@pytest.mark.parametrize("bad", [None, 0.5, "0", True])
def test_non_int_random_state_rejected(bad):
    with pytest.raises(TrainerError, match="random_state"):
        GradientBoostingTrainer(random_state=bad)


# --- reproducibility + capabilities -----------------------------------------

def test_same_seed_reproduces_within_tolerance():
    """Two independent fits on the same data reproduce within the declared band.
    (HistGradientBoosting is deterministic for the DOE regime, so this is tight.)"""
    x, y = _nonlinear_problem(n=160)
    a = GradientBoostingTrainer(random_state=0)
    a.fit(x, y)
    b = GradientBoostingTrainer(random_state=0)
    b.fit(x, y)
    diff = float(np.max(np.abs(a.predict(x) - b.predict(x))))
    assert diff <= GradientBoostingTrainer.reproduction_tolerance


def test_declares_capabilities():
    assert GradientBoostingTrainer.export_formats == frozenset({"pickle", "onnx"})
    tol = GradientBoostingTrainer.reproduction_tolerance
    assert isinstance(tol, float) and np.isfinite(tol) and tol > 0.0
    t = GradientBoostingTrainer()
    assert t.supports_export("onnx") and t.supports_export("pickle")
