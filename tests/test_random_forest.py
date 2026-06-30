"""RandomForestTrainer — random-forest surrogate via scikit-learn.

Mirrors test_gradient_boosting.py: registration, the BaseTrainer contract (shapes,
native multi-output, picklability), nonlinear-fit behaviour, hyperparameter
validation, and reproducibility. Unlike gradient-boosting, RandomForest's bootstrap
and feature subsampling make random_state genuinely matter, so this suite includes a
different-seed divergence test (the negative case with teeth).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.metrics import r2_score

from surroforge.errors import TrainerError
from surroforge.plugins import available_trainers, get_trainer
from surroforge.trainers.random_forest import RandomForestTrainer


def _nonlinear_problem(n=240, seed=0):
    """A nonlinear 2-D regression target a random forest fits well."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, (n, 2))
    y = (np.sin(2.0 * x[:, 0]) + x[:, 1] ** 2 - 0.5 * x[:, 0] * x[:, 1]).reshape(-1, 1)
    return x, y


# --- registration -----------------------------------------------------------

def test_registered_as_builtin():
    assert "random-forest" in available_trainers()
    assert get_trainer("random-forest") is RandomForestTrainer


# --- BaseTrainer contract ----------------------------------------------------

def test_predict_returns_2d_single_output():
    x, y = _nonlinear_problem()
    t = RandomForestTrainer()
    t.fit(x[:200], y[:200])
    preds = t.predict(x[200:])
    assert preds.ndim == 2 and preds.shape == (40, 1)


def test_native_multi_output():
    """RandomForestRegressor is natively multi-output (no MultiOutputRegressor wrapper)."""
    x, y1 = _nonlinear_problem()
    y = np.column_stack([y1[:, 0], x[:, 0] ** 2 - x[:, 1]])  # 2 outputs
    t = RandomForestTrainer()
    t.fit(x[:200], y[:200])
    preds = t.predict(x[200:])
    assert preds.shape == (40, 2)


def test_single_output_emits_no_dataconversion_warning():
    """fit() must extract a 1-D target for a single output so sklearn does not warn."""
    x, y = _nonlinear_problem(n=80)
    t = RandomForestTrainer(n_estimators=30)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        t.fit(x, y)


def test_picklable_round_trip():
    import pickle

    x, y = _nonlinear_problem(n=120)
    t = RandomForestTrainer(n_estimators=30)
    t.fit(x, y)
    restored = pickle.loads(pickle.dumps(t))
    np.testing.assert_array_equal(restored.predict(x), t.predict(x))


def test_learns_nonlinear_function():
    x, y = _nonlinear_problem()
    t = RandomForestTrainer()
    t.fit(x[:200], y[:200])
    score = r2_score(y[200:], t.predict(x[200:]))
    assert score > 0.9, f"R²={score:.3f} — random forest failed to fit a nonlinear target"


def test_predict_before_fit_raises():
    with pytest.raises(TrainerError, match="not fitted"):
        RandomForestTrainer().predict(np.zeros((2, 2)))


# --- hyperparameter validation ----------------------------------------------

def test_unknown_hyperparameter_rejected():
    with pytest.raises(TrainerError) as exc:
        RandomForestTrainer(hidden_layer_sizes=(8,))  # an mlp param, not a forest one
    msg = str(exc.value)
    assert "hidden_layer_sizes" in msg
    assert "Supported:" in msg and "n_estimators" in msg


def test_known_hyperparameters_accepted():
    t = RandomForestTrainer(n_estimators=50, max_depth=4, max_leaf_nodes=None, max_features="sqrt")
    assert t.n_estimators == 50
    assert t.max_depth == 4
    assert t.max_leaf_nodes is None
    assert t.max_features == "sqrt"


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"n_estimators": 0}, "n_estimators"),
        ({"max_depth": 0}, "max_depth"),
        ({"max_leaf_nodes": 1}, "max_leaf_nodes"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
    ],
)
def test_bad_numeric_hyperparameters_rejected(kwargs, field):
    with pytest.raises(TrainerError, match=field):
        RandomForestTrainer(**kwargs)


@pytest.mark.parametrize("good", ["sqrt", "log2", None, 1, 3, 0.5, 1.0])
def test_max_features_accepts_valid(good):
    RandomForestTrainer(max_features=good)  # must not raise


@pytest.mark.parametrize("bad", ["all", 0, -1, 1.5, 0.0, True])
def test_max_features_rejects_invalid(bad):
    with pytest.raises(TrainerError, match="max_features"):
        RandomForestTrainer(max_features=bad)


@pytest.mark.parametrize("bad", [None, 0.5, "0", True])
def test_non_int_random_state_rejected(bad):
    with pytest.raises(TrainerError, match="random_state"):
        RandomForestTrainer(random_state=bad)


# --- reproducibility + capabilities -----------------------------------------

def test_same_seed_reproduces_within_tolerance():
    """Two independent fits with the same seed reproduce within the declared band."""
    x, y = _nonlinear_problem(n=160)
    a = RandomForestTrainer(random_state=0)
    a.fit(x, y)
    b = RandomForestTrainer(random_state=0)
    b.fit(x, y)
    diff = float(np.max(np.abs(a.predict(x) - b.predict(x))))
    assert diff <= RandomForestTrainer.reproduction_tolerance


def test_different_seed_diverges_beyond_tolerance():
    """Negative case with teeth. Pins a wrong impl that ignores random_state (hardcodes
    it / never threads it into RandomForestRegressor): that impl would make these two
    fits identical, so this divergence assertion would fail and expose the bug."""
    x, y = _nonlinear_problem(n=160)
    a = RandomForestTrainer(random_state=0)
    b = RandomForestTrainer(random_state=1)
    a.fit(x, y)
    b.fit(x, y)
    diff = float(np.max(np.abs(a.predict(x) - b.predict(x))))
    assert diff > RandomForestTrainer.reproduction_tolerance


def test_declares_capabilities():
    assert RandomForestTrainer.export_formats == frozenset({"pickle", "onnx"})
    tol = RandomForestTrainer.reproduction_tolerance
    assert isinstance(tol, float) and np.isfinite(tol) and tol > 0.0
    t = RandomForestTrainer()
    assert t.supports_export("onnx") and t.supports_export("pickle")


# --- ONNX export -------------------------------------------------------------

import importlib.util  # noqa: E402
import sys  # noqa: E402

from surroforge.errors import ExportError  # noqa: E402

_needs_onnx = pytest.mark.skipif(
    importlib.util.find_spec("skl2onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="requires the 'onnx' optional backend (skl2onnx + onnxruntime)",
)


@_needs_onnx
@pytest.mark.parametrize("n_outputs", [1, 2])
def test_onnx_round_trip(n_outputs):
    import onnxruntime as ort

    x, y1 = _nonlinear_problem(n=160)
    y = y1 if n_outputs == 1 else np.column_stack([y1[:, 0], x[:, 0] * x[:, 1]])
    t = RandomForestTrainer(n_estimators=40)
    t.fit(x, y)

    blob = t.to_onnx(n_features=2)
    assert isinstance(blob, bytes) and blob

    sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x.astype(np.float32)})[0]
    ref = t.predict(x)
    np.testing.assert_allclose(onnx_out.reshape(ref.shape), ref, atol=1e-3)


def test_onnx_export_hint_when_backend_missing(monkeypatch):
    x, y = _nonlinear_problem(n=60)
    t = RandomForestTrainer(n_estimators=20)
    t.fit(x, y)
    monkeypatch.setitem(sys.modules, "skl2onnx", None)
    with pytest.raises(ExportError, match=r"surroforge\[onnx\]"):
        t.to_onnx(n_features=2)


def test_onnx_export_before_fit_raises():
    with pytest.raises(TrainerError, match="not fitted"):
        RandomForestTrainer().to_onnx(n_features=2)


def test_onnx_export_rejects_wrong_feature_count():
    x, y = _nonlinear_problem(n=60)  # 2 input features
    t = RandomForestTrainer(n_estimators=20)
    t.fit(x, y)
    with pytest.raises(ExportError, match="fitted on 2"):
        t.to_onnx(n_features=5)
