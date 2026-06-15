"""MLPTrainer — neural-network surrogate via scikit-learn MLPRegressor (Story R6.1).

Fixture diversity (project-context R5 retro): beyond the tidy single-output happy
path, this suite exercises multi-output targets, unknown-hyperparameter rejection,
the different-seed divergence *negative* case (the one with teeth), ONNX export +
its missing-backend hint, and the reduced-field POD-NN inner e2e. The mlp core path
needs only Core scikit-learn; the ONNX round-trip importorskips its backend.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings

import numpy as np
import pytest
from sklearn.metrics import r2_score

from surroforge.errors import ExportError, TrainerError
from surroforge.plugins import available_trainers, get_trainer
from surroforge.trainers.mlp import MLPTrainer
from tests.conftest import requires_reproducible_blas


def _smooth_problem(n=240, seed=0):
    """A smooth nonlinear 2-D regression target an MLP can fit well."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, (n, 2))
    y = (np.sin(2.0 * x[:, 0]) + x[:, 1] ** 2 - 0.5 * x[:, 0] * x[:, 1]).reshape(-1, 1)
    return x, y


# --- registration -----------------------------------------------------------

def test_registered_as_builtin():
    assert "mlp" in available_trainers()
    assert get_trainer("mlp") is MLPTrainer


# --- BaseTrainer contract: shapes, multi-output, picklability ----------------

def test_predict_returns_2d_single_output():
    x, y = _smooth_problem()
    t = MLPTrainer()
    t.fit(x[:200], y[:200])
    preds = t.predict(x[200:])
    assert preds.ndim == 2 and preds.shape == (40, 1)


def test_native_multi_output():
    """Unlike kriging/rbf (single-output in the MVP), the MLP fits multi-output natively."""
    x, y1 = _smooth_problem()
    y = np.column_stack([y1[:, 0], x[:, 0] ** 2 - x[:, 1]])  # 2 outputs
    t = MLPTrainer()
    t.fit(x[:200], y[:200])
    preds = t.predict(x[200:])
    assert preds.shape == (40, 2)


def test_single_output_emits_no_dataconversion_warning():
    """AC2: pass a 1-D target for a single output so sklearn does not warn.

    Wrong impl this pins: fitting the column-vector y directly (the obvious choice),
    which raises sklearn's DataConversionWarning on every single-output fit.
    """
    x, y = _smooth_problem(n=80)
    t = MLPTrainer(max_iter=200)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        t.fit(x, y)


def test_picklable_round_trip():
    import pickle

    x, y = _smooth_problem(n=120)
    t = MLPTrainer(max_iter=300)
    t.fit(x, y)
    restored = pickle.loads(pickle.dumps(t))
    np.testing.assert_array_equal(restored.predict(x), t.predict(x))


def test_learns_smooth_nonlinear_function():
    x, y = _smooth_problem()
    t = MLPTrainer()
    t.fit(x[:200], y[:200])
    score = r2_score(y[200:], t.predict(x[200:]))
    assert score > 0.9, f"R²={score:.3f} — MLP failed to fit a smooth nonlinear target"


# --- hyperparameter validation ----------------------------------------------

def test_unknown_hyperparameter_rejected():
    with pytest.raises(TrainerError) as exc:
        MLPTrainer(n_estimators=10)  # a Random-Forest param, not an MLP one
    msg = str(exc.value)
    assert "n_estimators" in msg
    assert "Supported:" in msg and "hidden_layer_sizes" in msg  # names the real options


def test_known_hyperparameters_accepted():
    t = MLPTrainer(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs", alpha=1e-3)
    assert t.hidden_layer_sizes == (8,)
    assert t.activation == "tanh"


def test_bad_max_iter_rejected():
    with pytest.raises(TrainerError, match="max_iter"):
        MLPTrainer(max_iter=0)


def test_invalid_activation_rejected():
    """A typo'd activation fails at construction with a domain error listing the
    valid choices — not a raw sklearn ValueError deep inside fit()."""
    with pytest.raises(TrainerError, match="activation"):
        MLPTrainer(activation="rectified-linear")


def test_invalid_solver_rejected():
    with pytest.raises(TrainerError, match="solver"):
        MLPTrainer(solver="newton")


@pytest.mark.parametrize("bad", [None, 0.5, "0", True])
def test_non_int_random_state_rejected(bad):
    """random_state must be a real int: None/RandomState/float would silently break
    determinism, which reproduction_tolerance and lineage rely on. (bool is excluded
    even though it subclasses int.)"""
    with pytest.raises(TrainerError, match="random_state"):
        MLPTrainer(random_state=bad)


@pytest.mark.parametrize(
    "kwargs, field",
    [({"alpha": -1e-3}, "alpha"), ({"tol": 0.0}, "tol"), ({"learning_rate_init": 0.0}, "learning")],
)
def test_bad_numeric_hyperparameters_rejected(kwargs, field):
    with pytest.raises(TrainerError, match=field):
        MLPTrainer(**kwargs)


def test_predict_before_fit_raises():
    with pytest.raises(TrainerError, match="not fitted"):
        MLPTrainer().predict(np.zeros((2, 2)))


# --- reproducibility: positive + the negative case with teeth ----------------

@requires_reproducible_blas
def test_same_seed_reproduces_within_tolerance():
    """Positive case. Pins the wrong impl that leans on the ambient global RNG
    (random_state=None): perturbing np.random between fits would then change the
    result, so two same-seed fits would differ and this assertion would fail."""
    x, y = _smooth_problem(n=160)
    a = MLPTrainer(random_state=0)
    a.fit(x, y)
    np.random.random(8)  # perturb ambient global RNG between fits
    b = MLPTrainer(random_state=0)
    b.fit(x, y)
    diff = float(np.max(np.abs(a.predict(x) - b.predict(x))))
    assert diff <= MLPTrainer.reproduction_tolerance


def test_different_seed_diverges_beyond_tolerance():
    """Negative case (the one with teeth). Pins the wrong impl that ignores the
    random_state hyperparameter (hardcodes it / never threads it into MLPRegressor):
    that impl would make these two fits identical, so this divergence assertion would
    fail — exposing the bug."""
    x, y = _smooth_problem(n=160)
    a = MLPTrainer(random_state=0)
    b = MLPTrainer(random_state=1)
    a.fit(x, y)
    b.fit(x, y)
    diff = float(np.max(np.abs(a.predict(x) - b.predict(x))))
    assert diff > MLPTrainer.reproduction_tolerance


def test_declares_capabilities():
    assert MLPTrainer.export_formats == frozenset({"pickle", "onnx"})
    tol = MLPTrainer.reproduction_tolerance
    assert isinstance(tol, float) and np.isfinite(tol) and tol > 0.0  # honest, not 0.0
    assert MLPTrainer().supports_export("onnx") and MLPTrainer().supports_export("pickle")


# --- ONNX export -------------------------------------------------------------

_needs_onnx = pytest.mark.skipif(
    importlib.util.find_spec("skl2onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="requires the 'onnx' optional backend (skl2onnx + onnxruntime)",
)


@_needs_onnx
@pytest.mark.parametrize("n_outputs", [1, 2])
def test_onnx_round_trip(n_outputs):
    import onnxruntime as ort

    x, y1 = _smooth_problem(n=160)
    y = y1 if n_outputs == 1 else np.column_stack([y1[:, 0], x[:, 0] * x[:, 1]])
    t = MLPTrainer(max_iter=500)
    t.fit(x, y)

    blob = t.to_onnx(n_features=2)
    assert isinstance(blob, bytes) and blob

    sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x.astype(np.float32)})[0]
    ref = t.predict(x)
    np.testing.assert_allclose(onnx_out.reshape(ref.shape), ref, atol=1e-4)


def test_onnx_export_hint_when_backend_missing(monkeypatch):
    """AC5: a missing onnx stack yields an actionable ExportError, not a raw ImportError.

    Runs regardless of whether skl2onnx is installed: forcing the module to None makes
    `from skl2onnx import ...` raise ImportError. Teeth: without the translation the test
    would see ImportError (or, with the backend present, a successful export), not the
    ExportError naming `pip install 'surroforge[onnx]'`.
    """
    x, y = _smooth_problem(n=60)
    t = MLPTrainer(max_iter=200)
    t.fit(x, y)
    monkeypatch.setitem(sys.modules, "skl2onnx", None)
    with pytest.raises(ExportError, match=r"surroforge\[onnx\]"):
        t.to_onnx(n_features=2)


def test_onnx_export_before_fit_raises():
    with pytest.raises(TrainerError, match="not fitted"):
        MLPTrainer().to_onnx(n_features=2)


def test_onnx_export_rejects_wrong_feature_count():
    """A wrong n_features would yield an ONNX graph that silently mispredicts; the
    guard refuses it before conversion (runs without the onnx backend)."""
    x, y = _smooth_problem(n=60)  # 2 input features
    t = MLPTrainer(max_iter=200)
    t.fit(x, y)
    with pytest.raises(ExportError, match="fitted on 2"):
        t.to_onnx(n_features=5)


# --- works as a reduced-field inner (POD-NN), unchanged ----------------------

def test_reduced_field_inner_mlp_end_to_end():
    """AC6: the spine theorem — mlp serves as a per-mode inner surrogate with no
    ROM-specific change, reconstructing a smooth low-rank field on a holdout."""
    from surroforge.trainers.reduced_field import ReducedFieldTrainer

    rng = np.random.default_rng(0)
    p = rng.uniform(-1.0, 1.0, (160, 2))
    phi, _ = np.linalg.qr(rng.standard_normal((30, 3)))  # 3 orthonormal spatial modes
    amps = np.column_stack([p[:, 0], p[:, 1] ** 2, p[:, 0] * p[:, 1]]) * np.array([5.0, 2.0, 1.0])
    field = amps @ phi.T

    t = ReducedFieldTrainer(
        reduction={"basis": "pod", "energy": 0.999},
        inner={"method": "mlp", "hidden_layer_sizes": (32, 32), "max_iter": 1000},
    )
    t.fit(p[:120], field[:120])
    recon = t.predict(p[120:])
    assert recon.shape == field[120:].shape
    score = r2_score(field[120:], recon)
    assert score > 0.9, f"POD-NN reconstruction R²={score:.3f}"
