"""TorchMLPTrainer — PyTorch torch-mlp surrogate plugin (Story R7.1).

Fixture diversity (project-context R5 retro): beyond the tidy single-output happy path,
this suite exercises multi-output, unknown/invalid-hyperparameter rejection, the
different-seed divergence *negative* case (the one with teeth), the shipped check_trainer
conformance suite, ONNX export + its guards, and the reduced-field POD-NN inner e2e. The
core path needs torch; the ONNX round-trip importorskips onnxruntime.
"""

from __future__ import annotations

import importlib.util
import io
import pickle

import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score

from surroforge.conformance import check_trainer
from surroforge.errors import ExportError, TrainerError
from surroforge.plugins import available_trainers, get_trainer
from surroforge_torch.trainer import TorchMLPTrainer


def _smooth_problem(n: int = 240, seed: int = 0):
    """A smooth nonlinear 2-D regression target an MLP can fit well."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, (n, 2))
    y = (np.sin(2.0 * x[:, 0]) + x[:, 1] ** 2 - 0.5 * x[:, 0] * x[:, 1]).reshape(-1, 1)
    return x, y


# --- registration (entry-point seam; needs the package pip-installed) --------

def test_registered_as_plugin():
    assert "torch-mlp" in available_trainers()
    assert get_trainer("torch-mlp") is TorchMLPTrainer


# --- BaseTrainer contract: shapes, multi-output, picklability ----------------

def test_predict_returns_2d_single_output():
    x, y = _smooth_problem()
    t = TorchMLPTrainer(max_iter=300)
    t.fit(x[:200], y[:200])
    preds = t.predict(x[200:])
    assert preds.ndim == 2 and preds.shape == (40, 1)


def test_native_multi_output():
    """One network, n_outputs columns — supported natively like the sklearn mlp."""
    x, y1 = _smooth_problem()
    y = np.column_stack([y1[:, 0], x[:, 0] ** 2 - x[:, 1]])  # 2 outputs
    t = TorchMLPTrainer(max_iter=300)
    t.fit(x[:200], y[:200])
    preds = t.predict(x[200:])
    assert preds.shape == (40, 2)


def test_picklable_round_trip_bit_identical():
    x, y = _smooth_problem(n=120)
    t = TorchMLPTrainer(max_iter=200)
    t.fit(x, y)
    restored = pickle.loads(pickle.dumps(t))
    np.testing.assert_array_equal(restored.predict(x), t.predict(x))


def test_learns_smooth_nonlinear_function():
    x, y = _smooth_problem()
    t = TorchMLPTrainer(max_iter=500)
    t.fit(x[:200], y[:200])
    score = r2_score(y[200:], t.predict(x[200:]))
    assert score > 0.9, f"R²={score:.3f} — torch-mlp failed to fit a smooth nonlinear target"


# --- hyperparameter validation ----------------------------------------------

def test_unknown_hyperparameter_rejected():
    with pytest.raises(TrainerError) as exc:
        TorchMLPTrainer(n_estimators=10)  # a Random-Forest param, not an MLP one
    msg = str(exc.value)
    assert "n_estimators" in msg
    assert "Supported:" in msg and "hidden_layer_sizes" in msg


def test_known_hyperparameters_accepted():
    t = TorchMLPTrainer(hidden_layer_sizes=(8,), activation="tanh", solver="adam", alpha=1e-3)
    assert t.hidden_layer_sizes == (8,)
    assert t.activation == "tanh"
    assert t.solver == "adam"


def test_bad_max_iter_rejected():
    with pytest.raises(TrainerError, match="max_iter"):
        TorchMLPTrainer(max_iter=0)


def test_invalid_activation_rejected():
    with pytest.raises(TrainerError, match="activation"):
        TorchMLPTrainer(activation="rectified-linear")


def test_invalid_solver_rejected():
    with pytest.raises(TrainerError, match="solver"):
        TorchMLPTrainer(solver="newton")


@pytest.mark.parametrize("bad", [None, 0.5, "0", True])
def test_non_int_random_state_rejected(bad):
    """random_state must be a real int: None/float would silently break determinism,
    which reproduction_tolerance and lineage rely on. (bool is excluded.)"""
    with pytest.raises(TrainerError, match="random_state"):
        TorchMLPTrainer(random_state=bad)


@pytest.mark.parametrize(
    "kwargs, field",
    [({"alpha": -1e-3}, "alpha"), ({"tol": 0.0}, "tol"), ({"learning_rate_init": 0.0}, "learning")],
)
def test_bad_numeric_hyperparameters_rejected(kwargs, field):
    with pytest.raises(TrainerError, match=field):
        TorchMLPTrainer(**kwargs)


def test_predict_before_fit_raises():
    with pytest.raises(TrainerError, match="not fitted"):
        TorchMLPTrainer().predict(np.zeros((2, 2)))


def test_predict_rejects_wrong_feature_count():
    """A wrong input width raises a domain TrainerError, not a bare torch RuntimeError."""
    x, y = _smooth_problem(n=60)  # 2 input features
    t = TorchMLPTrainer(max_iter=100)
    t.fit(x, y)
    with pytest.raises(TrainerError, match="fitted on 2"):
        t.predict(np.zeros((4, 3)))


# --- reproducibility: positive + the negative case with teeth ----------------

def test_same_seed_reproduces_within_tolerance():
    """The case with TEETH. torch's float32 band (1e-4) absorbs any platform ULP drift, so
    — unlike mlp's tight 1e-6 — this needs no BLAS guard, and a same-seed refit is
    bit-identical in practice. Perturbing both ambient RNGs between fits pins any wrong impl
    that ignores random_state or leaks the global RNG into weight init: such an impl would
    NOT reproduce bit-identically and would fail here."""
    x, y = _smooth_problem(n=160)
    a = TorchMLPTrainer(random_state=0, max_iter=300)
    a.fit(x, y)
    np.random.random(8)
    torch.rand(8)  # perturb both ambient RNGs between fits
    b = TorchMLPTrainer(random_state=0, max_iter=300)
    b.fit(x, y)
    diff = float(np.max(np.abs(a.predict(x) - b.predict(x))))
    assert diff <= TorchMLPTrainer.reproduction_tolerance


def test_different_seed_diverges_beyond_tolerance():
    """Guards against a vacuous (too-loose) tolerance and a degenerate constant fit:
    genuinely different seeds must produce genuinely different surfaces, diverging well
    beyond the band (~1e-2 here, ~100x the 1e-4 tolerance). (A seed-ignoring impl is caught
    by the same-seed test above, not here — an unseeded net still diverges across two fits.)"""
    x, y = _smooth_problem(n=160)
    a = TorchMLPTrainer(random_state=0, max_iter=300)
    b = TorchMLPTrainer(random_state=1, max_iter=300)
    a.fit(x, y)
    b.fit(x, y)
    diff = float(np.max(np.abs(a.predict(x) - b.predict(x))))
    assert diff > TorchMLPTrainer.reproduction_tolerance


def test_declares_capabilities():
    assert TorchMLPTrainer.export_formats == frozenset({"pickle", "onnx", "pt"})
    tol = TorchMLPTrainer.reproduction_tolerance
    assert isinstance(tol, float) and np.isfinite(tol) and tol > 0.0  # honest, not 0.0
    t = TorchMLPTrainer()
    assert t.supports_export("onnx") and t.supports_export("pickle") and t.supports_export("pt")


# --- the shipped conformance suite (all 12 checks) ---------------------------

def test_passes_conformance_suite():
    report = check_trainer(TorchMLPTrainer, hyperparameters={"max_iter": 300})
    assert report.passed, report.summary()
    report.raise_for_status()  # must not raise


# --- ONNX export -------------------------------------------------------------

_needs_ort = pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="requires onnxruntime (the surroforge-torch[onnx] extra)",
)


@_needs_ort
@pytest.mark.parametrize("n_outputs", [1, 2])
def test_onnx_round_trip(n_outputs):
    import onnxruntime as ort

    x, y1 = _smooth_problem(n=160)
    y = y1 if n_outputs == 1 else np.column_stack([y1[:, 0], x[:, 0] * x[:, 1]])
    t = TorchMLPTrainer(max_iter=300)
    t.fit(x, y)

    blob = t.to_onnx(n_features=2)
    assert isinstance(blob, bytes) and blob

    sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x.astype(np.float32)})[0]
    ref = t.predict(x)
    np.testing.assert_allclose(onnx_out.reshape(ref.shape), ref, atol=1e-4)


def test_onnx_export_before_fit_raises():
    with pytest.raises(TrainerError, match="not fitted"):
        TorchMLPTrainer().to_onnx(n_features=2)


def test_onnx_export_rejects_wrong_feature_count():
    """A wrong n_features would yield a graph that silently mispredicts; the guard
    refuses it before export."""
    x, y = _smooth_problem(n=60)  # 2 input features
    t = TorchMLPTrainer(max_iter=100)
    t.fit(x, y)
    with pytest.raises(ExportError, match="fitted on 2"):
        t.to_onnx(n_features=5)


def test_onnx_export_hint_when_onnx_missing(monkeypatch):
    """Without the `onnx` package (torch's exporter serialises through it), export raises an
    actionable ExportError naming the [onnx] extra — exercised via the real find_spec guard,
    not by faking torch.onnx.export. Teeth: an impl missing the guard would instead surface
    torch's OnnxExporterError without the install hint."""
    x, y = _smooth_problem(n=60)
    t = TorchMLPTrainer(max_iter=100)
    t.fit(x, y)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "onnx" else real_find_spec(name, *a, **k),
    )
    with pytest.raises(ExportError, match=r"surroforge-torch\[onnx\]"):
        t.to_onnx(n_features=2)


# --- native PyTorch .pt export -----------------------------------------------

def _rebuild_from_pt(checkpoint) -> "torch.nn.Module":
    """Reconstruct the fitted module from a to_pt() checkpoint, the way a consumer would."""
    from surroforge_torch.trainer import _ACTIVATIONS, _StandardizedMLP

    layers, prev = [], checkpoint["n_features"]
    for width in checkpoint["hidden_layer_sizes"]:
        layers += [torch.nn.Linear(prev, width), _ACTIVATIONS[checkpoint["activation"]]()]
        prev = width
    layers.append(torch.nn.Linear(prev, checkpoint["n_outputs"]))
    net = torch.nn.Sequential(*layers)
    xz, yz = torch.zeros(checkpoint["n_features"]), torch.zeros(checkpoint["n_outputs"])
    # Right-shaped buffer placeholders; load_state_dict fills in the saved values.
    model = _StandardizedMLP(net, xz.clone(), xz.clone(), yz.clone(), yz.clone())
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@pytest.mark.parametrize("n_outputs", [1, 2])
def test_pt_round_trip_reproduces_predict(n_outputs):
    """The .pt checkpoint loads under the safe weights_only path and, once the architecture
    is rebuilt from it, reproduces predict() bit-for-bit (standardisation buffers included)."""
    x, y1 = _smooth_problem(n=160)
    y = y1 if n_outputs == 1 else np.column_stack([y1[:, 0], x[:, 0] * x[:, 1]])
    t = TorchMLPTrainer(max_iter=300)
    t.fit(x, y)

    blob = t.to_pt(n_features=2)
    assert isinstance(blob, bytes) and blob

    checkpoint = torch.load(io.BytesIO(blob), weights_only=True)  # safe load, no code exec
    assert checkpoint["surroforge_torch_pt_version"] == 1
    model = _rebuild_from_pt(checkpoint)
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    with torch.no_grad():
        out = model(xt).numpy()
    np.testing.assert_allclose(out.reshape(t.predict(x).shape), t.predict(x), atol=1e-6)


def test_pt_export_before_fit_raises():
    with pytest.raises(TrainerError, match="not fitted"):
        TorchMLPTrainer().to_pt(n_features=2)


def test_pt_export_rejects_wrong_feature_count():
    x, y = _smooth_problem(n=60)  # 2 input features
    t = TorchMLPTrainer(max_iter=100)
    t.fit(x, y)
    with pytest.raises(ExportError, match="fitted on 2"):
        t.to_pt(n_features=5)


def test_pt_exporter_registered_on_seam():
    """The plugin registers the `pt` exporter on the Core seam, so Core resolves it without
    any built-in entry — and writes a real .pt file for a fitted torch-mlp."""
    from surroforge.plugins import available_exporters, get_exporter
    from surroforge_torch.pt_exporter import PtExporter

    assert "pt" in available_exporters()
    assert get_exporter("pt") is PtExporter


def test_pt_exporter_writes_file(tmp_path):
    x, y = _smooth_problem(n=80)
    t = TorchMLPTrainer(max_iter=100)
    t.fit(x, y)
    from surroforge_torch.pt_exporter import PtExporter

    dest = PtExporter().export(t, tmp_path / "model.pt", n_features=2)
    assert dest.exists() and dest.suffix == ".pt"
    checkpoint = torch.load(io.BytesIO(dest.read_bytes()), weights_only=True)
    assert set(checkpoint) >= {"state_dict", "hidden_layer_sizes", "activation", "n_features"}


# --- works as a reduced-field inner (POD-NN), unchanged ----------------------

def test_reduced_field_inner_torch_mlp_end_to_end():
    """The spine theorem from the outside: torch-mlp serves as a per-mode inner with no
    ROM-specific change, reconstructing a smooth low-rank field on a holdout."""
    from surroforge.trainers.reduced_field import ReducedFieldTrainer

    rng = np.random.default_rng(0)
    p = rng.uniform(-1.0, 1.0, (160, 2))
    phi, _ = np.linalg.qr(rng.standard_normal((30, 3)))  # 3 orthonormal spatial modes
    amps = np.column_stack([p[:, 0], p[:, 1] ** 2, p[:, 0] * p[:, 1]]) * np.array([5.0, 2.0, 1.0])
    field = amps @ phi.T

    t = ReducedFieldTrainer(
        reduction={"basis": "pod", "energy": 0.999},
        inner={"method": "torch-mlp", "hidden_layer_sizes": (32, 32), "max_iter": 500},
    )
    t.fit(p[:120], field[:120])
    recon = t.predict(p[120:])
    assert recon.shape == field[120:].shape
    score = r2_score(field[120:], recon)
    assert score > 0.9, f"POD-NN reconstruction R²={score:.3f}"
