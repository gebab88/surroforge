"""Model export: pickle round-trip and ONNX (incl. runtime parity) (PRD 4.4).

Core tests until Wave 2; the exporters are Professional now, so the suite lives
here. Direct class instantiation bypasses the licence seam by design (the gate
sits in registry resolution); everything going through ``export_model`` /
``Run.export`` runs with the ``licensed`` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from surroforge.core.config import parse_config
from surroforge.engine import run_training
from surroforge.errors import ExportError
from surroforge.persistence import load_model
from surroforge.services import export_model
from surroforge.trainers.kriging import KrigingTrainer
from surroforge.trainers.polynomial import PolynomialTrainer
from surroforge_professional.exporters.onnx_exporter import ONNXExporter
from surroforge_professional.exporters.pickle_exporter import PickleExporter


def _fitted_polynomial(n_features=3):
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (50, n_features))
    y = x @ np.arange(1, n_features + 1).reshape(-1, 1) + 0.5
    model = PolynomialTrainer(degree=2)
    model.fit(x, y)
    return model, x


def _kriging_run(project):
    pytest.importorskip("smt")
    config = parse_config(
        {
            "experiment": "beam-kriging",
            "data": {
                "source": "beam",
                "inputs": ["length", "width", "load"],
                "outputs": ["max_stress"],
                "split": {"test_fraction": 0.2, "seed": 42},
            },
            "model": {"method": "kriging"},
            "validation": {"cross_validation": 5, "metrics": ["rmse", "r2", "mae"]},
        }
    )
    return run_training(project, config)


def test_pickle_roundtrip_preserves_predictions(tmp_path):
    model, x = _fitted_polynomial()
    path = PickleExporter().export(model, tmp_path / "m.pkl", n_features=3)
    assert path.is_file()
    restored = load_model(path)
    assert np.allclose(restored.predict(x), model.predict(x))


def test_onnx_export_and_runtime_parity(tmp_path):
    pytest.importorskip("onnxruntime")
    model, x = _fitted_polynomial()
    path = ONNXExporter().export(model, tmp_path / "m.onnx", n_features=3)
    assert path.is_file()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path))
    onnx_out = sess.run(None, {"input": x.astype(np.float32)})[0]
    # ONNX runs in float32, so compare with a tolerance.
    assert np.allclose(onnx_out.reshape(-1), model.predict(x).reshape(-1), atol=1e-3)


def test_onnx_export_rejects_unsupported_method(tmp_path):
    pytest.importorskip("smt")
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (30, 1))
    model = KrigingTrainer()
    model.fit(x, 2 * x)
    with pytest.raises(ExportError):
        ONNXExporter().export(model, tmp_path / "m.onnx", n_features=1)


def test_export_model_service_default_path(licensed, trained_run):
    # No explicit output -> artifact lands inside the run directory.
    path = export_model(trained_run, "pickle")
    assert path == trained_run.run_dir / "model.pkl"
    assert path.is_file()


def test_run_export_method_binds_to_service(licensed, trained_run):
    # Run.export is the object-level binding of export_model: same default path/result.
    path = trained_run.export("pickle")
    assert path == trained_run.run_dir / "model.pkl"
    assert path.is_file()
    restored = load_model(path)
    assert restored is not None


def test_run_export_propagates_gate_error(licensed, project):
    # The binding carries the declaration-driven gate: kriging cannot export to ONNX.
    run = _kriging_run(project)
    with pytest.raises(ExportError):
        run.export("onnx")


def test_export_model_service_onnx_for_kriging_errors(licensed, project):
    run = _kriging_run(project)
    with pytest.raises(ExportError):
        export_model(run, "onnx")
