"""Model export: pickle round-trip and ONNX (incl. runtime parity) (PRD 4.4)."""

from __future__ import annotations

import numpy as np
import pytest

from surroforge.errors import ExportError
from surroforge.exporters.onnx_exporter import ONNXExporter
from surroforge.exporters.pickle_exporter import PickleExporter, load_pickle
from surroforge.services import export_model
from surroforge.trainers.kriging import KrigingTrainer
from surroforge.trainers.polynomial import PolynomialTrainer


def _fitted_polynomial(n_features=3):
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (50, n_features))
    y = x @ np.arange(1, n_features + 1).reshape(-1, 1) + 0.5
    model = PolynomialTrainer(degree=2)
    model.fit(x, y)
    return model, x


def test_pickle_roundtrip_preserves_predictions(tmp_path):
    model, x = _fitted_polynomial()
    path = PickleExporter().export(model, tmp_path / "m.pkl", n_features=3)
    assert path.is_file()
    restored = load_pickle(path)
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


def test_export_model_service_default_path(trained_run):
    # No explicit output -> artifact lands inside the run directory.
    path = export_model(trained_run, "pickle")
    assert path == trained_run.run_dir / "model.pkl"
    assert path.is_file()


def test_export_model_service_onnx_for_kriging_errors(project):
    pytest.importorskip("smt")
    from surroforge.core.config import parse_config
    from surroforge.engine import run_training
    from tests.conftest import poly_config

    run = run_training(project, parse_config(poly_config("kriging")))
    with pytest.raises(ExportError):
        export_model(run, "onnx")
