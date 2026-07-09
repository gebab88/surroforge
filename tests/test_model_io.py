"""Run-store model serialisation (Wave 2: decoupled from the Professional exporters)."""

from __future__ import annotations

import numpy as np

from surroforge.persistence import dump_model, load_model
from surroforge.trainers.polynomial import PolynomialTrainer


def test_model_roundtrip_preserves_predictions(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (50, 3))
    y = x @ np.array([[1.0], [2.0], [3.0]]) + 0.5
    model = PolynomialTrainer(degree=2)
    model.fit(x, y)

    path = dump_model(model, tmp_path / "model.pkl")
    assert path.is_file()
    restored = load_model(path)
    assert np.allclose(restored.predict(x), model.predict(x))
