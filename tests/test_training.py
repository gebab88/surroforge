"""End-to-end training, persistence, and run artifacts (PRD 4.1, 4.5, 7)."""

from __future__ import annotations

import numpy as np
import pytest

from surroforge.core.config import parse_config
from surroforge.engine import run_training
from surroforge.errors import DataError
from tests.conftest import poly_config


def test_polynomial_run_fits_well_and_persists(project):
    cfg = parse_config(poly_config("polynomial"))
    run = run_training(project, cfg)

    # Metrics present and good on this smooth data.
    assert set(run.metrics) >= {"train", "test", "cross_validation"}
    assert run.metrics["test"]["r2"] > 0.95
    assert run.metrics["n_inputs"] == 3
    assert run.metrics["n_outputs"] == 1

    # Artifacts on disk.
    assert run.model_path.is_file()
    assert (run.run_dir / "config.snapshot.yaml").is_file()
    assert (run.run_dir / "log.txt").is_file()

    # Lineage completeness (PRD 7.2).
    for key in (
        "run_id", "created_at", "config_hash", "data_hash", "seed",
        "python_version", "surroforge_version", "dependencies", "parent_run",
    ):
        assert key in run.lineage
    assert run.lineage["data_hash"] == project.dataset("beam").hash


def test_cross_validation_reports_per_fold(project):
    cfg = parse_config(poly_config("polynomial"))
    run = run_training(project, cfg)
    cv = run.metrics["cross_validation"]
    assert cv["folds"] == 5
    assert len(cv["rmse"]["per_fold"]) == 5
    assert cv["rmse"]["mean"] >= 0.0


def test_loaded_model_predicts(project):
    cfg = parse_config(poly_config("polynomial"))
    run = run_training(project, cfg)
    model = run.load_model()
    frame = project.load_dataset_frame(project.dataset("beam"))
    x = frame[["length", "width", "load"]].to_numpy(float)
    preds = model.predict(x)
    assert preds.shape == (len(frame), 1)
    assert np.isfinite(preds).all()


def test_sequential_run_ids(project):
    cfg = parse_config(poly_config("polynomial"))
    r1 = run_training(project, cfg)
    r2 = run_training(project, cfg)
    assert r1.run_id == "run-001"
    assert r2.run_id == "run-002"
    assert [r.run_id for r in project.runs()] == ["run-001", "run-002"]


def test_too_many_folds_raises_clear_error(project):
    raw = poly_config("polynomial")
    raw["validation"]["cross_validation"] = 1000  # more folds than samples
    cfg = parse_config(raw)
    with pytest.raises(DataError):
        run_training(project, cfg)


def test_file_source_without_registration(project, tmp_path):
    # data.source may be a path, not only a registered dataset name.
    raw = poly_config("polynomial")
    raw["data"]["source"] = str(project.dataset("beam").path)  # relative to project root
    cfg = parse_config(raw)
    run = run_training(project, cfg)
    assert run.metrics["test"]["r2"] > 0.95
