"""End-to-end training, persistence, and run artifacts (PRD 4.1, 4.5, 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surroforge.core.config import parse_config
from surroforge.core.project import Project
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


def test_onnx_export_is_opt_in_by_default(project):
    # ONNX conversion is the dominant fixed cost of a cheap method, so a plain
    # train no longer auto-exports it; the pickle artifact is still written.
    run = run_training(project, parse_config(poly_config("polynomial")))
    assert run.model_path.is_file()
    assert not run.has_onnx()
    assert run.lineage["onnx_exported"] is False


def test_onnx_exported_when_requested(project, licensed_export):
    pytest.importorskip("skl2onnx")  # needs the 'onnx' extra
    run = run_training(project, parse_config(poly_config("polynomial")), export_onnx=True)
    assert run.has_onnx()
    assert run.lineage["onnx_exported"] is True


def test_train_onnx_without_license_skips_gracefully(project, monkeypatch, tmp_path):
    # Wave 2: the ONNX exporter is Professional. An unlicensed train --onnx must
    # not fail the run — the best-effort block logs a skip instead. The assertion
    # sticks to the skip prefix so it holds both with surroforge_professional
    # installed (licence message) and without (unknown-format message).
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-licence-home"))
    run = run_training(project, parse_config(poly_config("polynomial")), export_onnx=True)
    assert not run.has_onnx()
    assert run.lineage["onnx_exported"] is False
    assert "ONNX export skipped:" in (run.run_dir / "log.txt").read_text()


def test_train_then_predict_without_license(project, monkeypatch, tmp_path):
    # Second lock on the Wave-2/3 invariant: the run store is licence-free —
    # an unlicensed user trains AND predicts from the stored model, end to end.
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-licence-home"))
    from surroforge.services import predict_with_run
    from tests.conftest import make_samples

    run = run_training(project, parse_config(poly_config("polynomial")))
    new_points = make_samples(n=5, seed=99).drop(columns=["max_stress"])
    preds = predict_with_run(run, new_points)
    assert "max_stress" in preds.columns
    assert len(preds) == 5


def test_holdout_actual_persisted_and_aligned_with_fingerprint(project):
    run = run_training(project, parse_config(poly_config("polynomial")))
    actual = np.asarray(run.holdout_actual, dtype=float)
    pred = np.asarray(run.holdout_fingerprint, dtype=float)
    assert "holdout_actual" in run.lineage
    assert actual.shape == pred.shape          # same rows AND same columns
    assert actual.ndim == 2 and actual.shape[1] == 1
    assert np.isfinite(actual).all()


def test_holdout_actual_absent_returns_empty_list(project):
    run = run_training(project, parse_config(poly_config("polynomial")))
    run.lineage.pop("holdout_actual")          # simulate a pre-feature run
    assert run.holdout_actual == []


def test_holdout_actual_multi_output_shape(tmp_path):
    rng = np.random.default_rng(1)
    n = 60
    a = rng.uniform(0.0, 1.0, n)
    b = rng.uniform(0.0, 1.0, n)
    df = pd.DataFrame({"a": a, "b": b, "y1": 2 * a + b, "y2": a - 3 * b + 0.5})
    csv = tmp_path / "multi.csv"
    df.to_csv(csv, index=False)
    proj = Project.init(tmp_path / "p", name="multi")
    proj.add_dataset(csv, name="multi")
    cfg = parse_config(
        {
            "experiment": "multi",
            "data": {
                "source": "multi",
                "inputs": ["a", "b"],
                "outputs": ["y1", "y2"],
                "split": {"test_fraction": 0.25, "seed": 1},
            },
            "model": {"method": "polynomial", "degree": 1},
            "validation": {"cross_validation": 4, "metrics": ["rmse", "r2"]},
        }
    )
    run = run_training(proj, cfg)
    actual = np.asarray(run.holdout_actual, dtype=float)
    pred = np.asarray(run.holdout_fingerprint, dtype=float)
    assert actual.shape == pred.shape
    assert actual.shape[1] == 2
