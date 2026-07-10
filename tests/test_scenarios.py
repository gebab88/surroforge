"""The three standard scenarios from the Phase 1 Definition of Done (PRD 9.1).

These are the acceptance scenarios external testers run: a greenfield workflow,
a method comparison, and a reproduction.
"""

from __future__ import annotations

import pytest

from surroforge import Project, Trainer
from surroforge.services import (
    assert_reproducible,
    compare_runs,
    export_model,
    generate_report,
    predict_from_file,
    reproduce,
)
from tests.conftest import make_samples, poly_config


def test_scenario_greenfield(tmp_path, licensed_export):
    """From an empty directory to a validated, exported, reported model."""
    # Export is Professional since Wave 2; the rest of the scenario is Core.
    csv = tmp_path / "samples.csv"
    make_samples().to_csv(csv, index=False)

    project = Project.init(tmp_path / "study", name="greenfield")
    project.add_dataset(csv, name="beam")

    run = Trainer.from_dict(poly_config("polynomial")).fit(project.dataset("beam"))
    assert run.metrics["test"]["r2"] > 0.95

    onnx_path = export_model(run, "onnx")
    assert onnx_path.is_file()

    report_path, text = generate_report(project, run)
    assert report_path.is_file()
    assert "SurroForge Report" in text

    # Predict on fresh points.
    new = tmp_path / "new.csv"
    make_samples(n=5, seed=99).drop(columns=["max_stress"]).to_csv(new, index=False)
    preds = predict_from_file(run, new)
    assert "max_stress" in preds.columns
    assert len(preds) == 5


def test_scenario_method_comparison(tmp_path):
    """Train three methods on the same data and pick the best."""
    pytest.importorskip("smt")  # the comparison includes Kriging/RBF (SMT backend)
    csv = tmp_path / "samples.csv"
    make_samples().to_csv(csv, index=False)
    project = Project.init(tmp_path / "study", name="compare")
    project.add_dataset(csv, name="beam")

    for method in ("polynomial", "kriging", "rbf"):
        Trainer.from_dict(poly_config(method)).fit(project)

    run_ids = [r.run_id for r in project.runs()]
    assert len(run_ids) == 3

    result = compare_runs(project, run_ids, metric="rmse")
    assert result["best"] in run_ids
    assert len(result["runs"]) == 3
    # Every row carries the requested metrics.
    assert set(result["runs"][0]["metrics"]) == {"rmse", "r2", "mae"}


def test_scenario_reproduction(tmp_path):
    """Reproduce a run months later: same config, same data, same model."""
    csv = tmp_path / "samples.csv"
    make_samples().to_csv(csv, index=False)
    project = Project.init(tmp_path / "study", name="reproduce")
    project.add_dataset(csv, name="beam")

    original = Trainer.from_dict(poly_config("polynomial")).fit(project)
    result = reproduce(project, original.run_id)

    assert result.bit_identical
    assert_reproducible(result)
    assert result.reproduction.parent_run == original.run_id
