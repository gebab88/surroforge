from __future__ import annotations

from types import SimpleNamespace

import pytest

from surroforge.core.config import parse_config
from surroforge.errors import PluginError
from surroforge.services import generate_report
from surroforge_professional.reports.html import HtmlReporter


def test_html_contains_required_sections(licensed, trained_run):
    text = HtmlReporter().render(trained_run)

    assert text.startswith("<!doctype html>")
    assert "<h1>SurroForge Report</h1>" in text
    for heading in (
        "Executive Summary",
        "Quality Snapshot",
        "Data And Model",
        "Validation Details",
        "Reproducibility Appendix",
    ):
        assert heading in text
    assert "<script" not in text.lower()
    assert "https://" not in text
    assert "http://" not in text


def test_html_reports_metrics_and_lineage(licensed, trained_run):
    text = HtmlReporter().render(trained_run)

    assert trained_run.run_id in text
    assert trained_run.config.experiment in text
    assert trained_run.config.model.method in text
    assert trained_run.data_hash in text
    assert trained_run.config_hash in text
    assert trained_run.lineage["prediction_hash"] in text
    assert f"surroforge reproduce {trained_run.run_id}" in text
    for metric in trained_run.config.validation.metrics:
        assert f">{metric}<" in text
    rmse = trained_run.metrics["cross_validation"]["rmse"]
    assert f">{trained_run.metrics['train']['rmse']:.5g}<" in text
    assert f">{trained_run.metrics['test']['rmse']:.5g}<" in text
    assert f">{rmse['mean']:.5g}<" in text
    for column in trained_run.config.data.inputs + trained_run.config.data.outputs:
        assert column in text


def test_generate_report_writes_html_under_reports_dir(licensed, project, trained_run):
    path, text = generate_report(project, trained_run, fmt="html")

    assert path == project.reports_dir / f"{trained_run.run_id}.html"
    assert path.read_text() == text
    assert "SurroForge Report" in text


def test_missing_required_train_metrics_fail_clearly(licensed, trained_run):
    trained_run.__dict__["metrics"] = {"test": {}, "cross_validation": {}}

    with pytest.raises(PluginError, match=r"metrics\['train'\]"):
        HtmlReporter().render(trained_run)


def test_missing_cross_validation_metric_fails_clearly(licensed, trained_run):
    metrics = dict(trained_run.metrics)
    metrics["cross_validation"] = {"folds": 5}
    trained_run.__dict__["metrics"] = metrics

    with pytest.raises(PluginError, match="cross-validation metric 'rmse'"):
        HtmlReporter().render(trained_run)


@pytest.mark.parametrize("field", ["data_hash", "config_hash", "prediction_hash"])
@pytest.mark.parametrize("value", [None, ""])
def test_missing_or_blank_identity_lineage_fails_clearly(licensed, trained_run, field, value):
    if value is None:
        trained_run.lineage.pop(field, None)
    else:
        trained_run.lineage[field] = value

    with pytest.raises(PluginError, match=field):
        HtmlReporter().render(trained_run)


def test_dynamic_report_values_are_escaped(licensed):
    config = parse_config(
        {
            "experiment": "<script>alert(1)</script>",
            "data": {
                "source": "<b>beam</b>",
                "inputs": ["length<script>"],
                "outputs": ["max_stress<b>"],
                "split": {"test_fraction": 0.2, "seed": 42},
            },
            "model": {"method": "polynomial", "degree": 2},
            "validation": {"cross_validation": 5, "metrics": ["rmse"]},
        }
    )
    run = SimpleNamespace(
        run_id="run-hostile",
        config=config,
        metrics={
            "train": {"rmse": 1.0},
            "test": {"rmse": 2.0},
            "cross_validation": {"rmse": {"mean": 1.5, "std": 0.1}, "folds": 5},
            "n_samples": 12,
            "n_inputs": 1,
            "n_outputs": 1,
        },
        lineage={
            "created_at": "2026-06-24T00:00:00Z",
            "data_source": "<b>beam</b>",
            "data_hash": "data-hash",
            "config_hash": "config-hash",
            "prediction_hash": "prediction-hash",
            "dependencies": ["pkg<script>==1", "pkg<b>==2"],
            "python_version": "3.13",
            "surroforge_version": "0.1",
            "seed": 42,
        },
        seed=42,
    )

    text = HtmlReporter().render(run)

    assert "<script" not in text.lower()
    assert "<b>" not in text
    assert "&lt;script&gt;" in text
    assert "&lt;b&gt;" in text
