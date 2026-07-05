"""Professional HTML comparison report (spec 2026-07-05)."""

from __future__ import annotations

import json
import re

import pytest

from surroforge.comparison import build_comparison
from surroforge.errors import PluginError
from surroforge_professional.reports.comparison import HtmlComparisonReporter


def _minimal_comparison(**overrides):
    base = {
        "metric": "rmse",
        "metric_names": ["rmse"],
        "runs": [
            {
                "run_id": "run-001",
                "method": "polynomial",
                "experiment": "exp",
                "created_at": "2026-07-05T12:00:00",
                "rank": 1,
                "delta": 0.0,
                "metrics": {
                    "rmse": {"train": 0.1, "test": 0.2, "cv_mean": 0.15, "cv_std": 0.01}
                },
                "lineage": {
                    "seed": 0, "data_hash": "abc123", "config_hash": "def456",
                    "parent_run": None,
                },
            }
        ],
        "best": "run-001",
        "best_per_metric": {"rmse": "run-001"},
        "config_diff": {},
        "lineage_summary": {"same_data": True, "data_hash_groups": {"abc123": ["run-001"]}},
        "warnings": [],
    }
    base.update(overrides)
    return base


@pytest.fixture
def comparison(trained_run, second_run):
    return build_comparison([trained_run, second_run], metric="rmse")


@pytest.fixture
def html(licensed, comparison):
    return HtmlComparisonReporter().render(comparison)


def test_requires_license(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(PluginError, match="license"):
        HtmlComparisonReporter().render(_minimal_comparison())


def test_smoke_and_sections(html):
    assert html.startswith("<!doctype html>")
    for heading in (
        "SurroForge Run Comparison",
        "Best Run",
        "Ranking",
        "Metric Charts",
        "All Metrics",
        "Configuration Differences",
        "Lineage",
    ):
        assert heading in html


def test_self_contained(html):
    for marker in ('src="http', 'href="http', "@import", "url(http"):
        assert marker not in html


def test_ranking_shows_every_run_and_marks_best(html, comparison):
    for entry in comparison["runs"]:
        assert entry["run_id"] in html
    assert comparison["best"] in html
    assert ">best</span>" in html


def test_config_diff_lists_differing_field(html):
    assert "model.degree" in html


def test_chart_json_embeds_all_runs(html, comparison):
    match = re.search(
        r'<script type="application/json" id="sf-comparison-charts">(.*?)</script>',
        html,
        re.S,
    )
    assert match
    data = json.loads(match.group(1))
    assert data["runs"] == [entry["run_id"] for entry in comparison["runs"]]
    assert set(data["metrics"]) <= set(comparison["metric_names"])
    first = data["per_metric"][data["metrics"][0]]
    assert len(first["test"]) == len(data["runs"])


def test_uplot_attribution_present(html):
    assert "uPlot v1.6.32 (MIT License)" in html


def test_warning_banner(licensed):
    text = HtmlComparisonReporter().render(
        _minimal_comparison(warnings=["Runs use different datasets (data_hash differs)."])
    )
    assert "Warnings" in text
    assert "different datasets" in text


def test_hostile_strings_are_escaped(licensed):
    comparison = _minimal_comparison()
    comparison["runs"][0]["experiment"] = '<script>alert("xss")</script>'
    comparison["config_diff"] = {
        "model.note": {"run-001": "<img src=x onerror=alert(1)>"}
    }
    text = HtmlComparisonReporter().render(comparison)
    assert '<script>alert("xss")</script>' not in text
    assert "<img src=x" not in text
    assert "&lt;script&gt;" in text


def test_single_run_comparison_renders(licensed):
    text = HtmlComparisonReporter().render(_minimal_comparison())
    assert text.startswith("<!doctype html>")


def test_charts_section_omitted_when_nothing_plottable(licensed):
    comparison = _minimal_comparison()
    comparison["runs"][0]["metrics"] = {
        "rmse": {"train": None, "test": None, "cv_mean": None, "cv_std": None}
    }
    text = HtmlComparisonReporter().render(comparison)
    assert "sf-comparison-charts" not in text
    assert "Metric Charts" not in text


def test_generate_comparison_report_end_to_end(licensed, project, trained_run, second_run):
    from surroforge.services import generate_comparison_report

    path = generate_comparison_report(
        project, [trained_run.run_id, second_run.run_id], metric="rmse", format="html"
    )
    assert path == project.reports_dir / "comparison.html"
    assert path.read_text().startswith("<!doctype html>")


def test_cli_compare_format_html_full_stack(licensed, project, trained_run, second_run, tmp_path):
    from typer.testing import CliRunner

    from surroforge.cli.main import app

    out = tmp_path / "cmp.html"
    result = CliRunner().invoke(
        app,
        [
            "--config-dir", str(project.root), "compare",
            "--runs", f"{trained_run.run_id},{second_run.run_id}",
            "--format", "html", "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.read_text().startswith("<!doctype html>")
