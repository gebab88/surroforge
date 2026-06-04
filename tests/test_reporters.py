"""Markdown report rendering (PRD 4.9)."""

from __future__ import annotations

from surroforge.reporters.markdown import MarkdownReporter
from surroforge.services import generate_report


def test_markdown_contains_required_sections(trained_run):
    text = MarkdownReporter().render(trained_run)
    for heading in (
        "# SurroForge Report",
        "## Data Overview",
        "## Model Specification",
        "## Validation Metrics",
        "## Configuration",
        "## Reproducibility",
    ):
        assert heading in text


def test_markdown_reports_metrics_and_lineage(trained_run):
    text = MarkdownReporter().render(trained_run)
    assert trained_run.run_id in text
    assert trained_run.config.experiment in text
    assert trained_run.data_hash in text
    assert trained_run.config_hash in text
    # Every requested metric appears as a table row.
    for metric in trained_run.config.validation.metrics:
        assert f"| {metric} |" in text
    # Each input/output column is listed.
    for col in trained_run.config.data.inputs + trained_run.config.data.outputs:
        assert col in text


def test_generate_report_writes_under_reports_dir(project, trained_run):
    path, text = generate_report(project, trained_run)
    assert path == project.reports_dir / f"{trained_run.run_id}.md"
    assert path.read_text() == text
    assert "SurroForge Report" in text


def test_generate_report_custom_output(project, trained_run, tmp_path):
    dest = tmp_path / "nested" / "out.md"
    path, _ = generate_report(project, trained_run, output=dest)
    assert path == dest
    assert dest.is_file()
