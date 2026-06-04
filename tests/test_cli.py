"""CLI behaviour via Typer's test runner (PRD 4.2)."""

from __future__ import annotations

import pandas as pd
import yaml
from typer.testing import CliRunner

from surroforge.cli.main import app
from tests.conftest import make_samples, poly_config

runner = CliRunner()


def _init_project_with_data(tmp_path):
    root = tmp_path / "proj"
    result = runner.invoke(app, ["init", str(root)])
    assert result.exit_code == 0, result.output

    csv = tmp_path / "samples.csv"
    make_samples().to_csv(csv, index=False)
    result = runner.invoke(
        app, ["--config-dir", str(root), "data", "add", str(csv), "--name", "beam"]
    )
    assert result.exit_code == 0, result.output
    return root


def _write_config(root, name="poly.yaml", method="polynomial"):
    path = root / "configs" / name
    path.write_text(yaml.safe_dump(poly_config(method)))
    return path


def test_init_creates_layout(tmp_path):
    root = tmp_path / "p"
    result = runner.invoke(app, ["init", str(root)])
    assert result.exit_code == 0
    for sub in ("data", "configs", "runs", "reports", "surroforge.yaml"):
        assert (root / sub).exists()


def test_data_add_and_list(tmp_path):
    root = _init_project_with_data(tmp_path)
    result = runner.invoke(app, ["--config-dir", str(root), "data", "list"])
    assert result.exit_code == 0
    assert "beam" in result.output


def test_train_validate_list(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    result = runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "run-001" in result.output

    result = runner.invoke(app, ["--config-dir", str(root), "list"])
    assert "run-001" in result.output
    assert "polynomial" in result.output

    result = runner.invoke(app, ["--config-dir", str(root), "validate", "run-001"])
    assert result.exit_code == 0
    assert "rmse" in result.output


def test_export_and_report(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])

    result = runner.invoke(
        app, ["--config-dir", str(root), "export", "run-001", "--format", "onnx"]
    )
    assert result.exit_code == 0
    assert (root / "runs" / "run-001" / "model.onnx").is_file()

    result = runner.invoke(app, ["--config-dir", str(root), "report", "run-001"])
    assert result.exit_code == 0
    assert (root / "reports" / "run-001.md").is_file()


def test_reproduce_cli(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])
    result = runner.invoke(app, ["--config-dir", str(root), "reproduce", "run-001"])
    assert result.exit_code == 0
    assert "identical" in result.output.lower()


def test_unknown_run_reports_clean_error(tmp_path):
    root = _init_project_with_data(tmp_path)
    result = runner.invoke(app, ["--config-dir", str(root), "validate", "run-999"])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_missing_project_reports_clean_error(tmp_path):
    result = runner.invoke(app, ["--config-dir", str(tmp_path / "nope"), "list"])
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_predict_to_file(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])

    new_points = tmp_path / "new.csv"
    make_samples(n=5, seed=9).drop(columns=["max_stress"]).to_csv(new_points, index=False)
    out = root / "preds.csv"
    result = runner.invoke(
        app,
        [
            "--config-dir", str(root), "predict", "run-001",
            "--input", str(new_points), "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    df = pd.read_csv(out)
    assert "max_stress" in df.columns
    assert len(df) == 5


def test_predict_missing_column_clean_error(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"length": [1.0, 2.0]}).to_csv(bad, index=False)  # missing width, load
    result = runner.invoke(
        app, ["--config-dir", str(root), "predict", "run-001", "--input", str(bad)]
    )
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_compare_cli(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])
    result = runner.invoke(app, ["--config-dir", str(root), "compare", "--runs", "run-001,run-002"])
    assert result.exit_code == 0, result.output
    assert "run-001" in result.output
    assert "Best by test rmse" in result.output


def test_info_cli(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])
    result = runner.invoke(app, ["--config-dir", str(root), "info", "run-001"])
    assert result.exit_code == 0
    assert "config_hash" in result.output
    assert "polynomial" in result.output


def test_version_cli():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "surroforge" in result.output


def test_data_add_unsupported_format_clean_error(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["init", str(root)])
    bad = tmp_path / "data.txt"
    bad.write_text("not a table")
    result = runner.invoke(app, ["--config-dir", str(root), "data", "add", str(bad)])
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_quiet_mode_suppresses_output(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    result = runner.invoke(app, ["--config-dir", str(root), "-q", "train", "--config", str(cfg)])
    assert result.exit_code == 0
    assert result.output.strip() == ""
    assert (root / "runs" / "run-001").is_dir()


def test_verbose_mode_shows_progress(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    result = runner.invoke(app, ["--config-dir", str(root), "-v", "train", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "·" in result.output  # per-step progress bullets
