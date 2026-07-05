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


def test_train_invalid_polynomial_degree_clean_error(tmp_path):
    root = _init_project_with_data(tmp_path)
    raw = poly_config("polynomial")
    raw["model"]["degree"] = "two"
    cfg = root / "configs" / "bad-poly.yaml"
    cfg.write_text(yaml.safe_dump(raw))

    result = runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "degree" in result.output
    assert "Traceback" not in result.output


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


def test_predict_non_numeric_input_clean_error(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])

    bad = tmp_path / "bad.csv"
    frame = make_samples(n=2, seed=9).drop(columns=["max_stress"])
    frame["width"] = frame["width"].astype(object)
    frame.loc[0, "width"] = "wide"
    frame.to_csv(bad, index=False)
    result = runner.invoke(
        app, ["--config-dir", str(root), "predict", "run-001", "--input", str(bad)]
    )
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Non-numeric input columns" in result.output
    assert "Traceback" not in result.output


def test_predict_missing_value_clean_error(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])

    bad = tmp_path / "bad.csv"
    frame = make_samples(n=2, seed=9).drop(columns=["max_stress"])
    frame.loc[0, "width"] = None
    frame.to_csv(bad, index=False)
    result = runner.invoke(
        app, ["--config-dir", str(root), "predict", "run-001", "--input", str(bad)]
    )
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "missing values" in result.output
    assert "Traceback" not in result.output


def test_compare_cli(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])
    result = runner.invoke(app, ["--config-dir", str(root), "compare", "--runs", "run-001,run-002"])
    assert result.exit_code == 0, result.output
    assert "Ranking by test rmse" in result.output
    assert "Test metrics" in result.output
    assert "Configurations are identical." in result.output  # two identical-config runs
    assert "Lineage" in result.output
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


def test_compare_format_html_calls_service(tmp_path, monkeypatch):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])

    calls = {}

    def fake_generate(project, run_ids, *, metric="rmse", format="html", output=None):
        calls.update(run_ids=run_ids, metric=metric, format=format, output=output)
        out = root / "reports" / "comparison.html"
        out.write_text("<!doctype html>")
        return out

    monkeypatch.setattr("surroforge.services.generate_comparison_report", fake_generate)
    result = runner.invoke(
        app, ["--config-dir", str(root), "compare", "--runs", "run-001", "-f", "html"]
    )
    assert result.exit_code == 0, result.output
    assert "comparison.html" in result.output
    assert calls == {"run_ids": ["run-001"], "metric": "rmse", "format": "html", "output": None}


def test_compare_terminal_with_output_errors(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])
    result = runner.invoke(
        app,
        ["--config-dir", str(root), "compare", "--runs", "run-001", "--output", "x.html"],
    )
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "--format" in result.output
    assert "Traceback" not in result.output


# --- data add-openfoam (Story R5.2) --------------------------------------------------


def _of_scalar(path, values):
    body = "\n".join(f"{v:.10g}" for v in values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "FoamFile\n{\n    class volScalarField;\n    object p;\n}\n"
        f"internalField nonuniform List<scalar>\n{len(values)}\n(\n{body}\n)\n;\n"
    )


def _make_of_sweep(tmp_path):
    """Three OpenFOAM cases of a 3-cell field 'p' + a DoE samples table."""
    case_root = tmp_path / "cases"
    rows = {"case_00": [1.0, 2.0, 3.0], "case_01": [4.0, 5.0, 6.0], "case_02": [7.0, 8.0, 9.0]}
    for case, values in rows.items():
        _of_scalar(case_root / case / "10" / "p", values)
    samples = tmp_path / "samples.csv"
    pd.DataFrame(
        {"case": list(rows), "u_in": [1.0, 2.0, 3.0], "aoa": [0.0, 1.0, 2.0]}
    ).to_csv(samples, index=False)
    return case_root, samples


def test_data_add_openfoam_registers(tmp_path):
    root = tmp_path / "proj"
    assert runner.invoke(app, ["init", str(root)]).exit_code == 0
    case_root, samples = _make_of_sweep(tmp_path)
    result = runner.invoke(
        app,
        ["--config-dir", str(root), "data", "add-openfoam", str(case_root),
         "--field", "p", "--samples", str(samples), "--name", "cfd"],
    )
    assert result.exit_code == 0, result.output
    assert "Registered dataset 'cfd'" in result.output
    assert "3 rows" in result.output  # 3 cases; cols = u_in, aoa, p_0000..p_0002
    listing = runner.invoke(app, ["--config-dir", str(root), "data", "list"])
    assert "cfd" in listing.output


def test_data_add_openfoam_deterministic_hash(tmp_path):
    from surroforge.core.project import Project

    case_root, samples = _make_of_sweep(tmp_path)
    hashes = []
    for i in range(2):
        root = tmp_path / f"proj{i}"
        runner.invoke(app, ["init", str(root)])
        result = runner.invoke(
            app,
            ["--config-dir", str(root), "data", "add-openfoam", str(case_root),
             "--field", "p", "--samples", str(samples), "--name", "cfd"],
        )
        assert result.exit_code == 0, result.output
        hashes.append(Project.open(root).dataset("cfd").hash)
    assert hashes[0] == hashes[1]


def test_data_add_openfoam_bad_samples_errors(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["init", str(root)])
    case_root, _ = _make_of_sweep(tmp_path)
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["case_00"], "u_in": [1.0]}).to_csv(bad, index=False)  # no 'case' column
    result = runner.invoke(
        app,
        ["--config-dir", str(root), "data", "add-openfoam", str(case_root),
         "--field", "p", "--samples", str(bad), "--name", "cfd"],
    )
    assert result.exit_code == 1
    assert "Error:" in result.output and "'case' column" in result.output


def test_data_add_openfoam_name_collision_preserves_existing(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["init", str(root)])
    pre = tmp_path / "pre.csv"
    pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).to_csv(pre, index=False)
    assert runner.invoke(
        app, ["--config-dir", str(root), "data", "add", str(pre), "--name", "cfd"]
    ).exit_code == 0
    backing = root / "data" / "cfd.csv"
    before = backing.read_text()
    case_root, samples = _make_of_sweep(tmp_path)
    result = runner.invoke(
        app,
        ["--config-dir", str(root), "data", "add-openfoam", str(case_root),
         "--field", "p", "--samples", str(samples), "--name", "cfd"],
    )
    assert result.exit_code == 1
    assert "already registered" in result.output
    assert backing.read_text() == before  # existing backing file must NOT be clobbered


def test_data_add_openfoam_integer_case_names(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["init", str(root)])
    case_root = tmp_path / "cases"
    for i in range(3):
        _of_scalar(case_root / str(i) / "10" / "p", [float(i), float(i) + 0.5])
    samples = tmp_path / "samples.csv"
    pd.DataFrame({"case": [0, 1, 2], "u": [1.0, 2.0, 3.0]}).to_csv(samples, index=False)
    result = runner.invoke(
        app,
        ["--config-dir", str(root), "data", "add-openfoam", str(case_root),
         "--field", "p", "--samples", str(samples), "--name", "cfd"],
    )
    assert result.exit_code == 0, result.output
    assert "Registered dataset 'cfd'" in result.output


def test_data_add_openfoam_case_mismatch_errors(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["init", str(root)])
    case_root, _ = _make_of_sweep(tmp_path)  # dirs case_00..case_02
    bad = tmp_path / "mismatch.csv"
    mismatch = pd.DataFrame({"case": ["nope_0", "nope_1", "nope_2"], "u": [1.0, 2.0, 3.0]})
    mismatch.to_csv(bad, index=False)
    result = runner.invoke(
        app,
        ["--config-dir", str(root), "data", "add-openfoam", str(case_root),
         "--field", "p", "--samples", str(bad), "--name", "cfd"],
    )
    assert result.exit_code == 1
    assert "Error:" in result.output and "No parameters provided" in result.output


def test_data_add_openfoam_empty_samples_errors(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["init", str(root)])
    case_root, _ = _make_of_sweep(tmp_path)
    empty = tmp_path / "empty.csv"
    pd.DataFrame({"case": [], "u": []}).to_csv(empty, index=False)
    result = runner.invoke(
        app,
        ["--config-dir", str(root), "data", "add-openfoam", str(case_root),
         "--field", "p", "--samples", str(empty), "--name", "cfd"],
    )
    assert result.exit_code == 1
    assert "no rows" in result.output


def test_data_add_openfoam_unsupported_output_errors(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["init", str(root)])
    case_root, samples = _make_of_sweep(tmp_path)
    result = runner.invoke(
        app,
        ["--config-dir", str(root), "data", "add-openfoam", str(case_root),
         "--field", "p", "--samples", str(samples), "--name", "cfd",
         "--output", str(tmp_path / "out.txt")],
    )
    assert result.exit_code == 1
    assert "Unsupported --output format" in result.output


def test_data_add_openfoam_bad_time_errors(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["init", str(root)])
    case_root, samples = _make_of_sweep(tmp_path)
    result = runner.invoke(
        app,
        ["--config-dir", str(root), "data", "add-openfoam", str(case_root),
         "--field", "p", "--samples", str(samples), "--name", "cfd", "--time", "999"],
    )
    assert result.exit_code == 1
    assert "Error:" in result.output
