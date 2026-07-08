from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from surroforge.errors import PluginError
from surroforge_professional.commands.doe import doe_app


def _spec_file(tmp_path):
    path = tmp_path / "doe.yaml"
    path.write_text(
        "design: full-factorial\n"
        "factors:\n"
        "  - {name: aoa, type: discrete, levels: [0, 4, 8]}\n"
        "  - {name: speed, type: discrete, levels: [20, 40]}\n"
    )
    return path


def test_run_writes_outputs(licensed, tmp_path):
    spec = _spec_file(tmp_path)
    out = tmp_path / "design"
    result = CliRunner().invoke(doe_app, ["run", str(spec), "--out", str(out)])

    assert result.exit_code == 0, result.stdout
    frame = pd.read_csv(out / "samples.csv")
    assert list(frame.columns) == ["case", "aoa", "speed"]
    assert (out / "doe.manifest.json").is_file()


def test_run_requires_license_before_reading_spec(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    missing = tmp_path / "does-not-exist.yaml"

    # License is checked first: a missing-license error, not a missing-file error.
    with pytest.raises(PluginError, match="Professional license required"):
        from surroforge_professional.commands.doe import run as run_cmd

        run_cmd(spec=missing, out=tmp_path, force=False)


def test_doe_registered_as_command():
    from surroforge.plugins import available_command_names

    assert "doe" in available_command_names()


def test_surroforge_doe_run_end_to_end(licensed, tmp_path):
    # The plugin must be installed (editable) for the entry point to resolve.
    import importlib
    import sys

    import surroforge.cli.main  # noqa: F401 — ensure the module is in sys.modules

    cli_main = sys.modules["surroforge.cli.main"]
    importlib.reload(cli_main)  # re-mount commands now that the entry point exists
    spec = _spec_file(tmp_path)
    out = tmp_path / "design"
    result = CliRunner().invoke(cli_main.app, ["doe", "run", str(spec), "--out", str(out)])

    assert result.exit_code == 0, result.stdout
    assert (out / "samples.csv").is_file()
