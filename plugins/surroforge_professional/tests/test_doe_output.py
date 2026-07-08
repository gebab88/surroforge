from __future__ import annotations

import json

import pandas as pd
import pytest

from surroforge_professional.doe.output import write_design
from surroforge_professional.doe.spec import DoeError, DoeSpec


def _spec():
    return DoeSpec(
        design="full-factorial",
        factors=[
            {"name": "aoa", "type": "discrete", "levels": [0, 4, 8]},
            {"name": "speed", "type": "discrete", "levels": [20, 40]},
        ],
    )


def test_writes_samples_csv_with_case_column(tmp_path):
    csv_path, manifest_path = write_design(_spec(), tmp_path)

    assert csv_path == tmp_path / "samples.csv"
    assert manifest_path == tmp_path / "doe.manifest.json"
    frame = pd.read_csv(csv_path)
    assert list(frame.columns) == ["case", "aoa", "speed"]
    assert len(frame) == 6
    assert frame["case"].iloc[0] == "case_000"
    assert frame["case"].iloc[-1] == "case_005"


def test_manifest_records_design_seed_and_hash(tmp_path):
    _, manifest_path = write_design(_spec(), tmp_path)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["design"] == "full-factorial"
    assert manifest["samples"] == 6
    assert manifest["seed"] == 0
    assert len(manifest["factors"]) == 2
    assert isinstance(manifest["table_sha256"], str) and len(manifest["table_sha256"]) == 64
    assert "plugin_version" in manifest


def test_case_padding_widens_for_large_designs(tmp_path):
    spec = DoeSpec(
        design="lhs",
        samples=1000,
        factors=[{"name": "a", "type": "continuous", "bounds": [0, 1]}],
    )
    csv_path, _ = write_design(spec, tmp_path)
    frame = pd.read_csv(csv_path)
    assert frame["case"].iloc[0] == "case_0000"
    assert frame["case"].iloc[-1] == "case_0999"


def test_refuses_to_overwrite_without_force(tmp_path):
    write_design(_spec(), tmp_path)
    with pytest.raises(DoeError, match="already exists"):
        write_design(_spec(), tmp_path)


def test_force_overwrites(tmp_path):
    write_design(_spec(), tmp_path)
    csv_path, _ = write_design(_spec(), tmp_path, force=True)
    assert csv_path.is_file()


def test_output_columns_match_openfoam_ingest_contract(tmp_path):
    # `read_openfoam_sweep` / `_openfoam_params` expect a 'case' column plus
    # one numeric parameter column per factor. Confirm the contract shape.
    csv_path, _ = write_design(_spec(), tmp_path)
    frame = pd.read_csv(csv_path)
    assert frame.columns[0] == "case"
    assert {"aoa", "speed"}.issubset(set(frame.columns))
    assert frame["case"].is_unique
