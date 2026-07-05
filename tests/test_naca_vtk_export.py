from __future__ import annotations

import base64
import importlib.util
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "examples" / "naca0012_openfoam" / "surrogate_vtk.py"


def _load_surrogate_vtk_module():
    spec = importlib.util.spec_from_file_location("naca_surrogate_vtk", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vtk_payload(values: list[float], dtype: str = "<f4") -> str:
    array = np.asarray(values, dtype=dtype)
    raw = struct.pack("<Q", array.nbytes) + array.tobytes()
    return base64.b64encode(raw).decode("ascii")


def _write_minimal_vtu(path: Path) -> None:
    path.write_text(
        "<?xml version='1.0'?>\n"
        "<VTKFile type='UnstructuredGrid' version='0.1' byte_order='LittleEndian' "
        "header_type='UInt64'>\n"
        "  <UnstructuredGrid>\n"
        "    <Piece NumberOfPoints='4' NumberOfCells='3'>\n"
        "      <CellData>\n"
        "        <DataArray type='Float32' Name='p' format='binary'>\n"
        f"{_vtk_payload([1.0, 2.0, 3.0])}\n"
        "        </DataArray>\n"
        "      </CellData>\n"
        "      <PointData>\n"
        "        <DataArray type='Float32' Name='p' format='binary'>\n"
        f"{_vtk_payload([10.0, 20.0, 30.0, 40.0])}\n"
        "        </DataArray>\n"
        "      </PointData>\n"
        "      <Points>\n"
        "        <DataArray type='Float32' Name='Points' NumberOfComponents='3' format='binary'>\n"
        f"{_vtk_payload([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0])}\n"
        "        </DataArray>\n"
        "      </Points>\n"
        "      <Cells>\n"
        "        <DataArray type='Int32' Name='connectivity' format='binary'>\n"
        f"{_vtk_payload([0, 1, 2, 0, 2, 3, 0, 1, 3], '<i4')}\n"
        "        </DataArray>\n"
        "        <DataArray type='Int32' Name='offsets' format='binary'>\n"
        f"{_vtk_payload([3, 6, 9], '<i4')}\n"
        "        </DataArray>\n"
        "        <DataArray type='UInt8' Name='types' format='binary'>\n"
        f"{_vtk_payload([5, 5, 5], 'u1')}\n"
        "        </DataArray>\n"
        "      </Cells>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n"
    )


def test_surrogate_vtkhdf_writes_single_time_series_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    module = _load_surrogate_vtk_module()
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "surrogate_vtk"
    samples = pd.DataFrame(
        {
            "case": ["case_000", "case_001"],
            "aoa": [-5.0, -4.0],
            "speed": [20.0, 20.0],
        }
    )

    for case in samples["case"]:
        template = run_dir / case / "VTK" / f"{case}_10" / "internal.vtu"
        template.parent.mkdir(parents=True)
        _write_minimal_vtu(template)

    output = out_dir / "surrogate_fields.vtkhdf"
    written = module.write_surrogate_vtkhdf(
        run_dir,
        output,
        samples,
        np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]),
        field_name="p",
    )

    assert written == output
    assert pd.read_csv(out_dir / "samples.csv").to_dict("list") == samples.to_dict("list")
    with h5py.File(output, "r") as h5:
        vtk = h5["VTKHDF"]
        assert vtk.attrs["Type"] == b"UnstructuredGrid"
        assert vtk.attrs["Version"].tolist() == [2, 5]
        assert vtk["NumberOfPoints"][...].tolist() == [4, 4]
        assert vtk["NumberOfCells"][...].tolist() == [3, 3]
        assert vtk["NumberOfConnectivityIds"][...].tolist() == [9, 9]
        assert vtk["Offsets"][...].tolist() == [0, 3, 6, 9, 0, 3, 6, 9]
        assert vtk["CellData/p"][...].tolist() == [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        assert vtk["FieldData/TimeValue"][...].tolist() == [0.0, 1.0]
        assert vtk["Steps"].attrs["NSteps"] == 2
        assert vtk["Steps/Values"][...].tolist() == [0.0, 1.0]
        assert vtk["Steps/PointOffsets"][...].tolist() == [0, 4]
        assert vtk["Steps/ConnectivityIdOffsets"][...].tolist() == [0, 9]
        assert vtk["Steps/CellDataOffsets/p"][...].tolist() == [0, 3]
