#!/usr/bin/env python3
"""Ingest a real OpenFOAM NACA 0012 sweep and train the SurroForge surrogates.

Run this after ``run_sweep.py`` has produced a sweep directory. It reads each case's
converged pressure field with ``surroforge.data.read_openfoam_sweep`` and the **real**
lift and drag coefficients ``Cl``/``Cd`` that ``forceCoeffs`` integrated, then trains the
same two surrogates as the synthetic CI example (``examples/09_naca0012_openfoam_surrogate.py``):

  * a **field** surrogate (``reduced-field`` POD + neural inner): (AoA, speed) -> pressure field;
  * a **scalar** surrogate: (AoA, speed) -> [Cl, Cd] (lift and drag coefficients).

Both inputs come from the 2-D DoE ``run_sweep.py`` wrote (angle × speed). Both models use
Core ``mlp``; the PyTorch ``torch-mlp`` backend joins the comparison if the
``surroforge-torch`` plugin is installed. Unlike example 09 this needs no solver here —
it consumes the output ``run_sweep.py`` already produced.

    python examples/naca0012_openfoam/ingest_and_train.py run
    python examples/naca0012_openfoam/ingest_and_train.py run --out ./ex09_real   # save the run

With ``--out`` the SurroForge project (datasets, runs, lineage, POD basis, models) **and** the
exported artifacts are saved under that directory, so the real run is inspectable/reproducible
afterwards (``surroforge --config-dir <out>/naca list``). The best field surrogate is also
evaluated on **every** case from ``samples.csv`` and written as one VTKHDF time-series file
``<out>/surrogate_vtk/surrogate_fields.vtkhdf`` (or
``<run_dir>/surrogate_vtk/surrogate_fields.vtkhdf`` without ``--out``), plus ``samples.csv`` for
the case/AoA/speed mapping. Without ``--out`` the project is built in a temp dir and removed, and
only the exported artifacts plus the surrogate VTKHDF file persist.
"""

from __future__ import annotations

import argparse
import base64
import shutil
import struct
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from surroforge import Project, Trainer
from surroforge.data import read_openfoam_sweep
from surroforge.errors import ExportError
from surroforge.plugins import available_trainers
from surroforge.services import compare_runs

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # benign lbfgs iteration cap
_HAS_TORCH = "torch-mlp" in available_trainers()

# A case counts as settled if its C_L spread over the last CL_WINDOW writes is within
# CL_RTOL of |C_L| (plus a small absolute floor for near-zero lift).
CL_WINDOW = 20
CL_RTOL = 1e-2
CL_ATOL = 1e-4


class CoeffReading(NamedTuple):
    cl: float  # lift coefficient at the last (most converged) time
    cd: float  # drag coefficient at the last time
    spread: float  # peak-to-peak of C_L over the last CL_WINDOW writes
    settled: bool  # C_L spread within CL_RTOL*|cl| + CL_ATOL — a genuine plateau


class VtuMesh(NamedTuple):
    points: np.ndarray
    connectivity: np.ndarray
    offsets: np.ndarray
    types: np.ndarray


def read_coeffs(case_dir: Path) -> CoeffReading:
    """Parse Cl and Cd from a case's forceCoeffs output, and check the lift has settled.

    Reads ``postProcessing/forceCoeffs1/<t>/coefficient.dat``, takes ``Cl``/``Cd`` at the last
    (most converged) time, and inspects C_L's peak-to-peak spread over the final ``CL_WINDOW``
    writes: a still-swinging C_L means the run had not reached a steady value even if simpleFoam
    stopped (e.g. residual convergence on a marginally unsteady case).
    """
    # Sort time directories numerically (a restart could make '100' precede '20' lexically).
    dat = sorted(
        case_dir.glob("postProcessing/forceCoeffs1/*/coefficient.dat"),
        key=lambda p: float(p.parent.name),
    )
    if not dat:
        raise FileNotFoundError(f"no forceCoeffs coefficient.dat under {case_dir}")
    header, data_rows = None, []
    for line in dat[-1].read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            cols = stripped.lstrip("#").split()
            if "Cl" in cols and "Cd" in cols:  # the column-name header (not the meta lines)
                header = cols
        elif stripped:
            data_rows.append(stripped.split())
    if header is None or not data_rows:
        raise ValueError(f"could not parse Cl/Cd header from {dat[-1]}")
    cl_idx, cd_idx = header.index("Cl"), header.index("Cd")
    if max(cl_idx, cd_idx) >= len(data_rows[-1]):
        raise ValueError(f"truncated data row (no Cl/Cd value) in {dat[-1]}")
    try:
        cl_series = [float(r[cl_idx]) for r in data_rows if cl_idx < len(r)]
        cd = float(data_rows[-1][cd_idx])
    except ValueError as exc:
        raise ValueError(f"non-numeric Cl/Cd value in {dat[-1]}") from exc
    cl = cl_series[-1]
    window = cl_series[-CL_WINDOW:]
    spread = max(window) - min(window)
    settled = spread <= CL_RTOL * abs(cl) + CL_ATOL
    return CoeffReading(cl=cl, cd=cd, spread=spread, settled=settled)


def _nn(method: str, widths: list[int]) -> dict:
    return {"method": method, "hidden_layer_sizes": widths, "activation": "tanh", "max_iter": 5000}


def _candidates(widths: list[int]) -> list[tuple[str, dict]]:
    cands = [("mlp", _nn("mlp", widths))]
    if _HAS_TORCH:
        cands.append(("torch-mlp", _nn("torch-mlp", widths)))
    cands.append(("polynomial", {"method": "polynomial", "degree": 2}))
    return cands


def _key(aoa: float, speed: float) -> tuple[float, float]:
    """A robust (AoA, speed) lookup key — rounded so equal DoE values match exactly."""
    return (round(float(aoa), 6), round(float(speed), 6))


def _vtk_float32_payload(values) -> str:
    """VTK XML inline-binary payload with a UInt64 byte-count header."""
    array = np.ascontiguousarray(values, dtype="<f4")
    raw = struct.pack("<Q", array.nbytes) + array.tobytes()
    return base64.b64encode(raw).decode("ascii")


def _data_array(parent: ET.Element, name: str) -> ET.Element | None:
    for child in parent.findall("DataArray"):
        if child.attrib.get("Name") == name:
            return child
    return None


def _vtk_dtype(vtk_type: str, byte_order: str) -> np.dtype:
    endian = ">" if byte_order == "BigEndian" else "<"
    if vtk_type == "Float32":
        return np.dtype(f"{endian}f4")
    if vtk_type == "Float64":
        return np.dtype(f"{endian}f8")
    if vtk_type == "Int32":
        return np.dtype(f"{endian}i4")
    if vtk_type == "Int64":
        return np.dtype(f"{endian}i8")
    if vtk_type == "UInt8":
        return np.dtype("u1")
    if vtk_type == "UInt32":
        return np.dtype(f"{endian}u4")
    if vtk_type == "UInt64":
        return np.dtype(f"{endian}u8")
    raise ValueError(f"unsupported VTK DataArray type {vtk_type!r}")


def _decode_vtk_data_array(
    data_array: ET.Element,
    *,
    byte_order: str,
    header_type: str,
) -> np.ndarray:
    dtype = _vtk_dtype(data_array.attrib["type"], byte_order)
    text = "".join((data_array.text or "").split())
    fmt = data_array.attrib.get("format", "ascii")
    if fmt == "binary":
        raw = base64.b64decode(text)
        header_dtype = _vtk_dtype(header_type, byte_order)
        header_nbytes = header_dtype.itemsize
        n_bytes = int(np.frombuffer(raw[:header_nbytes], dtype=header_dtype, count=1)[0])
        data = np.frombuffer(raw[header_nbytes : header_nbytes + n_bytes], dtype=dtype)
    elif fmt == "ascii":
        data = np.fromstring(data_array.text or "", dtype=dtype, sep=" ")
    else:
        raise ValueError(f"unsupported VTK DataArray format {fmt!r}")

    components = int(data_array.attrib.get("NumberOfComponents", "1"))
    if components > 1:
        if data.size % components:
            raise ValueError("VTK DataArray length is not divisible by NumberOfComponents")
        data = data.reshape((-1, components))
    return data


def _read_vtu_mesh(path: Path) -> VtuMesh:
    root = ET.parse(path).getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"{path}: no VTK Piece element found")

    n_points = int(piece.attrib["NumberOfPoints"])
    n_cells = int(piece.attrib["NumberOfCells"])
    byte_order = root.attrib.get("byte_order", "LittleEndian")
    header_type = root.attrib.get("header_type", "UInt32")

    points_array = piece.find("Points/DataArray")
    cells = piece.find("Cells")
    if points_array is None or cells is None:
        raise ValueError(f"{path}: VTU template must contain Points and Cells")
    connectivity_array = _data_array(cells, "connectivity")
    offsets_array = _data_array(cells, "offsets")
    types_array = _data_array(cells, "types")
    if connectivity_array is None or offsets_array is None or types_array is None:
        raise ValueError(f"{path}: VTU Cells must contain connectivity, offsets and types arrays")

    decode = {"byte_order": byte_order, "header_type": header_type}
    points = np.asarray(_decode_vtk_data_array(points_array, **decode), dtype=np.float32)
    if points.ndim == 1:
        if points.size % 3:
            raise ValueError(f"{path}: point coordinate array is not divisible by 3")
        points = points.reshape((-1, 3))
    if points.shape != (n_points, 3):
        raise ValueError(f"{path}: expected {n_points} 3D points, got shape {points.shape}")

    connectivity = np.asarray(
        _decode_vtk_data_array(connectivity_array, **decode).reshape(-1), dtype=np.int64
    )
    vtu_offsets = np.asarray(
        _decode_vtk_data_array(offsets_array, **decode).reshape(-1), dtype=np.int64
    )
    if vtu_offsets.size != n_cells:
        raise ValueError(f"{path}: expected {n_cells} cell offsets, got {vtu_offsets.size}")
    offsets = np.empty(n_cells + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = vtu_offsets

    types = np.asarray(_decode_vtk_data_array(types_array, **decode).reshape(-1), dtype=np.uint8)
    if types.size != n_cells:
        raise ValueError(f"{path}: expected {n_cells} cell types, got {types.size}")
    return VtuMesh(points=points, connectivity=connectivity, offsets=offsets, types=types)


def write_surrogate_internal_vtu(
    template: Path,
    output: Path,
    values,
    *,
    field_name: str = "p",
) -> Path:
    """Write a VTU copy whose cell field is replaced by surrogate values.

    ``foamToVTK`` writes both cell-centered and interpolated point fields. The surrogate
    predicts the OpenFOAM ``internalField`` values, i.e. cell data, so the stale point field
    with the same name is removed to avoid ParaView showing original-CFD values by accident.
    """
    tree = ET.parse(template)
    root = tree.getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"{template}: no VTK Piece element found")
    n_cells = int(piece.attrib["NumberOfCells"])
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != n_cells:
        raise ValueError(
            f"{template}: surrogate field has {array.size} values, "
            f"but VTU template has {n_cells} cells"
        )

    cell_data = piece.find("CellData")
    if cell_data is None:
        cell_data = ET.SubElement(piece, "CellData")
    cell_field = _data_array(cell_data, field_name)
    if cell_field is None:
        cell_field = ET.SubElement(cell_data, "DataArray")
    cell_field.attrib.clear()
    cell_field.attrib.update({"type": "Float32", "Name": field_name, "format": "binary"})
    cell_field.text = "\n" + _vtk_float32_payload(array) + "\n        "

    point_data = piece.find("PointData")
    point_field = _data_array(point_data, field_name) if point_data is not None else None
    if point_data is not None and point_field is not None:
        point_data.remove(point_field)

    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def _internal_vtu_template(case_dir: Path) -> Path:
    templates = sorted((case_dir / "VTK").glob(f"{case_dir.name}_*/internal.vtu"))
    if not templates:
        raise FileNotFoundError(f"no internal.vtu template under {case_dir / 'VTK'}")
    return templates[-1]


def _write_pvd_collection(path: Path, files: list[Path]) -> Path:
    root = ET.Element(
        "VTKFile",
        {"type": "Collection", "version": "0.1", "byte_order": "LittleEndian"},
    )
    collection = ET.SubElement(root, "Collection")
    for i, file in enumerate(files):
        ET.SubElement(
            collection,
            "DataSet",
            {
                "timestep": str(i),
                "group": "",
                "part": "0",
                "file": file.name,
            },
        )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def write_surrogate_vtk_cases(
    run_dir: Path,
    output_dir: Path,
    samples: pd.DataFrame,
    fields,
    *,
    field_name: str = "p",
) -> list[Path]:
    """Write one surrogate ``.vtu`` file per case in ``samples``."""
    if "case" not in samples.columns:
        raise ValueError("samples must include a 'case' column")
    matrix = np.asarray(fields, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(samples):
        raise ValueError(
            f"surrogate fields must be an (n_cases, n_cells) matrix; got {matrix.shape} "
            f"for {len(samples)} sample row(s)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_dir / "samples.csv", index=False)
    written: list[Path] = []
    for i, row in enumerate(samples.itertuples(index=False)):
        case = str(row.case)
        template = _internal_vtu_template(run_dir / case)
        output = output_dir / f"{case}.vtu"
        written.append(
            write_surrogate_internal_vtu(template, output, matrix[i], field_name=field_name)
        )
    _write_pvd_collection(output_dir / "surrogate_fields.pvd", written)
    return written


def _step_offsets(counts: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        offsets[1:] = np.cumsum(counts[:-1], dtype=np.int64)
    return offsets


def write_surrogate_vtkhdf(
    run_dir: Path,
    output: Path,
    samples: pd.DataFrame,
    fields,
    *,
    field_name: str = "p",
) -> Path:
    """Write all surrogate fields to one ParaView-readable VTKHDF time-series file."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "VTKHDF export requires h5py; install the optional dependency with "
            "`pip install 'surroforge[formats]'`."
        ) from exc

    if "case" not in samples.columns:
        raise ValueError("samples must include a 'case' column")
    matrix = np.asarray(fields, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(samples):
        raise ValueError(
            f"surrogate fields must be an (n_cases, n_cells) matrix; got {matrix.shape} "
            f"for {len(samples)} sample row(s)"
    )

    meshes: list[VtuMesh] = []
    for row in samples.itertuples(index=False):
        case = str(row.case)
        mesh = _read_vtu_mesh(_internal_vtu_template(run_dir / case))
        if matrix.shape[1] != mesh.types.size:
            raise ValueError(
                f"{case}: surrogate field has {matrix.shape[1]} values, "
                f"but VTU template has {mesh.types.size} cells"
            )
        meshes.append(mesh)

    output.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output.parent / "samples.csv", index=False)

    n_steps = len(meshes)
    point_counts = np.asarray([mesh.points.shape[0] for mesh in meshes], dtype=np.int64)
    cell_counts = np.asarray([mesh.types.size for mesh in meshes], dtype=np.int64)
    connectivity_counts = np.asarray([mesh.connectivity.size for mesh in meshes], dtype=np.int64)
    times = np.arange(n_steps, dtype=np.float32)

    with h5py.File(output, "w") as h5:
        vtk = h5.create_group("VTKHDF")
        vtk.attrs["Type"] = np.bytes_("UnstructuredGrid")
        vtk.attrs["Version"] = np.asarray([2, 5], dtype=np.int64)

        vtk.create_dataset("NumberOfPoints", data=point_counts)
        vtk.create_dataset("NumberOfCells", data=cell_counts)
        vtk.create_dataset("NumberOfConnectivityIds", data=connectivity_counts)
        vtk.create_dataset(
            "Points",
            data=np.concatenate([mesh.points for mesh in meshes], axis=0).astype(np.float32),
        )
        vtk.create_dataset(
            "Connectivity",
            data=np.concatenate([mesh.connectivity for mesh in meshes]).astype(np.int64),
        )
        vtk.create_dataset(
            "Offsets",
            data=np.concatenate([mesh.offsets for mesh in meshes]).astype(np.int64),
        )
        vtk.create_dataset(
            "Types",
            data=np.concatenate([mesh.types for mesh in meshes]).astype(np.uint8),
        )

        cell_data = vtk.create_group("CellData")
        cell_data.create_dataset(field_name, data=matrix.reshape(-1).astype(np.float32))
        vtk.create_group("PointData")
        field_data = vtk.create_group("FieldData")
        field_data.create_dataset("TimeValue", data=times)

        steps = vtk.create_group("Steps")
        steps.attrs["NSteps"] = np.int64(n_steps)
        steps.create_dataset("Values", data=times)
        steps.create_dataset("PartOffsets", data=np.zeros(n_steps, dtype=np.int64))
        steps.create_dataset("PointOffsets", data=_step_offsets(point_counts))
        steps.create_dataset("CellOffsets", data=np.zeros(n_steps, dtype=np.int64))
        steps.create_dataset("ConnectivityIdOffsets", data=_step_offsets(connectivity_counts))
        steps.create_group("PointDataOffsets")
        cell_data_offsets = steps.create_group("CellDataOffsets")
        cell_data_offsets.create_dataset(field_name, data=_step_offsets(cell_counts))
        field_data_offsets = steps.create_group("FieldDataOffsets")
        field_data_offsets.create_dataset("TimeValue", data=np.arange(n_steps, dtype=np.int64))
        field_data_sizes = steps.create_group("FieldDataSizes")
        field_data_sizes.create_dataset(
            "TimeValue", data=np.ones((n_steps, 2), dtype=np.int64)
        )

    return output


def main(run_dir: Path, out_dir: Path | None = None) -> None:
    # With --out the run is saved: the SurroForge project (datasets, runs, lineage, POD basis,
    # models) under <out_dir>/naca and the exported artifacts alongside it, so it is inspectable
    # and reproducible afterwards (surroforge --config-dir <out_dir>/naca list). Without it the
    # project lives in a temp dir and is removed, and the artifacts go into the sweep directory.
    export_dir = out_dir if out_dir is not None else run_dir
    doe = pd.read_csv(run_dir / "samples.csv").set_index("case")
    snapshots = read_openfoam_sweep(
        run_dir, field="p",
        params=lambda case: {
            "aoa": float(doe.loc[case.name, "aoa"]),
            "speed": float(doe.loc[case.name, "speed"]),
        },
    )
    # The real Cl/Cd forceCoeffs reported, per case. Align by (AoA, speed) — robust to row order
    # or a case the field reader dropped — rather than positional order.
    case_names = sorted(
        d.name for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("case_")
    )
    coeffs, unsettled = {}, []
    for name in case_names:
        aoa, speed = float(doe.loc[name, "aoa"]), float(doe.loc[name, "speed"])
        reading = read_coeffs(run_dir / name)
        coeffs[_key(aoa, speed)] = reading
        if not reading.settled:
            unsettled.append((aoa, speed, reading.spread))
    keys = [_key(a, s) for a, s in zip(snapshots["aoa"], snapshots["speed"], strict=True)]
    if any(k not in coeffs for k in keys):
        raise RuntimeError("could not match real Cl/Cd to every ingested case by (AoA, speed)")
    snapshots["cl"] = [coeffs[k].cl for k in keys]
    snapshots["cd"] = [coeffs[k].cd for k in keys]
    field_cols = [c for c in snapshots.columns if c.startswith("p_")]
    print(f"Ingested {len(snapshots)} real cases: {len(field_cols)} pressure DOFs + Cl + Cd")
    if unsettled:
        listed = ", ".join(f"({a:+.1f}°,{s:.4g}) ±{sp / 2:.3g}" for a, s, sp in unsettled)
        print(f"⚠  {len(unsettled)}/{len(case_names)} case(s) have a still-swinging C_L over the "
              f"last {CL_WINDOW} writes: {listed}\n"
              f"   Their lift had not settled — re-run those cases with more --iters.")
    print()

    with tempfile.TemporaryDirectory(prefix="surroforge-naca-") as tmp:
        csv = Path(tmp) / "snapshots.csv"
        snapshots.to_csv(csv, index=False)
        # Persist the project under <out_dir>/naca when --out is given (start it fresh on a re-run);
        # otherwise build it in the temp dir so it is removed on exit.
        if out_dir is not None:
            project_root = out_dir / "naca"
            if project_root.exists():
                shutil.rmtree(project_root)
        else:
            project_root = Path(tmp) / "naca"
        project = Project.init(project_root, name="naca0012-cfd")
        project.add_dataset(csv, name="sweep")

        split = {"test_fraction": 0.25, "seed": 1}
        validation = {"cross_validation": 3, "metrics": ["rmse", "r2", "mae"]}
        inputs = ["aoa", "speed"]

        print("=== Field surrogate: (AoA, speed) -> pressure field (POD + inner) ===")
        field_runs = {
            label: Trainer.from_dict({
                "experiment": f"field-{label}",
                "data": {"source": "sweep", "inputs": inputs, "outputs": ["p_*"], "split": split},
                "model": {"method": "reduced-field",
                          "reduction": {"basis": "pod", "energy": 0.999}, "inner": inner},
                "validation": validation,
            }).fit(project)
            for label, inner in _candidates([48, 48])
        }
        _report(project, field_runs, "inner")

        print("\n=== Scalar surrogate: (AoA, speed) -> [Cl, Cd] ===")
        cl_runs = {
            label: Trainer.from_dict({
                "experiment": f"coeff-{label}",
                "data": {"source": "sweep", "inputs": inputs, "outputs": ["cl", "cd"],
                         "split": split},
                "model": model,
                "validation": validation,
            }).fit(project)
            for label, model in _candidates([48, 48])
        }
        best_cl = cl_runs[_report(project, cl_runs, "method")]

        # Export into the (persistent) sweep directory, one artifact per format: the best
        # coefficient model to ONNX, the torch-mlp coefficient model to a native PyTorch .pt
        # (when the plugin is installed), and the field surrogate to pickle (reduced-field only).
        print("\n=== Export ===")
        try:
            out = best_cl.export("onnx", export_dir / "coeffs_surrogate.onnx")
            print(f"  onnx  -> {out.name} ({out.stat().st_size} bytes)  [best: {best_cl.method}]")
        except ExportError as exc:
            print(f"  onnx  unavailable — {exc}")
        torch_run = cl_runs.get("torch-mlp")
        if torch_run is not None:
            out = torch_run.export("pt", export_dir / "coeffs_surrogate.pt")
            print(f"  pt    -> {out.name} ({out.stat().st_size} bytes)  [torch-mlp native]")
        else:
            print("  pt    skipped — install ./plugins/surroforge-torch for the torch-mlp backend")
        field_best = field_runs[_best_label(project, field_runs)]
        field_out = field_best.export("pickle", export_dir / "field_surrogate.pkl")
        print(f"  field surrogate -> {field_out.name} (pickle; reduced-field is pickle-only)")

        print("\n=== Surrogate field VTK ===")
        prediction_cases = doe.loc[case_names, ["aoa", "speed"]].reset_index()
        predicted_fields = field_best.load_model().predict(
            prediction_cases[inputs].to_numpy(dtype=float)
        )
        vtk_dir = export_dir / "surrogate_vtk"
        vtkhdf = write_surrogate_vtkhdf(
            run_dir,
            vtk_dir / "surrogate_fields.vtkhdf",
            prediction_cases,
            predicted_fields,
        )
        print(
            f"  surrogate p fields -> {vtkhdf} "
            f"({len(prediction_cases)} time steps + samples.csv)"
        )
        if out_dir is not None:
            print(f"\nArtifacts written to {export_dir}/")
            print(f"Run saved to {project_root}/ — inspect with: "
                  f"surroforge --config-dir {project_root} list")


def _best_label(project: Project, runs: dict) -> str:
    result = compare_runs(project, [r.run_id for r in runs.values()], metric="rmse")
    return next(label for label, r in runs.items() if r.run_id == result["best"])


def _report(project: Project, runs: dict, label_kind: str) -> str:
    label_by_id = {r.run_id: name for name, r in runs.items()}
    result = compare_runs(project, list(label_by_id), metric="rmse")
    print(f"{label_kind:<12}{'test R²':>10}{'test RMSE':>12}")
    for row in result["rows"]:
        marker = "  <- best" if row["run_id"] == result["best"] else ""
        print(f"{label_by_id[row['run_id']]:<12}{row['test']['r2']:>10.4f}"
              f"{row['test']['rmse']:>12.4g}{marker}")
    best_row = next(r for r in result["rows"] if r["run_id"] == result["best"])
    if best_row["test"]["r2"] < 0:
        print("  (all test R² < 0 — data-starved sweep; '<- best' is lowest-RMSE only. "
              "Use more angles and --iters; see this directory's README.)")
    return label_by_id[result["best"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default="run",
                        help="sweep directory run_sweep.py produced (samples.csv + case_* dirs)")
    parser.add_argument(
        "--out", metavar="DIR", default=None,
        help="save the run to DIR: the SurroForge project (DIR/naca — runs, lineage, POD "
             "basis, models) plus the exported artifacts; without it the project goes to a "
             "temp dir and is removed, and only the artifacts (in the sweep dir) persist.",
    )
    args = parser.parse_args()
    target = Path(args.run_dir)
    if not (target / "samples.csv").is_file():
        sys.exit(f"No sweep found at {target}/ (expected samples.csv). Run run_sweep.py first.")
    out_dir = Path(args.out) if args.out else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    main(target, out_dir)
