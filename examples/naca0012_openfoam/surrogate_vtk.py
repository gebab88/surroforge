"""VTK/VTKHDF I/O for surrogate fields on an OpenFOAM sweep.

Standalone helper for ``ingest_and_train.py`` — no SurroForge imports. It reads the
``internal.vtu`` mesh that ``foamToVTK`` wrote for each case (``run_sweep.py --vtk``) and
writes the surrogate-predicted cell fields of **all** cases into a single ParaView-readable
VTKHDF (UnstructuredGrid, format version 2.5) time-series file — one time step per case —
plus a ``samples.csv`` mapping time step -> case/parameters next to it.

Only ``write_surrogate_vtkhdf`` is public. The VTKHDF write needs ``h5py`` (imported
lazily); install it with ``pip install 'surroforge[formats]'``.
"""

from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


class VtuMesh(NamedTuple):
    points: np.ndarray
    connectivity: np.ndarray
    offsets: np.ndarray
    types: np.ndarray


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


def _internal_vtu_template(case_dir: Path) -> Path:
    templates = sorted((case_dir / "VTK").glob(f"{case_dir.name}_*/internal.vtu"))
    if not templates:
        raise FileNotFoundError(f"no internal.vtu template under {case_dir / 'VTK'}")
    return templates[-1]


def _step_offsets(counts: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        offsets[1:] = np.cumsum(counts[:-1], dtype=np.int64)
    return offsets


def write_surrogate_vtkhdf(
    run_dir: Path,
    output: Path,
    samples: pd.DataFrame,
    fields: Mapping[str, np.ndarray],
) -> Path:
    """Write surrogate cell fields of all cases to one ParaView-readable VTKHDF file.

    ``fields`` maps each field name to an ``(n_cases, n_cells * n_components)`` matrix,
    vector components laid out cell-major (``[x0 y0 z0 x1 …]``) — exactly the column
    order ``read_openfoam_sweep`` produces. Scalar fields (1 component) are written
    flat under ``CellData/<name>``; vector fields (3 components) as an
    ``(n_steps·n_cells, 3)`` dataset, which ParaView reads as a vector array
    (magnitude colouring, glyphs, streamlines).
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "VTKHDF export requires h5py; install the optional dependency with "
            "`pip install 'surroforge[formats]'`."
        ) from exc

    if "case" not in samples.columns:
        raise ValueError("samples must include a 'case' column")
    if not fields:
        raise ValueError("fields must contain at least one named field matrix")
    matrices: dict[str, np.ndarray] = {}
    for name, values in fields.items():
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(samples):
            raise ValueError(
                f"field '{name}' must be an (n_cases, n_dof) matrix; got {matrix.shape} "
                f"for {len(samples)} sample row(s)"
            )
        matrices[name] = matrix

    components: dict[str, int] = {}
    meshes: list[VtuMesh] = []
    for row in samples.itertuples(index=False):
        case = str(row.case)
        mesh = _read_vtu_mesh(_internal_vtu_template(run_dir / case))
        n_cells = int(mesh.types.size)
        for name, matrix in matrices.items():
            n_dof = matrix.shape[1]
            if n_dof % n_cells or n_dof // n_cells not in (1, 3):
                raise ValueError(
                    f"{case}: field '{name}' has {n_dof} values, but the VTU template "
                    f"has {n_cells} cells; expected n_cells (scalar) or 3*n_cells "
                    "(vector) values"
                )
            inferred = n_dof // n_cells
            if components.setdefault(name, inferred) != inferred:
                raise ValueError(
                    f"{case}: field '{name}' implies {inferred} component(s) on this "
                    f"mesh, but an earlier case implied {components[name]}"
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
        for name, matrix in matrices.items():
            if components[name] == 1:
                cell_data.create_dataset(name, data=matrix.reshape(-1))
            else:
                cell_data.create_dataset(name, data=matrix.reshape(-1, 3))
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
        for name in matrices:
            # Offsets count rows (cells) for scalar and vector datasets alike.
            cell_data_offsets.create_dataset(name, data=_step_offsets(cell_counts))
        field_data_offsets = steps.create_group("FieldDataOffsets")
        field_data_offsets.create_dataset("TimeValue", data=np.arange(n_steps, dtype=np.int64))
        field_data_sizes = steps.create_group("FieldDataSizes")
        field_data_sizes.create_dataset(
            "TimeValue", data=np.ones((n_steps, 2), dtype=np.int64)
        )

    return output
