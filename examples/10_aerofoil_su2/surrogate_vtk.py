"""VTK/VTKHDF I/O for surrogate fields on an SU2 sweep.

Standalone helper for ``ingest_and_train.py`` — no SurroForge imports. It reads the
``flow.vtu`` files SU2 writes per case (``run_sweep.py``) and writes the
surrogate-predicted **point** fields of all cases into a single ParaView-readable
VTKHDF (UnstructuredGrid, format version 2.5) time-series file — one time step per
case — plus a ``samples.csv`` mapping time step -> case/parameters next to it.

Unlike example 09 (OpenFOAM cell data via ``foamToVTK``), SU2's finite-volume scheme
is node-centred: the solution lives on the mesh points, so fields are read from and
written to ``PointData``. The VTU parser here additionally understands the
``appended``/raw-binary encoding SU2 emits (which trips up ``meshio``), next to the
inline-base64 and ascii encodings example 09's parser handled.

Public API: ``read_point_field`` (used at ingest) and ``write_surrogate_vtkhdf``.
The VTKHDF write needs ``h5py`` (imported lazily); install it with
``pip install 'surroforge[formats]'``.
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


class _VtuFile(NamedTuple):
    root: ET.Element
    appended: bytes  # raw appended-data blob (b"" if the file has none)


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


def _parse_vtu(path: Path) -> _VtuFile:
    """Parse a .vtu file, splitting off any raw appended-data blob first.

    With ``<AppendedData encoding="raw">`` the payload is raw binary embedded in the
    XML document (after a leading ``_``), which no XML parser accepts — so the blob is
    cut out before parsing and kept for offset-based lookup.
    """
    raw = path.read_bytes()
    marker = raw.find(b"<AppendedData")
    if marker < 0:
        return _VtuFile(ET.fromstring(raw), b"")
    tag_end = raw.index(b">", marker)
    underscore = raw.index(b"_", tag_end)
    end = raw.rfind(b"</AppendedData>")
    xml_bytes = raw[:marker] + b"</VTKFile>"
    encoding = b'encoding="base64"' in raw[marker:tag_end]
    blob = raw[underscore + 1 : end]
    if encoding:
        blob = base64.b64decode(bytes(blob.split()))
    return _VtuFile(ET.fromstring(xml_bytes), blob)


def _decode_vtk_data_array(
    data_array: ET.Element,
    vtu: _VtuFile,
) -> np.ndarray:
    byte_order = vtu.root.attrib.get("byte_order", "LittleEndian")
    header_type = vtu.root.attrib.get("header_type", "UInt32")
    dtype = _vtk_dtype(data_array.attrib["type"], byte_order)
    header_dtype = _vtk_dtype(header_type, byte_order)
    fmt = data_array.attrib.get("format", "ascii")

    if fmt == "appended":
        if not vtu.appended:
            raise ValueError("DataArray is 'appended' but the file has no AppendedData")
        offset = int(data_array.attrib.get("offset", "0"))
        n_bytes = int(np.frombuffer(vtu.appended[offset : offset + header_dtype.itemsize],
                                    dtype=header_dtype, count=1)[0])
        start = offset + header_dtype.itemsize
        data = np.frombuffer(vtu.appended[start : start + n_bytes], dtype=dtype)
    elif fmt == "binary":
        raw = base64.b64decode("".join((data_array.text or "").split()))
        header_nbytes = header_dtype.itemsize
        n_bytes = int(np.frombuffer(raw[:header_nbytes], dtype=header_dtype, count=1)[0])
        data = np.frombuffer(raw[header_nbytes : header_nbytes + n_bytes], dtype=dtype)
    elif fmt == "ascii":
        data = np.array((data_array.text or "").split(), dtype=dtype)
    else:
        raise ValueError(f"unsupported VTK DataArray format {fmt!r}")

    components = int(data_array.attrib.get("NumberOfComponents", "1"))
    if components > 1:
        if data.size % components:
            raise ValueError("VTK DataArray length is not divisible by NumberOfComponents")
        data = data.reshape((-1, components))
    return data


def read_point_field(path: Path, name: str) -> np.ndarray:
    """Read a named PointData array from a .vtu file, flattened cell-major.

    Scalars come back as ``(n_points,)``; vectors as ``(n_points * n_components,)``
    laid out ``[x0 y0 z0 x1 …]`` — the column order ``ingest_and_train.py`` builds its
    wide snapshot table in (matching example 09's ``read_openfoam_sweep`` layout).
    """
    vtu = _parse_vtu(Path(path))
    point_data = vtu.root.find(".//Piece/PointData")
    if point_data is None:
        raise ValueError(f"{path}: no PointData element found")
    array = _data_array(point_data, name)
    if array is None:
        available = [c.attrib.get("Name") for c in point_data.findall("DataArray")]
        raise ValueError(f"{path}: no PointData array {name!r} (available: {available})")
    return np.asarray(_decode_vtk_data_array(array, vtu), dtype=np.float64).reshape(-1)


def _read_vtu_mesh(path: Path) -> VtuMesh:
    vtu = _parse_vtu(Path(path))
    piece = vtu.root.find(".//Piece")
    if piece is None:
        raise ValueError(f"{path}: no VTK Piece element found")

    n_points = int(piece.attrib["NumberOfPoints"])
    n_cells = int(piece.attrib["NumberOfCells"])

    points_array = piece.find("Points/DataArray")
    cells = piece.find("Cells")
    if points_array is None or cells is None:
        raise ValueError(f"{path}: VTU file must contain Points and Cells")
    connectivity_array = _data_array(cells, "connectivity")
    offsets_array = _data_array(cells, "offsets")
    types_array = _data_array(cells, "types")
    if connectivity_array is None or offsets_array is None or types_array is None:
        raise ValueError(f"{path}: VTU Cells must contain connectivity, offsets and types arrays")

    points = np.asarray(_decode_vtk_data_array(points_array, vtu), dtype=np.float32)
    if points.ndim == 1:
        if points.size % 3:
            raise ValueError(f"{path}: point coordinate array is not divisible by 3")
        points = points.reshape((-1, 3))
    if points.shape != (n_points, 3):
        raise ValueError(f"{path}: expected {n_points} 3D points, got shape {points.shape}")

    connectivity = np.asarray(
        _decode_vtk_data_array(connectivity_array, vtu).reshape(-1), dtype=np.int64
    )
    vtu_offsets = np.asarray(
        _decode_vtk_data_array(offsets_array, vtu).reshape(-1), dtype=np.int64
    )
    if vtu_offsets.size != n_cells:
        raise ValueError(f"{path}: expected {n_cells} cell offsets, got {vtu_offsets.size}")
    offsets = np.empty(n_cells + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = vtu_offsets

    types = np.asarray(_decode_vtk_data_array(types_array, vtu).reshape(-1), dtype=np.uint8)
    if types.size != n_cells:
        raise ValueError(f"{path}: expected {n_cells} cell types, got {types.size}")
    return VtuMesh(points=points, connectivity=connectivity, offsets=offsets, types=types)


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
    """Write surrogate point fields of all cases to one ParaView-readable VTKHDF file.

    ``fields`` maps each field name to an ``(n_cases, n_points * n_components)`` matrix,
    vector components laid out point-major (``[x0 y0 z0 x1 …]``) — exactly the column
    order ``read_point_field`` produces. Scalar fields (1 component) are written flat
    under ``PointData/<name>``; vector fields (3 components) as an
    ``(n_steps·n_points, 3)`` dataset, which ParaView reads as a vector array
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
        mesh = _read_vtu_mesh(run_dir / case / "flow.vtu")
        n_points = int(mesh.points.shape[0])
        for name, matrix in matrices.items():
            n_dof = matrix.shape[1]
            if n_dof % n_points or n_dof // n_points not in (1, 3):
                raise ValueError(
                    f"{case}: field '{name}' has {n_dof} values, but flow.vtu has "
                    f"{n_points} points; expected n_points (scalar) or 3*n_points "
                    "(vector) values"
                )
            inferred = n_dof // n_points
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

        point_data = vtk.create_group("PointData")
        for name, matrix in matrices.items():
            if components[name] == 1:
                point_data.create_dataset(name, data=matrix.reshape(-1))
            else:
                point_data.create_dataset(name, data=matrix.reshape(-1, 3))
        vtk.create_group("CellData")
        field_data = vtk.create_group("FieldData")
        field_data.create_dataset("TimeValue", data=times)

        steps = vtk.create_group("Steps")
        steps.attrs["NSteps"] = np.int64(n_steps)
        steps.create_dataset("Values", data=times)
        steps.create_dataset("PartOffsets", data=np.zeros(n_steps, dtype=np.int64))
        steps.create_dataset("PointOffsets", data=_step_offsets(point_counts))
        steps.create_dataset("CellOffsets", data=np.zeros(n_steps, dtype=np.int64))
        steps.create_dataset("ConnectivityIdOffsets", data=_step_offsets(connectivity_counts))
        point_data_offsets = steps.create_group("PointDataOffsets")
        for name in matrices:
            # Offsets count rows (points) for scalar and vector datasets alike.
            point_data_offsets.create_dataset(name, data=_step_offsets(point_counts))
        steps.create_group("CellDataOffsets")
        field_data_offsets = steps.create_group("FieldDataOffsets")
        field_data_offsets.create_dataset("TimeValue", data=np.arange(n_steps, dtype=np.int64))
        field_data_sizes = steps.create_group("FieldDataSizes")
        field_data_sizes.create_dataset(
            "TimeValue", data=np.ones((n_steps, 2), dtype=np.int64)
        )

    return output
