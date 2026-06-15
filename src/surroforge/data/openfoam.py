"""OpenFOAM field ingestion — solver-native, mesh-free (Story R5.1, FR-R1 extension).

A field is "columns that move together": an OpenFOAM parametric study is one case
directory per DOE point, each holding a converged field (``p``, ``T``, ``U``, …) as a
per-cell ``internalField`` list. This module reads those ASCII fields straight into the
wide ``(n_samples × n_dof)`` snapshot matrix the unchanged ``reduced-field`` pipeline
trains on — reading **only** field values, never coordinates or connectivity (the mesh-free
invariant, AR-R5).

ASCII parsing is pure-Python (stdlib :mod:`re`) and adds **no new dependency** — Core stays
lean and permissively licensed. Binary OpenFOAM format and any ``meshio``/``pyvista``-backed
reader are deliberately deferred to the (currently empty) ``surroforge[rom]`` extra, to be
added later behind the same lazy-import + install-hint pattern as ``[smt]``/``[onnx]``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

from surroforge.errors import DataError

if TYPE_CHECKING:
    from collections.abc import Callable

    # Per-case parameters: a {case_name: {param: value}} mapping, or a callable resolving a
    # case directory to its parameters. (Never evaluated at runtime — annotations are strings.)
    ParamSource = Mapping[str, Mapping[str, float]] | Callable[[Path], Mapping[str, float]] | None

# A scalar field is one value per cell; a vector field flattens to three components per cell,
# laid out **cell-major**: [x0 y0 z0 x1 y1 z1 …]. The column namer mirrors this interleave.
_KIND_COMPONENTS = {"scalar": 1, "vector": 3}
_VECTOR_AXES = ("x", "y", "z")

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_LIST_HEADER = re.compile(
    r"\binternalField\b\s+nonuniform\s+List<(?P<kind>scalar|vector)>\s*(?P<n>\d+)?\s*\(",
)
_UNIFORM = re.compile(r"\binternalField\b\s+uniform\s+(?P<rest>[^;]+);", re.DOTALL)


class _Field(NamedTuple):
    values: np.ndarray  # flat, length n_cells * n_components, cell-major
    n_cells: int
    n_components: int  # 1 (scalar) or 3 (vector)


def read_openfoam_field(path: str | Path) -> np.ndarray:
    """Read an OpenFOAM ``internalField`` into a flat 1-D ``np.ndarray``.

    A ``volScalarField`` returns ``n_cells`` values; a ``volVectorField`` flattens to
    ``3 * n_cells`` values in **cell-major** order (``[x0 y0 z0 x1 …]``). Both
    ``nonuniform List<…>`` and ``uniform <value>`` forms are supported. Raises
    :class:`~surroforge.errors.DataError`, naming the file, on any malformation.
    """
    return _read_field(Path(path)).values


def read_openfoam_dimensioned_scalar(path: str | Path, keyword: str) -> float:
    """Recover a dimensioned scalar (e.g. ``nu [0 2 -1 0 0 0 0] 1.5e-05;``) from a dict.

    Lets a sweep parameter come straight from a case dictionary (``constant/
    transportProperties``, …) instead of only a DoE table. Raises :class:`DataError`
    if the file is absent or the keyword is not found.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"OpenFOAM dictionary not found: {path}")
    text = _strip_comments(path.read_text())
    match = re.search(
        rf"\b{re.escape(keyword)}\b[^;]*?(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*;",
        text,
    )
    if match is None:
        raise DataError(f"Keyword '{keyword}' not found in {path}")
    return float(match.group("value"))


def latest_time_field(case_dir: str | Path, field: str) -> Path:
    """Path to ``field`` in the highest-numbered time directory of an OpenFOAM case."""
    case_dir = Path(case_dir)
    times = [d for d in case_dir.iterdir() if d.is_dir() and _is_time_dir(d.name)]
    if not times:
        raise DataError(f"No numeric time directories in OpenFOAM case: {case_dir}")
    latest = max(times, key=lambda d: float(d.name))
    field_path = latest / field
    if not field_path.is_file():
        raise DataError(f"Field '{field}' not found in latest time directory {latest}")
    return field_path


def read_openfoam_sweep(
    case_root: str | Path,
    *,
    field: str,
    params: ParamSource = None,
    time: str | None = None,
) -> pd.DataFrame:
    """Ingest an OpenFOAM case sweep into a wide snapshot DataFrame.

    ``case_root`` holds one sub-directory per DOE point. For each case, ``field`` is read
    (from ``time`` if given, else the **highest-numbered** time directory — the converged
    solution; the reader never falls back to an earlier time) and laid out as a column group
    ``<field>_0000 … <field>_NNNN`` (vector components suffixed ``_x``/``_y``/``_z``).
    ``params`` attaches parameter columns: either a ``{case_name: {param: value}}`` mapping
    or a callable ``case_dir -> {param: value}``; every case must yield the same parameter keys.

    The result has a **deterministic** layout — rows sorted by case name, parameter columns
    first (first-case key order) then the field group in native DOF order — so the registered
    dataset hash is reproducible (``hash_dataframe`` is column- and row-order-sensitive).
    Raises :class:`DataError`, naming the offending case, if DOF counts or parameter keys
    disagree across cases.
    """
    root = Path(case_root)
    if not root.is_dir():
        raise DataError(f"OpenFOAM case root not found: {root}")
    cases = sorted(
        (d for d in root.iterdir() if d.is_dir() and _has_field(d, field, time)),
        key=lambda d: d.name,
    )
    if not cases:
        case_dirs = [d for d in root.iterdir() if d.is_dir()]
        if case_dirs:
            where = f"at time '{time}'" if time is not None else "in its latest time directory"
            raise DataError(
                f"No case under {root} has field '{field}' {where} (checked {len(case_dirs)} "
                "case dir(s)). The field may exist only in earlier time directories, or the "
                "field name may be wrong."
            )
        raise DataError(f"No case directories found under {root}.")

    columns: list[str] | None = None
    field_rows: list[np.ndarray] = []
    param_rows: list[Mapping[str, float]] = []
    for case in cases:
        field_path = _timed_field(case, time, field) if time is not None else latest_time_field(
            case, field
        )
        data = _read_field(field_path)
        if columns is None:
            columns = _field_columns(field, data.n_cells, data.n_components)
        elif data.values.size != len(columns):
            raise DataError(
                f"Case '{case.name}': field '{field}' has {data.values.size} DOF, but "
                f"'{cases[0].name}' has {len(columns)}. All cases must share one DOF count."
            )
        field_rows.append(data.values)
        param_rows.append(_resolve_params(params, case) if params is not None else {})

    assert columns is not None  # cases is non-empty by the guard above
    # Every case must declare the same parameter keys — otherwise a key present only in a
    # non-first case would be silently dropped (taking columns from the first case alone).
    expected_keys = set(param_rows[0])
    for case, row in zip(cases, param_rows, strict=True):
        if set(row) != expected_keys:
            raise DataError(
                f"Case '{case.name}' has parameter keys {sorted(row)}, but '{cases[0].name}' "
                f"has {sorted(expected_keys)}. All cases must share the same parameters."
            )
    frame: dict[str, list[float] | np.ndarray] = {}
    for name in param_rows[0]:
        frame[name] = [float(row[name]) for row in param_rows]
    matrix = np.asarray(field_rows, dtype=float)
    for index, column in enumerate(columns):
        frame[column] = matrix[:, index]
    return pd.DataFrame(frame)


def read_mesh_field(path: str | Path, field: str, *, location: str = "point") -> np.ndarray:
    """Read a named field from a binary/mesh file via ``meshio`` as a flat array.

    Handles any format :mod:`meshio` supports (VTK/VTU/XDMF, binary OpenFOAM, …) — the
    deferred Phase-2 binary path. ``location='point'`` reads ``mesh.point_data[field]``;
    ``location='cell'`` concatenates ``mesh.cell_data[field]`` across cell blocks. Vector
    fields flatten cell-major (``[x0 y0 z0 …]``), matching :func:`read_openfoam_field`.

    ``meshio`` is imported lazily; if it is not installed this raises a :class:`DataError`
    with a ``pip install 'surroforge[rom]'`` hint (mirroring ``[smt]``/``[onnx]``).
    """
    meshio = _require_meshio()
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Mesh field file not found: {path}")
    try:
        mesh = meshio.read(str(path))
    except Exception as exc:  # meshio raises a variety of reader-specific errors
        raise DataError(f"{path}: meshio could not read this file: {exc}") from exc

    if location == "point":
        store = mesh.point_data
    elif location == "cell":
        store = mesh.cell_data
    else:
        raise DataError(f"location must be 'point' or 'cell', got {location!r}")
    if field not in store:
        raise DataError(f"{path}: field '{field}' not found; available: {sorted(store)}")

    raw = store[field]
    try:
        if location == "cell":
            # cell_data[field] is one array per cell block — concatenate in block order.
            if not list(raw):
                raise DataError(f"{path}: field '{field}' has no cell-data blocks")
            array = np.concatenate([np.asarray(block, dtype=float) for block in raw])
        else:
            array = np.asarray(raw, dtype=float)
    except (ValueError, TypeError) as exc:
        raise DataError(f"{path}: could not read field '{field}' as a float array: {exc}") from exc
    # Only scalar (1 component) or vector (3) fields map to a DOF column group — reject
    # tensors etc. rather than silently flattening them (mirrors read_openfoam_field).
    components = 1 if array.ndim == 1 else array.shape[-1]
    if array.ndim > 2 or components not in (1, 3):
        raise DataError(
            f"{path}: field '{field}' has shape {array.shape}; only scalar (n,) or vector "
            "(n, 3) fields are supported"
        )
    values = array.reshape(-1)  # (n, 3) -> cell-major [x0 y0 z0 …]; (n,) -> as-is
    if values.size == 0:
        raise DataError(f"{path}: field '{field}' is empty")
    if not np.all(np.isfinite(values)):
        raise DataError(f"{path}: non-finite (nan/inf) value in field '{field}'")
    return values


def _require_meshio():
    try:
        import meshio  # lazy: keeps meshio out of the base install
    except ImportError as exc:
        raise DataError(
            "Reading binary/mesh field files requires the 'meshio' backend. "
            "Install it with: pip install 'surroforge[rom]'"
        ) from exc
    return meshio


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _strip_comments(text: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))


def _matching_paren(text: str, open_idx: int) -> int:
    """Index of the ``)`` closing the ``(`` at ``open_idx`` (handles nested vector parens)."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced parentheses")


def _floats(blob: str, path: Path) -> np.ndarray:
    """Parse every number in ``blob`` (parens already removed) into a finite float array."""
    try:
        values = np.asarray(blob.replace("(", " ").replace(")", " ").split(), dtype=float)
    except ValueError as exc:
        raise DataError(f"{path}: non-numeric value in internalField") from exc
    if values.size and not np.all(np.isfinite(values)):
        # A diverged solver dump can write nan/inf, which float() accepts silently — refuse it
        # rather than poison the snapshot matrix / POD downstream.
        raise DataError(f"{path}: non-finite (nan/inf) value in internalField")
    return values


def _read_field(path: Path) -> _Field:
    if not path.is_file():
        raise DataError(f"OpenFOAM field file not found: {path}")
    text = _strip_comments(path.read_text())

    header = _LIST_HEADER.search(text)
    if header is not None:
        kind = header.group("kind")
        n_components = _KIND_COMPONENTS[kind]
        open_idx = header.end() - 1  # the '(' captured at the end of the pattern
        try:
            close_idx = _matching_paren(text, open_idx)
        except ValueError as exc:
            raise DataError(f"{path}: unbalanced parentheses in internalField list") from exc
        body = text[open_idx + 1 : close_idx]
        declared = None if header.group("n") is None else int(header.group("n"))
        if n_components == 1:
            values = _floats(body, path)
            if values.size == 0:
                raise DataError(f"{path}: internalField list is empty")
            n_cells = values.size
        else:
            # Parse per-cell paren groups so a ragged vector (e.g. `(1 2 3)(4 5)`) is rejected,
            # not silently re-grouped by a blind flat parse.
            groups = re.findall(r"\(([^()]*)\)", body)
            if not groups:
                raise DataError(f"{path}: internalField list is empty")
            cells = []
            for group in groups:
                components = _floats(group, path)
                if components.size != n_components:
                    raise DataError(
                        f"{path}: vector entry has {components.size} component(s), "
                        f"expected {n_components}"
                    )
                cells.append(components)
            values = np.concatenate(cells)
            n_cells = len(groups)
        if declared is not None and n_cells != declared:
            raise DataError(f"{path}: declared {declared} {kind}(s) but parsed {n_cells}")
        return _Field(values=values, n_cells=n_cells, n_components=n_components)

    uniform = _UNIFORM.search(text)
    if uniform is not None:
        values = _floats(uniform.group("rest"), path)
        if values.size not in (1, 3):
            raise DataError(
                f"{path}: unsupported uniform field with {values.size} component(s) "
                "(only scalar and vector are supported)"
            )
        return _Field(values=values, n_cells=1, n_components=int(values.size))

    # An internalField of a recognised-but-unsupported element type (symmTensor, tensor, …)
    # would otherwise fall through to a misleading "no internalField found".
    unsupported = re.search(r"\binternalField\b\s+nonuniform\s+List<(?P<kind>\w+)>", text)
    if unsupported is not None:
        raise DataError(
            f"{path}: unsupported field type '{unsupported.group('kind')}'; "
            "only scalar and vector are supported"
        )
    raise DataError(f"{path}: no internalField found")


def _field_columns(field: str, n_cells: int, n_components: int) -> list[str]:
    # Pad to the actual cell count (≥4) so names stay lexically sortable past 10k DOF.
    width = max(4, len(str(n_cells - 1))) if n_cells > 0 else 4
    if n_components == 1:
        return [f"{field}_{cell:0{width}d}" for cell in range(n_cells)]
    return [
        f"{field}_{cell:0{width}d}_{_VECTOR_AXES[axis]}"
        for cell in range(n_cells)
        for axis in range(n_components)
    ]


def _is_time_dir(name: str) -> bool:
    try:
        # Reject non-finite names (`nan`/`inf`): float() accepts them but they would corrupt the
        # max() that picks the latest time.
        return math.isfinite(float(name))
    except ValueError:
        return False


def _timed_field(case: Path, time: str, field: str) -> Path:
    field_path = case / time / field
    if not field_path.is_file():
        raise DataError(f"Field '{field}' not found at time '{time}' in {case}")
    return field_path


def _has_field(case: Path, field: str, time: str | None) -> bool:
    if time is not None:
        return (case / time / field).is_file()
    try:
        latest_time_field(case, field)
    except DataError:
        return False
    return True


def _resolve_params(params: object, case: Path) -> Mapping[str, float]:
    if callable(params):
        try:
            result = params(case)
        except DataError:
            raise  # already a named, actionable domain error
        except Exception as exc:  # surface any callback failure as a named, case-tagged error
            raise DataError(f"params callable failed for case '{case.name}': {exc}") from exc
        if not isinstance(result, Mapping):
            raise DataError(
                f"params callable returned {type(result).__name__} for case '{case.name}'; "
                "expected a mapping {param: value}."
            )
        return result
    if isinstance(params, Mapping):
        try:
            return params[case.name]
        except KeyError:
            raise DataError(f"No parameters provided for case '{case.name}'.") from None
    raise DataError("`params` must be a mapping {case: params} or a callable(case_dir).")
