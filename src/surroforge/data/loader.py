"""Tabular data loading and schema inference (PRD 4.4).

Supported inputs: CSV (always), Parquet and HDF5 (with the ``formats`` extra).
The loader keeps the optional dependencies lazy so the core installs lean.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd

from surroforge.errors import DataError

SUPPORTED_SUFFIXES = {".csv", ".parquet", ".pq", ".h5", ".hdf5"}

# Glob metacharacters that turn a ``data.outputs`` entry into a field pattern; an entry
# with none of these is a literal column name (full back-compat). (Story R1.1, #19/#53.)
_GLOB_CHARS = "*?["


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a tabular file into a DataFrame, dispatching on the suffix."""
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return _read_with_optional(path, "read_parquet", "parquet", "pyarrow")
    if suffix in {".h5", ".hdf5"}:
        return _read_hdf(path)
    raise DataError(
        f"Unsupported data format '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}"
    )


def _read_with_optional(path: Path, reader: str, fmt: str, extra: str) -> pd.DataFrame:
    try:
        return getattr(pd, reader)(path)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DataError(
            f"Reading {fmt} files requires the '{extra}' package. "
            f"Install it with: pip install 'surroforge[formats]'"
        ) from exc


def _read_hdf(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_hdf(path)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DataError(
            "Reading HDF5 files requires the 'h5py'/'tables' package. "
            "Install it with: pip install 'surroforge[formats]'"
        ) from exc
    if not isinstance(frame, pd.DataFrame):
        raise DataError(f"HDF5 file {path} does not contain a single table.")
    return frame


def infer_schema(frame: pd.DataFrame) -> dict[str, str]:
    """Return an ordered ``column -> dtype`` map used as the dataset schema."""
    return {str(col): str(dtype) for col, dtype in frame.dtypes.items()}


def expand_field_columns(frame: pd.DataFrame, outputs: list[str]) -> list[str]:
    """Resolve ``data.outputs`` entries into concrete column names (Story R1.1).

    An entry that exactly equals an existing column name is always a **literal** (so a
    bracketed-index DOF like ``disp[0]`` / ``stress[xx]`` is not misread as a ``[…]``
    character class). Otherwise, an entry containing a glob metacharacter (``*``, ``?``,
    ``[``) is a **field pattern** matched against the frame's columns with
    :func:`fnmatch.fnmatchcase` (case-sensitive, deterministic); a remaining plain entry is
    a literal kept verbatim (and caught downstream if absent). This is how a field is
    expressed as a column-group — a wide ``(n_samples × n_dof)`` snapshot matrix,
    **mesh-free**: only column *names* are read, never coordinates or connectivity (#19/#53).

    Ordering: literals keep the order the user wrote them (back-compat with an explicit
    list); a pattern expands to its matching columns **in the frame's column order** — the
    snapshot matrix's native DOF order, which the author controls via the file layout. (We
    deliberately do not sort by a parsed numeric suffix: that would assume ``<stem><int>``
    naming and silently reorder fields like ``stress_xx/_yy/_zz``.)

    Raises :class:`DataError` naming the offenders when a pattern matches **no** columns, or
    when a column is reached by **more than one** entry (overlapping/duplicate DOF). A
    literal that does not exist in the frame is left in place and caught by the
    missing-columns guard in :func:`select_columns`.
    """
    columns = list(frame.columns)
    column_names = {str(col) for col in columns}
    resolved: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for entry in outputs:
        if entry in column_names:
            # An exact column name is always a literal — even if it contains a glob
            # metacharacter. Scientific fields commonly index DOF as ``disp[0]`` /
            # ``stress[xx]``; treating those as a literal (not a ``[…]`` character class)
            # is what the user means, and avoids a spurious "matched no columns".
            members = [entry]
        elif any(ch in entry for ch in _GLOB_CHARS):
            matches = [str(col) for col in columns if fnmatch.fnmatchcase(str(col), entry)]
            if not matches:
                raise DataError(
                    f"Output field pattern '{entry}' matched no columns. "
                    f"Available: {[str(c) for c in columns]}"
                )
            members = matches
        else:
            members = [entry]
        for col in members:
            if col in seen:
                duplicates.append(col)
            else:
                seen.add(col)
                resolved.append(col)
    if duplicates:
        raise DataError(
            "Output columns matched by more than one outputs entry "
            f"(overlapping/duplicate field DOF): {sorted(set(duplicates))}."
        )
    return resolved


def select_columns(
    frame: pd.DataFrame, inputs: list[str], outputs: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract input/output matrices, validating that all columns exist and are numeric.

    ``outputs`` may contain field patterns (e.g. ``temp_*``); they are resolved to concrete
    columns first (:func:`expand_field_columns`, Story R1.1). Returns ``X`` of shape
    ``(n, n_inputs)`` and ``y`` of shape ``(n, n_outputs)``.
    """
    outputs = expand_field_columns(frame, outputs)
    missing = [c for c in (*inputs, *outputs) if c not in frame.columns]
    if missing:
        raise DataError(
            f"Columns not found in data: {missing}. Available: {list(frame.columns)}"
        )
    overlap = set(inputs) & set(outputs)
    if overlap:
        raise DataError(f"Columns used as both input and output: {sorted(overlap)}")

    _require_numeric(frame, [*inputs, *outputs])

    if frame[[*inputs, *outputs]].isnull().any().any():
        raise DataError("Data contains missing values in the selected columns.")

    x = frame[inputs].to_numpy(dtype=float)
    y = frame[outputs].to_numpy(dtype=float)
    return x, y


def _require_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise DataError(
            f"Non-numeric columns selected as inputs/outputs: {non_numeric}. "
            "Surrogate models require numeric data."
        )
