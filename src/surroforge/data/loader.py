"""Tabular data loading and schema inference (PRD 4.4).

Supported inputs: CSV (always), Parquet and HDF5 (with the ``formats`` extra).
The loader keeps the optional dependencies lazy so the core installs lean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from surroforge.errors import DataError

SUPPORTED_SUFFIXES = {".csv", ".parquet", ".pq", ".h5", ".hdf5"}


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


def select_columns(
    frame: pd.DataFrame, inputs: list[str], outputs: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract input/output matrices, validating that all columns exist and are numeric.

    Returns ``X`` of shape ``(n, n_inputs)`` and ``y`` of shape ``(n, n_outputs)``.
    """
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
