"""Data loading, hashing, and column selection (PRD 4.4, 4.5)."""

from __future__ import annotations

import numpy as np
import pytest

from surroforge.data.hashing import hash_dataframe, hash_file
from surroforge.data.loader import load_table, select_columns
from surroforge.errors import DataError
from tests.conftest import make_samples


def test_hash_dataframe_is_stable_and_content_sensitive():
    a = make_samples(seed=1)
    b = make_samples(seed=1)
    assert hash_dataframe(a) == hash_dataframe(b)

    c = a.copy()
    c.loc[0, "load"] += 1e-6
    assert hash_dataframe(c) != hash_dataframe(a)


def test_hash_stable_across_reload(tmp_path):
    # Re-loading the same file must yield the same hash (the property that makes
    # the data hash usable for reproducibility checks).
    frame = make_samples()
    csv = tmp_path / "d.csv"
    frame.to_csv(csv, index=False)
    assert hash_dataframe(load_table(csv)) == hash_dataframe(load_table(csv))

    # A lossless round-trip (Parquet) preserves the hash of the in-memory frame.
    parquet = tmp_path / "d.parquet"
    frame.to_parquet(parquet, index=False)
    assert hash_dataframe(load_table(parquet)) == hash_dataframe(frame)


def test_hash_file_differs_from_content_hash(tmp_path):
    frame = make_samples()
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    assert hash_file(path) != hash_dataframe(frame)  # byte hash vs content hash


def test_select_columns_returns_2d_arrays():
    frame = make_samples()
    x, y = select_columns(frame, ["length", "width", "load"], ["max_stress"])
    assert x.shape == (len(frame), 3)
    assert y.shape == (len(frame), 1)
    assert x.dtype == np.float64


def test_missing_column_raises():
    frame = make_samples()
    with pytest.raises(DataError):
        select_columns(frame, ["length", "nope"], ["max_stress"])


def test_input_output_overlap_raises():
    frame = make_samples()
    with pytest.raises(DataError):
        select_columns(frame, ["length", "load"], ["load"])


def test_non_numeric_column_raises(tmp_path):
    frame = make_samples()
    frame["label"] = "x"
    with pytest.raises(DataError):
        select_columns(frame, ["length", "label"], ["max_stress"])


def test_unsupported_format_raises(tmp_path):
    bad = tmp_path / "data.txt"
    bad.write_text("nope")
    with pytest.raises(DataError):
        load_table(bad)
