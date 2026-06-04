"""SHA-256 hashing for data and configurations (PRD 4.5, 5.2).

Hashes are the backbone of reproducibility: the data hash detects silent changes
to an input file, and the config hash pins the exact specification of a run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

_CHUNK = 1 << 20  # 1 MiB


def hash_file(path: str | Path) -> str:
    """SHA-256 over the raw bytes of a file, streamed to bound memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_dataframe(frame: pd.DataFrame) -> str:
    """SHA-256 over a dataframe's content (values, column names, and dtypes).

    Hashing the in-memory frame rather than the source file makes the hash
    stable across lossless re-encodings (e.g. Parquet) of the same data and
    sensitive to any change in values, columns, or types — which is what a
    reproducibility check cares about. Note that CSV is a lossy text format, so a
    CSV round-trip is not guaranteed to preserve the hash to the last ULP.
    """
    digest = hashlib.sha256()
    # Column names and dtypes, so a rename or retype changes the hash.
    for name in frame.columns:
        digest.update(str(name).encode("utf-8"))
        digest.update(str(frame[name].dtype).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(frame, index=False).values
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()
