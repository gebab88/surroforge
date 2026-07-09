"""Run-store model serialisation (the ``model.pkl`` artifact).

Deliberately Core persistence, not an exporter: train writes and predict reads
the stored model regardless of which editions are installed (PRD 3.2: export is
Professional, the run store is not). The Professional ``pickle`` *exporter*
delegates to :func:`dump_model`, so a user-facing pickle export stays
byte-compatible with the run store.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from surroforge.trainers.base import BaseTrainer


def dump_model(trainer: BaseTrainer, path: Path) -> Path:
    """Write the fitted ``trainer`` to ``path`` (highest pickle protocol)."""
    path = Path(path)
    with open(path, "wb") as handle:
        pickle.dump(trainer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model(path: str | Path) -> BaseTrainer:
    """Load a trainer previously written by :func:`dump_model`."""
    with open(path, "rb") as handle:
        return pickle.load(handle)
