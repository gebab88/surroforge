"""Pickle exporter — the canonical, always-available artifact (PRD 4.4)."""

from __future__ import annotations

import pickle
from pathlib import Path

from surroforge.exporters.base import BaseExporter
from surroforge.trainers.base import BaseTrainer


class PickleExporter(BaseExporter):
    name = "pickle"
    extension = ".pkl"

    def export(self, trainer: BaseTrainer, destination: Path, *, n_features: int) -> Path:
        destination = Path(destination)
        with open(destination, "wb") as handle:
            pickle.dump(trainer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return destination


def load_pickle(path: str | Path) -> BaseTrainer:
    """Load a trainer previously written by :class:`PickleExporter`."""
    with open(path, "rb") as handle:
        return pickle.load(handle)
