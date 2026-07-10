"""Pickle exporter — Professional since Wave 2 (PRD 3.2, 4.4).

Delegates to Core's run-store serialisation (``persistence.model_io``), so a
user-facing pickle export stays byte-compatible with the ``model.pkl`` the run
store writes at train time. The gate sits in registry resolution (W1 seam);
the class itself carries no licence logic.
"""

from __future__ import annotations

from pathlib import Path

from surroforge.exporters.base import BaseExporter
from surroforge.persistence.model_io import dump_model
from surroforge.trainers.base import BaseTrainer


class PickleExporter(BaseExporter):
    name = "pickle"
    extension = ".pkl"
    required_tier = "professional"

    def export(self, trainer: BaseTrainer, destination: Path, *, n_features: int) -> Path:
        return dump_model(trainer, Path(destination))
