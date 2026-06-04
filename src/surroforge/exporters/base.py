"""Exporter interface (PRD 4.4, 4.11)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from surroforge.trainers.base import BaseTrainer


class BaseExporter(ABC):
    """Serialise a fitted trainer to a file in a specific format."""

    #: Registry name / ``--format`` value.
    name: str = "base"
    #: File extension (including the dot) for the default artifact name.
    extension: str = ""

    @abstractmethod
    def export(self, trainer: BaseTrainer, destination: Path, *, n_features: int) -> Path:
        """Write ``trainer`` to ``destination`` and return the path written."""
