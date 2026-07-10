"""ONNX exporter — Professional since Wave 2 (PRD 3.2, 4.4).

Delegates to the trainer's :meth:`to_onnx`; methods that cannot be represented
in ONNX (e.g. SMT Kriging/RBF) raise a clear :class:`ExportError`.
"""

from __future__ import annotations

from pathlib import Path

from surroforge.errors import ExportError
from surroforge.exporters.base import BaseExporter
from surroforge.trainers.base import BaseTrainer


class ONNXExporter(BaseExporter):
    name = "onnx"
    extension = ".onnx"
    required_tier = "professional"

    def export(self, trainer: BaseTrainer, destination: Path, *, n_features: int) -> Path:
        if not trainer.supports_export("onnx"):
            raise ExportError(
                f"Method '{trainer.name}' cannot be exported to ONNX. "
                "Use --format pickle instead."
            )
        payload = trainer.to_onnx(n_features)
        destination = Path(destination)
        destination.write_bytes(payload)
        return destination
