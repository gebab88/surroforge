"""Native PyTorch ``.pt`` exporter — part of surroforge_professional (needs the [torch] extra).

Registered through the ``surroforge.exporters`` entry point only (never Core's
``_BUILTIN`` table), so Core stays torch-free and only gains the ``pt`` format when this
package is installed. It mirrors the built-in ONNX exporter: it delegates to the
trainer's :meth:`to_pt`, so only a method that declares ``pt`` in ``export_formats``
(today, ``torch-mlp``) can be exported this way.
"""

from __future__ import annotations

from pathlib import Path

from surroforge.errors import ExportError
from surroforge.exporters.base import BaseExporter
from surroforge.trainers.base import BaseTrainer


class PtExporter(BaseExporter):
    name = "pt"
    extension = ".pt"

    def export(self, trainer: BaseTrainer, destination: Path, *, n_features: int) -> Path:
        if not trainer.supports_export("pt"):
            raise ExportError(
                f"Method '{trainer.name}' cannot be exported to a PyTorch .pt file. "
                "The 'pt' format is specific to torch-backed methods (e.g. torch-mlp); "
                "use --format pickle or onnx instead."
            )
        payload = trainer.to_pt(n_features)  # type: ignore[attr-defined]  # gated above
        destination = Path(destination)
        destination.write_bytes(payload)
        return destination
