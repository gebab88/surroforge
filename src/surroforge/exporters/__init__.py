"""Exporter interface. The concrete exporters (pickle/onnx/pt) are Professional (Wave 2)."""

from surroforge.exporters.base import BaseExporter

__all__ = ["BaseExporter"]
