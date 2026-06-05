"""Trainer interface shared by built-in methods and third-party plugins (PRD 4.11).

A trainer is constructed with method-specific hyperparameters, fitted on numeric
arrays, and must be picklable so it can be persisted as the run artifact. Export
capability is advertised via :meth:`supports_export`, and a method declares the
numerical tolerance within which it reproduces via :attr:`reproduction_tolerance`
(Story 5.4) — both queried rather than hard-coded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from surroforge.errors import ExportError


class BaseTrainer(ABC):
    """Abstract base for all surrogate trainers.

    Subclasses set :attr:`name` and accept their hyperparameters as keyword
    arguments. They must keep ``__init__`` cheap and free of heavy imports so the
    registry can list methods without importing optional backends.
    """

    #: Registry name of the method (matches ``model.method`` in the config).
    name: str = "base"
    #: Export formats this method supports in addition to the universal "pickle"
    #: (every trainer is picklable). Queried by the engine and exporters via
    #: :meth:`supports_export` (Story 5.4) — declare formats here instead of
    #: relying on a hard-coded list. ONNX-capable methods add ``"onnx"``.
    export_formats: frozenset[str] = frozenset({"pickle"})
    #: Numerical band within which two independent fits on the same data count as
    #: reproduced (Story 5.4). ``0.0`` declares a deterministic method (must
    #: reproduce bit-for-bit); a positive value suits an iterative/optimiser core
    #: (e.g. SMT). ``reproduce()`` compares against the value recorded at train
    #: time, so a method's verdict follows its own declaration, not a global rule.
    reproduction_tolerance: float = 1e-9

    def __init__(self, **hyperparameters: Any) -> None:
        self.hyperparameters: dict[str, Any] = dict(hyperparameters)

    def supports_export(self, fmt: str) -> bool:
        """Whether this method can be exported to ``fmt`` (Story 5.4)."""
        return fmt in self.export_formats

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the surrogate on inputs ``x`` and outputs ``y`` (both 2-D)."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict outputs for ``x``; returns a 2-D array ``(n, n_outputs)``."""

    def to_onnx(self, n_features: int) -> bytes:
        """Serialise the fitted model to ONNX. Override in ONNX-capable trainers."""
        raise ExportError(f"Method '{self.name}' does not support ONNX export.")

    @staticmethod
    def _as_2d(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=float)
        return array.reshape(-1, 1) if array.ndim == 1 else array
