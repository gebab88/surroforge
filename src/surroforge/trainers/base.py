"""Trainer interface shared by built-in methods and third-party plugins (PRD 4.11).

A trainer is constructed with method-specific hyperparameters, fitted on numeric
arrays, and must be picklable so it can be persisted as the run artifact. ONNX
export is optional and advertised via :attr:`supports_onnx`.
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
    #: Whether :meth:`to_onnx` is implemented.
    supports_onnx: bool = False

    def __init__(self, **hyperparameters: Any) -> None:
        self.hyperparameters: dict[str, Any] = dict(hyperparameters)

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
