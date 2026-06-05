"""Radial Basis Function interpolation via SMT (PRD 4.6).

Like Kriging, SMT is imported lazily and the method is single-output in the MVP.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from surroforge.errors import TrainerError
from surroforge.trainers.base import BaseTrainer

_SUPPORTED = {"d0", "poly_degree", "reg"}


class RBFTrainer(BaseTrainer):
    """Radial Basis Function surrogate.

    Hyperparameters (all optional, passed through to SMT):
        d0 (float): basis-function width (default 1.0).
        poly_degree (int): degree of the added polynomial tail (-1, 0, or 1).
        reg (float): regularisation parameter.
    """

    name = "rbf"
    # Like kriging: an SMT-backed method that reproduces within a tolerance and has
    # no ONNX path, so it keeps the default pickle-only export set (Story 5.4).
    reproduction_tolerance = 1e-9

    def __init__(self, **hyperparameters: Any) -> None:
        unknown = set(hyperparameters) - _SUPPORTED
        if unknown:
            raise TrainerError(
                f"Unknown hyperparameter(s) for rbf: {sorted(unknown)}. "
                f"Supported: {sorted(_SUPPORTED)}."
            )
        super().__init__(**hyperparameters)
        self._sm: Any = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        try:
            from smt.surrogate_models import RBF
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise TrainerError(
                "RBF requires the 'smt' package. "
                "Install it with: pip install 'surroforge[smt]'"
            ) from exc

        x = self._as_2d(x)
        y = self._as_2d(y)
        if y.shape[1] != 1:
            raise TrainerError(
                "RBF (SMT) supports a single output in the MVP; "
                f"got {y.shape[1]} outputs. Train one model per output."
            )
        options: dict[str, Any] = {"print_global": False, **self.hyperparameters}
        self._sm = RBF(**options)
        self._sm.set_training_values(x, y[:, 0])
        self._sm.train()

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._sm is None:
            raise TrainerError("Model is not fitted.")
        preds = self._sm.predict_values(self._as_2d(x))
        return self._as_2d(preds)
