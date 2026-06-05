"""Kriging (Gaussian-process interpolation) via SMT (PRD 4.6).

SMT is an optional dependency: the import lives inside :meth:`fit` so the method
appears in the registry even when SMT is absent, failing with a clear message
only when actually used. SMT's Kriging is single-output in the MVP.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from surroforge.errors import TrainerError
from surroforge.trainers.base import BaseTrainer

_SUPPORTED = {"corr", "poly", "theta0", "nugget"}


class KrigingTrainer(BaseTrainer):
    """Ordinary Kriging surrogate.

    Hyperparameters (all optional, passed through to SMT):
        corr (str): correlation kernel, e.g. ``squar_exp`` (default), ``abs_exp``.
        poly (str): trend, e.g. ``constant`` (default), ``linear``, ``quadratic``.
        theta0 (list[float]): initial length-scale hyperparameters.
        nugget (float): regularisation added to the diagonal.
    """

    name = "kriging"
    # SMT's optimiser converges to within a small numerical band, not bit-for-bit,
    # so kriging declares a tolerance rather than determinism (Story 5.4). It has
    # no ONNX path, so it keeps the default pickle-only export set.
    reproduction_tolerance = 1e-9

    def __init__(self, **hyperparameters: Any) -> None:
        unknown = set(hyperparameters) - _SUPPORTED
        if unknown:
            raise TrainerError(
                f"Unknown hyperparameter(s) for kriging: {sorted(unknown)}. "
                f"Supported: {sorted(_SUPPORTED)}."
            )
        super().__init__(**hyperparameters)
        self._sm: Any = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        try:
            from smt.surrogate_models import KRG
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise TrainerError(
                "Kriging requires the 'smt' package. "
                "Install it with: pip install 'surroforge[smt]'"
            ) from exc

        x = self._as_2d(x)
        y = self._as_2d(y)
        if y.shape[1] != 1:
            raise TrainerError(
                "Kriging (SMT) supports a single output in the MVP; "
                f"got {y.shape[1]} outputs. Train one model per output."
            )
        options: dict[str, Any] = {"print_global": False, **self.hyperparameters}
        self._sm = KRG(**options)
        self._sm.set_training_values(x, y[:, 0])
        self._sm.train()

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._sm is None:
            raise TrainerError("Model is not fitted.")
        preds = self._sm.predict_values(self._as_2d(x))
        return self._as_2d(preds)
