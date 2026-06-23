"""Polynomial regression via scikit-learn (PRD 4.6).

Always available (scikit-learn is a core dependency) and the only built-in
method with ONNX export, which makes it the natural smoke-test method.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from surroforge.errors import TrainerError
from surroforge.trainers.base import BaseTrainer


class PolynomialTrainer(BaseTrainer):
    """Ordinary least-squares regression on polynomial features.

    Hyperparameters:
        degree (int): polynomial degree (default 2).
        include_bias (bool): include the intercept feature (default True).
        interaction_only (bool): only interaction terms, no powers (default False).
    """

    name = "polynomial"
    # Closed-form OLS on polynomial features: deterministic, so it must reproduce
    # bit-for-bit, and it is the one built-in with ONNX export (Story 5.4).
    export_formats = frozenset({"pickle", "onnx"})
    reproduction_tolerance = 0.0

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        interaction_only: bool = False,
        **extra: Any,
    ) -> None:
        if extra:
            raise TrainerError(
                f"Unknown hyperparameter(s) for polynomial: {sorted(extra)}. "
                "Supported: degree, include_bias, interaction_only."
            )
        if not isinstance(degree, int) or isinstance(degree, bool):
            raise TrainerError(f"polynomial 'degree' must be an int >= 1; got {degree!r}.")
        if degree < 1:
            raise TrainerError("polynomial 'degree' must be >= 1.")
        if not isinstance(include_bias, bool):
            raise TrainerError(
                f"polynomial 'include_bias' must be a bool; got {include_bias!r}."
            )
        if not isinstance(interaction_only, bool):
            raise TrainerError(
                f"polynomial 'interaction_only' must be a bool; got {interaction_only!r}."
            )
        super().__init__(
            degree=degree, include_bias=include_bias, interaction_only=interaction_only
        )
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self._model: Any = None

    def _build(self) -> Any:
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures

        return make_pipeline(
            PolynomialFeatures(
                degree=self.degree,
                include_bias=self.include_bias,
                interaction_only=self.interaction_only,
            ),
            LinearRegression(),
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = self._as_2d(x)
        y = self._as_2d(y)
        self._model = self._build()
        self._model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise TrainerError("Model is not fitted.")
        preds = self._model.predict(self._as_2d(x))
        return self._as_2d(preds)

    def to_onnx(self, n_features: int) -> bytes:
        if self._model is None:
            raise TrainerError("Model is not fitted.")
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError as exc:  # pragma: no cover - depends on environment
            from surroforge.errors import ExportError

            raise ExportError(
                "ONNX export requires the 'onnx' extra. "
                "Install it with: pip install 'surroforge[onnx]'"
            ) from exc

        # skl2onnx's stub types convert_sklearn as returning a (model, topology)
        # tuple; it returns the ModelProto. cast so the .SerializeToString() call
        # type-checks against the real return.
        onnx_model = cast(Any, convert_sklearn(
            self._model,
            initial_types=[("input", FloatTensorType([None, n_features]))],
        ))
        return onnx_model.SerializeToString()
