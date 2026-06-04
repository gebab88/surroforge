"""Polynomial regression via scikit-learn (PRD 4.6).

Always available (scikit-learn is a core dependency) and the only built-in
method with ONNX export, which makes it the natural smoke-test method.
"""

from __future__ import annotations

from typing import Any

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
    supports_onnx = True

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
        if degree < 1:
            raise TrainerError("polynomial 'degree' must be >= 1.")
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

        onnx_model = convert_sklearn(
            self._model,
            initial_types=[("input", FloatTensorType([None, n_features]))],
        )
        return onnx_model.SerializeToString()
