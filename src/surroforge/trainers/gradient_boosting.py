"""Gradient-boosting surrogate — scikit-learn HistGradientBoostingRegressor (PRD 4.6).

A tree-ensemble surrogate, the boosting counterpart to ``mlp``. Backed by scikit-learn
(already Core, BSD → NFR8), so it adds **no new dependency**. Two deliberate differences
from ``mlp``:

* **No input scaler.** Tree ensembles are scale-invariant, so the estimator is used
  directly rather than wrapped in a ``StandardScaler`` pipeline — simpler model, simpler
  ONNX graph.
* **Multi-output via a per-target wrapper.** ``HistGradientBoostingRegressor`` is
  single-output, so >1 target is fitted with ``MultiOutputRegressor`` (one independent
  booster per column); a single target uses the bare regressor on a 1-D vector (avoiding
  sklearn's ``DataConversionWarning``).

``early_stopping`` is disabled so there is no internal validation split and the fit is
deterministic for the small-DOE regime. The model is persisted via pickle (the universal
artifact format); ONNX export is not offered, because the current ``skl2onnx``/``onnx``
stack cannot convert tree-ensemble regressors (``mlp``/``polynomial`` still export to ONNX).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from surroforge.errors import TrainerError
from surroforge.trainers.base import BaseTrainer

_SUPPORTED = {
    "learning_rate",
    "max_iter",
    "max_leaf_nodes",
    "max_depth",
    "min_samples_leaf",
    "l2_regularization",
    "random_state",
}


class GradientBoostingTrainer(BaseTrainer):
    """Gradient-boosting surrogate (scikit-learn ``HistGradientBoostingRegressor``).

    Hyperparameters (all optional, passed through to the estimator):
        learning_rate (float): shrinkage per boosting stage (default ``0.1``, ``>0``).
        max_iter (int): number of boosting stages (default ``100``, ``>=1``).
        max_leaf_nodes (int | None): leaves per tree (default ``31``, ``>=2`` or ``None``).
        max_depth (int | None): max tree depth (default ``None``, ``>=1`` or ``None``).
        min_samples_leaf (int): min samples per leaf (default ``20``, ``>=1``).
        l2_regularization (float): L2 penalty on leaf values (default ``0.0``, ``>=0``).
        random_state (int): seed (default ``0``) — fixed so the method is reproducible.
    """

    name = "gradient-boosting"
    # Tree ensemble with a fixed seed and early_stopping off: deterministic in-process.
    # Declares a small honest tolerance (like mlp) rather than 0.0 to absorb any
    # thread/platform float differences in histogram summation.
    reproduction_tolerance = 1e-6
    # Pickle-only: skl2onnx's TreeEnsembleRegressor converter emits numpy bool/uint32
    # attributes that onnx>=1.21 rejects, so tree-ensemble ONNX export is unavailable in
    # this stack. ONNX can be re-added once skl2onnx/onnx convert tree ensembles cleanly.
    export_formats = frozenset({"pickle"})

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: int | None = 31,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        random_state: int = 0,
        **extra: Any,
    ) -> None:
        if extra:
            raise TrainerError(
                f"Unknown hyperparameter(s) for gradient-boosting: {sorted(extra)}. "
                f"Supported: {sorted(_SUPPORTED)}."
            )
        if learning_rate <= 0:
            raise TrainerError("gradient-boosting 'learning_rate' must be > 0.")
        if max_iter < 1:
            raise TrainerError("gradient-boosting 'max_iter' must be >= 1.")
        if max_leaf_nodes is not None and max_leaf_nodes < 2:
            raise TrainerError("gradient-boosting 'max_leaf_nodes' must be >= 2 or None.")
        if max_depth is not None and max_depth < 1:
            raise TrainerError("gradient-boosting 'max_depth' must be >= 1 or None.")
        if min_samples_leaf < 1:
            raise TrainerError("gradient-boosting 'min_samples_leaf' must be >= 1.")
        if l2_regularization < 0:
            raise TrainerError("gradient-boosting 'l2_regularization' must be >= 0.")
        # A fixed integer seed is what makes the method reproducible; None/RandomState/float
        # would silently break that (bool excluded though it subclasses int).
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise TrainerError(
                f"gradient-boosting 'random_state' must be an int; got {random_state!r}."
            )
        super().__init__(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self._model: Any = None

    def _build_regressor(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=self.random_state,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = self._as_2d(x)
        y = self._as_2d(y)
        if y.shape[1] == 1:
            # Single output: a 1-D target avoids sklearn's DataConversionWarning.
            model: Any = self._build_regressor()
            model.fit(x, y[:, 0])
        else:
            # HistGradientBoostingRegressor is single-output; wrap for multi-output.
            from sklearn.multioutput import MultiOutputRegressor

            model = MultiOutputRegressor(self._build_regressor())
            model.fit(x, y)
        # Assign only after a successful fit, so a failed fit can't leave a half-built
        # model that predict() would treat as fitted.
        self._model = model

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise TrainerError("Model is not fitted.")
        preds = self._model.predict(self._as_2d(x))
        return self._as_2d(preds)
