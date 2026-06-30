"""Random-forest surrogate — scikit-learn RandomForestRegressor (PRD 4.6).

A bagged tree-ensemble method, completing the Phase 2 extended methods alongside
``mlp`` (NN), ``kriging`` (GP), and ``gradient-boosting``. Backed by scikit-learn
(already Core, BSD → NFR8), so it adds **no new dependency**. Two notes:

* **No input scaler.** Tree ensembles are scale-invariant, so the estimator is used
  directly — a simpler model.
* **Native multi-output.** Unlike ``gradient-boosting`` (which wraps a single-output
  booster in ``MultiOutputRegressor``), ``RandomForestRegressor`` predicts multiple
  targets directly, so a 2-D ``y`` is passed straight through; a single target uses a
  1-D vector (avoiding sklearn's ``DataConversionWarning``).

A fixed integer ``random_state`` makes the forest deterministic (bootstrap and feature
subsampling are seeded). Like ``mlp``/``gradient-boosting`` it exports to ONNX via
``skl2onnx`` (the RandomForest converter is unaffected by the tree-ensemble issue that
made ``gradient-boosting`` pickle-only).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from surroforge.errors import TrainerError
from surroforge.trainers.base import BaseTrainer

_SUPPORTED = {
    "n_estimators",
    "max_depth",
    "max_leaf_nodes",
    "min_samples_leaf",
    "max_features",
    "random_state",
}


def _valid_max_features(value: Any) -> bool:
    if value is None or value in ("sqrt", "log2"):
        return True
    if isinstance(value, bool):  # bool subclasses int; not a valid max_features
        return False
    if isinstance(value, int):
        return value >= 1
    if isinstance(value, float):
        return 0.0 < value <= 1.0
    return False


class RandomForestTrainer(BaseTrainer):
    """Random-forest surrogate (scikit-learn ``RandomForestRegressor``).

    Hyperparameters (all optional, passed through to the estimator):
        n_estimators (int): number of trees (default ``100``, ``>=1``).
        max_depth (int | None): max tree depth (default ``None``, ``>=1`` or ``None``).
        max_leaf_nodes (int | None): leaves per tree (default ``None``, ``>=2`` or ``None``).
        min_samples_leaf (int): min samples per leaf (default ``1``, ``>=1``).
        max_features: features considered per split — ``"sqrt"``/``"log2"``/``None``,
            an ``int >= 1``, or a ``float`` in ``(0, 1]`` (default ``1.0``).
        random_state (int): seed (default ``0``) — fixed so the method is reproducible.
    """

    name = "random-forest"
    # Bootstrap + feature subsampling are seeded, so a fixed random_state is
    # deterministic. Declares a small honest tolerance (like mlp/gradient-boosting)
    # rather than 0.0 to absorb thread/platform float differences in tree averaging.
    reproduction_tolerance = 1e-6
    # Backed by scikit-learn → exports to ONNX via skl2onnx. The RandomForest converter
    # emits clean integer attributes (unlike HistGradientBoosting), so ONNX works here.
    export_formats = frozenset({"pickle", "onnx"})

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_leaf_nodes: int | None = None,
        min_samples_leaf: int = 1,
        max_features: Any = 1.0,
        random_state: int = 0,
        **extra: Any,
    ) -> None:
        if extra:
            raise TrainerError(
                f"Unknown hyperparameter(s) for random-forest: {sorted(extra)}. "
                f"Supported: {sorted(_SUPPORTED)}."
            )
        if n_estimators < 1:
            raise TrainerError("random-forest 'n_estimators' must be >= 1.")
        if max_depth is not None and max_depth < 1:
            raise TrainerError("random-forest 'max_depth' must be >= 1 or None.")
        if max_leaf_nodes is not None and max_leaf_nodes < 2:
            raise TrainerError("random-forest 'max_leaf_nodes' must be >= 2 or None.")
        if min_samples_leaf < 1:
            raise TrainerError("random-forest 'min_samples_leaf' must be >= 1.")
        if not _valid_max_features(max_features):
            raise TrainerError(
                "random-forest 'max_features' must be 'sqrt', 'log2', None, an int >= 1, "
                f"or a float in (0, 1]; got {max_features!r}."
            )
        # A fixed integer seed is what makes the method reproducible; None/RandomState/float
        # would silently break that (bool excluded though it subclasses int).
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise TrainerError(
                f"random-forest 'random_state' must be an int; got {random_state!r}."
            )
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._model: Any = None

    def _build_regressor(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = self._as_2d(x)
        y = self._as_2d(y)
        # RandomForestRegressor is natively multi-output; pass 2-D y directly. For a
        # single output, a 1-D target avoids sklearn's DataConversionWarning.
        target = y[:, 0] if y.shape[1] == 1 else y
        model = self._build_regressor()
        model.fit(x, target)
        # Assign only after a successful fit, so a failed fit can't leave a half-built
        # model that predict() would treat as fitted.
        self._model = model

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise TrainerError("Model is not fitted.")
        preds = self._model.predict(self._as_2d(x))
        return self._as_2d(preds)

    def to_onnx(self, n_features: int) -> bytes:
        from surroforge.errors import ExportError

        if self._model is None:
            raise TrainerError("Model is not fitted.")
        # The model was fitted on a fixed input width; a mismatched n_features would
        # build a graph that silently mispredicts at inference, so refuse it.
        trained = int(self._model.n_features_in_)
        if n_features != trained:
            raise ExportError(
                f"ONNX export was asked for {n_features} input feature(s) but the model was "
                f"fitted on {trained}. Pass the training feature count."
            )
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ExportError(
                "ONNX export requires the 'onnx' extra. "
                "Install it with: pip install 'surroforge[onnx]'"
            ) from exc

        try:
            onnx_model = cast(Any, convert_sklearn(
                self._model,
                initial_types=[("input", FloatTensorType([None, n_features]))],
            ))
        except Exception as exc:  # skl2onnx raises various conversion errors
            raise ExportError(
                f"ONNX conversion of the random-forest model failed: {exc}"
            ) from exc
        return onnx_model.SerializeToString()
