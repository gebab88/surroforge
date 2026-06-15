"""Neural-network surrogate — a multi-layer perceptron via scikit-learn (PRD 4.6).

The first neural-network method. It is a genuine MLP, but backed by
``sklearn.neural_network.MLPRegressor`` rather than a deep-learning framework, so
it adds **no new dependency** (scikit-learn is already Core, BSD-licensed → NFR8),
plugs into the project's sklearn-style integer ``random_state`` reproducibility
model, and — like ``polynomial`` — exports to ONNX via ``skl2onnx``. A heavyweight
framework (PyTorch/DeepONet/FNO) for the large, strongly-nonlinear field regime
belongs on the external entry-point plugin seam, not in Core.

Two domain choices differ from scikit-learn's defaults on purpose:

* ``solver="lbfgs"`` — the surrogate regime is small-data (a DOE of dozens–hundreds
  of points), where scikit-learn itself recommends lbfgs; it is also far more
  reproducible (a same-``random_state`` lbfgs refit is bit-identical in-process).
* an input :class:`~sklearn.preprocessing.StandardScaler` — MLPs need scaled inputs;
  it is deterministic and ``skl2onnx`` converts the pipeline cleanly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from surroforge.errors import TrainerError
from surroforge.trainers.base import BaseTrainer

_SUPPORTED = {
    "hidden_layer_sizes",
    "activation",
    "solver",
    "alpha",
    "max_iter",
    "tol",
    "learning_rate_init",
    "random_state",
}

# Enumerable choices are validated at construction so a typo fails fast with a
# domain error, not a raw scikit-learn ValueError deep inside fit().
_ACTIVATIONS = {"identity", "logistic", "tanh", "relu"}
_SOLVERS = {"lbfgs", "sgd", "adam"}


class MLPTrainer(BaseTrainer):
    """Multi-layer-perceptron surrogate (scikit-learn ``MLPRegressor``).

    Inputs are standardised and the network is wrapped in a single pipeline, which
    is the persisted artifact and the unit ONNX export operates on. Multi-output
    targets are supported natively (one network, ``n_outputs`` columns).

    Hyperparameters (all optional, passed through to ``MLPRegressor``):
        hidden_layer_sizes (tuple[int, ...]): layer widths (default ``(64, 64)``).
        activation (str): ``relu`` (default), ``tanh``, ``logistic``, ``identity``.
        solver (str): ``lbfgs`` (default — small-data + reproducible), ``adam``, ``sgd``.
        alpha (float): L2 regularisation strength (default ``1e-4``).
        max_iter (int): optimiser iteration budget (default ``2000``).
        tol (float): optimiser convergence tolerance (default ``1e-4``).
        learning_rate_init (float): initial step for ``adam``/``sgd`` (default ``1e-3``).
        random_state (int): seed for weight init / shuffling (default ``0``) — fixed so
            the method is deterministic out of the box; override in the config.
    """

    name = "mlp"
    # An iterative, non-convex optimiser: deterministic in-process with a fixed
    # random_state (a same-seed lbfgs refit is bit-identical), but not guaranteed
    # bit-for-bit across BLAS thread counts / platforms. So it declares a small,
    # honest tolerance rather than determinism (Story 5.4 / R6.1) — looser than the
    # SMT methods' 1e-9, since the loss surface is non-convex. A changed seed moves
    # predictions by ~1e-2, three orders above this band, so a real divergence still
    # fails the verdict.
    reproduction_tolerance = 1e-6
    # Backed by scikit-learn, so it exports to ONNX via skl2onnx (the second method
    # to do so, after polynomial). The StandardScaler + MLPRegressor pipeline
    # converts for single- and multi-output alike.
    export_formats = frozenset({"pickle", "onnx"})

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 64),
        activation: str = "relu",
        solver: str = "lbfgs",
        alpha: float = 1e-4,
        max_iter: int = 2000,
        tol: float = 1e-4,
        learning_rate_init: float = 1e-3,
        random_state: int = 0,
        **extra: Any,
    ) -> None:
        if extra:
            raise TrainerError(
                f"Unknown hyperparameter(s) for mlp: {sorted(extra)}. "
                f"Supported: {sorted(_SUPPORTED)}."
            )
        if activation not in _ACTIVATIONS:
            raise TrainerError(
                f"mlp 'activation' must be one of {sorted(_ACTIVATIONS)}; got {activation!r}."
            )
        if solver not in _SOLVERS:
            raise TrainerError(f"mlp 'solver' must be one of {sorted(_SOLVERS)}; got {solver!r}.")
        # A fixed integer seed is what makes the method reproducible (and what
        # reproduction_tolerance/lineage assume); None or a RandomState would silently
        # break that, so reject them here rather than discover non-determinism later.
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise TrainerError(f"mlp 'random_state' must be an int; got {random_state!r}.")
        if max_iter < 1:
            raise TrainerError("mlp 'max_iter' must be >= 1.")
        if alpha < 0:
            raise TrainerError("mlp 'alpha' (L2 regularisation) must be >= 0.")
        if tol <= 0:
            raise TrainerError("mlp 'tol' must be > 0.")
        if learning_rate_init <= 0:
            raise TrainerError("mlp 'learning_rate_init' must be > 0.")
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            learning_rate_init=learning_rate_init,
            random_state=random_state,
        )
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state
        self._model: Any = None

    def _build(self) -> Any:
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                learning_rate_init=self.learning_rate_init,
                random_state=self.random_state,
            ),
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = self._as_2d(x)
        y = self._as_2d(y)
        # MLPRegressor expects a 1-D target for a single output; passing the
        # column-vector triggers a DataConversionWarning. predict() re-widens to 2-D.
        target = y[:, 0] if y.shape[1] == 1 else y
        model = self._build()
        model.fit(x, target)
        # Assign only after a successful fit, so a failed fit can't leave a
        # half-built pipeline that predict() would treat as fitted.
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
        # The pipeline was fitted on a fixed input width; a mismatched n_features
        # would build a graph that silently mispredicts at inference, so refuse it.
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
            onnx_model = convert_sklearn(
                self._model,
                initial_types=[("input", FloatTensorType([None, n_features]))],
            )
        except Exception as exc:  # skl2onnx raises various conversion errors
            raise ExportError(f"ONNX conversion of the mlp pipeline failed: {exc}") from exc
        return onnx_model.SerializeToString()
