"""PyTorch neural-network surrogate — ``torch-mlp`` in surroforge_professional (Epic R7).

This is the heavyweight-framework counterpart of ``mlp``. Since the Wave-1
editions split it ships inside ``surroforge_professional`` behind the optional
``[torch]`` extra and registers ``torch-mlp`` through the ``surroforge.trainers``
entry point (via the guard module ``trainers/torch_mlp.py``, resolvable without
torch installed) — never Core's ``_BUILTIN`` table. Core stays torch-free.
Switching backends is a one-line config edit: ``torch-mlp`` mirrors ``mlp``'s
exact hyperparameter surface, so the same YAML trains either scikit-learn or
PyTorch.

Design choices for a reproducibility-first surrogate tool:

* **Deterministic by construction** — weights are initialised under a
  saved/seeded/restored torch RNG and training is full-batch (no shuffling), so a fixed
  ``random_state`` fully pins the result (a same-seed refit is bit-identical in-process).
  The trainer carries its own ``random_state`` because the engine does not thread the
  run seed into trainers.
* **``reproduction_tolerance = 1e-4``** — honest for a float32 network (looser than
  ``mlp``'s 1e-6), yet ~two orders below a real (different-seed) divergence.
* **``solver="lbfgs"`` default** — full-batch L-BFGS with a strong-Wolfe line search
  suits the small-data DOE regime (mirrors ``mlp``); ``adam``/``sgd`` are available via
  an epoch loop with a ``tol`` early-stop.
* **Standardisation baked into the graph** — input/output mean-std live as module
  buffers applied inside ``forward``, so the exported ONNX graph is self-contained and
  matches :meth:`TorchMLPTrainer.predict`.
"""

from __future__ import annotations

import importlib.util
import io
import warnings
from typing import Any

import numpy as np
import torch
from torch import nn

from surroforge.errors import ExportError, TrainerError
from surroforge.trainers.base import BaseTrainer

# torch-mlp mirrors mlp's curated surface so switching backend is a one-line edit.
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

# Validated at construction so a typo fails fast with a domain error, not a raw torch
# error deep inside fit().
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "identity": nn.Identity,
    "logistic": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}
_SOLVERS = {"lbfgs", "adam", "sgd"}


class _StandardizedMLP(nn.Module):
    """An MLP that standardises inputs and de-standardises outputs inside forward.

    The mean/std are buffers (not parameters), so they persist through pickling and are
    emitted into the ONNX graph — the exported model therefore matches
    :meth:`TorchMLPTrainer.predict` rather than expecting pre-normalised inputs. Defined
    at module level so the trainer (which holds an instance) is picklable.
    """

    # Annotate the registered buffers so the type checker treats them as tensors
    # (nn.Module.__getattr__ otherwise widens attribute access to Tensor | Module).
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    def __init__(
        self,
        net: nn.Module,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.x_mean) / self.x_std
        return self.net(z) * self.y_std + self.y_mean


class TorchMLPTrainer(BaseTrainer):
    """Multi-layer-perceptron surrogate backed by PyTorch (the ``torch-mlp`` method).

    Mirrors the built-in ``mlp``'s hyperparameter surface so the two are interchangeable
    via ``model.method``. Multi-output targets are supported natively (one network,
    ``n_outputs`` columns).

    Hyperparameters (all optional, mirroring ``mlp``):
        hidden_layer_sizes (tuple[int, ...]): layer widths (default ``(64, 64)``).
        activation (str): ``relu`` (default), ``tanh``, ``logistic``, ``identity``.
        solver (str): ``lbfgs`` (default — full-batch, small-data), ``adam``, ``sgd``.
        alpha (float): L2 regularisation / weight decay (default ``1e-4``).
        max_iter (int): optimiser iteration budget (default ``2000``).
        tol (float): convergence tolerance (default ``1e-4``).
        learning_rate_init (float): initial step for ``adam``/``sgd`` (default ``1e-3``).
        random_state (int): seed for weight init (default ``0``) — fixed so the method is
            deterministic out of the box; override in the config.
    """

    name = "torch-mlp"
    required_tier = "professional"
    # float32 network: a same-seed refit is bit-identical in-process, but float32
    # reductions are not bit-stable across threads/platforms — so it declares a small,
    # honest band (looser than mlp's 1e-6). A changed seed moves predictions by ~1e-2
    # (~100x this band) on the test fixtures, so a real divergence still fails the verdict.
    reproduction_tolerance = 1e-4
    export_formats = frozenset({"pickle", "onnx", "pt"})

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
                f"Unknown hyperparameter(s) for torch-mlp: {sorted(extra)}. "
                f"Supported: {sorted(_SUPPORTED)}."
            )
        if activation not in _ACTIVATIONS:
            raise TrainerError(
                f"torch-mlp 'activation' must be one of {sorted(_ACTIVATIONS)}; "
                f"got {activation!r}."
            )
        if solver not in _SOLVERS:
            raise TrainerError(
                f"torch-mlp 'solver' must be one of {sorted(_SOLVERS)}; got {solver!r}."
            )
        # A fixed integer seed is what makes the method reproducible (and what
        # reproduction_tolerance/lineage assume); None or a float would silently break
        # that, so reject them here rather than discover non-determinism later.
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise TrainerError(f"torch-mlp 'random_state' must be an int; got {random_state!r}.")
        # torch.manual_seed accepts a 64-bit seed; reject an out-of-range value here with a
        # domain error rather than let manual_seed raise a bare ValueError deep in fit().
        if not -(2**63) <= random_state < 2**64:
            raise TrainerError(
                f"torch-mlp 'random_state' must be in [-(2**63), 2**64 - 1]; got {random_state!r}."
            )
        if max_iter < 1:
            raise TrainerError("torch-mlp 'max_iter' must be >= 1.")
        if alpha < 0:
            raise TrainerError("torch-mlp 'alpha' (L2 regularisation) must be >= 0.")
        if tol <= 0:
            raise TrainerError("torch-mlp 'tol' must be > 0.")
        if learning_rate_init <= 0:
            raise TrainerError("torch-mlp 'learning_rate_init' must be > 0.")
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
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state
        self._model: _StandardizedMLP | None = None
        self._n_features: int | None = None
        self._n_outputs: int | None = None

    def _build_net(self, n_in: int, n_out: int) -> nn.Module:
        # Save / seed / restore so weight init is pinned by random_state without
        # clobbering the caller's global torch RNG. Full-batch training adds no further
        # randomness, so the seed fully determines the fit.
        state = torch.get_rng_state()
        torch.manual_seed(self.random_state)
        try:
            layers: list[nn.Module] = []
            prev = n_in
            for width in self.hidden_layer_sizes:
                layers.append(nn.Linear(prev, width))
                layers.append(_ACTIVATIONS[self.activation]())
                prev = width
            layers.append(nn.Linear(prev, n_out))
            return nn.Sequential(*layers)
        finally:
            torch.set_rng_state(state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x2 = self._as_2d(x)
        y2 = self._as_2d(y)
        xt = torch.from_numpy(np.ascontiguousarray(x2, dtype=np.float32))
        yt = torch.from_numpy(np.ascontiguousarray(y2, dtype=np.float32))
        # Standardisation stats (std == 0 -> 1 so a constant column is a no-op shift).
        x_mean = xt.mean(0)
        x_std = xt.std(0, unbiased=False)
        x_std[x_std == 0] = 1.0
        y_mean = yt.mean(0)
        y_std = yt.std(0, unbiased=False)
        y_std[y_std == 0] = 1.0
        net = self._build_net(x2.shape[1], y2.shape[1])
        model = _StandardizedMLP(net, x_mean, x_std, y_mean, y_std)
        model.train()
        self._optimise(model, xt, yt)
        model.eval()
        # Assign only after a successful fit (build-then-assign), mirroring mlp.
        self._model = model
        self._n_features = x2.shape[1]
        self._n_outputs = y2.shape[1]

    def _optimise(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> None:
        loss_fn = nn.MSELoss()
        if self.solver == "lbfgs":
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                max_iter=self.max_iter,
                tolerance_grad=self.tol,
                tolerance_change=self.tol * self.tol,
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                loss = loss_fn(model(x), y) + self._l2_penalty(model)
                loss.backward()
                return loss

            lbfgs.step(closure)  # L-BFGS iterates internally up to max_iter
            return
        # adam / sgd: a full-batch epoch loop with a simple tol early-stop. alpha is the L2
        # weight_decay, applied to weights only (not biases) — matching the lbfgs penalty
        # and scikit-learn, so `alpha` means the same thing across solvers.
        decay = [p for name, p in model.named_parameters() if name.endswith("weight")]
        no_decay = [p for name, p in model.named_parameters() if not name.endswith("weight")]
        groups = [
            {"params": decay, "weight_decay": self.alpha},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if self.solver == "adam":
            opt: torch.optim.Optimizer = torch.optim.Adam(groups, lr=self.learning_rate_init)
        else:
            opt = torch.optim.SGD(groups, lr=self.learning_rate_init)
        prev: float | None = None
        for _ in range(self.max_iter):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            current = loss.item()  # .item() detaches; float(loss) would warn on a grad tensor
            if prev is not None and abs(prev - current) < self.tol:
                break
            prev = current

    def _l2_penalty(self, model: nn.Module) -> torch.Tensor:
        # L-BFGS has no weight_decay, so regularise the layer weights explicitly (biases and
        # the standardisation buffers are excluded). The 0.5 factor makes the gradient
        # alpha*w — the same effective L2 the adam/sgd weight_decay applies — so `alpha`
        # means the same thing across solvers. alpha == 0 -> no-op.
        penalty = torch.zeros(())
        for name, param in model.named_parameters():
            if name.endswith("weight"):
                penalty = penalty + param.pow(2).sum()
        return 0.5 * self.alpha * penalty

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise TrainerError("Model is not fitted.")
        x2 = self._as_2d(x)
        # Guard the input width (to_onnx does the same): a mismatch would otherwise raise a
        # bare torch RuntimeError, or — for a 1-D input — silently mispredict.
        if x2.shape[1] != self._n_features:
            raise TrainerError(
                f"torch-mlp was fitted on {self._n_features} input feature(s) but predict got "
                f"{x2.shape[1]}. Pass an (n_samples, {self._n_features}) array."
            )
        self._model.eval()
        xt = torch.from_numpy(np.ascontiguousarray(x2, dtype=np.float32))
        with torch.no_grad():
            preds = self._model(xt).detach().numpy()
        return self._as_2d(preds)

    def to_onnx(self, n_features: int) -> bytes:
        if self._model is None:
            raise TrainerError("Model is not fitted.")
        # The module was fitted on a fixed input width; a mismatched n_features would
        # build a graph that silently mispredicts at inference, so refuse it.
        if n_features != self._n_features:
            raise ExportError(
                f"ONNX export was asked for {n_features} input feature(s) but the model was "
                f"fitted on {self._n_features}. Pass the training feature count."
            )
        # torch's exporter serialises the graph through the `onnx` package; surface a
        # missing one as an actionable ExportError naming the extra, rather than the deep
        # torch OnnxExporterError it would otherwise raise.
        if importlib.util.find_spec("onnx") is None:
            raise ExportError(
                "ONNX export requires the 'onnx' package (and 'onnxruntime' to run the model). "
                "Install them with: pip install 'surroforge_professional[onnx]'"
            )
        self._model.eval()
        buffer = io.BytesIO()
        dummy = torch.zeros(1, n_features, dtype=torch.float32)
        try:
            # dynamo=False selects torch's legacy (TorchScript) exporter, which needs only
            # torch + onnx (the default dynamo exporter would also require onnxscript). Its
            # deprecation notice is a deliberate, non-actionable choice here, so we keep it
            # out of the user's output (a future torch may move us to dynamo).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                torch.onnx.export(
                    self._model,
                    (dummy,),
                    buffer,  # type: ignore[arg-type]  # legacy exporter accepts a file-like
                    dynamo=False,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "n"}, "output": {0: "n"}},
                    opset_version=17,
                )
        except Exception as exc:  # torch raises various tracing/conversion errors
            raise ExportError(f"ONNX conversion of the torch-mlp module failed: {exc}") from exc
        return buffer.getvalue()

    def to_pt(self, n_features: int) -> bytes:
        """Serialise the fitted module to a native PyTorch ``.pt`` checkpoint.

        Unlike ONNX (a framework-agnostic graph) this is a torch-native artifact: a
        ``torch.save`` payload carrying the module's ``state_dict`` plus the architecture
        needed to rebuild it. It deliberately stores the *state dict* (weights + the
        standardisation buffers), not a pickled module, so it loads under the safe
        ``torch.load(..., weights_only=True)`` path without importing this package.
        """
        if self._model is None:
            raise TrainerError("Model is not fitted.")
        # Mirror to_onnx: the module was fitted on a fixed input width, so a mismatched
        # n_features would describe a checkpoint that cannot be rebuilt correctly.
        if n_features != self._n_features:
            raise ExportError(
                f"PT export was asked for {n_features} input feature(s) but the model was "
                f"fitted on {self._n_features}. Pass the training feature count."
            )
        self._model.eval()
        # Both are set together with _model in fit(); assert so the int() calls below are typed.
        assert self._n_features is not None and self._n_outputs is not None
        checkpoint = {
            "surroforge_torch_pt_version": 1,
            "state_dict": self._model.state_dict(),
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "activation": self.activation,
            "n_features": int(self._n_features),
            "n_outputs": int(self._n_outputs),
        }
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        return buffer.getvalue()
