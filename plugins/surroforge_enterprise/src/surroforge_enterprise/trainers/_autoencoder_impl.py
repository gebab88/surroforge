from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
import torch
from torch import nn

from surroforge.errors import TrainerError
from surroforge.trainers.base import BaseTrainer


_SUPPORTED = {
    "latent_dim",
    "encoder_hidden_layer_sizes",
    "decoder_hidden_layer_sizes",
    "activation",
    "alpha",
    "max_iter",
    "tol",
    "learning_rate_init",
    "random_state",
    "inner",
}
_ACTIVATIONS = {"identity", "logistic", "tanh", "relu"}
_TORCH_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "identity": nn.Identity,
    "logistic": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}


def _positive_int_tuple(name: str, value: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or isinstance(value, (str, bytes)):
        raise TrainerError(f"{name} must be a tuple/list of positive ints; got {value!r}.")
    result = tuple(value)
    if any(isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in result):
        raise TrainerError(f"{name} must contain only positive ints; got {value!r}.")
    return result


def _nonnegative_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value < 0:
        raise TrainerError(f"autoencoder-field '{name}' must be a number >= 0; got {value!r}.")
    return float(value)


def _positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value <= 0:
        raise TrainerError(f"autoencoder-field '{name}' must be a number > 0; got {value!r}.")
    return float(value)


def _build_network(widths: tuple[int, ...], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_features, out_features in zip(widths[:-1], widths[1:], strict=True):
        layers.append(nn.Linear(in_features, out_features))
        if out_features != widths[-1]:
            layers.append(_TORCH_ACTIVATIONS[activation]())
    return nn.Sequential(*layers)


class _AutoencoderModule(nn.Module):
    field_mean: torch.Tensor
    field_std: torch.Tensor

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        field_mean: torch.Tensor,
        field_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.register_buffer("field_mean", field_mean)
        self.register_buffer("field_std", field_std)

    def standardize_field(self, field: torch.Tensor) -> torch.Tensor:
        return (field - self.field_mean) / self.field_std

    def encode_standardized(self, standardized: torch.Tensor) -> torch.Tensor:
        return self.encoder(standardized)

    def decode_standardized(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def encode_field(self, field: torch.Tensor) -> torch.Tensor:
        return self.encode_standardized(self.standardize_field(field))

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode_standardized(latent) * self.field_std + self.field_mean

    def forward(self, standardized: torch.Tensor) -> torch.Tensor:
        return self.decode_standardized(self.encode_standardized(standardized))


class AutoencoderFieldTrainer(BaseTrainer):
    name = "autoencoder-field"
    required_tier = "enterprise"
    export_formats = frozenset({"pickle"})
    reproduction_tolerance = 1e-4

    def __init__(
        self,
        latent_dim: int = 5,
        encoder_hidden_layer_sizes: tuple[int, ...] | list[int] = (128, 64),
        decoder_hidden_layer_sizes: tuple[int, ...] | list[int] | None = None,
        activation: str = "relu",
        alpha: float = 1e-4,
        max_iter: int = 2000,
        tol: float = 1e-5,
        learning_rate_init: float = 1e-3,
        random_state: int = 0,
        inner: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        if extra:
            raise TrainerError(
                f"Unknown hyperparameter(s) for autoencoder-field: {sorted(extra)}. "
                f"Supported: {sorted(_SUPPORTED)}."
            )
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim < 1:
            raise TrainerError(
                f"autoencoder-field 'latent_dim' must be a positive int; got {latent_dim!r}."
            )
        if activation not in _ACTIVATIONS:
            raise TrainerError(
                f"autoencoder-field 'activation' must be one of {sorted(_ACTIVATIONS)}; "
                f"got {activation!r}."
            )
        alpha_value = _nonnegative_float("alpha", alpha)
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise TrainerError("autoencoder-field 'max_iter' must be a positive int.")
        tol_value = _positive_float("tol", tol)
        learning_rate_value = _positive_float("learning_rate_init", learning_rate_init)
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise TrainerError(
                f"autoencoder-field 'random_state' must be an int; got {random_state!r}."
            )
        if not -(2**63) <= random_state < 2**64:
            raise TrainerError(
                "autoencoder-field 'random_state' must be in "
                f"[-(2**63), 2**64 - 1]; got {random_state!r}."
            )
        if inner is not None and not isinstance(inner, dict):
            raise TrainerError(
                f"autoencoder-field 'inner' must be a mapping of trainer hyperparameters; "
                f"got {inner!r}."
            )

        encoder_layers = _positive_int_tuple(
            "autoencoder-field 'encoder_hidden_layer_sizes'", encoder_hidden_layer_sizes
        )
        decoder_layers = (
            tuple(reversed(encoder_layers))
            if decoder_hidden_layer_sizes is None
            else _positive_int_tuple(
                "autoencoder-field 'decoder_hidden_layer_sizes'", decoder_hidden_layer_sizes
            )
        )
        inner_spec = dict(inner or {"method": "mlp"})
        self._inner_method = str(inner_spec.pop("method", "mlp"))
        self._inner_hyper = inner_spec

        super().__init__(
            latent_dim=latent_dim,
            encoder_hidden_layer_sizes=encoder_layers,
            decoder_hidden_layer_sizes=decoder_layers,
            activation=activation,
            alpha=alpha_value,
            max_iter=max_iter,
            tol=tol_value,
            learning_rate_init=learning_rate_value,
            random_state=random_state,
            inner={"method": self._inner_method, **self._inner_hyper},
        )
        self.latent_dim = latent_dim
        self.encoder_hidden_layer_sizes = encoder_layers
        self.decoder_hidden_layer_sizes = decoder_layers
        self.activation = activation
        self.alpha = alpha_value
        self.max_iter = max_iter
        self.tol = tol_value
        self.learning_rate_init = learning_rate_value
        self.random_state = random_state
        self._model: _AutoencoderModule | None = None
        self._latent_model: BaseTrainer | None = None
        self._n_features: int | None = None
        self._n_outputs: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        from surroforge.plugins import get_trainer

        x2 = self._as_2d(x)
        y2 = self._as_2d(y)
        yt = torch.from_numpy(np.ascontiguousarray(y2, dtype=np.float32))

        field_mean = yt.mean(0)
        field_std = yt.std(0, unbiased=False)
        field_std[field_std == 0] = 1.0
        standardized = (yt - field_mean) / field_std

        state = torch.get_rng_state()
        torch.manual_seed(self.random_state)
        try:
            encoder = _build_network(
                (y2.shape[1], *self.encoder_hidden_layer_sizes, self.latent_dim),
                self.activation,
            )
            decoder = _build_network(
                (self.latent_dim, *self.decoder_hidden_layer_sizes, y2.shape[1]),
                self.activation,
            )
        finally:
            torch.set_rng_state(state)

        model = _AutoencoderModule(encoder, decoder, field_mean, field_std)
        model.train()
        self._optimise_autoencoder(model, standardized)
        model.eval()

        with torch.no_grad():
            latent = model.encode_standardized(standardized).detach().numpy()

        latent_model = get_trainer(self._inner_method)(**self._inner_hyper)
        latent_model.fit(x2, latent)

        self._model = model
        self._latent_model = latent_model
        self._n_features = x2.shape[1]
        self._n_outputs = y2.shape[1]

    def _optimise_autoencoder(self, model: _AutoencoderModule, standardized: torch.Tensor) -> None:
        loss_fn = nn.MSELoss()
        decay = [param for name, param in model.named_parameters() if name.endswith("weight")]
        no_decay = [param for name, param in model.named_parameters() if not name.endswith("weight")]
        optimizer = torch.optim.Adam(
            [
                {"params": decay, "weight_decay": self.alpha},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.learning_rate_init,
        )
        previous: float | None = None
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            loss = loss_fn(model(standardized), standardized)
            loss.backward()
            optimizer.step()
            current = loss.item()
            if previous is not None and abs(previous - current) < self.tol:
                break
            previous = current

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None or self._latent_model is None:
            raise TrainerError("Model is not fitted.")
        x2 = self._as_2d(x)
        if x2.shape[1] != self._n_features:
            raise TrainerError(
                f"autoencoder-field was fitted on {self._n_features} input feature(s) "
                f"but predict got {x2.shape[1]}."
            )
        latent = self._latent_model.predict(x2)
        if latent.shape[1] != self.latent_dim:
            raise TrainerError(
                f"inner model predicted {latent.shape[1]} latent column(s), "
                f"expected {self.latent_dim}."
            )
        zt = torch.from_numpy(np.ascontiguousarray(latent, dtype=np.float32))
        self._model.eval()
        with torch.no_grad():
            decoded = self._model.decode_latent(zt).detach().numpy()
        return self._as_2d(decoded)
