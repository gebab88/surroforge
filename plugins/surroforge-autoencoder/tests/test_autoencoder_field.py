from __future__ import annotations

from pathlib import Path
import pickle
import tomllib

import numpy as np
import pytest
from sklearn.metrics import r2_score

from surroforge.errors import TrainerError
from surroforge_autoencoder import AutoencoderFieldTrainer


def smooth_field_fixture():
    rng = np.random.default_rng(4)
    x = rng.uniform(-1.0, 1.0, (120, 2))
    grid = np.linspace(-1.0, 1.0, 18)
    mode_a = np.sin(np.pi * grid)
    mode_b = np.cos(0.5 * np.pi * grid)
    latent_a = 1.8 * x[:, 0] + 0.4 * x[:, 1] ** 2
    latent_b = -0.8 * x[:, 1] + 0.6 * x[:, 0] * x[:, 1]
    y = (
        latent_a[:, None] * mode_a[None, :]
        + latent_b[:, None] * mode_b[None, :]
        + 0.15 * np.tanh(latent_a[:, None] * mode_b[None, :])
    )
    return x, y


def test_pyproject_declares_autoencoder_field_entry_point():
    data = tomllib.loads(Path("plugins/surroforge-autoencoder/pyproject.toml").read_text())
    trainers = data["project"]["entry-points"]["surroforge.trainers"]
    assert trainers == {
        "autoencoder-field": "surroforge_autoencoder.trainer:AutoencoderFieldTrainer"
    }


def test_constructor_rejects_invalid_hyperparameters():
    with pytest.raises(TrainerError, match="Unknown hyperparameter"):
        AutoencoderFieldTrainer(bogus=1)
    with pytest.raises(TrainerError, match="latent_dim"):
        AutoencoderFieldTrainer(latent_dim=0)
    with pytest.raises(TrainerError, match="activation"):
        AutoencoderFieldTrainer(activation="swish")
    with pytest.raises(TrainerError, match="encoder_hidden_layer_sizes"):
        AutoencoderFieldTrainer(encoder_hidden_layer_sizes=(16, 0))
    with pytest.raises(TrainerError, match="inner"):
        AutoencoderFieldTrainer(inner="mlp")
    with pytest.raises(TrainerError, match="alpha"):
        AutoencoderFieldTrainer(alpha=True)
    with pytest.raises(TrainerError, match="tol"):
        AutoencoderFieldTrainer(tol=True)


def test_predict_before_fit_raises_trainer_error():
    trainer = AutoencoderFieldTrainer(max_iter=20)
    with pytest.raises(TrainerError, match="not fitted"):
        trainer.predict(np.zeros((2, 2)))


def test_autoencoder_field_predicts_smooth_holdout_field():
    x, y = smooth_field_fixture()
    trainer = AutoencoderFieldTrainer(
        latent_dim=2,
        encoder_hidden_layer_sizes=(16,),
        decoder_hidden_layer_sizes=(16,),
        activation="tanh",
        max_iter=800,
        learning_rate_init=5.0e-3,
        random_state=1,
        inner={"method": "mlp", "hidden_layer_sizes": (32, 32), "max_iter": 2000},
    )
    trainer.fit(x[:90], y[:90])
    pred = trainer.predict(x[90:])
    assert pred.shape == y[90:].shape
    assert r2_score(y[90:], pred) > 0.85


def test_pickle_round_trip_preserves_predictions():
    x, y = smooth_field_fixture()
    trainer = AutoencoderFieldTrainer(
        latent_dim=2,
        encoder_hidden_layer_sizes=(16,),
        decoder_hidden_layer_sizes=(16,),
        activation="tanh",
        max_iter=400,
        learning_rate_init=5.0e-3,
        random_state=2,
        inner={"method": "mlp", "hidden_layer_sizes": (24, 24), "max_iter": 1500},
    )
    trainer.fit(x[:90], y[:90])
    before = trainer.predict(x[90:95])
    loaded = pickle.loads(pickle.dumps(trainer))
    np.testing.assert_allclose(loaded.predict(x[90:95]), before, rtol=1.0e-6, atol=1.0e-6)


def test_public_metadata_matches_surroforge_contract():
    assert AutoencoderFieldTrainer.name == "autoencoder-field"
    assert AutoencoderFieldTrainer.export_formats == frozenset({"pickle"})
    assert AutoencoderFieldTrainer.reproduction_tolerance > 0
