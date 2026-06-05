"""ReducedFieldTrainer — POD field reduction + per-mode inner surrogate (Story R1.3).

The spine-theorem check (#36): a field run goes through the *unchanged* engine
end-to-end, because the reducer is an ordinary BaseTrainer emitting (n, n_dof).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surroforge.conformance import check_trainer
from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import run_training
from surroforge.errors import TrainerError
from surroforge.plugins import available_trainers, get_trainer
from surroforge.trainers.reduced_field import ReducedFieldTrainer


def low_rank_field(n_samples=80, n_dof=40, k_true=3, seed=0):
    """A field of intrinsic rank k_true: smooth (≤ degree-2) amplitudes × spatial modes."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(-1.0, 1.0, (n_samples, 2))
    phi, _ = np.linalg.qr(rng.standard_normal((n_dof, k_true)))  # orthonormal modes
    amps = np.column_stack([p[:, 0], p[:, 1] ** 2, p[:, 0] * p[:, 1]])[:, :k_true]
    amps = amps * np.array([5.0, 2.0, 1.0])[:k_true]
    return p, amps @ phi.T


# --- registration -----------------------------------------------------------

def test_registered_as_builtin():
    assert "reduced-field" in available_trainers()
    assert get_trainer("reduced-field") is ReducedFieldTrainer


# --- reduction & reconstruction ---------------------------------------------

def test_reconstructs_low_rank_field_on_holdout():
    p, field = low_rank_field()
    t = ReducedFieldTrainer(
        reduction={"basis": "pod", "energy": 0.999},
        inner={"method": "polynomial", "degree": 2},
    )
    t.fit(p[:60], field[:60])
    pred = t.predict(p[60:])
    assert pred.shape == field[60:].shape
    rel = np.linalg.norm(pred - field[60:]) / np.linalg.norm(field[60:])
    assert rel < 1e-2, f"relative reconstruction error {rel:.2e}"
    assert t.rank <= 4  # energy 0.999 on a rank-3 field keeps ~3 modes


def test_rank_target_selects_exact_rank():
    p, field = low_rank_field()
    t = ReducedFieldTrainer(reduction={"rank": 2})
    t.fit(p, field)
    assert t.rank == 2
    assert len(t._models) == 2  # one inner surrogate per retained mode


def test_higher_energy_keeps_at_least_as_many_modes():
    p, field = low_rank_field()
    lo = ReducedFieldTrainer(reduction={"energy": 0.5})
    hi = ReducedFieldTrainer(reduction={"energy": 0.9999})
    lo.fit(p, field)
    hi.fit(p, field)
    assert hi.rank >= lo.rank >= 1


def test_predict_shape_is_field_width():
    p, field = low_rank_field(n_dof=25)
    t = ReducedFieldTrainer()
    t.fit(p, field)
    assert t.predict(p[:5]).shape == (5, 25)


def test_basis_is_deterministic_and_sign_canonical():
    p, field = low_rank_field()
    a = ReducedFieldTrainer(reduction={"rank": 3})
    b = ReducedFieldTrainer(reduction={"rank": 3})
    a.fit(p, field)
    b.fit(p, field)
    assert np.array_equal(a._basis, b._basis)  # deterministic basis
    for k in range(a._basis.shape[1]):  # each mode's max-abs entry is positive (#16)
        peak = int(np.argmax(np.abs(a._basis[:, k])))
        assert a._basis[peak, k] > 0


# --- contract / error handling ----------------------------------------------

def test_predict_before_fit_raises_trainer_error():
    with pytest.raises(TrainerError):
        ReducedFieldTrainer().predict(np.zeros((3, 2)))


def test_unknown_hyperparameter_raises():
    with pytest.raises(TrainerError):
        ReducedFieldTrainer(bogus=1)


def test_unknown_reduction_key_raises():
    with pytest.raises(TrainerError):
        ReducedFieldTrainer(reduction={"nonsense": 1})


def test_energy_and_rank_conflict_raises():
    with pytest.raises(TrainerError):
        ReducedFieldTrainer(reduction={"energy": 0.99, "rank": 3})


def test_non_pod_basis_raises():
    with pytest.raises(TrainerError):
        ReducedFieldTrainer(reduction={"basis": "autoencoder"})


def test_passes_conformance_suite():
    assert check_trainer(ReducedFieldTrainer).passed
    report = check_trainer(
        ReducedFieldTrainer, hyperparameters={"inner": {"method": "polynomial", "degree": 2}}
    )
    assert report.passed, report.summary()


# --- spine theorem: unchanged engine, end to end ----------------------------

def test_end_to_end_through_unchanged_engine(tmp_path):
    p, field = low_rank_field(n_samples=80, n_dof=6, k_true=3, seed=1)
    cols = {"p0": p[:, 0], "p1": p[:, 1]}
    for j in range(field.shape[1]):
        cols[f"f{j}"] = field[:, j]
    csv = tmp_path / "field.csv"
    pd.DataFrame(cols).to_csv(csv, index=False)

    proj = Project.init(tmp_path / "proj", name="rom")
    proj.add_dataset(csv, name="snaps")
    cfg = parse_config(
        {
            "experiment": "rom-smoke",
            "data": {
                "source": "snaps",
                "inputs": ["p0", "p1"],
                "outputs": [f"f{j}" for j in range(6)],  # the field is a column group
                "split": {"test_fraction": 0.25, "seed": 1},
            },
            "model": {
                "method": "reduced-field",
                "reduction": {"basis": "pod", "energy": 0.999},
                "inner": {"method": "polynomial", "degree": 2},
            },
            "validation": {"cross_validation": 3, "metrics": ["rmse", "r2", "mae"]},
        }
    )
    run = run_training(proj, cfg)

    assert run.metrics["n_outputs"] == 6
    assert run.metrics["test"]["r2"] > 0.95
    assert run.model_path.is_file()
    # The persisted reduced-field model reloads and reconstructs the field.
    model = run.load_model()
    assert model.predict(p[:4]).shape == (4, 6)
