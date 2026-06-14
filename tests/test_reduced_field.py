"""ReducedFieldTrainer — POD field reduction + per-mode inner surrogate (Story R1.3).

The spine-theorem check (#36): a field run goes through the *unchanged* engine
end-to-end, because the reducer is an ordinary BaseTrainer emitting (n, n_dof).
Story R1.4 extends the proof to all three built-in inner methods: kriging/rbf/
polynomial serve as per-mode inner surrogates with zero changes to the inner
trainers, the engine, or the conformance suite.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from surroforge.conformance import check_trainer
from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import run_training
from surroforge.errors import TrainerError
from surroforge.plugins import available_trainers, get_trainer
from surroforge.services import assert_reproducible, reproduce
from surroforge.trainers.reduced_field import ReducedFieldTrainer
from tests.conftest import blas_is_bit_reproducible, requires_reproducible_blas

# Kriging and RBF need the SMT backend; skip them where it is not installed
# (mirrors tests/test_conformance.py — replicated, not imported, to keep
# cross-test-module coupling to the single low_rank_field precedent).
_needs_smt = pytest.mark.skipif(
    importlib.util.find_spec("smt") is None,
    reason="requires the 'smt' optional backend",
)


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


def test_missing_rom_backend_yields_install_hint(monkeypatch):
    """R4.4: training reduced-field without the ROM backend gives an actionable hint.

    The trainer stays registered (mirroring [smt]/[onnx]); the missing-backend case
    surfaces a clean TrainerError naming `pip install 'surroforge[rom]'` at fit time —
    NOT a raw ImportError. `sys.modules["scipy"] = None` makes the guard's `import
    scipy.linalg` raise, simulating an install without the ROM stack.
    """
    monkeypatch.setitem(sys.modules, "scipy", None)
    p, field = low_rank_field(n_samples=20, n_dof=6, k_true=2)
    t = ReducedFieldTrainer(reduction={"energy": 0.99}, inner={"method": "polynomial"})
    with pytest.raises(TrainerError, match=r"surroforge\[rom\]"):
        t.fit(p, field)


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


# --- Story R1.4: built-ins as inner surrogates, unmodified -------------------

# Per-mode targets are single-output (n, 1) slices — squarely inside the SMT MVP
# envelope (Kriging/RBF reject only y.shape[1] != 1, which the reducer never
# produces). Small fixture on purpose: each mode x each CV fold x holdout x final
# is a separate SMT fit, so r~2 with 3 folds keeps a kriging run at ~10 fits.

INNER_METHODS = [
    # degree 3 on purpose: it differs from BOTH the default-inner-method fallback
    # AND PolynomialTrainer's own default (2), so the hyper pass-through assertion
    # below has teeth for the one param row that runs everywhere.
    pytest.param("polynomial", {"degree": 3}, id="polynomial"),
    pytest.param("kriging", {}, id="kriging", marks=_needs_smt),
    pytest.param("rbf", {}, id="rbf", marks=_needs_smt),
]


def _inner_field_run(tmp_path, method: str, hyper: dict):
    """Train a reduced-field run with the given inner method through the unchanged engine."""
    p, field = low_rank_field(n_samples=60, n_dof=6, k_true=2, seed=1)
    cols = {"p0": p[:, 0], "p1": p[:, 1]}
    for j in range(field.shape[1]):
        cols[f"f{j}"] = field[:, j]
    csv = tmp_path / "field.csv"
    pd.DataFrame(cols).to_csv(csv, index=False)
    proj = Project.init(tmp_path / "proj", name="rom")
    proj.add_dataset(csv, name="snaps")
    cfg = parse_config(
        {
            "experiment": f"rom-inner-{method}",
            "data": {
                "source": "snaps",
                "inputs": ["p0", "p1"],
                "outputs": [f"f{j}" for j in range(6)],
                "split": {"test_fraction": 0.25, "seed": 1},
            },
            "model": {
                "method": "reduced-field",
                "reduction": {"basis": "pod", "energy": 0.99},
                "inner": {"method": method, **hyper},
            },
            "validation": {"cross_validation": 3, "metrics": ["rmse", "r2", "mae"]},
        }
    )
    return proj, run_training(proj, cfg), p


@pytest.mark.parametrize(("method", "hyper"), INNER_METHODS)
def test_each_builtin_trains_as_inner_through_unchanged_engine(tmp_path, method, hyper):
    proj, run, p = _inner_field_run(tmp_path, method, hyper)
    assert run.metrics["n_outputs"] == 6
    # 0.99, not 0.9: mode 1 alone holds ~0.945 of the fixture's energy, so a rank
    # collapse (or a garbage mode-2 fit) would clear a 0.9 floor. Real runs sit
    # at r² ≈ 1 - 5e-13 for all three methods.
    assert run.metrics["test"]["r2"] > 0.99
    model = run.load_model()
    assert model.predict(p[:4]).shape == (4, 6)
    # The discriminating assertions: metrics/shape above are identical under the
    # default-polynomial fallback, so they alone cannot prove `inner.method` was
    # honored. Pin the exact rank (k_true=2; mode energies 0.945/1.0 straddle the
    # 0.99 target with wide margin), one inner instance per retained mode of the
    # configured method, and the configured inner hyperparameters' pass-through.
    assert model.rank == 2
    assert len(model._models) == model.rank
    inner_cls = get_trainer(method)
    assert all(isinstance(im, inner_cls) for im in model._models)
    for key, value in hyper.items():
        assert all(im.hyperparameters[key] == value for im in model._models)


@pytest.mark.parametrize(
    "method",
    [pytest.param("kriging", marks=_needs_smt), pytest.param("rbf", marks=_needs_smt)],
)
def test_conformance_with_smt_inner(method):
    # The polynomial inner is covered by test_passes_conformance_suite above; this
    # adds the two SMT methods. check_trainer drives the trainer DIRECTLY (trainer
    # kwargs, bypassing the R1.2 config layer), complementing the config-path e2e.
    report = check_trainer(
        ReducedFieldTrainer,
        hyperparameters={"reduction": {"energy": 0.99}, "inner": {"method": method}},
    )
    if method == "kriging" and not blas_is_bit_reproducible():
        # Apple Accelerate's LAPACK is not bit-reproducible, so any check that fits
        # gradient-MLE Kriging TWICE legitimately diverges here: `deterministic` (two
        # independent fits) and `mesh_free` (a second fit on permuted DOFs). Both probe
        # fit-to-fit stability, which the 1e-9 guarantee only promises on OpenBLAS/CI.
        # `reconstruction_round_trip` is a SINGLE fit (projector identity) and `picklable`
        # re-predicts ONE fitted model — both stay bit-stable, so nothing else may fail.
        assert {r.name for r in report.failures} <= {"deterministic", "mesh_free"}, (
            report.summary()
        )
        return
    assert report.passed, report.summary()


@_needs_smt
@requires_reproducible_blas
def test_kriging_inner_field_run_reproduces(tmp_path):
    # Reproducibility for a tolerant SMT inner: reproduce() re-trains from the
    # snapshot, doubling the kriging fit count — hence the small Task-1 fixture.
    # Double-guarded: on Accelerate this measurably returns `diverged` (fit-to-fit
    # divergence is real there), so the BLAS guard is mandatory, not defensive.
    proj, run, _ = _inner_field_run(tmp_path, "kriging", {})
    result = reproduce(proj, run.run_id)
    assert result.verdict in {"reproduced", "reproduced_within_tolerance"}
    assert_reproducible(result)  # must not raise
