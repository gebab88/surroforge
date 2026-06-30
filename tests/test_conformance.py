"""The shipped trainer conformance suite (Story 5.3 / AR10).

Core runs the same suite plugin authors use against its three built-in trainers,
and we assert the suite has teeth by running it against deliberately broken ones.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from surroforge.conformance import ConformanceError, check_trainer
from surroforge.errors import TrainerError
from surroforge.trainers.base import BaseTrainer
from surroforge.trainers.gradient_boosting import GradientBoostingTrainer
from surroforge.trainers.kriging import KrigingTrainer
from surroforge.trainers.mlp import MLPTrainer
from surroforge.trainers.polynomial import PolynomialTrainer
from surroforge.trainers.random_forest import RandomForestTrainer
from surroforge.trainers.rbf import RBFTrainer
from surroforge.trainers.reduced_field import ReducedFieldTrainer
from tests.conftest import blas_is_bit_reproducible

# Kriging and RBF need the SMT backend; skip them where it is not installed
# (the conformance suite itself stays backend-agnostic, and CI ships SMT).
_needs_smt = pytest.mark.skipif(
    importlib.util.find_spec("smt") is None,
    reason="requires the 'smt' optional backend",
)

BUILTINS = [
    pytest.param(PolynomialTrainer, id="polynomial"),
    pytest.param(KrigingTrainer, id="kriging", marks=_needs_smt),
    pytest.param(RBFTrainer, id="rbf", marks=_needs_smt),
    # MLP is scikit-learn-backed (Core), so it is conformant without any extra.
    pytest.param(MLPTrainer, id="mlp"),
    pytest.param(GradientBoostingTrainer, id="gradient-boosting"),
    pytest.param(RandomForestTrainer, id="random-forest"),
    # Default inner is polynomial, so reduced-field is conformant without SMT.
    pytest.param(ReducedFieldTrainer, id="reduced-field"),
]

# The full set of contract checks the suite is expected to run.
EXPECTED_CHECKS = {
    "subclass_of_base_trainer",
    "constructs_from_hyperparameters",
    "fit_returns_none",
    "predict_returns_2d",
    "picklable",
    "deterministic",
    "fingerprint_participation",
    "surroforge_error_only",
    "invalid_hyperparameter_error_only",
    "declares_capabilities",
    "reconstruction_round_trip",
    "mesh_free",
}


@pytest.mark.parametrize("trainer_cls", BUILTINS)
def test_builtin_trainer_is_conformant(trainer_cls):
    report = check_trainer(trainer_cls)
    if trainer_cls is KrigingTrainer and not blas_is_bit_reproducible():
        # Apple Accelerate's LAPACK is not bit-reproducible, so gradient-MLE
        # Kriging's determinism probe (two independent fits) legitimately fails
        # here — the 1e-9 guarantee holds on OpenBLAS/CI, not on macOS/Accelerate.
        # Every other contract check must still pass: the failure is *only*
        # allowed to be 'deterministic', nothing else.
        assert {r.name for r in report.failures} <= {"deterministic"}, report.summary()
        return
    assert report.passed, report.summary()
    report.raise_for_status()  # must not raise


@pytest.mark.parametrize("trainer_cls", BUILTINS)
def test_suite_runs_every_contract_check(trainer_cls):
    report = check_trainer(trainer_cls)
    assert {r.name for r in report.results} == EXPECTED_CHECKS


# --- The suite must catch broken trainers (teeth) ----------------------------


class _OneDimTrainer(BaseTrainer):
    """Violates the 2-D predict contract by returning a flat array."""

    name = "bad-1d"

    def fit(self, x, y):
        self._mean = float(np.mean(self._as_2d(y)))

    def predict(self, x):
        return np.full(self._as_2d(x).shape[0], self._mean)  # 1-D, wrong


class _BareErrorTrainer(BaseTrainer):
    """Fails with a bare ValueError instead of a SurroForgeError."""

    name = "bad-error"

    def fit(self, x, y):
        self._fitted = True

    def predict(self, x):
        if not getattr(self, "_fitted", False):
            raise ValueError("not fitted")  # should be a TrainerError
        return self._as_2d(np.zeros(self._as_2d(x).shape[0]))


class _NoisyTrainer(BaseTrainer):
    """Non-deterministic: fresh randomness on every predict."""

    name = "bad-nondeterministic"

    def fit(self, x, y):
        self._fitted = True

    def predict(self, x):
        n = self._as_2d(x).shape[0]
        if not getattr(self, "_fitted", False):
            raise TrainerError("not fitted")
        return self._as_2d(np.random.default_rng().uniform(size=n))


class _WrongColsTrainer(BaseTrainer):
    """Returns the wrong number of output columns (3 for a single-output target)."""

    name = "bad-cols"

    def fit(self, x, y):
        self._fitted = True

    def predict(self, x):
        if not getattr(self, "_fitted", False):
            raise TrainerError("not fitted")
        return np.zeros((self._as_2d(x).shape[0], 3))  # wrong arity


class _BareHyperErrorTrainer(BaseTrainer):
    """Validates hyperparameters but with a bare ValueError, not a SurroForgeError."""

    name = "bad-hyper-error"

    def __init__(self, **hp):
        if hp:  # rejects unknown hyperparameters — but with the wrong error type
            raise ValueError(f"unknown hyperparameter(s): {sorted(hp)}")
        super().__init__()

    def fit(self, x, y):
        self._fitted = True

    def predict(self, x):
        if not getattr(self, "_fitted", False):
            raise TrainerError("not fitted")
        return self._as_2d(np.zeros(self._as_2d(x).shape[0]))


class _NoPickleTrainer(BaseTrainer):
    """Fails to declare the universal 'pickle' export capability (Story 5.4)."""

    name = "bad-no-pickle"
    export_formats = frozenset()  # must always include "pickle"

    def fit(self, x, y):
        self._fitted = True

    def predict(self, x):
        if not getattr(self, "_fitted", False):
            raise TrainerError("not fitted")
        return self._as_2d(np.zeros(self._as_2d(x).shape[0]))


class _NegativeToleranceTrainer(BaseTrainer):
    """Declares an invalid (negative) reproduction tolerance (Story 5.4)."""

    name = "bad-tolerance"
    reproduction_tolerance = -1.0

    def fit(self, x, y):
        self._fitted = True

    def predict(self, x):
        if not getattr(self, "_fitted", False):
            raise TrainerError("not fitted")
        return self._as_2d(np.zeros(self._as_2d(x).shape[0]))


def test_suite_flags_non_2d_predict():
    report = check_trainer(_OneDimTrainer)
    assert "predict_returns_2d" in {r.name for r in report.failures}
    with pytest.raises(ConformanceError, match="predict_returns_2d"):
        report.raise_for_status()


def test_suite_flags_non_surroforge_error():
    report = check_trainer(_BareErrorTrainer)
    failed = {r.name for r in report.failures}
    assert "surroforge_error_only" in failed


def test_suite_flags_nondeterminism():
    report = check_trainer(_NoisyTrainer)
    assert "deterministic" in {r.name for r in report.failures}


def test_report_summary_lists_each_check():
    report = check_trainer(PolynomialTrainer)
    summary = report.summary()
    for name in EXPECTED_CHECKS:
        assert name in summary
    assert "PASS" in summary


def test_suite_flags_wrong_output_columns():
    report = check_trainer(_WrongColsTrainer)
    assert "predict_returns_2d" in {r.name for r in report.failures}


def test_suite_flags_non_surroforge_hyperparameter_error():
    report = check_trainer(_BareHyperErrorTrainer)
    assert "invalid_hyperparameter_error_only" in {r.name for r in report.failures}


@pytest.mark.parametrize("bad", [None, 42, "polynomial", object()])
def test_check_trainer_does_not_crash_on_non_class(bad):
    # A non-class argument must produce a clean failing report, never an exception.
    report = check_trainer(bad)
    assert not report.passed
    assert "subclass_of_base_trainer" in {r.name for r in report.failures}


def test_suite_flags_undeclared_pickle_export():
    report = check_trainer(_NoPickleTrainer)
    assert "declares_capabilities" in {r.name for r in report.failures}


def test_suite_flags_invalid_reproduction_tolerance():
    report = check_trainer(_NegativeToleranceTrainer)
    assert "declares_capabilities" in {r.name for r in report.failures}


# --- Story R4.3: reconstruction round-trip + no-mesh checks (teeth) -----------


class _NonOrthonormalReducer(ReducedFieldTrainer):
    """Reports a scaled (non-orthonormal) basis: Bᵀ·B ≠ I, so the round-trip must fail.

    ``predict_coefficients`` stays consistent with the true orthonormal basis, so the
    failure is purely the reported basis being a non-faithful projector — exactly the
    sign/normalization/scaling class of bug the check targets.
    """

    name = "bad-nonorthonormal-reducer"

    def reduction_artifacts(self):
        artifacts = dict(super().reduction_artifacts())
        artifacts["basis"] = np.asarray(artifacts["basis"], dtype=float) * 2.0
        return artifacts


class _GeometryDependentReducer(ReducedFieldTrainer):
    """Adds a fixed DOF-index ramp on reconstruction — breaks permutation equivariance.

    A faithful mesh-free reducer treats DOFs as exchangeable; this one makes the
    reconstructed value depend on the column index, so permuting the snapshot columns no
    longer permutes the field identically.
    """

    name = "bad-geometry-reducer"

    def predict(self, x):
        field = super().predict(x)
        return field + np.arange(field.shape[1], dtype=float)


class _NaNBasisReducer(ReducedFieldTrainer):
    """Reports a basis containing a NaN — must FAIL the round-trip, not slip through.

    Without the finiteness guard, `diff` is NaN and `nan > tol` is False, so a corrupt
    basis would PASS. predict() uses the true basis, so only the round-trip is affected.
    """

    name = "bad-nan-basis-reducer"

    def reduction_artifacts(self):
        artifacts = dict(super().reduction_artifacts())
        basis = np.array(artifacts["basis"], dtype=float)  # writable copy of the frozen view
        basis[0, 0] = np.nan
        artifacts["basis"] = basis
        return artifacts


def test_suite_flags_non_orthonormal_basis():
    report = check_trainer(_NonOrthonormalReducer)
    assert report.passed is False
    assert "reconstruction_round_trip" in {r.name for r in report.failures}
    # The bug is isolated to the projector: mesh-freeness still holds (predict uses the
    # true basis), so this discriminates a non-orthonormal basis from a broken reducer.
    assert "mesh_free" not in {r.name for r in report.failures}


def test_suite_flags_geometry_dependent_reducer():
    report = check_trainer(_GeometryDependentReducer)
    assert report.passed is False
    assert "mesh_free" in {r.name for r in report.failures}
    # The projector itself is fine (round-trip reads the basis, not predict), so the
    # failure is specifically the geometry dependence, not reducer breakage in general.
    assert "reconstruction_round_trip" not in {r.name for r in report.failures}


def test_suite_flags_nan_in_basis():
    # Teeth for the finiteness guard: a NaN basis must FAIL, not pass via `nan > tol == False`.
    report = check_trainer(_NaNBasisReducer)
    assert report.passed is False
    assert "reconstruction_round_trip" in {r.name for r in report.failures}


def test_correct_reducer_passes_both_rom_checks():
    """Trap guard: the faithful built-in reducer PASSES both checks the broken ones fail."""
    report = check_trainer(ReducedFieldTrainer)
    by_name = {r.name: r for r in report.results}
    assert by_name["reconstruction_round_trip"].passed, report.summary()
    assert by_name["mesh_free"].passed, report.summary()


def test_scalar_trainer_rom_checks_not_applicable():
    report = check_trainer(PolynomialTrainer)
    by_name = {r.name: r for r in report.results}
    for name in ("reconstruction_round_trip", "mesh_free"):
        assert by_name[name].passed
        assert "not a reduced-field trainer" in by_name[name].detail
    assert report.passed is True
