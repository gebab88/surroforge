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
from surroforge.trainers.kriging import KrigingTrainer
from surroforge.trainers.polynomial import PolynomialTrainer
from surroforge.trainers.rbf import RBFTrainer

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
}


@pytest.mark.parametrize("trainer_cls", BUILTINS)
def test_builtin_trainer_is_conformant(trainer_cls):
    report = check_trainer(trainer_cls)
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
