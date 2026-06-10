"""Field metrics as registered callables (Story R3.1, PRD 4.6).

``energy_l2`` and ``truncation_error`` are plain ``(y_true, y_pred) -> float``
entries in ``AVAILABLE_METRICS``, so they flow through the engine, CV, compare,
and the report with no special handling. The pinned rank-3 fixture below makes
truncation discriminating: a rank-2 model on it has a floor ~0.15 while the
in-span (surrogate) error is ~0 — the two-source split r3-2 builds on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surroforge import services
from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import run_training
from surroforge.errors import ConfigError
from surroforge.services import compare_runs, generate_report
from surroforge.trainers.polynomial import PolynomialTrainer
from surroforge.trainers.reduced_field import ReducedFieldTrainer
from surroforge.validators.metrics import (
    AVAILABLE_METRICS,
    FIELD_ENERGY_FLOOR,
    compute_metrics,
    validate_metric_names,
)


def _rank3_field(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Pinned discriminating fixture (story Dev Notes — do not weaken the third mode).

    Rank-3 field over orthonormal spatial modes; the third amplitude is deliberately
    strong (2, not 1) so the rank-2 truncation floor stays well above the 0.1
    assertion on any faithful replication (0.153 on the pinned seed 42; 0.16-0.30
    across seeds 0-19).
    """
    rng = np.random.default_rng(seed)
    n_dof, n = 40, 80
    modes = np.linalg.qr(rng.standard_normal((n_dof, 3)))[0]
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    amps = np.column_stack([5.0 * x[:, 0], 2.0 * x[:, 1] ** 2, 2.0 * x[:, 0] * x[:, 1]])
    return x, amps @ modes.T


def _both(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    m = compute_metrics(y_true, y_pred, ["energy_l2", "truncation_error"])
    return m["energy_l2"], m["truncation_error"]


# --- registration + unknown-metric regression (AC 1, 2) -------------------------------

def test_new_metrics_are_registered():
    assert "energy_l2" in AVAILABLE_METRICS
    assert "truncation_error" in AVAILABLE_METRICS


def test_unknown_metric_error_lists_the_new_names():
    # Recomputed from the registry on purpose — no golden string literal.
    with pytest.raises(ConfigError, match="bogus") as excinfo:
        validate_metric_names(["bogus"])
    message = str(excinfo.value)
    assert str(sorted(AVAILABLE_METRICS)) in message
    assert "energy_l2" in message and "truncation_error" in message


# --- energy_l2 (AC 3) ------------------------------------------------------------------

def test_energy_l2_is_the_relative_frobenius_energy_error():
    rng = np.random.default_rng(7)
    y_true = rng.standard_normal((30, 5))
    delta = rng.standard_normal((30, 5)) * 0.01
    expected = np.linalg.norm(delta) / np.linalg.norm(y_true)  # recomputed, not golden
    value = AVAILABLE_METRICS["energy_l2"](y_true, y_true + delta)
    assert value == pytest.approx(expected, rel=1e-12)


def test_energy_l2_is_global_not_per_column_average():
    # One loud column among quiet ones: a per-column uniform_average would weight
    # the relative errors equally; the global Frobenius norm (the verdict's norm)
    # is dominated by the energetic column.
    y_true = np.column_stack([np.full(20, 100.0), np.full(20, 0.01)])
    y_pred = np.column_stack([np.full(20, 101.0), np.full(20, 0.02)])
    expected = np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true)
    assert AVAILABLE_METRICS["energy_l2"](y_true, y_pred) == pytest.approx(expected)


def test_energy_l2_zero_field_guard():
    zeros = np.zeros((5, 4))
    assert AVAILABLE_METRICS["energy_l2"](zeros, zeros) == 0.0


def test_field_energy_floor_relocated_single_source():
    # services re-exports the relocated constant — the verdict norm and the metric
    # share one definition (AC 3), byte-identical behavior.
    assert services.FIELD_ENERGY_FLOOR is FIELD_ENERGY_FLOOR
    assert FIELD_ENERGY_FLOOR == 1e-300


# --- truncation_error (AC 4) -------------------------------------------------------------

def test_truncated_model_shows_the_floor_full_rank_does_not():
    x, y = _rank3_field()
    x_train, x_test, y_train, y_test = x[:60], x[60:], y[:60], y[60:]

    truncated = ReducedFieldTrainer(
        reduction={"basis": "pod", "rank": 2},
        inner={"method": "polynomial", "degree": 2},
    )
    truncated.fit(x_train, y_train)
    energy, trunc = _both(y_test, truncated.predict(x_test))
    assert trunc > 0.1  # measured 0.153 on the pinned seed 42
    # Pythagoras gives trunc <= energy always; the exact polynomial inner leaves no
    # in-span error, so the two coincide. Margins, not bit-equality (BLAS discipline):
    # the DERIVED gap sqrt(energy² − trunc²) amplifies 1-ulp rounding noise on two
    # ~0.15 values to ~1e-8, so it must not be asserted against a ~1e-12 threshold.
    assert trunc <= energy + 1e-12
    assert trunc == pytest.approx(energy, abs=1e-9)
    in_span = np.sqrt(max(energy**2 - trunc**2, 0.0))
    assert in_span < 1e-6  # still 5 orders below the 0.1 floor: ALL error is truncation

    full_rank = ReducedFieldTrainer(
        reduction={"basis": "pod", "rank": 3},
        inner={"method": "polynomial", "degree": 2},
    )
    full_rank.fit(x_train, y_train)
    _, trunc_full = _both(y_test, full_rank.predict(x_test))
    assert trunc_full < 1e-12


def test_scalar_model_has_no_truncation_floor():
    rng = np.random.default_rng(3)
    x = rng.uniform(-1.0, 1.0, size=(80, 2))
    y = np.column_stack([3.0 * x[:, 0] + 1.0, x[:, 1] ** 2])
    model = PolynomialTrainer(degree=2)
    model.fit(x[:60], y[:60])
    _, trunc = _both(y[60:], model.predict(x[60:]))
    assert trunc < 1e-12


def test_all_zero_predictions_take_the_empty_basis_path():
    rng = np.random.default_rng(5)
    y_true = rng.standard_normal((10, 4))
    y_pred = np.zeros((10, 4))
    # Empty basis -> the metric is the full relative residual of y_true.
    assert AVAILABLE_METRICS["truncation_error"](y_true, y_pred) == pytest.approx(1.0)


def test_single_sample_is_defined_not_crashing():
    y_true = np.array([[1.0, 2.0, 3.0]])
    y_pred = np.array([[1.0, 1.0, 3.0]])
    # Centered predictions are zero -> empty basis -> full residual against the
    # prediction anchor; identical to energy_l2 in this degenerate case.
    energy, trunc = _both(y_true, y_pred)
    assert np.isfinite(trunc)
    assert trunc == pytest.approx(energy)


def test_zero_field_no_crash_both_metrics():
    zeros = np.zeros((6, 8))
    energy, trunc = _both(zeros, zeros)
    assert energy == 0.0
    assert trunc == 0.0


@pytest.mark.parametrize("metric", ["energy_l2", "truncation_error"])
def test_field_metrics_reject_contract_violations(metric):
    # The sklearn-backed metrics raise on all of these via check_array; the field
    # metrics must not silently broadcast, return nan, or leak a raw LinAlgError.
    fn = AVAILABLE_METRICS[metric]
    good = np.ones((4, 3))
    with pytest.raises(ValueError, match="2-D"):
        fn(np.ones(4), np.ones(4))  # 1-D: clear error, not LinAlgError
    with pytest.raises(ValueError, match="2-D"):
        fn(np.ones((4, 2)), good)  # column mismatch: no silent broadcasting
    with pytest.raises(ValueError, match="non-empty"):
        fn(np.empty((0, 3)), np.empty((0, 3)))  # no perfect score for no data
    with pytest.raises(ValueError, match="finite"):
        fn(good, np.full((4, 3), np.nan))
    with pytest.raises(ValueError, match="finite"):
        fn(np.full((4, 3), np.inf), good)


# --- e2e: engine, CV, report, compare — all unchanged (AC 1, 5) --------------------------

def test_field_metrics_flow_through_unchanged_engine_report_and_compare(tmp_path):
    x, y = _rank3_field()
    cols: dict[str, np.ndarray] = {"p0": x[:, 0], "p1": x[:, 1]}
    for j in range(y.shape[1]):
        cols[f"f{j}"] = y[:, j]
    csv = tmp_path / "field.csv"
    pd.DataFrame(cols).to_csv(csv, index=False)
    proj = Project.init(tmp_path / "proj", name="rom")
    proj.add_dataset(csv, name="snaps")

    def _cfg(experiment: str, rank: int):
        return parse_config(
            {
                "experiment": experiment,
                "data": {
                    "source": "snaps",
                    "inputs": ["p0", "p1"],
                    "outputs": ["f*"],
                    "split": {"test_fraction": 0.25, "seed": 1},
                },
                "model": {
                    "method": "reduced-field",
                    "reduction": {"basis": "pod", "rank": rank},
                    "inner": {"method": "polynomial", "degree": 2},
                },
                "validation": {
                    "cross_validation": 3,
                    "metrics": ["energy_l2", "truncation_error", "rmse"],
                },
            }
        )

    run = run_training(proj, _cfg("rom-field-metrics", rank=2))  # truncating on purpose

    # All three metrics in train/test/CV, exactly like rmse today (AC 1).
    for split in ("train", "test"):
        for name in ("energy_l2", "truncation_error", "rmse"):
            assert name in run.metrics[split]
    cv = run.metrics["cross_validation"]
    assert cv["folds"] == 3
    for name in ("energy_l2", "truncation_error", "rmse"):
        assert {"mean", "std"} <= cv[name].keys()
        assert len(cv[name]["per_fold"]) == 3

    # Test-split values satisfy the Pythagorean relation; the exact polynomial
    # inner means essentially all error is truncation (in-span ~0).
    energy = run.metrics["test"]["energy_l2"]
    trunc = run.metrics["test"]["truncation_error"]
    assert 0.05 < trunc <= energy + 1e-12
    assert trunc == pytest.approx(energy, rel=1e-6)

    # Report renders the new names with zero reporter changes (AC 5).
    _path, text = generate_report(proj, run)
    assert "energy_l2" in text
    assert "truncation_error" in text

    # compare under the lower-is-better default (AC 5): a second, full-rank run has
    # decisively lower energy_l2, so best-selection is discriminating — a future flip
    # of the .get(metric, True) default would pick the truncated run and fail here.
    run_full = run_training(proj, _cfg("rom-field-metrics-full", rank=3))
    assert run_full.metrics["test"]["energy_l2"] < trunc / 100
    result = compare_runs(proj, [run.run_id, run_full.run_id], metric="energy_l2")
    assert result["best"] == run_full.run_id
