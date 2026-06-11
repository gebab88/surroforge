"""Honest-basis cross-validation (Story R3.3, PRD 4.6, #40).

The per-fold fresh-factory contract (``cross_validate`` builds a fresh trainer per
fold) already re-derives mean + POD basis from each fold's training snapshots only.
These tests PIN that honesty: a deliberately LEAKY cross-validation — global basis
fit on ALL snapshots, otherwise byte-for-byte the same splits and inner fits — is
demonstrably more optimistic. The a3 = 1.8 fixture's amplitudes are degree-2
polynomials in x, so the degree-2 inner is exact (per-fold surrogate ~0) and the
honest-vs-leaky gap isolates basis leakage alone. The report surfaces the per-fold
basis spread as a labeled first-class line, pinned here on the degree-1 / a3 = 0.8
fixture where truncation floors and energies differ at 5g (the exact-inner fixture
is value-degenerate: floors == energies bitwise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import run_training
from surroforge.plugins import get_trainer
from surroforge.services import generate_report
from surroforge.validators.metrics import AVAILABLE_METRICS, two_source_split

LABEL = "Basis variance across folds (honest CV)"


# --- pinned fixture (story Dev Notes; same family as test_two_source_report.py) ---------

def _field_data(a3):
    rng = np.random.default_rng(42)
    n_dof, n = 40, 80
    modes = np.linalg.qr(rng.standard_normal((n_dof, 3)))[0]
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    amps = np.column_stack([5.0 * x[:, 0], 2.0 * x[:, 1] ** 2, a3 * x[:, 0] * x[:, 1]])
    return x, amps @ modes.T


def _field_run(tmp_path, a3, degree=2, model=None):
    """Pinned rank-3 field, trained rank-2 through the real engine path."""
    x, y = _field_data(a3)
    cols: dict[str, np.ndarray] = {"p0": x[:, 0], "p1": x[:, 1]}
    for j in range(y.shape[1]):
        cols[f"f{j}"] = y[:, j]
    csv = tmp_path / "field.csv"
    pd.DataFrame(cols).to_csv(csv, index=False)
    proj = Project.init(tmp_path / "proj", name="rom")
    proj.add_dataset(csv, name="snaps")
    cfg = parse_config(
        {
            "experiment": "rom-honest-cv",
            "data": {
                "source": "snaps",
                "inputs": ["p0", "p1"],
                "outputs": ["f*"],
                "split": {"test_fraction": 0.25, "seed": 1},
            },
            "model": model
            or {
                "method": "reduced-field",
                "reduction": {"basis": "pod", "rank": 2},
                "inner": {"method": "polynomial", "degree": degree},
            },
            "validation": {
                "cross_validation": 3,
                "metrics": ["energy_l2", "truncation_error", "rmse", "r2"],
            },
        }
    )
    return proj, run_training(proj, cfg)


def _leaky_cv_energies(a3):
    """Global-basis (LEAKY) CV: identical splits and inner fits, but mean + basis come
    from ALL snapshots, so every fold's test block helped shape the basis.

    Lives in the test module only — shipping a leaky CV path in src/, even "for
    comparison", would be an attractive nuisance (story constraint).
    """
    x, y = _field_data(a3)
    mean = y.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(y - mean, full_matrices=False)
    basis = vt[:2].T  # global rank-2 basis: it has SEEN every fold's test snapshots
    energy_l2 = AVAILABLE_METRICS["energy_l2"]
    # Engine replication: run_training passes seed=split.seed (= 1 in this fixture)
    # to cross_validate, NOT the run-config seed; CrossValidationConfig.shuffle
    # defaults to True.
    kfold = KFold(n_splits=3, shuffle=True, random_state=1)
    values = []
    for train_idx, test_idx in kfold.split(x):
        coeffs_train = (y[train_idx] - mean) @ basis
        pred_coeffs = np.empty((test_idx.size, basis.shape[1]))
        for k in range(basis.shape[1]):
            inner = get_trainer("polynomial")(degree=2)
            inner.fit(x[train_idx], coeffs_train[:, k : k + 1])
            pred_coeffs[:, k] = inner.predict(x[test_idx]).reshape(-1)
        values.append(energy_l2(y[test_idx], pred_coeffs @ basis.T + mean))
    return values


# --- the honesty pin (AC 1) ---------------------------------------------------------------

def test_leaky_global_basis_cv_is_more_optimistic(tmp_path):
    _proj, run = _field_run(tmp_path, a3=1.8)
    cv = run.metrics["cross_validation"]
    honest_e = cv["energy_l2"]["per_fold"]
    honest_t = cv["truncation_error"]["per_fold"]

    # Exact inner (degree-2 amplitudes): the per-fold surrogate is ~0, so the
    # total error IS the truncation floor and the honest-vs-leaky gap below
    # measures basis leakage alone — nothing else differs between the two CVs.
    for e, t in zip(honest_e, honest_t, strict=True):
        assert two_source_split(e, t)["surrogate"] < 1e-6

    # Engine-measured (documentation, asserted with margin — never 5g-string
    # equality on SVD-derived outputs): honest per-fold = [0.19243, 0.17422, 0.18693].
    assert honest_e == pytest.approx([0.19243, 0.17422, 0.18693], rel=1e-3)

    leaky = _leaky_cv_energies(a3=1.8)
    honest_mean = float(np.mean(honest_e))
    leaky_mean = float(np.mean(leaky))
    # Measured optimism: honest 0.18453 vs leaky 0.17047 (ratio 1.082, diff 0.0141);
    # both thresholds carry 2x headroom. A leaky cross_validate would give
    # honest == leaky (ratio 1.0) and fail decisively.
    assert honest_mean > leaky_mean * 1.04
    assert honest_mean - leaky_mean > 0.007


# --- basis variance as a first-class report line (AC 2) ------------------------------------

def _basis_variance_line(text: str) -> str:
    """The single labeled line — scoped extraction is mandatory: the std string also
    appears in the CV-std cells of both report tables (4x), so whole-report substring
    assertions are vacant (validation finding)."""
    matches = [line for line in text.splitlines() if line.startswith(LABEL)]
    assert len(matches) == 1, f"expected exactly one basis-variance line, got {len(matches)}"
    return matches[0]


def test_report_renders_per_fold_basis_variance_line(tmp_path):
    # Degree-1 / a3=0.8: the discriminating fixture — floors differ from energies
    # at 5g, so wiring the line to energy_l2 per_fold (or cv_std.total) fails.
    proj, run = _field_run(tmp_path, a3=0.8, degree=1)
    _path, text = generate_report(proj, run)

    cv = run.metrics["cross_validation"]
    floors = cv["truncation_error"]["per_fold"]
    energies = cv["energy_l2"]["per_fold"]
    assert {f"{v:.5g}" for v in floors}.isdisjoint(f"{v:.5g}" for v in energies)
    assert f"{np.std(floors):.5g}" != f"{np.std(energies):.5g}"

    # Exact full-line equality, recomputed from run.metrics (not golden): pins
    # order, count, and structure — per-value substring checks would tolerate a
    # scrambled or duplicated join. BLAS-safe: both sides render the same floats
    # from the same record with the same {:.5g}.
    joined = ", ".join(f"{v:.5g}" for v in floors)
    expected = f"{LABEL}: per-fold truncation floors {joined} (std {np.std(floors):.5g})."
    assert _basis_variance_line(text) == expected


def test_unreduced_method_gets_no_basis_variance_line(tmp_path):
    # Field metrics on a plain polynomial run (no lineage["reduction"]): the section
    # renders, but there is no per-fold basis to speak of — no basis-variance line.
    proj, run = _field_run(tmp_path, a3=2.0, model={"method": "polynomial", "degree": 2})
    assert "reduction" not in run.lineage
    _path, text = generate_report(proj, run)
    assert "## Two-Source Error Split" in text
    assert LABEL not in text
