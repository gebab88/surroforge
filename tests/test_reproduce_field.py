"""Field-primary reproducibility verdict (Story R2.3).

The verdict for a *field run* is the relative field-energy difference of the
reconstructed out-of-sample holdout field; the basis hash and canonicalized
coefficients are a diagnostic lens, never a gate (spec-field-rom-fingerprint.md).
A *scalar run* keeps the absolute byte/diff behaviour (the R2.4 guard).

These lift the R2.1 spike's three claims to the production verdict:
  - field metric is gauge-invariant (re-gauge → reproduced; basis hash is not the gate);
  - a degenerate cluster is compared as a subspace, never per-mode;
  - the field metric has teeth (a genuinely different field diverges).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import _spec_hash, run_training
from surroforge.errors import ReproducibilityError
from surroforge.services import (
    ReproduceResult,
    _basis_regauged,
    _coefficient_diagnostic,
    _field_energy_relative_diff,
    assert_reproducible,
    reproduce,
)
from tests.conftest import poly_config


@pytest.fixture(autouse=True)
def _grant_license(granted_license):
    """reproduce is Professional since Wave 3; this module tests the Core math behind it."""


N_DOF = 12


def make_field_samples(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """A smooth, low-rank field: 3 spatial modes mixed by smooth functions of (a, b).

    POD recovers a handful of modes, so the reduced-field path is genuinely exercised.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 3.0, n)
    b = rng.uniform(0.5, 1.5, n)
    nodes = np.linspace(0.0, 1.0, N_DOF)
    m1, m2, m3 = np.sin(np.pi * nodes), np.cos(2 * np.pi * nodes), nodes**2
    field = (
        a[:, None] * m1 + (a * b)[:, None] * m2 + b[:, None] * m3
        + rng.normal(0.0, 1e-3, (n, N_DOF))
    )
    cols: dict[str, np.ndarray] = {"a": a, "b": b}
    for j in range(N_DOF):
        cols[f"u{j}"] = field[:, j]
    return pd.DataFrame(cols)


def field_config(energy: float = 0.999) -> dict:
    """A reduced-field config over the registered 'field' dataset (deterministic inner)."""
    return {
        "experiment": "field-rom",
        "data": {
            "source": "field",
            "inputs": ["a", "b"],
            "outputs": [f"u{j}" for j in range(N_DOF)],
            "split": {"test_fraction": 0.25, "seed": 42},
        },
        "model": {
            "method": "reduced-field",
            "reduction": {"basis": "pod", "energy": energy},
            "inner": {"method": "polynomial", "degree": 2},
        },
        "validation": {"cross_validation": 3, "metrics": ["rmse", "r2", "mae"]},
    }


@pytest.fixture
def field_project(tmp_path):
    proj = Project.init(tmp_path / "fproj", name="field-proj")
    csv = tmp_path / "field.csv"
    make_field_samples().to_csv(csv, index=False)
    proj.add_dataset(csv, name="field")
    return proj


class _FakeRun:
    """Minimal stand-in exposing only what `_coefficient_diagnostic` reads."""

    def __init__(self, coeffs: np.ndarray) -> None:
        self.holdout_coefficients = np.asarray(coeffs, dtype=float).tolist()


# --- Diagnostics are recorded for field runs only (AC1, AC2) -----------------

def test_field_run_records_diagnostics(field_project):
    run = run_training(field_project, parse_config(field_config()))
    assert run.basis_hash  # non-empty hex digest
    assert len(run.basis_hash) == 64
    assert np.asarray(run.holdout_coefficients, dtype=float).ndim == 2
    assert run.reduction["basis"] == "pod"
    assert run.reduction["rank"] >= 1
    assert 0.0 < run.reduction["energy_captured"] <= 1.0
    assert run.spec_hash and len(run.spec_hash) == 64


def test_scalar_run_records_no_field_diagnostics(project):
    # A scalar (single-output) run must not grow any field lineage — the engine stays
    # ROM-agnostic and the scalar path is unchanged (AC5).
    run = run_training(project, parse_config(poly_config("polynomial")))
    assert run.basis_hash == ""
    assert run.holdout_coefficients == []
    assert run.reduction == {}
    assert run.spec_hash == ""


# --- The field verdict (AC1) -------------------------------------------------

def test_field_run_reproduces_within_tolerance(field_project):
    run = run_training(field_project, parse_config(field_config()))
    result = reproduce(field_project, run.run_id)
    assert result.verdict in {"reproduced", "reproduced_within_tolerance"}
    # The field-energy-relative metric drove the verdict (not the scalar abs diff).
    assert result.field_relative_diff is not None
    assert result.field_relative_diff <= result.tolerance
    assert result.data_hash_match
    assert_reproducible(result)  # must not raise


def test_scalar_run_uses_absolute_metric_not_field(project):
    # AC5: a scalar reproduce leaves field_relative_diff None and judges by max_abs_diff.
    run = run_training(project, parse_config(poly_config("polynomial")))
    result = reproduce(project, run.run_id)
    assert result.field_relative_diff is None
    assert result.basis_hash_match is None
    assert not result.basis_regauged
    assert result.verdict == "reproduced"


# --- Gauge invariance: a re-gauged basis still reproduces (AC2, AC3) ----------

def test_basis_regauged_flag_logic():
    # AC2 annotation: differ + reproduced → re-gauged; match/None or diverged → not.
    assert _basis_regauged(False, "reproduced") is True
    assert _basis_regauged(False, "reproduced_within_tolerance") is True
    assert _basis_regauged(False, "diverged") is False
    assert _basis_regauged(True, "reproduced") is False     # bytes matched → not re-gauged
    assert _basis_regauged(None, "reproduced") is False     # a hash was missing → unknown


def test_field_metric_is_gauge_invariant():
    # An arbitrary orthogonal re-gauge (sign + reorder + intra-subspace rotation) leaves
    # the reconstructed field identical, so the field-energy-relative diff is ~0 →
    # "reproduced" — proving the verdict does NOT key on the basis bytes (which change).
    rng = np.random.default_rng(3)
    n_dof, r, m = 30, 4, 15
    basis, _ = np.linalg.qr(rng.standard_normal((n_dof, r)))
    coeffs = rng.standard_normal((m, r))
    field = coeffs @ basis.T

    gauge, _ = np.linalg.qr(rng.standard_normal((r, r)))  # orthogonal: field-invariant
    regauged_field = (coeffs @ gauge) @ (basis @ gauge).T

    assert _field_energy_relative_diff(field, regauged_field) < 1e-10
    assert not np.allclose(basis @ gauge, basis)  # the basis genuinely re-gauged


# --- Degenerate cluster: subspace, never per-mode (AC3) ----------------------

def test_coefficient_diagnostic_uses_subspace_for_degenerate_cluster():
    # σ = [10, 5, 5, 1]: modes 1,2 are degenerate. Rotate that pair 40°: per-mode would
    # false-diverge, but the principal-angle subspace test reads ~0; the non-degenerate
    # modes (0, 3) are untouched and report ~0 per-mode diff.
    rng = np.random.default_rng(5)
    sigma = np.array([10.0, 5.0, 5.0, 1.0])
    frame, _ = np.linalg.qr(rng.standard_normal((40, sigma.size)))
    pc = frame * sigma
    g = np.eye(sigma.size)
    c, s = np.cos(np.radians(40.0)), np.sin(np.radians(40.0))
    g[1, 1], g[1, 2], g[2, 1], g[2, 2] = c, -s, s, c
    rc = pc @ g

    diag = _coefficient_diagnostic(_FakeRun(pc), _FakeRun(rc))
    assert diag is not None
    assert sorted(diag["degenerate_clusters"][0]) == [1, 2]
    assert diag["max_subspace_angle"] < 1e-6                 # subspace reproduced
    assert set(diag["per_mode_relative_diff"]) == {0, 3}     # degenerate modes excluded
    assert all(v < 1e-9 for v in diag["per_mode_relative_diff"].values())


def test_coefficient_diagnostic_none_without_coefficients():
    assert _coefficient_diagnostic(_FakeRun(np.empty((0, 0))), _FakeRun(np.empty((0, 0)))) is None


def test_coefficient_diagnostic_survives_cluster_wider_than_samples():
    # A degenerate cluster wider than the holdout sample count must NOT crash the verdict:
    # principal angles need cols ≤ rows. 2 samples × 4 equal-norm modes → one width-4
    # cluster > 2 rows; the over-wide cluster is skipped, the diagnostic degrades softly.
    rng = np.random.default_rng(7)
    pc = rng.standard_normal((2, 4))
    pc = pc / np.linalg.norm(pc, axis=0)  # equal column norms → one degenerate cluster
    diag = _coefficient_diagnostic(_FakeRun(pc), _FakeRun(pc))
    assert diag is not None
    assert diag["max_subspace_angle"] == 0.0  # over-wide cluster skipped, no ValueError


# --- Teeth: a genuinely different field diverges (AC6) -----------------------

def test_field_metric_detects_a_genuinely_different_field(field_project):
    # Same data + split, very different reduction (energy 0.5 keeps far fewer modes) →
    # the reconstructed fields genuinely differ, well beyond the tolerance.
    rich = run_training(field_project, parse_config(field_config(energy=0.99999)))
    lean = run_training(field_project, parse_config(field_config(energy=0.5)))
    fa = np.asarray(rich.holdout_fingerprint, dtype=float)
    fb = np.asarray(lean.holdout_fingerprint, dtype=float)
    assert fa.shape == fb.shape
    assert _field_energy_relative_diff(fa, fb) > rich.lineage["reproduction_tolerance"]


def test_assert_reproducible_raises_on_field_divergence():
    # Teeth: the FIELD metric drives the raise. max_abs_diff is 0.0, so this fails only
    # if assert_reproducible reverts to the scalar metric for a field run.
    result = ReproduceResult(
        parent=None, reproduction=None, bit_identical=False,
        max_abs_diff=0.0, data_hash_match=True, verdict="diverged",
        field_relative_diff=1e-2, tolerance=1e-9,
    )
    with pytest.raises(ReproducibilityError):
        assert_reproducible(result)


# --- The tolerance is folded into the hashed spec (AC4) ----------------------

def test_spec_hash_folds_in_tolerance():
    cfg = parse_config(field_config())
    base = _spec_hash(cfg, 1e-9)
    assert base == _spec_hash(cfg, 1e-9)               # deterministic for identical inputs
    # Widening the tolerance is a DIFFERENT run (re-hash), not a looser compare (AC4).
    assert base != _spec_hash(cfg, 1e-6)
    # A different reduction request lives in the config, so it re-hashes too.
    assert base != _spec_hash(parse_config(field_config(energy=0.5)), 1e-9)


def test_two_identical_field_runs_share_spec_hash(field_project):
    a = run_training(field_project, parse_config(field_config()))
    b = run_training(field_project, parse_config(field_config()))
    # spec_hash = config + tolerance, so it is bit-reproducible on any BLAS.
    assert a.spec_hash == b.spec_hash
    # basis_hash is recorded for both; its run-to-run equality is BLAS-dependent, so we
    # only assert it exists — the field verdict, not the basis hash, is the gate.
    assert a.basis_hash and b.basis_hash
