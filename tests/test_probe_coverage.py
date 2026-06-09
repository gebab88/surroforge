"""Probe coverage + low-coverage warning (Story R2.5).

R2.3 ships the field verdict as the **global field-energy norm over the full
reconstructed holdout field** — there are no probe-node subsets in v1, so "the probes"
are the whole field and the only coverage gap is modal truncation. This file pins:

  - a field run records ``probe_coverage_energy`` (the retained-mode energy share) in
    lineage; a scalar run records none (byte-identity, R2.4 spirit);
  - ``reproduce()`` warns (``FieldCoverageWarning``) when coverage is low, but the
    coverage is **advisory, never a gate** — the verdict and ``assert_reproducible`` are
    unchanged (the warns-but-doesn't-gate invariant, distinct from the repro-FAILS teeth
    in ``test_reproduce_field.py``);
  - the CLI renders ``FieldCoverageWarning`` as a sober ``Warning:`` line.

BLAS note: coverage is SVD-derived (ULP-sensitive on Apple Accelerate across runs), so the
warn/no-warn checks use wide margins (~0.84 vs 0.99; ~0.999998 vs 0.99) and never assert a
bit-exact coverage float — no ``requires_reproducible_blas`` marker is needed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.core.run import LINEAGE_FILE, FieldCoverageWarning
from surroforge.engine import run_training
from surroforge.persistence import write_json
from surroforge.services import (
    PROBE_COVERAGE_WARN_THRESHOLD,
    assert_reproducible,
    reproduce,
)
from tests.conftest import poly_config
from tests.test_reproduce_field import field_config, make_field_samples


def _const_field_config() -> dict:
    """A reduced-field config over a constant (zero-fluctuation) field on 'cfield'."""
    return {
        "experiment": "field-const",
        "data": {
            "source": "cfield",
            "inputs": ["a", "b"],
            "outputs": [f"u{j}" for j in range(6)],
            "split": {"test_fraction": 0.25, "seed": 42},
        },
        "model": {
            "method": "reduced-field",
            "reduction": {"basis": "pod", "energy": 0.999},
            "inner": {"method": "polynomial", "degree": 2},
        },
        # rmse/mae only: a constant target makes r2's ss_tot zero (0/0), out of scope here.
        "validation": {"cross_validation": 2, "metrics": ["rmse", "mae"]},
    }


@pytest.fixture
def const_field_project(tmp_path):
    proj = Project.init(tmp_path / "cproj", name="const-proj")
    rng = np.random.default_rng(0)
    n = 60
    cols: dict[str, np.ndarray] = {"a": rng.uniform(1.0, 3.0, n), "b": rng.uniform(0.5, 1.5, n)}
    for j in range(6):  # each output column is constant across rows -> zero fluctuation
        cols[f"u{j}"] = np.full(n, float(j + 1))
    csv = tmp_path / "cfield.csv"
    pd.DataFrame(cols).to_csv(csv, index=False)
    proj.add_dataset(csv, name="cfield")
    return proj


@pytest.fixture
def field_project(tmp_path):
    proj = Project.init(tmp_path / "fproj", name="field-proj")
    csv = tmp_path / "field.csv"
    make_field_samples().to_csv(csv, index=False)
    proj.add_dataset(csv, name="field")
    return proj


# --- AC1: coverage is recorded for field runs, absent for scalar runs ----------------

def test_field_run_records_probe_coverage(field_project):
    run = run_training(field_project, parse_config(field_config()))
    # Assert on the RAW lineage (the accessor would mask absence with its 1.0 default).
    assert "probe_coverage_energy" in run.lineage
    coverage = run.lineage["probe_coverage_energy"]
    assert 0.0 < coverage <= 1.0
    # In v1 the probes are the full field, so coverage == the retained-mode energy share.
    assert coverage == pytest.approx(run.reduction["energy_captured"])


def test_scalar_run_records_no_probe_coverage(project):
    run = run_training(project, parse_config(poly_config("polynomial")))
    assert "probe_coverage_energy" not in run.lineage  # field-gated, scalar unchanged (R2.4)
    assert run.probe_coverage_energy == 1.0  # accessor default: a scalar run self-covers


# --- AC3: high coverage does not warn; low coverage warns but does not gate -----------

def test_high_coverage_field_run_does_not_warn(field_project):
    run = run_training(field_project, parse_config(field_config(energy=0.99999)))
    assert run.probe_coverage_energy >= PROBE_COVERAGE_WARN_THRESHOLD
    # pytest.warns(None) was removed in pytest 9 — record and assert absence instead.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reproduce(field_project, run.run_id)
    assert not any(isinstance(w.message, FieldCoverageWarning) for w in caught)
    assert result.verdict in {"reproduced", "reproduced_within_tolerance"}


def test_low_coverage_field_run_warns(field_project):
    # energy=0.5 is the *requested* energy, not the coverage: searchsorted lands on the
    # first (dominant) mode, which already captures ~84% — well below the 0.99 threshold.
    run = run_training(field_project, parse_config(field_config(energy=0.5)))
    assert run.probe_coverage_energy < PROBE_COVERAGE_WARN_THRESHOLD
    with pytest.warns(FieldCoverageWarning, match="of the field energy"):
        result = reproduce(field_project, run.run_id)
    # Warns-but-doesn't-gate: a low-coverage run still reproduces and does NOT raise.
    assert result.verdict in {"reproduced", "reproduced_within_tolerance"}
    assert_reproducible(result)


# --- Review patch #1: a constant (zero-energy) field is fully covered, never warns -----

def test_constant_field_reports_full_coverage_and_does_not_warn(const_field_project):
    # A constant / zero-fluctuation field has no modal energy to truncate; the
    # reconstruction (the mean) captures it exactly, so coverage is 1.0 — NOT 0.0, which
    # would mis-warn "0% verified / 100% truncated" on a run that reproduces perfectly.
    run = run_training(const_field_project, parse_config(_const_field_config()))
    assert run.probe_coverage_energy == 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reproduce(const_field_project, run.run_id)
    assert not any(isinstance(w.message, FieldCoverageWarning) for w in caught)
    assert result.verdict in {"reproduced", "reproduced_within_tolerance"}


# --- Review patch #2: no coverage warning when nothing was verified --------------------

def test_no_coverage_warning_on_unverifiable_field_run(field_project):
    # The warning is meaningful only when the field was actually compared. On an
    # ``unverifiable`` verdict (no comparable holdout fingerprint) "verifies only X%" is
    # contradictory noise, so it must NOT fire even though coverage is below threshold.
    run = run_training(field_project, parse_config(field_config(energy=0.5)))
    assert run.probe_coverage_energy < PROBE_COVERAGE_WARN_THRESHOLD

    # Make the parent unverifiable: an empty holdout fingerprint (kept basis_hash + low
    # coverage). reproduce() re-reads the run fresh, so the on-disk edit takes effect.
    lineage = dict(run.lineage)
    lineage["holdout_fingerprint"] = []
    write_json(run.run_dir / LINEAGE_FILE, lineage)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reproduce(field_project, run.run_id)
    assert result.verdict == "unverifiable"
    assert not any(isinstance(w.message, FieldCoverageWarning) for w in caught)


# --- AC3: the verdict carries the coverage (additive field), None for scalar ----------

def test_reproduce_result_exposes_probe_coverage(field_project, project):
    frun = run_training(field_project, parse_config(field_config()))
    fres = reproduce(field_project, frun.run_id)
    assert fres.probe_coverage_energy == pytest.approx(frun.probe_coverage_energy)

    srun = run_training(project, parse_config(poly_config("polynomial")))
    sres = reproduce(project, srun.run_id)
    assert sres.probe_coverage_energy is None  # scalar-shaped result


# --- AC3: the CLI renders FieldCoverageWarning as a sober "Warning:" line --------------

def test_render_warnings_renders_field_coverage_as_sober_line(capsys):
    # CliRunner.invoke(app, ...) bypasses main()/_render_warnings (see test_cli.py), so
    # unit-test the widened predicate directly. catch_warnings restores showwarning on exit.
    from surroforge.cli.main import _render_warnings

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        _render_warnings()
        warnings.warn("verifies only 50.0% of the field energy", FieldCoverageWarning, stacklevel=1)
    err = capsys.readouterr().err
    assert "Warning:" in err
    assert "field energy" in err


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_render_warnings_passes_through_foreign_warnings(capsys):
    # A non-SurroForge warning must NOT be reformatted as our sober line. The filterwarnings
    # mark keeps the deliberately-emitted foreign UserWarning from tripping a future strict
    # ``filterwarnings = error`` config (the warning escapes catch_warnings via the default
    # handler that _render_warnings restores for foreign categories).
    from surroforge.cli.main import _render_warnings

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        _render_warnings()
        warnings.warn("some library warning", UserWarning, stacklevel=1)
    err = capsys.readouterr().err
    assert "Warning:" not in err
