"""ROM inspect service and CLI tests (Story R4.2, PRD §4.8 #30).

Covers: inspect_rom_run service (AC 1–4) and ``surroforge rom inspect`` CLI
command (AC 2–3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from surroforge.cli.main import app
from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import run_training
from surroforge.errors import ProjectError
from surroforge.services import inspect_rom_run

runner = CliRunner()

N_DOF = 40


# --- shared fixture helpers (same pattern as test_predict_field.py) --------------------------

def _field_data(n=80):
    rng = np.random.default_rng(42)
    modes = np.linalg.qr(rng.standard_normal((N_DOF, 3)))[0]
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    amps = np.column_stack([5.0 * x[:, 0], 2.0 * x[:, 1] ** 2, 1.8 * x[:, 0] * x[:, 1]])
    return x, amps @ modes.T


def _write_snapshots(path):
    x, y = _field_data()
    cols: dict[str, np.ndarray] = {"p0": x[:, 0], "p1": x[:, 1]}
    for j in range(y.shape[1]):
        cols[f"f{j}"] = y[:, j]
    pd.DataFrame(cols).to_csv(path, index=False)
    return path


def _field_config():
    return parse_config(
        {
            "experiment": "rom-inspect",
            "data": {
                "source": "snaps",
                "inputs": ["p0", "p1"],
                "outputs": ["f*"],
                "split": {"test_fraction": 0.25, "seed": 1},
            },
            "model": {
                "method": "reduced-field",
                "reduction": {"basis": "pod", "rank": 2},
                "inner": {"method": "polynomial", "degree": 2},
            },
            "validation": {"cross_validation": 3, "metrics": ["rmse", "r2"]},
        }
    )


def _scalar_config():
    return parse_config(
        {
            "experiment": "rom-inspect-scalar",
            "data": {
                "source": "snaps",
                "inputs": ["p0", "p1"],
                "outputs": ["f0"],
                "split": {"test_fraction": 0.25, "seed": 1},
            },
            "model": {"method": "polynomial", "degree": 2},
            "validation": {"cross_validation": 3, "metrics": ["rmse", "r2"]},
        }
    )


@pytest.fixture(scope="module")
def field_run(tmp_path_factory):
    """One trained field run shared across read-only tests."""
    root = tmp_path_factory.mktemp("inspect-field") / "proj"
    csv = _write_snapshots(root.parent / "snaps.csv")
    proj = Project.init(root, name="rom-inspect")
    proj.add_dataset(csv, name="snaps")
    run = run_training(proj, _field_config())
    # Verify the fixture has rank < total_modes before writing rank-dependent pins.
    assert int(run.reduction["rank"]) < N_DOF, (
        f"fixture sanity: rank {run.reduction['rank']} should be < {N_DOF}"
    )
    return proj, run


@pytest.fixture(scope="module")
def scalar_run(tmp_path_factory):
    """One trained scalar (polynomial) run for the non-reduced path."""
    root = tmp_path_factory.mktemp("inspect-scalar") / "proj"
    x, y = _field_data()
    csv = root.parent / "snaps-scalar.csv"
    pd.DataFrame({"p0": x[:, 0], "p1": x[:, 1], "f0": y[:, 0]}).to_csv(csv, index=False)
    proj = Project.init(root, name="rom-scalar")
    proj.add_dataset(csv, name="snaps")
    run = run_training(proj, _scalar_config())
    return proj, run


# --- AC 1: service returns correct spectrum --------------------------------------------------

def test_service_correct_spectrum(field_run):
    _proj, run = field_run
    result = inspect_rom_run(run)

    assert result.is_reduced is True
    assert result.rank == int(run.reduction["rank"])
    assert result.total_modes > result.rank  # teeth: rank < full spectrum length

    # Independent recomputation pin (catches aliasing, value degeneracy,
    # presence-without-structure — three trap classes from vacant-pin discipline).
    npz = np.load(str(run.basis_path))
    sv = npz["singular_values"]
    expected_energy = (sv**2 / (sv**2).sum()).tolist()
    assert result.per_mode_energy == expected_energy  # same-process exact equality

    assert sum(result.per_mode_energy) == pytest.approx(1.0, rel=1e-6)
    assert result.cumulative_energy[-1] == pytest.approx(1.0, rel=1e-6)

    # probe_coverage_energy from lineage — exact, not approx (it's just a float read)
    assert result.probe_coverage_energy == run.probe_coverage_energy

    # energy_captured from lineage (holdout model), NOT re-derived from final npz spectrum.
    # This is the critical guard against a dev agent conflating holdout vs. final model.
    assert result.energy_captured == pytest.approx(
        float(run.reduction["energy_captured"]), rel=1e-9
    )

    assert result.basis == "pod"
    assert result.total_modes == len(sv)


# --- AC 3 (service): scalar run → is_reduced=False ------------------------------------------

def test_scalar_run_returns_not_reduced(scalar_run):
    _proj, run = scalar_run
    result = inspect_rom_run(run)
    assert result.is_reduced is False  # exact False, not truthy check
    assert result.rank == 0


# --- AC 4: pre-R4.1 field run → ProjectError with "re-train" --------------------------------

def test_pre_r41_run_raises_project_error(field_run):
    _proj, run = field_run
    # A freshly-loaded run object so mutation doesn't affect other tests.
    fresh_run = _proj.run(run.run_id)
    fresh_run.lineage.pop("basis_file", None)  # simulate pre-R4.1 lineage (no basis_file key)
    with pytest.raises(ProjectError) as exc_info:
        inspect_rom_run(fresh_run)
    assert "re-train" in exc_info.value.args[0].lower()


# --- AC 2: CLI e2e — field run ---------------------------------------------------------------

def test_cli_field_run_inspect(field_run):
    proj, run = field_run
    run_id = run.run_id
    result = runner.invoke(
        app,
        ["--config-dir", str(proj.root), "rom", "inspect", run_id],
    )
    assert result.exit_code == 0, result.output

    # Header must contain the run_id (discriminates a generic header that omits the id).
    assert f"ROM inspection: {run_id}" in result.output

    # [retained] count == result.rank — discriminates "any marker present" from correct count.
    service_result = inspect_rom_run(run)
    retained_count = result.output.count("[retained]")
    assert retained_count == service_result.rank

    # Coverage line present.
    assert "probe_coverage" in result.output


# --- AC 3 (CLI): scalar run → exit 0 with message -------------------------------------------

def test_cli_scalar_run_inspect(scalar_run):
    proj, run = scalar_run
    run_id = run.run_id
    result = runner.invoke(
        app,
        ["--config-dir", str(proj.root), "rom", "inspect", run_id],
    )
    assert result.exit_code == 0, result.output
    # Message must contain the run_id (not just generic "not a reduced-field run").
    assert "not a reduced-field run" in result.output
    assert run_id in result.output


# --- Review round 2 (2026-06-13): corrupt on-disk state → ProjectError, never a traceback ---
# _guard catches only SurroForgeError, so every read of untrusted lineage/store data must
# surface as ProjectError. These pin the three guards added in code review (services.py).

def test_malformed_reduction_keys_raise_project_error(field_run):
    """A reduction block missing a required numeric key → ProjectError, not bare KeyError.

    Teeth: without the defensive read, ``int(run.reduction["rank"])`` raises ``KeyError``,
    which escapes ``_guard`` as a raw traceback. The npz loads fine here (intact shared
    store) — only the reduction lineage is corrupt, so this isolates the key-read guard.
    """
    _proj, run = field_run
    fresh = _proj.run(run.run_id)  # independent lineage; safe to mutate
    fresh.lineage["reduction"].pop("rank")
    with pytest.raises(ProjectError) as exc_info:
        inspect_rom_run(fresh)
    assert "re-train" in exc_info.value.args[0].lower()


def test_non_numeric_reduction_rank_raises_project_error(field_run):
    """A non-numeric rank (hand-edited lineage) → ProjectError, not bare ValueError."""
    _proj, run = field_run
    fresh = _proj.run(run.run_id)
    fresh.lineage["reduction"]["rank"] = "not-an-int"
    with pytest.raises(ProjectError) as exc_info:
        inspect_rom_run(fresh)
    assert "re-train" in exc_info.value.args[0].lower()


def test_corrupt_basis_store_raises_project_error(tmp_path_factory):
    """A truncated zip and an npz missing 'singular_values' both surface as ProjectError.

    Own isolated project (the corruption mutates the store on disk) so the module-scoped
    ``field_run`` store stays intact. Both modes share one trained run to pay the cost once.
    """
    root = tmp_path_factory.mktemp("inspect-corrupt") / "proj"
    csv = _write_snapshots(root.parent / "snaps.csv")
    proj = Project.init(root, name="rom-corrupt")
    proj.add_dataset(csv, name="snaps")
    run = run_training(proj, _field_config())
    path = run.basis_path
    assert path is not None

    # (a) truncated archive (partial copy / interrupted CloudDocs sync) → BadZipFile
    good = path.read_bytes()
    path.write_bytes(good[: len(good) // 2])
    with pytest.raises(ProjectError) as exc_info:
        inspect_rom_run(proj.run(run.run_id))
    assert "re-train" in exc_info.value.args[0].lower()

    # (b) valid npz, but missing the 'singular_values' key (foreign/partial store) → KeyError
    np.savez(str(path), wrong_key=np.array([1.0, 2.0]))
    with pytest.raises(ProjectError) as exc_info:
        inspect_rom_run(proj.run(run.run_id))
    assert "re-train" in exc_info.value.args[0].lower()
