"""Predict coefficients by default; reconstruct on --field (Story R4.1, PRD 4.7).

Covers the #35 flip (compact modal coefficients are the default prediction for
reduced runs; the full field only on demand), the #21 write-back with
round-trippable column names, and the #32/#22/#47 storage surface: a
content-addressed, project-level basis store (address = sha256 over the ARRAY
bytes, each framed with key/dtype/shape, in sorted key order — never the npz
file bytes, never the holdout-diagnostic ``basis_hash``) plus the size pin
(run dir < raw snapshots).

Bitwise pins are safe throughout: every equality compares same-process reads of
the same persisted arrays (or one recomputation on identical bytes) — no
cross-run golden literals (BLAS discipline).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from surroforge.cli.main import _save_frame, app
from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.core.run import Run
from surroforge.data.loader import expand_field_columns, load_table
from surroforge.engine import run_training
from surroforge.errors import DataError, ProjectError, TrainerError
from surroforge.persistence import read_json, write_json
from surroforge.services import predict_from_file, predict_with_run
from surroforge.trainers.reduced_field import ReducedFieldTrainer

runner = CliRunner()

N_DOF = 40


# --- pinned fixture (story Dev Notes; r3 family) -------------------------------------------

def _field_data(a3=1.8, n=80):
    rng = np.random.default_rng(42)
    modes = np.linalg.qr(rng.standard_normal((N_DOF, 3)))[0]
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    amps = np.column_stack([5.0 * x[:, 0], 2.0 * x[:, 1] ** 2, a3 * x[:, 0] * x[:, 1]])
    return x, amps @ modes.T


def _write_snapshots(path):
    x, y = _field_data()
    cols: dict[str, np.ndarray] = {"p0": x[:, 0], "p1": x[:, 1]}
    for j in range(y.shape[1]):
        cols[f"f{j}"] = y[:, j]
    pd.DataFrame(cols).to_csv(path, index=False)
    return path


def _config(model=None):
    return parse_config(
        {
            "experiment": "rom-predict",
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
                "inner": {"method": "polynomial", "degree": 2},
            },
            "validation": {"cross_validation": 3, "metrics": ["rmse", "r2"]},
        }
    )


def _train(root, model=None):
    csv = _write_snapshots(root.parent / f"snaps-{root.name}.csv")
    proj = Project.init(root, name="rom")
    proj.add_dataset(csv, name="snaps")
    return proj, run_training(proj, _config(model)), csv


@pytest.fixture(scope="module")
def field_run(tmp_path_factory):
    """One shared reduced-field engine run for the read-only tests."""
    root = tmp_path_factory.mktemp("predict") / "proj"
    proj, run, csv = _train(root)
    new = root.parent / "new.csv"
    grid = np.column_stack([np.linspace(-0.9, 0.9, 7), np.linspace(0.8, -0.8, 7)])
    pd.DataFrame({"p0": grid[:, 0], "p1": grid[:, 1]}).to_csv(new, index=False)
    # Re-read so the bitwise pins compare against EXACTLY the bytes the service
    # parses (the CSV round-trip can differ from the in-memory floats by an ulp).
    x_new = pd.read_csv(new)[["p0", "p1"]].to_numpy(dtype=float)
    return proj, run, csv, new, x_new


# --- trainer surface (Task 1) ---------------------------------------------------------------

def test_unfitted_trainer_raises_on_new_surface():
    t = ReducedFieldTrainer()
    with pytest.raises(TrainerError):
        t.predict_coefficients(np.zeros((3, 2)))
    with pytest.raises(TrainerError):
        t.reduction_artifacts()


def test_artifacts_shapes_and_coefficient_identity(field_run):
    _proj, run, _csv, _new, x_new = field_run
    model = run.load_model()
    coeffs = model.predict_coefficients(x_new)
    assert coeffs.shape == (len(x_new), 2)
    artifacts = model.reduction_artifacts()
    assert set(artifacts) == {"basis", "mean", "singular_values"}
    basis, mean = artifacts["basis"], artifacts["mean"]
    assert basis.shape == (N_DOF, 2)
    assert mean.shape == (1, N_DOF)
    # Full spectrum retained at fit (feeds R4.2): min(n_samples, n_dof) values.
    assert artifacts["singular_values"].shape == (N_DOF,)
    # The pre-reconstruction step IS predict (bitwise; same process, same arrays).
    assert np.array_equal(coeffs @ basis.T + mean, model.predict(x_new))
    # Orthonormal columns (approx — SVD-derived, never bit-pinned cross-run).
    assert basis.T @ basis == pytest.approx(np.eye(2), abs=1e-12)


# --- coefficients by default (AC 1) ---------------------------------------------------------

def test_predict_defaults_to_coefficients(field_run):
    _proj, run, _csv, new, x_new = field_run
    frame = predict_from_file(run, new)
    assert list(frame.columns) == ["p0", "p1", "mode_0", "mode_1"]
    model = run.load_model()
    assert np.array_equal(
        frame[["mode_0", "mode_1"]].to_numpy(), model.predict_coefficients(x_new)
    )


# --- --field reconstructs with round-trippable names (AC 2) ---------------------------------

def test_field_flag_reconstructs_with_roundtrip_columns(field_run):
    _proj, run, _csv, new, x_new = field_run
    expected_names = [f"f{j}" for j in range(N_DOF)]
    assert run.output_columns == expected_names  # recorded at train time

    frame = predict_with_run(run, pd.read_csv(new), field=True)
    field_cols = [c for c in frame.columns if c not in ("p0", "p1")]
    assert field_cols == expected_names
    assert np.array_equal(
        frame[expected_names].to_numpy(), run.load_model().predict(x_new)
    )
    # Round-trip teeth (vacant-pin check): the written frame re-ingests through the
    # run's own glob pattern — an `output_{i}` fallback matches nothing under "f*".
    assert expand_field_columns(frame, ["f*"]) == expected_names


def test_legacy_lineage_without_output_columns_falls_back(tmp_path):
    _proj, run, _csv = _train(tmp_path / "proj")
    lineage = read_json(run.run_dir / "lineage.json")
    del lineage["output_columns"]
    write_json(run.run_dir / "lineage.json", lineage)
    run = Project.open(tmp_path / "proj").run(run.run_id)  # fresh, uncached read
    frame = predict_with_run(run, pd.DataFrame({"p0": [0.1], "p1": [0.2]}), field=True)
    assert "output_0" in frame.columns and f"output_{N_DOF - 1}" in frame.columns


# --- scalar behavior unchanged (AC 3) --------------------------------------------------------

def test_scalar_run_predict_unchanged_with_and_without_field(tmp_path):
    _proj, run, _csv = _train(
        tmp_path / "proj", model={"method": "polynomial", "degree": 2}
    )
    new = pd.DataFrame({"p0": [0.0, 0.5], "p1": [0.2, -0.3]})
    plain = predict_with_run(run, new)
    flagged = predict_with_run(run, new, field=True)
    assert plain.equals(flagged)  # the flag is meaningless and silent on scalar runs
    assert "mode_0" not in plain.columns
    lineage = read_json(run.run_dir / "lineage.json")
    assert "basis_file" not in lineage and "output_columns" not in lineage
    assert not (tmp_path / "proj" / "basis").exists()  # no store for scalar projects


# --- content-addressed basis store (AC 4) ----------------------------------------------------

def test_basis_store_content_addressed(field_run):
    proj, run, _csv, _new, _x = field_run
    assert run.basis_file.startswith("basis/") and run.basis_file.endswith(".npz")
    path = run.basis_path
    assert path is not None and path.is_file() and path.parent == proj.basis_dir

    model = run.load_model()
    artifacts = model.reduction_artifacts()
    with np.load(path) as npz:
        stored = {key: npz[key] for key in npz.files}
    assert set(stored) == {"basis", "mean", "singular_values"}
    for key, value in artifacts.items():  # the FINAL model's arrays, bitwise
        assert np.array_equal(stored[key], value), key

    # The address spec itself (validation Major): filename stem == sha256 over the
    # contiguous ARRAY bytes, each framed with key/dtype/shape, in sorted key
    # order, recomputed from the loaded arrays. Kills file-bytes (zip timestamps),
    # partial-array, wrong-order, and shape/dtype-ambiguous addressing —
    # recomputed, never golden.
    digest = hashlib.sha256()
    for key in sorted(stored):
        array = np.ascontiguousarray(stored[key])
        digest.update(f"{key}:{array.dtype.str}:{array.shape};".encode())
        digest.update(array.tobytes())
    assert path.stem == digest.hexdigest()

    # Storage identity != holdout diagnostic: the lineage basis_hash comes from the
    # HOLDOUT model (60-row split, canonicalized); wiring the store name to it is
    # THE likely wrong implementation and fails here.
    assert path.stem != run.basis_hash


def test_basis_store_dedups_across_runs(tmp_path):
    proj, first, _csv = _train(tmp_path / "proj")
    second = run_training(proj, _config())
    npz_files = list(proj.basis_dir.glob("*.npz"))
    assert len(npz_files) == 1  # same-process retrain is bit-identical (measured)
    assert first.basis_file == second.basis_file


# --- run dir smaller than the raw snapshots (AC 5) -------------------------------------------

def test_run_dir_smaller_than_raw_snapshots(field_run):
    _proj, run, csv, _new, _x = field_run
    run_size = sum(p.stat().st_size for p in run.run_dir.rglob("*") if p.is_file())
    npz_size = run.basis_path.stat().st_size
    # Engine-measured ratio 0.46 incl. the npz — >2x headroom on the strict bound.
    assert run_size + npz_size < csv.stat().st_size


# --- CLI surface (AC 1, 2) -------------------------------------------------------------------

def test_cli_predict_coefficients_and_field(field_run):
    proj, _run, _csv, new, _x = field_run
    out = proj.root.parent / "coeffs.csv"
    result = runner.invoke(
        app,
        ["--config-dir", str(proj.root), "predict", "run-001",
         "--input", str(new), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert list(pd.read_csv(out).columns) == ["p0", "p1", "mode_0", "mode_1"]

    out_field = proj.root.parent / "field.csv"
    result = runner.invoke(
        app,
        ["--config-dir", str(proj.root), "predict", "run-001",
         "--input", str(new), "--output", str(out_field), "--field"],
    )
    assert result.exit_code == 0, result.output
    written = pd.read_csv(out_field)
    assert expand_field_columns(written, ["f*"]) == [f"f{j}" for j in range(N_DOF)]


def test_cli_field_writes_hdf5_column_group(field_run):
    h5py = pytest.importorskip("h5py")
    proj, run, _csv, new, x_new = field_run
    out = proj.root.parent / "field.h5"
    result = runner.invoke(
        app,
        ["--config-dir", str(proj.root), "predict", "run-001",
         "--input", str(new), "--output", str(out), "--field"],
    )
    assert result.exit_code == 0, result.output
    preds = run.load_model().predict(x_new)
    with h5py.File(out, "r") as fh:
        # List (not set) equality: track_order preserves the frame's column
        # order — lexicographic HDF5 iteration would yield f10 before f2.
        assert list(fh.keys()) == ["p0", "p1", *(f"f{j}" for j in range(N_DOF))]
        for j in (0, N_DOF - 1):
            assert np.array_equal(fh[f"f{j}"][...], preds[:, j])
    loaded = load_table(out)
    assert list(loaded.columns) == ["p0", "p1", *(f"f{j}" for j in range(N_DOF))]
    assert np.array_equal(loaded[f"f{N_DOF - 1}"].to_numpy(), preds[:, N_DOF - 1])


# --- h5 writer degradation & validation (AC 2; review patches) -------------------------------

def test_h5_write_without_h5py_raises_actionable_dataerror(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("h5py deliberately blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(DataError, match=r"surroforge\[formats\]"):
        _save_frame(pd.DataFrame({"a": [1.0]}), tmp_path / "out.h5")
    assert not (tmp_path / "out.h5").exists()


def test_h5_write_rejects_non_numeric_columns(tmp_path):
    pytest.importorskip("h5py")
    # Predict carries every input column through verbatim — a string case-id
    # column must yield a clean DataError, not h5py's raw object-dtype TypeError.
    frame = pd.DataFrame({"p0": [0.1], "case": ["a"]})
    with pytest.raises(DataError, match="case"):
        _save_frame(frame, tmp_path / "out.h5")
    assert list(tmp_path.iterdir()) == []  # validation precedes any write; no tmp residue


def test_h5_write_rejects_slash_and_duplicate_columns(tmp_path):
    pytest.importorskip("h5py")
    with pytest.raises(DataError, match="disp/x"):  # h5py would silently nest groups
        _save_frame(pd.DataFrame({"disp/x": [1.0]}), tmp_path / "out.h5")
    with pytest.raises(DataError, match="not valid"):  # illegal HDF5 link name
        _save_frame(pd.DataFrame({"": [1.0]}), tmp_path / "out.h5")
    duplicated = pd.DataFrame([[1.0, 2.0]], columns=["a", "a"])
    with pytest.raises(DataError, match="duplicate"):
        _save_frame(duplicated, tmp_path / "out.h5")
    assert list(tmp_path.iterdir()) == []


# --- lineage basis_file validation (review patch) ---------------------------------------------

def test_tampered_basis_file_lineage_raises_clean_error(tmp_path):
    _proj, run, _csv = _train(tmp_path / "proj")
    lineage = read_json(run.run_dir / "lineage.json")
    lineage["basis_file"] = "../../outside/evil.npz"
    write_json(run.run_dir / "lineage.json", lineage)
    run = Project.open(tmp_path / "proj").run(run.run_id)  # fresh, uncached read
    with pytest.raises(ProjectError, match="basis_file"):
        _ = run.basis_path


def test_missing_store_file_raises_retrain_guidance(tmp_path):
    proj, run, _csv = _train(tmp_path / "proj")
    path = run.basis_path
    assert path is not None
    path.unlink()  # a deleted basis/ dir (partial copy) must not surface raw IO errors
    run = Project.open(tmp_path / "proj").run(run.run_id)
    with pytest.raises(ProjectError, match="[Rr]e-train"):
        _ = run.basis_path


def test_corrupt_output_columns_lineage_falls_back(tmp_path):
    _proj, run, _csv = _train(tmp_path / "proj")
    lineage = read_json(run.run_dir / "lineage.json")
    lineage["output_columns"] = "f0"  # a stray string must not become ['f', '0']
    write_json(run.run_dir / "lineage.json", lineage)
    run = Project.open(tmp_path / "proj").run(run.run_id)
    assert run.output_columns == []
    frame = predict_with_run(run, pd.DataFrame({"p0": [0.1], "p1": [0.2]}), field=True)
    assert "output_0" in frame.columns  # legacy chain, not garbage names


# --- duck-typed coefficient contract (review patch) --------------------------------------------

def test_duck_typed_coefficients_shape_contract(field_run, monkeypatch):
    _proj, run, _csv, new, _x = field_run
    frame = pd.read_csv(new)

    class _Bad:
        def __init__(self, value):
            self._value = value

        def predict_coefficients(self, x):
            return self._value

    for bad in (np.float64(1.0), np.zeros((2, 2, 2)), np.zeros((len(frame) + 1, 2))):
        monkeypatch.setattr(Run, "load_model", lambda self, _v=bad: _Bad(_v))
        with pytest.raises(TrainerError, match="predict_coefficients"):
            predict_with_run(run, frame)
