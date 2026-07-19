#!/usr/bin/env python3
"""Ingest a real SU2 NACA 6412 sweep and train the SurroForge surrogates.

Run this after ``run_sweep.py`` has produced a sweep directory. It reads each case's
converged pressure and velocity point fields from the ``flow.vtu`` SU2 wrote and the
**real** lift and drag coefficients ``CL``/``CD`` from SU2's ``history.csv``, then
trains three surrogates:

  * a **field** surrogate (``reduced-field`` POD + neural inner): (AoA, speed) -> pressure field;
  * a **velocity** surrogate (same construction): (AoA, speed) -> velocity field (Ux, Uy —
    SU2 pads the 2-D vector with an exact-zero z component, dropped at ingest,
    re-padded on export);
  * a **scalar** surrogate: (AoA, speed) -> [Cl, Cd] (lift and drag coefficients).

Both inputs come from the 2-D DoE ``run_sweep.py`` wrote (angle × speed). This is the
same pipeline as example 09 — only the data source changes: SU2 point fields from VTU
instead of OpenFOAM cell fields, ``history.csv`` instead of ``forceCoeffs``.

    python examples/10_aerofoil_su2/ingest_and_train.py run
    python examples/10_aerofoil_su2/ingest_and_train.py run --out ./ex10_real   # save the run

With ``--out`` the SurroForge project (datasets, runs, lineage, POD basis, models) **and**
the exported artifacts are saved under that directory, so the real run is inspectable and
reproducible afterwards (``surroforge --config-dir <out>/naca list``). The best pressure-
and velocity-field surrogates are also evaluated on **every** case from ``samples.csv``
and written into one VTKHDF time-series file (point arrays ``p`` and ``U``)
``<out>/surrogate_vtk/surrogate_fields.vtkhdf`` (or
``<run_dir>/surrogate_vtk/surrogate_fields.vtkhdf`` without ``--out``), plus
``samples.csv`` for the case/AoA/speed mapping. Without ``--out`` the project is built in
a temp dir and removed, and only the exported artifacts plus the surrogate VTKHDF persist.

Note (editions): the neural-network candidates (``mlp``, ``torch-mlp``) are
Professional methods — ``pip install 'surroforge_professional[torch]'`` and set
``SURROFORGE_PRO_LICENSE``. The polynomial candidates run on Core alone.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from surrogate_vtk import read_point_field, write_surrogate_vtkhdf

from surroforge import Project, Trainer
from surroforge.errors import ExportError
from surroforge.plugins import available_trainers
from surroforge.services import compare_runs

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # benign lbfgs iteration cap
_HAS_TORCH = "torch-mlp" in available_trainers()

# A case counts as settled if its C_L spread over the last CL_WINDOW iterations is
# within CL_RTOL of |C_L| (plus a small absolute floor for near-zero lift).
CL_WINDOW = 20
CL_RTOL = 1e-2
CL_ATOL = 1e-4


class CoeffReading(NamedTuple):
    cl: float  # lift coefficient at the last (most converged) iteration
    cd: float  # drag coefficient at the last iteration
    spread: float  # peak-to-peak of C_L over the last CL_WINDOW iterations
    settled: bool  # C_L spread within CL_RTOL*|cl| + CL_ATOL — a genuine plateau


def read_coeffs(case_dir: Path) -> CoeffReading:
    """Parse CL and CD from a case's SU2 history, and check the lift has settled.

    Reads ``history.csv`` (written every inner iteration), takes ``CL``/``CD`` at the
    last (most converged) iteration, and inspects C_L's peak-to-peak spread over the
    final ``CL_WINDOW`` iterations: a still-swinging C_L means the run had not reached
    a steady value even if SU2 stopped — the residual limit cycle on this stretched
    C-grid makes this check, not SU2's residual verdict, the authority on usability.
    """
    history = case_dir / "history.csv"
    if not history.is_file():
        raise FileNotFoundError(f"no SU2 history.csv under {case_dir}")
    frame = pd.read_csv(history)
    frame.columns = [c.strip().strip('"') for c in frame.columns]
    if "CL" not in frame.columns or "CD" not in frame.columns:
        raise ValueError(f"could not find CL/CD columns in {history}")
    if frame.empty:
        raise ValueError(f"empty history in {history}")
    cl_series = frame["CL"].astype(float)
    cl = float(cl_series.iloc[-1])
    cd = float(frame["CD"].iloc[-1])
    window = cl_series.tail(CL_WINDOW)
    spread = float(window.max() - window.min())
    settled = spread <= CL_RTOL * abs(cl) + CL_ATOL
    return CoeffReading(cl=cl, cd=cd, spread=spread, settled=settled)


def read_su2_sweep(run_dir: Path, doe: pd.DataFrame) -> pd.DataFrame:
    """Read every case's pressure and velocity point fields into one wide table.

    Returns one row per case with ``aoa``/``speed`` plus ``p_*`` columns (pressure at
    every mesh point) and ``U_*_x``/``U_*_y`` columns — SU2's 2-D velocity carries an
    exact-zero z component in the VTU, which is dropped here (a zero-variance output
    column would wreck the uniform-average R², exactly as in example 09) and re-padded
    as exact zero on export. Each ``flow.vtu`` is parsed once for both fields.
    """
    rows: list[np.ndarray] = []
    params: list[tuple[float, float]] = []
    n_points: int | None = None
    for name in doe.index:
        vtu = run_dir / name / "flow.vtu"
        if not vtu.is_file():
            raise FileNotFoundError(
                f"{vtu} is missing — did SU2 finish this case? Re-run the sweep for it."
            )
        pressure = read_point_field(vtu, "Pressure")
        velocity = read_point_field(vtu, "Velocity")
        if n_points is None:
            n_points = pressure.size
        if pressure.size != n_points or velocity.size != 3 * n_points:
            raise ValueError(
                f"{vtu}: expected {n_points} pressure and {3 * n_points} velocity "
                f"values (fixed mesh), got {pressure.size}/{velocity.size}"
            )
        if np.abs(velocity[2::3]).max() > 1e-12:
            raise ValueError(f"{vtu}: nonzero z velocity in a 2-D case")
        rows.append(np.concatenate([pressure, velocity[0::3], velocity[1::3]]))
        params.append((float(doe.loc[name, "aoa"]), float(doe.loc[name, "speed"])))
    assert n_points is not None
    columns = (
        [f"p_{i:05d}" for i in range(n_points)]
        + [f"U_{i:05d}_x" for i in range(n_points)]
        + [f"U_{i:05d}_y" for i in range(n_points)]
    )
    table = pd.DataFrame(np.vstack(rows), columns=columns)
    table.insert(0, "aoa", [a for a, _ in params])
    table.insert(1, "speed", [s for _, s in params])
    return table


def _nn(method: str, widths: list[int]) -> dict:
    return {"method": method, "hidden_layer_sizes": widths, "activation": "tanh", "max_iter": 5000}


def _candidates(widths: list[int]) -> list[tuple[str, dict]]:
    cands = [("mlp", _nn("mlp", widths))]
    if _HAS_TORCH:
        cands.append(("torch-mlp", _nn("torch-mlp", widths)))
    cands.append(("polynomial", {"method": "polynomial", "degree": 2}))
    return cands


def main(run_dir: Path, out_dir: Path | None = None) -> None:
    # With --out the run is saved: the SurroForge project (datasets, runs, lineage, POD
    # basis, models) under <out_dir>/naca and the exported artifacts alongside it, so it
    # is inspectable and reproducible afterwards (surroforge --config-dir <out_dir>/naca
    # list). Without it the project lives in a temp dir and is removed, and the
    # artifacts go into the sweep directory.
    export_dir = out_dir if out_dir is not None else run_dir
    doe = pd.read_csv(run_dir / "samples.csv").set_index("case")
    snapshots = read_su2_sweep(run_dir, doe)

    # The real CL/CD SU2 reported, per case (same order as the field table: doe order).
    coeffs, unsettled = [], []
    for name in doe.index:
        reading = read_coeffs(run_dir / name)
        coeffs.append(reading)
        if not reading.settled:
            unsettled.append((float(doe.loc[name, "aoa"]),
                              float(doe.loc[name, "speed"]), reading.spread))
    snapshots["cl"] = [r.cl for r in coeffs]
    snapshots["cd"] = [r.cd for r in coeffs]
    field_cols = [c for c in snapshots.columns if c.startswith("p_")]
    velocity_cols = [c for c in snapshots.columns if c.startswith("U_")]
    print(f"Ingested {len(snapshots)} real cases: {len(field_cols)} pressure DOFs "
          f"+ {len(velocity_cols)} velocity DOFs + Cl + Cd")
    if unsettled:
        listed = ", ".join(f"({a:+.1f}°,{s:.4g}) ±{sp / 2:.3g}" for a, s, sp in unsettled)
        print(f"⚠  {len(unsettled)}/{len(doe)} case(s) have a still-swinging C_L over the "
              f"last {CL_WINDOW} iterations: {listed}\n"
              f"   Their lift had not settled — re-run those cases with more --iters.")
    print()

    with tempfile.TemporaryDirectory(prefix="surroforge-naca-") as tmp:
        csv = Path(tmp) / "snapshots.csv"
        snapshots.to_csv(csv, index=False)
        # Persist the project under <out_dir>/naca when --out is given (start it fresh on
        # a re-run); otherwise build it in the temp dir so it is removed on exit.
        if out_dir is not None:
            project_root = out_dir / "naca"
            if project_root.exists():
                shutil.rmtree(project_root)
        else:
            project_root = Path(tmp) / "naca"
        project = Project.init(project_root, name="naca6412-su2-cfd")
        project.add_dataset(csv, name="sweep")

        split = {"test_fraction": 0.25, "seed": 1}
        validation = {"cross_validation": 3, "metrics": ["rmse", "r2", "mae"]}
        inputs = ["aoa", "speed"]

        print("=== Field surrogate: (AoA, speed) -> pressure field (POD + inner) ===")
        field_runs = {
            label: Trainer.from_dict({
                "experiment": f"field-{label}",
                "data": {"source": "sweep", "inputs": inputs, "outputs": ["p_*"], "split": split},
                "model": {"method": "reduced-field",
                          "reduction": {"basis": "pod", "energy": 0.999}, "inner": inner},
                "validation": validation,
            }).fit(project)
            for label, inner in _candidates([48, 48])
        }
        _report(project, field_runs, "inner")

        print("\n=== Field surrogate: (AoA, speed) -> velocity field (POD + inner) ===")
        velocity_runs = {
            label: Trainer.from_dict({
                "experiment": f"velocity-{label}",
                "data": {"source": "sweep", "inputs": inputs, "outputs": ["U_*"], "split": split},
                "model": {"method": "reduced-field",
                          "reduction": {"basis": "pod", "energy": 0.999}, "inner": inner},
                "validation": validation,
            }).fit(project)
            for label, inner in _candidates([48, 48])
        }
        _report(project, velocity_runs, "inner")

        print("\n=== Scalar surrogate: (AoA, speed) -> [Cl, Cd] ===")
        cl_runs = {
            label: Trainer.from_dict({
                "experiment": f"coeff-{label}",
                "data": {"source": "sweep", "inputs": inputs, "outputs": ["cl", "cd"],
                         "split": split},
                "model": model,
                "validation": validation,
            }).fit(project)
            for label, model in _candidates([48, 48])
        }
        best_cl = cl_runs[_report(project, cl_runs, "method")]

        # Export into the (persistent) sweep directory, one artifact per format: the best
        # coefficient model to ONNX, the torch-mlp coefficient model to a native PyTorch
        # .pt (when the plugin is installed), and the field surrogates to pickle
        # (reduced-field only).
        print("\n=== Export ===")
        try:
            out = best_cl.export("onnx", export_dir / "coeffs_surrogate.onnx")
            print(f"  onnx  -> {out.name} ({out.stat().st_size} bytes)  [best: {best_cl.method}]")
        except ExportError as exc:
            print(f"  onnx  unavailable — {exc}")
        torch_run = cl_runs.get("torch-mlp")
        if torch_run is not None:
            out = torch_run.export("pt", export_dir / "coeffs_surrogate.pt")
            print(f"  pt    -> {out.name} ({out.stat().st_size} bytes)  [torch-mlp native]")
        else:
            print(
                "  pt    skipped — pip install 'surroforge_professional[torch]' "
                "for the torch-mlp backend"
            )
        field_best = field_runs[_best_label(project, field_runs)]
        field_out = field_best.export("pickle", export_dir / "field_surrogate.pkl")
        print(f"  field surrogate -> {field_out.name} (pickle; reduced-field is pickle-only)")
        velocity_best = velocity_runs[_best_label(project, velocity_runs)]
        velocity_out = velocity_best.export("pickle", export_dir / "velocity_surrogate.pkl")
        print(f"  velocity surrogate -> {velocity_out.name} "
              "(pickle; reduced-field is pickle-only)")

        print("\n=== Surrogate field VTK ===")
        prediction_cases = doe[["aoa", "speed"]].reset_index()
        prediction_inputs = prediction_cases[inputs].to_numpy(dtype=float)
        predicted_velocity_xy = velocity_best.load_model().predict(prediction_inputs)
        # Re-pad the dropped z component (exact zero in this 2-D case) so ParaView still
        # gets a 3-component vector field. The wide table groups columns [Ux… | Uy…], so
        # de-interleave block-wise into point-major [x0 y0 z0 x1 …], as the writer expects.
        n_points = predicted_velocity_xy.shape[1] // 2
        predicted_velocity = np.zeros((predicted_velocity_xy.shape[0], n_points * 3))
        predicted_velocity[:, 0::3] = predicted_velocity_xy[:, :n_points]
        predicted_velocity[:, 1::3] = predicted_velocity_xy[:, n_points:]
        vtk_dir = export_dir / "surrogate_vtk"
        vtkhdf = write_surrogate_vtkhdf(
            run_dir,
            vtk_dir / "surrogate_fields.vtkhdf",
            prediction_cases,
            {
                "p": field_best.load_model().predict(prediction_inputs),
                "U": predicted_velocity,
            },
        )
        print(
            f"  surrogate p+U fields -> {vtkhdf} "
            f"({len(prediction_cases)} time steps + samples.csv)"
        )
        if out_dir is not None:
            print(f"\nArtifacts written to {export_dir}/")
            print(f"Run saved to {project_root}/ — inspect with: "
                  f"surroforge --config-dir {project_root} list")


def _best_label(project: Project, runs: dict) -> str:
    result = compare_runs(project, [r.run_id for r in runs.values()], metric="rmse")
    return next(label for label, r in runs.items() if r.run_id == result["best"])


def _report(project: Project, runs: dict, label_kind: str) -> str:
    label_by_id = {r.run_id: name for name, r in runs.items()}
    result = compare_runs(project, list(label_by_id), metric="rmse")
    print(f"{label_kind:<12}{'test R²':>10}{'test RMSE':>12}")
    for row in result["runs"]:
        marker = "  <- best" if row["run_id"] == result["best"] else ""
        print(f"{label_by_id[row['run_id']]:<12}{row['metrics']['r2']['test']:>10.4f}"
              f"{row['metrics']['rmse']['test']:>12.4g}{marker}")
    best_row = next(r for r in result["runs"] if r["run_id"] == result["best"])
    if best_row["metrics"]["r2"]["test"] < 0:
        print("  (all test R² < 0 — data-starved sweep; '<- best' is lowest-RMSE only. "
              "Use more angles and --iters; see this directory's README.)")
    return label_by_id[result["best"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default="run",
                        help="sweep directory run_sweep.py produced (samples.csv + case_* dirs)")
    parser.add_argument(
        "--out", metavar="DIR", default=None,
        help="save the run to DIR: the SurroForge project (DIR/naca — runs, lineage, POD "
             "basis, models) plus the exported artifacts; without it the project goes to a "
             "temp dir and is removed, and only the artifacts (in the sweep dir) persist.",
    )
    args = parser.parse_args()
    target = Path(args.run_dir)
    if not (target / "samples.csv").is_file():
        sys.exit(f"No sweep found at {target}/ (expected samples.csv). Run run_sweep.py first.")
    out_dir = Path(args.out) if args.out else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    main(target, out_dir)
