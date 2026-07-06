#!/usr/bin/env python3
"""Ingest a real OpenFOAM NACA 0012 sweep and train the SurroForge surrogates.

Run this after ``run_sweep.py`` has produced a sweep directory. It reads each case's
converged pressure and velocity fields with ``surroforge.data.read_openfoam_sweep`` and the
**real** lift and drag coefficients ``Cl``/``Cd`` that ``forceCoeffs`` integrated, then trains
the two surrogates of the synthetic CI example (``examples/09_naca0012_openfoam_surrogate.py``)
plus a velocity-field surrogate:

  * a **field** surrogate (``reduced-field`` POD + neural inner): (AoA, speed) -> pressure field;
  * a **velocity** surrogate (same construction): (AoA, speed) -> velocity field (Ux, Uy, Uz);
  * a **scalar** surrogate: (AoA, speed) -> [Cl, Cd] (lift and drag coefficients).

Both inputs come from the 2-D DoE ``run_sweep.py`` wrote (angle × speed). Both models use
Core ``mlp``; the PyTorch ``torch-mlp`` backend joins the comparison if the
``surroforge-torch`` plugin is installed. Unlike example 09 this needs no solver here —
it consumes the output ``run_sweep.py`` already produced.

    python examples/naca0012_openfoam/ingest_and_train.py run
    python examples/naca0012_openfoam/ingest_and_train.py run --out ./ex09_real   # save the run

With ``--out`` the SurroForge project (datasets, runs, lineage, POD basis, models) **and** the
exported artifacts are saved under that directory, so the real run is inspectable/reproducible
afterwards (``surroforge --config-dir <out>/naca list``). The best pressure- and velocity-field
surrogates are also evaluated on **every** case from ``samples.csv`` and written into one VTKHDF
time-series file (cell arrays ``p`` and ``U``)
``<out>/surrogate_vtk/surrogate_fields.vtkhdf`` (or
``<run_dir>/surrogate_vtk/surrogate_fields.vtkhdf`` without ``--out``), plus ``samples.csv`` for
the case/AoA/speed mapping. Without ``--out`` the project is built in a temp dir and removed, and
only the exported artifacts plus the surrogate VTKHDF file persist.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from surrogate_vtk import write_surrogate_vtkhdf

from surroforge import Project, Trainer
from surroforge.data import read_openfoam_sweep
from surroforge.errors import ExportError
from surroforge.plugins import available_trainers
from surroforge.services import compare_runs

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # benign lbfgs iteration cap
_HAS_TORCH = "torch-mlp" in available_trainers()

# A case counts as settled if its C_L spread over the last CL_WINDOW writes is within
# CL_RTOL of |C_L| (plus a small absolute floor for near-zero lift).
CL_WINDOW = 20
CL_RTOL = 1e-2
CL_ATOL = 1e-4


class CoeffReading(NamedTuple):
    cl: float  # lift coefficient at the last (most converged) time
    cd: float  # drag coefficient at the last time
    spread: float  # peak-to-peak of C_L over the last CL_WINDOW writes
    settled: bool  # C_L spread within CL_RTOL*|cl| + CL_ATOL — a genuine plateau


def read_coeffs(case_dir: Path) -> CoeffReading:
    """Parse Cl and Cd from a case's forceCoeffs output, and check the lift has settled.

    Reads ``postProcessing/forceCoeffs1/<t>/coefficient.dat``, takes ``Cl``/``Cd`` at the last
    (most converged) time, and inspects C_L's peak-to-peak spread over the final ``CL_WINDOW``
    writes: a still-swinging C_L means the run had not reached a steady value even if simpleFoam
    stopped (e.g. residual convergence on a marginally unsteady case).
    """
    # Sort time directories numerically (a restart could make '100' precede '20' lexically).
    dat = sorted(
        case_dir.glob("postProcessing/forceCoeffs1/*/coefficient.dat"),
        key=lambda p: float(p.parent.name),
    )
    if not dat:
        raise FileNotFoundError(f"no forceCoeffs coefficient.dat under {case_dir}")
    header, data_rows = None, []
    for line in dat[-1].read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            cols = stripped.lstrip("#").split()
            if "Cl" in cols and "Cd" in cols:  # the column-name header (not the meta lines)
                header = cols
        elif stripped:
            data_rows.append(stripped.split())
    if header is None or not data_rows:
        raise ValueError(f"could not parse Cl/Cd header from {dat[-1]}")
    cl_idx, cd_idx = header.index("Cl"), header.index("Cd")
    if max(cl_idx, cd_idx) >= len(data_rows[-1]):
        raise ValueError(f"truncated data row (no Cl/Cd value) in {dat[-1]}")
    try:
        cl_series = [float(r[cl_idx]) for r in data_rows if cl_idx < len(r)]
        cd = float(data_rows[-1][cd_idx])
    except ValueError as exc:
        raise ValueError(f"non-numeric Cl/Cd value in {dat[-1]}") from exc
    cl = cl_series[-1]
    window = cl_series[-CL_WINDOW:]
    spread = max(window) - min(window)
    settled = spread <= CL_RTOL * abs(cl) + CL_ATOL
    return CoeffReading(cl=cl, cd=cd, spread=spread, settled=settled)


def _nn(method: str, widths: list[int]) -> dict:
    return {"method": method, "hidden_layer_sizes": widths, "activation": "tanh", "max_iter": 5000}


def _candidates(widths: list[int]) -> list[tuple[str, dict]]:
    cands = [("mlp", _nn("mlp", widths))]
    if _HAS_TORCH:
        cands.append(("torch-mlp", _nn("torch-mlp", widths)))
    cands.append(("polynomial", {"method": "polynomial", "degree": 2}))
    return cands


def _key(aoa: float, speed: float) -> tuple[float, float]:
    """A robust (AoA, speed) lookup key — rounded so equal DoE values match exactly."""
    return (round(float(aoa), 6), round(float(speed), 6))


def main(run_dir: Path, out_dir: Path | None = None) -> None:
    # With --out the run is saved: the SurroForge project (datasets, runs, lineage, POD basis,
    # models) under <out_dir>/naca and the exported artifacts alongside it, so it is inspectable
    # and reproducible afterwards (surroforge --config-dir <out_dir>/naca list). Without it the
    # project lives in a temp dir and is removed, and the artifacts go into the sweep directory.
    export_dir = out_dir if out_dir is not None else run_dir
    doe = pd.read_csv(run_dir / "samples.csv").set_index("case")
    snapshots = read_openfoam_sweep(
        run_dir, field="p",
        params=lambda case: {
            "aoa": float(doe.loc[case.name, "aoa"]),
            "speed": float(doe.loc[case.name, "speed"]),
        },
    )
    velocity = read_openfoam_sweep(
        run_dir, field="U",
        params=lambda case: {
            "aoa": float(doe.loc[case.name, "aoa"]),
            "speed": float(doe.loc[case.name, "speed"]),
        },
    )
    # Both frames are sorted by case name, but a case that has `p` while missing `U`
    # (or vice versa) would silently shift the rows — refuse instead of misaligning.
    if not snapshots[["aoa", "speed"]].equals(velocity[["aoa", "speed"]]):
        raise RuntimeError(
            "pressure and velocity sweeps disagree on their (AoA, speed) rows — "
            "some case has one field but not the other; re-run the sweep for that case"
        )
    velocity_cols = [c for c in velocity.columns if c.startswith("U_")]
    snapshots = pd.concat([snapshots, velocity[velocity_cols]], axis=1)
    # The real Cl/Cd forceCoeffs reported, per case. Align by (AoA, speed) — robust to row order
    # or a case the field reader dropped — rather than positional order.
    case_names = sorted(
        d.name for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("case_")
    )
    coeffs, unsettled = {}, []
    for name in case_names:
        aoa, speed = float(doe.loc[name, "aoa"]), float(doe.loc[name, "speed"])
        reading = read_coeffs(run_dir / name)
        coeffs[_key(aoa, speed)] = reading
        if not reading.settled:
            unsettled.append((aoa, speed, reading.spread))
    keys = [_key(a, s) for a, s in zip(snapshots["aoa"], snapshots["speed"], strict=True)]
    if any(k not in coeffs for k in keys):
        raise RuntimeError("could not match real Cl/Cd to every ingested case by (AoA, speed)")
    snapshots["cl"] = [coeffs[k].cl for k in keys]
    snapshots["cd"] = [coeffs[k].cd for k in keys]
    field_cols = [c for c in snapshots.columns if c.startswith("p_")]
    print(f"Ingested {len(snapshots)} real cases: {len(field_cols)} pressure DOFs "
          f"+ {len(velocity_cols)} velocity DOFs + Cl + Cd")
    if unsettled:
        listed = ", ".join(f"({a:+.1f}°,{s:.4g}) ±{sp / 2:.3g}" for a, s, sp in unsettled)
        print(f"⚠  {len(unsettled)}/{len(case_names)} case(s) have a still-swinging C_L over the "
              f"last {CL_WINDOW} writes: {listed}\n"
              f"   Their lift had not settled — re-run those cases with more --iters.")
    print()

    with tempfile.TemporaryDirectory(prefix="surroforge-naca-") as tmp:
        csv = Path(tmp) / "snapshots.csv"
        snapshots.to_csv(csv, index=False)
        # Persist the project under <out_dir>/naca when --out is given (start it fresh on a re-run);
        # otherwise build it in the temp dir so it is removed on exit.
        if out_dir is not None:
            project_root = out_dir / "naca"
            if project_root.exists():
                shutil.rmtree(project_root)
        else:
            project_root = Path(tmp) / "naca"
        project = Project.init(project_root, name="naca0012-cfd")
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
        # coefficient model to ONNX, the torch-mlp coefficient model to a native PyTorch .pt
        # (when the plugin is installed), and the field surrogate to pickle (reduced-field only).
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
            print("  pt    skipped — install ./plugins/surroforge-torch for the torch-mlp backend")
        field_best = field_runs[_best_label(project, field_runs)]
        field_out = field_best.export("pickle", export_dir / "field_surrogate.pkl")
        print(f"  field surrogate -> {field_out.name} (pickle; reduced-field is pickle-only)")
        velocity_best = velocity_runs[_best_label(project, velocity_runs)]
        velocity_out = velocity_best.export("pickle", export_dir / "velocity_surrogate.pkl")
        print(f"  velocity surrogate -> {velocity_out.name} "
              "(pickle; reduced-field is pickle-only)")

        print("\n=== Surrogate field VTK ===")
        prediction_cases = doe.loc[case_names, ["aoa", "speed"]].reset_index()
        prediction_inputs = prediction_cases[inputs].to_numpy(dtype=float)
        vtk_dir = export_dir / "surrogate_vtk"
        vtkhdf = write_surrogate_vtkhdf(
            run_dir,
            vtk_dir / "surrogate_fields.vtkhdf",
            prediction_cases,
            {
                "p": field_best.load_model().predict(prediction_inputs),
                "U": velocity_best.load_model().predict(prediction_inputs),
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
