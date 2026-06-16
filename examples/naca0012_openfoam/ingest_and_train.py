#!/usr/bin/env python3
"""Ingest a real OpenFOAM NACA 0012 sweep and train the SurroForge surrogates.

Run this after ``run_sweep.py`` has produced a sweep directory. It reads each case's
converged pressure field with ``surroforge.data.read_openfoam_sweep`` and the **real**
lift coefficient ``Cl`` that ``forceCoeffs`` integrated, then trains the same two
surrogates as the synthetic CI example (``examples/09_naca0012_openfoam_surrogate.py``):

  * a **field** surrogate (``reduced-field`` POD + neural inner): AoA -> pressure field;
  * a **scalar** surrogate: AoA -> lift coefficient Cl.

Both use Core ``mlp``; the PyTorch ``torch-mlp`` backend joins the comparison if the
``surroforge-torch`` plugin is installed. Unlike example 09 this needs no solver here —
it consumes the output ``run_sweep.py`` already produced.

    python examples/naca0012_openfoam/ingest_and_train.py run
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from surroforge import Project, Trainer
from surroforge.data import read_openfoam_sweep
from surroforge.plugins import available_trainers
from surroforge.services import compare_runs

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # benign lbfgs iteration cap
_HAS_TORCH = "torch-mlp" in available_trainers()

# A case's C_L counts as settled if its spread over the last CL_WINDOW writes is within
# CL_RTOL of |C_L| (plus a small absolute floor for near-zero lift).
CL_WINDOW = 20
CL_RTOL = 1e-2
CL_ATOL = 1e-4


class ClReading(NamedTuple):
    value: float  # C_L at the last (most converged) time
    spread: float  # peak-to-peak of C_L over the last CL_WINDOW writes
    settled: bool  # spread within CL_RTOL*|value| + CL_ATOL — a genuine plateau
    n_used: int  # how many writes the plateau check looked at


def read_cl(case_dir: Path) -> ClReading:
    """Parse the lift coefficient from a case's forceCoeffs output, and check it has settled.

    Reads ``postProcessing/forceCoeffs1/<t>/coefficient.dat``, takes ``Cl`` at the last (most
    converged) time, and inspects its peak-to-peak spread over the final ``CL_WINDOW`` writes:
    a still-swinging C_L means the run had not yet reached a steady value even if simpleFoam
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
            if "Cl" in cols:  # the column-name header (not the meta lines)
                header = cols
        elif stripped:
            data_rows.append(stripped.split())
    if header is None or not data_rows:
        raise ValueError(f"could not parse Cl from {dat[-1]}")
    idx = header.index("Cl")
    if idx >= len(data_rows[-1]):
        raise ValueError(f"truncated data row (no Cl value) in {dat[-1]}")
    try:
        cl_series = [float(r[idx]) for r in data_rows if idx < len(r)]
    except ValueError as exc:
        raise ValueError(f"non-numeric Cl value in {dat[-1]}") from exc
    value = cl_series[-1]
    window = cl_series[-CL_WINDOW:]
    spread = max(window) - min(window)
    settled = spread <= CL_RTOL * abs(value) + CL_ATOL
    return ClReading(value=value, spread=spread, settled=settled, n_used=len(window))


def _nn(method: str, widths: list[int]) -> dict:
    return {"method": method, "hidden_layer_sizes": widths, "activation": "tanh", "max_iter": 5000}


def _candidates(widths: list[int]) -> list[tuple[str, dict]]:
    cands = [("mlp", _nn("mlp", widths))]
    if _HAS_TORCH:
        cands.append(("torch-mlp", _nn("torch-mlp", widths)))
    cands.append(("polynomial", {"method": "polynomial", "degree": 2}))
    return cands


def main(run_dir: Path) -> None:
    doe = pd.read_csv(run_dir / "samples.csv").set_index("case")
    snapshots = read_openfoam_sweep(
        run_dir, field="p", params=lambda case: {"aoa": float(doe.loc[case.name, "aoa"])}
    )
    # The real integrated lift coefficient forceCoeffs reported, per case. Align it by angle of
    # attack (robust to row order or a case the field reader dropped), not by positional order.
    case_names = sorted(
        d.name for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("aoa_")
    )
    cl_by_aoa, unsettled = {}, []
    for name in case_names:
        aoa = float(doe.loc[name, "aoa"])
        reading = read_cl(run_dir / name)
        cl_by_aoa[aoa] = reading.value
        if not reading.settled:
            unsettled.append((aoa, reading.spread))
    snapshots["cl"] = snapshots["aoa"].map(cl_by_aoa)
    if snapshots["cl"].isna().any():
        raise RuntimeError("could not match a real Cl to every ingested case by angle of attack")
    field_cols = [c for c in snapshots.columns if c.startswith("p_")]
    print(f"Ingested {len(snapshots)} real cases: {len(field_cols)} pressure DOFs + Cl")
    if unsettled:
        listed = ", ".join(f"{a:+.1f}° (±{s / 2:.3g})" for a, s in unsettled)
        print(f"⚠  {len(unsettled)}/{len(case_names)} case(s) have a still-swinging C_L over the "
              f"last {CL_WINDOW} writes: {listed}\n"
              f"   Their lift had not settled — re-run those angles with more --iters.")
    print()

    with tempfile.TemporaryDirectory(prefix="surroforge-naca-") as tmp:
        csv = Path(tmp) / "snapshots.csv"
        snapshots.to_csv(csv, index=False)
        project = Project.init(Path(tmp) / "naca", name="naca0012-cfd")
        project.add_dataset(csv, name="sweep")

        split = {"test_fraction": 0.25, "seed": 1}
        validation = {"cross_validation": 3, "metrics": ["rmse", "r2", "mae"]}

        print("=== Field surrogate: AoA -> pressure field (POD + inner) ===")
        field_runs = {
            label: Trainer.from_dict({
                "experiment": f"field-{label}",
                "data": {"source": "sweep", "inputs": ["aoa"], "outputs": ["p_*"], "split": split},
                "model": {"method": "reduced-field",
                          "reduction": {"basis": "pod", "energy": 0.999}, "inner": inner},
                "validation": validation,
            }).fit(project)
            for label, inner in _candidates([48, 48])
        }
        _report(project, field_runs, "inner")

        print("\n=== Scalar surrogate: AoA -> lift coefficient Cl ===")
        cl_runs = {
            label: Trainer.from_dict({
                "experiment": f"cl-{label}",
                "data": {"source": "sweep", "inputs": ["aoa"], "outputs": ["cl"], "split": split},
                "model": model,
                "validation": validation,
            }).fit(project)
            for label, model in _candidates([32, 32])
        }
        _report(project, cl_runs, "method")

    if not _HAS_TORCH:
        print("\n(Install ./plugins/surroforge-torch to add the PyTorch `torch-mlp` backend.)")


def _report(project: Project, runs: dict, label_kind: str) -> None:
    label_by_id = {r.run_id: name for name, r in runs.items()}
    result = compare_runs(project, list(label_by_id), metric="rmse")
    print(f"{label_kind:<12}{'test R²':>10}{'test RMSE':>12}")
    for row in result["rows"]:
        marker = "  <- best" if row["run_id"] == result["best"] else ""
        print(f"{label_by_id[row['run_id']]:<12}{row['test']['r2']:>10.4f}"
              f"{row['test']['rmse']:>12.4g}{marker}")
    best_row = next(r for r in result["rows"] if r["run_id"] == result["best"])
    if best_row["test"]["r2"] < 0:
        print("  (all test R² < 0 — data-starved sweep; '<- best' is lowest-RMSE only. "
              "Use more angles and --iters; see this directory's README.)")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("run")
    if not (target / "samples.csv").is_file():
        sys.exit(f"No sweep found at {target}/ (expected samples.csv). Run run_sweep.py first.")
    main(target)
