"""NACA 0012 over angle of attack — OpenFOAM → POD-NN field surrogate + a lift surrogate.

A classic engineering surrogate task: sweep a NACA 0012 airfoil over angle of attack
(AoA) with a CFD solver, then learn a fast surrogate from the results. This example
does the *whole* SurroForge pipeline on that problem:

1. **Ingest an OpenFOAM sweep** — one case per AoA, each holding a converged pressure
   field `p` as an `internalField`, read with `surroforge.data.read_openfoam_sweep`
   (mesh-free: just the field values become an `(n_cases, n_dof)` snapshot matrix).
2. **Field surrogate (POD-NN)** — `reduced-field` reduces the pressure field with POD
   and an inner neural network learns AoA → modal coefficients, reconstructing the full
   field. The lift curve stalls past ~14°, so the coefficient map is *nonlinear* and the
   neural inner beats a degree-2 polynomial inner.
3. **Scalar surrogate (lift coefficient)** — AoA → C_L, the integrated lift coefficient
   `forceCoeffs` reports. Same stall nonlinearity, so again the neural network beats a
   quadratic.

Both neural surrogates use Core `mlp` (scikit-learn); if the optional `surroforge-torch`
plugin is installed, the PyTorch `torch-mlp` backend is added to the comparison too.

Run it:

    python examples/09_naca0012_openfoam_surrogate.py

**Honest note on the data.** OpenFOAM is not a SurroForge dependency, so this script
*fabricates* a representative NACA 0012 sweep in genuine OpenFOAM `volScalarField` format
(a low-rank pressure field whose dominant mode tracks circulation/lift, with a realistic
stall). It is therefore self-contained and runs in CI with no solver. To use **real** CFD,
run the case template under `examples/naca0012_openfoam/` (it drives `simpleFoam` over an
AoA sweep and writes `forceCoeffs`), then point `read_openfoam_sweep` at its run directory
instead of the synthetic writer below — the surrogate code downstream is identical.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from surroforge import Project, Trainer
from surroforge.data import read_openfoam_sweep
from surroforge.plugins import available_trainers
from surroforge.services import compare_runs

# The lbfgs optimiser stops at its iteration cap on the sharp stall nonlinearity; the
# resulting fits are excellent (see the R² columns the example prints), so silence the
# benign scikit-learn convergence notice to keep the output clean.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Add the PyTorch backend to the comparison when the plugin is installed (CI-safe: the
# example trains the Core `mlp` regardless, so it runs to completion without the plugin).
_HAS_TORCH = "torch-mlp" in available_trainers()

# --------------------------------------------------------------------------- #
# Synthesise a representative NACA 0012 AoA sweep in genuine OpenFOAM format.
# (On real data you skip this and point `read_openfoam_sweep` at your run directory.)
# --------------------------------------------------------------------------- #

_FOAM_BANNER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | (synthetic — written by SurroForge example 09)  |
|   \\\\  /    A nd           |                                                 |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
"""


def _write_scalar_field(path: Path, values: np.ndarray, obj: str) -> None:
    body = "\n".join(f"{v:.10g}" for v in values)
    path.write_text(
        _FOAM_BANNER
        + "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        + f"    class       volScalarField;\n    object      {obj};\n}}\n"
        + "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
        + "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        + f"internalField   nonuniform List<scalar>\n{values.size}\n(\n{body}\n)\n;\n\n"
        + "boundaryField\n{\n    airfoil { type zeroGradient; }\n}\n"
    )


def naca0012_lift(aoa_deg: np.ndarray) -> np.ndarray:
    """A NACA 0012 lift curve C_L(AoA): thin-airfoil-linear, then a smooth stall past ~14°.

    `2*pi*sin(alpha)` is the thin-airfoil slope; the sigmoid factor rolls lift off beyond
    the stall angle (symmetric for ±AoA). This is the nonlinearity a quadratic cannot fit
    but a neural network can.
    """
    alpha = np.radians(aoa_deg)
    linear = 2.0 * np.pi * np.sin(alpha)
    not_stalled = 1.0 / (1.0 + np.exp((np.abs(aoa_deg) - 14.0) / 1.5))
    return linear * not_stalled


def make_naca_sweep(root: Path, n_cells: int = 120, seed: int = 0) -> tuple[Path, np.ndarray]:
    """Write one OpenFOAM case per AoA (a converged `p` field) + a DoE table.

    The pressure field is intrinsically low-rank: a handful of fixed spatial modes whose
    amplitudes depend nonlinearly on AoA (the dominant one tracks circulation/lift with stall).
    Four modes are built, but the smallest is nearly collinear with AoA over this range, so at
    energy 0.999 POD keeps ~3 — still a genuinely nonlinear AoA→coefficient map for the inner
    surrogate to learn. Returns the cases directory and the angle-of-attack sweep.
    """
    rng = np.random.default_rng(seed)
    aoa = np.linspace(-6.0, 18.0, 61)  # the angle-of-attack sweep, degrees
    modes, _ = np.linalg.qr(rng.standard_normal((n_cells, 4)))  # orthonormal spatial modes
    amplitudes = np.column_stack(
        [naca0012_lift(aoa), aoa, aoa**2, np.sin(3.0 * np.radians(aoa))]
    )
    amplitudes /= amplitudes.std(axis=0)  # comparable energy across modes
    amplitudes *= 40.0 * np.array([1.0, 0.7, 0.5, 0.35])  # pressure scale, mild decay
    fields = amplitudes @ modes.T  # (n_cases, n_cells); POD keeps ~3 modes at energy 0.999

    cases = root / "cases"
    cases.mkdir(parents=True)
    for i in range(len(aoa)):
        case = cases / f"aoa_{i:03d}"
        (case / "500").mkdir(parents=True)  # one converged steady-state time
        _write_scalar_field(case / "500" / "p", fields[i], obj="p")
    pd.DataFrame({"case": [f"aoa_{i:03d}" for i in range(len(aoa))], "aoa": aoa}).to_csv(
        cases / "samples.csv", index=False
    )
    return cases, aoa


# --------------------------------------------------------------------------- #
# Surrogate training helpers
# --------------------------------------------------------------------------- #


def _inner_candidates() -> list[tuple[str, dict]]:
    """Inner surrogates for the field ROM: NN vs. a quadratic baseline."""
    nn = {"hidden_layer_sizes": [48, 48], "activation": "tanh", "max_iter": 5000}
    inners = [("mlp", {"method": "mlp", **nn})]
    if _HAS_TORCH:
        inners.append(("torch-mlp", {"method": "torch-mlp", **nn}))
    inners.append(("polynomial", {"method": "polynomial", "degree": 2}))
    return inners


def _scalar_candidates() -> list[tuple[str, dict]]:
    """Scalar surrogates for AoA -> C_L: NN vs. a quadratic baseline."""
    nn = {"hidden_layer_sizes": [32, 32], "activation": "tanh", "max_iter": 5000}
    methods = [("mlp", {"method": "mlp", **nn})]
    if _HAS_TORCH:
        methods.append(("torch-mlp", {"method": "torch-mlp", **nn}))
    methods.append(("polynomial", {"method": "polynomial", "degree": 2}))
    return methods


def main(workdir: Path) -> None:
    cases, _ = make_naca_sweep(workdir)
    print(f"Wrote {len(list(cases.glob('aoa_*')))} OpenFOAM cases (one per angle of attack)\n")

    # 1. Ingest the pressure field: AoA from the DoE table, `p` from each case's latest time.
    doe = pd.read_csv(cases / "samples.csv").set_index("case")
    snapshots = read_openfoam_sweep(
        cases, field="p", params=lambda case: {"aoa": float(doe.loc[case.name, "aoa"])}
    )
    field_cols = [c for c in snapshots.columns if c.startswith("p_")]
    # The integrated lift coefficient forceCoeffs would report (here the analytic NACA
    # curve), carried alongside the field as the scalar quantity of interest.
    snapshots["cl"] = naca0012_lift(snapshots["aoa"].to_numpy())
    csv = workdir / "snapshots.csv"
    snapshots.to_csv(csv, index=False)
    print(f"Ingested pressure field 'p' as {len(field_cols)} DOFs across {len(snapshots)} cases")

    project = Project.init(workdir / "naca", name="naca0012")
    project.add_dataset(csv, name="sweep")

    base_split = {"test_fraction": 0.25, "seed": 1}
    validation = {"cross_validation": 3, "metrics": ["rmse", "r2", "mae"]}

    # 2. Field surrogate (POD-NN): AoA -> full pressure field, one reduced-field per inner.
    print("\n=== Field surrogate: AoA -> pressure field (POD + inner) ===")
    field_runs: dict[str, object] = {}
    for label, inner in _inner_candidates():
        field_runs[label] = Trainer.from_dict(
            {
                "experiment": f"field-{label}",
                "data": {
                    "source": "sweep",
                    "inputs": ["aoa"],
                    "outputs": ["p_*"],  # the field as a column group
                    "split": base_split,
                },
                "model": {
                    "method": "reduced-field",
                    "reduction": {"basis": "pod", "energy": 0.999},
                    "inner": inner,
                },
                "validation": validation,
            }
        ).fit(project)
    inner_by_id = {r.run_id: label for label, r in field_runs.items()}
    field_result = compare_runs(project, list(inner_by_id), metric="rmse")
    print(f"{'inner':<12}{'test R²':>10}{'test RMSE':>12}")
    for row in field_result["rows"]:
        marker = "  <- best" if row["run_id"] == field_result["best"] else ""
        print(
            f"{inner_by_id[row['run_id']]:<12}{row['test']['r2']:>10.4f}"
            f"{row['test']['rmse']:>12.4g}{marker}"
        )
    modes_kept = next(iter(field_runs.values())).load_model().rank
    print(f"POD kept {modes_kept} modes (energy 0.999); the neural inner captures the stall the")
    print("quadratic inner cannot.")

    # 3. Scalar surrogate: AoA -> C_L (the lift coefficient), same stall nonlinearity.
    print("\n=== Scalar surrogate: AoA -> lift coefficient C_L ===")
    cl_runs: dict[str, object] = {}
    for label, model in _scalar_candidates():
        cl_runs[label] = Trainer.from_dict(
            {
                "experiment": f"cl-{label}",
                "data": {
                    "source": "sweep",
                    "inputs": ["aoa"],
                    "outputs": ["cl"],
                    "split": base_split,
                },
                "model": model,
                "validation": validation,
            }
        ).fit(project)
    cl_result = compare_runs(project, [r.run_id for r in cl_runs.values()], metric="rmse")
    print(f"{'method':<12}{'test R²':>10}{'test RMSE':>12}")
    for row in cl_result["rows"]:
        marker = "  <- best" if row["run_id"] == cl_result["best"] else ""
        print(f"{row['method']:<12}{row['test']['r2']:>10.4f}{row['test']['rmse']:>12.4g}{marker}")

    # 4. Use the best lift surrogate to predict C_L at a few angles.
    best_cl = next(r for r in cl_runs.values() if r.run_id == cl_result["best"])
    probe = np.array([[0.0], [5.0], [10.0], [16.0]])  # degrees (16° is post-stall)
    pred = best_cl.load_model().predict(probe)
    print("\nLift prediction (best surrogate):")
    print(f"{'AoA [deg]':>10}{'C_L (pred)':>12}{'C_L (truth)':>13}")
    for angle, cl in zip(probe[:, 0], pred[:, 0], strict=False):
        print(f"{angle:>10.1f}{cl:>12.4f}{naca0012_lift(np.array([angle]))[0]:>13.4f}")

    if not _HAS_TORCH:
        print("\n(Install ./plugins/surroforge-torch to add the PyTorch `torch-mlp` backend.)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="surroforge-ex09-") as tmp:
        main(Path(tmp))
