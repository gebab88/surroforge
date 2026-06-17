"""NACA 0012 over angle of attack and speed — OpenFOAM → POD-NN field + scalar surrogates.

A classic engineering surrogate task: sweep a NACA 0012 airfoil over angle of attack
(AoA) **and freestream speed** with a CFD solver, then learn a fast surrogate from the
results. Both inputs `(AoA, speed)` feed both models. This example does the *whole*
SurroForge pipeline on that problem:

1. **Ingest an OpenFOAM sweep** — one case per `(AoA, speed)` sample, each holding a
   converged pressure field `p` as an `internalField`, read with
   `surroforge.data.read_openfoam_sweep` (mesh-free: the field values become an
   `(n_cases, n_dof)` snapshot matrix).
2. **Field surrogate (POD-NN)** — `reduced-field` reduces the pressure field with POD and
   an inner neural network learns `(AoA, speed) → modal coefficients`, reconstructing the
   full field. The field is `Cp(AoA)·q(U)`: the angle shapes it (nonlinear, with stall),
   the speed scales its magnitude with the dynamic pressure `q = ½ρU²`.
3. **Scalar surrogate** — `(AoA, speed) → [C_L, C_D]`: the lift and drag *coefficients*, one
   network with two outputs. Both carry the stall nonlinearity (the stall angle drifts with
   speed, a Reynolds stand-in) and sit on a comparable O(1) scale, so a single shared network
   fits both — and the stall means it beats a degree-2 polynomial.

Both neural surrogates use Core `mlp` (scikit-learn); if the optional `surroforge-torch`
plugin is installed, the PyTorch `torch-mlp` backend is added to the comparison too.

Run it:

    python examples/09_naca0012_openfoam_surrogate.py
    python examples/09_naca0012_openfoam_surrogate.py --out ./ex09_output  # keep the artifacts

**Honest note on the data.** OpenFOAM is not a SurroForge dependency, so this script
*fabricates* a representative NACA 0012 sweep in genuine OpenFOAM `volScalarField` format
(a low-rank pressure field whose dominant mode tracks circulation/lift, with a realistic
stall). It is therefore self-contained and runs in CI with no solver. To use **real** CFD,
run the case template under `examples/naca0012_openfoam/` (it drives `simpleFoam` over an
AoA sweep and writes `forceCoeffs`), then point `read_openfoam_sweep` at its run directory
instead of the synthetic writer below — the surrogate code downstream is identical.
"""

from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from surroforge import Project, Trainer
from surroforge.data import read_openfoam_sweep
from surroforge.errors import ExportError
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


RHO = 1.0  # freestream density (incompressible; matches the real template's rhoInf=1)
SPEED_REF = 30.0  # reference freestream speed [m/s] at which the stall angle is ~14°


def dynamic_pressure(speed: np.ndarray) -> np.ndarray:
    """Freestream dynamic pressure q = ½·ρ·U² — what the pressure field scales with."""
    return 0.5 * RHO * np.asarray(speed, dtype=float) ** 2


def naca0012_cl(aoa_deg: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """NACA 0012 lift coefficient C_L(AoA, U): thin-airfoil-linear, then a smooth stall.

    `2*pi*sin(alpha)` is the thin-airfoil slope; the sigmoid rolls lift off past the stall
    angle (symmetric for ±AoA). The stall angle drifts up with speed — a stand-in for the
    Reynolds-number effect (faster flow ⇒ higher Re ⇒ slightly delayed stall) — so the
    *coefficient* genuinely depends on U too, not only the angle.
    """
    aoa_deg = np.asarray(aoa_deg, dtype=float)
    speed = np.asarray(speed, dtype=float)
    linear = 2.0 * np.pi * np.sin(np.radians(aoa_deg))
    stall_deg = 14.0 + 2.0 * np.log(speed / SPEED_REF)  # Reynolds stall delay
    not_stalled = 1.0 / (1.0 + np.exp((np.abs(aoa_deg) - stall_deg) / 1.5))
    return linear * not_stalled


def _stall_fraction(aoa_deg: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """0 (fully attached) → 1 (fully separated), with the same Reynolds stall drift as C_L."""
    aoa_deg = np.asarray(aoa_deg, dtype=float)
    stall_deg = 14.0 + 2.0 * np.log(np.asarray(speed, dtype=float) / SPEED_REF)
    return 1.0 - 1.0 / (1.0 + np.exp((np.abs(aoa_deg) - stall_deg) / 1.5))


def naca0012_cd(aoa_deg: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """NACA 0012 drag coefficient C_D(AoA, U): a simple drag polar with a post-stall rise.

    `cd0 + k·C_L²` is the classic parasitic-plus-drag-due-to-lift polar; a separation term
    makes C_D climb steeply past stall. Both C_L and C_D therefore share the same stall
    nonlinearity and sit on a comparable O(1) scale, so one network fits both cleanly. The
    speed enters via the Reynolds stall drift (in C_L and the separation term) and a mild
    Reynolds reduction of the parasitic drag.
    """
    cl = naca0012_cl(aoa_deg, speed)
    cd0 = 0.02 - 0.005 * np.log(np.asarray(speed, dtype=float) / SPEED_REF)  # Reynolds skin drag
    return cd0 + 0.05 * cl**2 + 1.1 * _stall_fraction(aoa_deg, speed)  # parasitic + induced + sep.


def make_naca_sweep(root: Path, n_cells: int = 120, seed: int = 0) -> Path:
    """Write one OpenFOAM case per (AoA, speed) sample (a converged `p` field) + a DoE table.

    The DoE is a 2-D grid over angle of attack **and** freestream speed. The pressure field
    factorises the way real aerodynamics does: a dimensionless pressure-coefficient pattern
    set by the angle, scaled by the dynamic pressure q = ½ρU² set by the speed —
    ``p = Cp(AoA) · q(U)``. So both inputs genuinely drive the field (the angle shapes it, the
    speed scales its magnitude with U²). The Cp pattern is intrinsically low-rank: a handful of
    fixed spatial modes whose amplitudes depend nonlinearly on AoA (the dominant one tracks
    circulation/lift with stall); at energy 0.999 POD keeps ~3. Returns the cases directory.
    """
    rng = np.random.default_rng(seed)
    aoa_axis = np.linspace(-6.0, 18.0, 17)  # angle-of-attack axis, degrees
    speed_axis = np.linspace(20.0, 60.0, 7)  # freestream-speed axis, m/s
    aoa_grid, speed_grid = np.meshgrid(aoa_axis, speed_axis)
    aoa, speed = aoa_grid.ravel(), speed_grid.ravel()  # 119 (AoA, speed) samples

    modes, _ = np.linalg.qr(rng.standard_normal((n_cells, 4)))  # orthonormal spatial modes
    cp_amp = np.column_stack(  # AoA-dependent, dimensionless pressure-coefficient amplitudes
        [naca0012_cl(aoa, SPEED_REF), aoa, aoa**2, np.sin(3.0 * np.radians(aoa))]
    )
    cp_amp /= cp_amp.std(axis=0)  # comparable energy across modes
    cp_amp *= np.array([1.0, 0.7, 0.5, 0.35])  # Cp ~ O(1), mild modal decay
    cp_field = cp_amp @ modes.T  # (n_cases, n_cells) dimensionless Cp distribution
    fields = dynamic_pressure(speed)[:, None] * cp_field  # pressure p = Cp · q(U)

    cases = root / "cases"
    cases.mkdir(parents=True)
    for i in range(len(aoa)):
        case = cases / f"case_{i:03d}"
        (case / "500").mkdir(parents=True)  # one converged steady-state time
        _write_scalar_field(case / "500" / "p", fields[i], obj="p")
    pd.DataFrame(
        {"case": [f"case_{i:03d}" for i in range(len(aoa))], "aoa": aoa, "speed": speed}
    ).to_csv(cases / "samples.csv", index=False)
    return cases


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
    """Scalar surrogates for (AoA, speed) -> [C_L, C_D]: NN vs. a quadratic baseline."""
    nn = {"hidden_layer_sizes": [48, 48], "activation": "tanh", "max_iter": 5000}
    methods = [("mlp", {"method": "mlp", **nn})]
    if _HAS_TORCH:
        methods.append(("torch-mlp", {"method": "torch-mlp", **nn}))
    methods.append(("polynomial", {"method": "polynomial", "degree": 2}))
    return methods


def main(workdir: Path, out_dir: Path | None = None) -> None:
    # Scratch data (synthetic cases, project, runs) lives under workdir; the trained-model
    # artifacts go to out_dir when given (a persistent output folder), else alongside it.
    export_dir = out_dir if out_dir is not None else workdir
    cases = make_naca_sweep(workdir)
    print(f"Wrote {len(list(cases.glob('case_*')))} OpenFOAM cases (AoA × speed DoE)\n")

    # 1. Ingest the pressure field: AoA and speed from the DoE table, `p` from each case.
    doe = pd.read_csv(cases / "samples.csv").set_index("case")
    snapshots = read_openfoam_sweep(
        cases, field="p",
        params=lambda case: {
            "aoa": float(doe.loc[case.name, "aoa"]),
            "speed": float(doe.loc[case.name, "speed"]),
        },
    )
    field_cols = [c for c in snapshots.columns if c.startswith("p_")]
    # The scalar quantities of interest, carried alongside the field: the lift and drag
    # coefficients C_L and C_D (what forceCoeffs reports), both on a comparable O(1) scale.
    aoa_arr, speed_arr = snapshots["aoa"].to_numpy(), snapshots["speed"].to_numpy()
    snapshots["cl"] = naca0012_cl(aoa_arr, speed_arr)
    snapshots["cd"] = naca0012_cd(aoa_arr, speed_arr)
    csv = workdir / "snapshots.csv"
    snapshots.to_csv(csv, index=False)
    print(f"Ingested pressure field 'p' as {len(field_cols)} DOFs across {len(snapshots)} cases")

    project = Project.init(workdir / "naca", name="naca0012")
    project.add_dataset(csv, name="sweep")

    base_split = {"test_fraction": 0.25, "seed": 1}
    validation = {"cross_validation": 3, "metrics": ["rmse", "r2", "mae"]}

    # 2. Field surrogate (POD-NN): (AoA, speed) -> full pressure field, one reduced-field per inner.
    print("\n=== Field surrogate: (AoA, speed) -> pressure field (POD + inner) ===")
    field_runs: dict[str, object] = {}
    for label, inner in _inner_candidates():
        field_runs[label] = Trainer.from_dict(
            {
                "experiment": f"field-{label}",
                "data": {
                    "source": "sweep",
                    "inputs": ["aoa", "speed"],
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

    # 3. Scalar surrogate: (AoA, speed) -> [C_L, C_D]. Both coefficients share the stall
    #    nonlinearity and sit on a comparable scale, so one network fits both well.
    print("\n=== Scalar surrogate: (AoA, speed) -> [C_L, C_D] ===")
    cl_runs: dict[str, object] = {}
    for label, model in _scalar_candidates():
        cl_runs[label] = Trainer.from_dict(
            {
                "experiment": f"coeff-{label}",
                "data": {
                    "source": "sweep",
                    "inputs": ["aoa", "speed"],
                    "outputs": ["cl", "cd"],
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

    # 4. Use the best scalar surrogate to predict [C_L, C_D] at a few (AoA, speed) points.
    best_cl = next(r for r in cl_runs.values() if r.run_id == cl_result["best"])
    probe = np.array([[5.0, 30.0], [10.0, 30.0], [10.0, 50.0], [16.0, 40.0]])  # (AoA°, U m/s)
    pred = best_cl.load_model().predict(probe)
    print("\nScalar prediction (best surrogate) — pred vs truth:")
    print(f"{'AoA':>5}{'U':>6}{'C_L pred':>10}{'C_L true':>10}{'C_D pred':>10}{'C_D true':>10}")
    for (angle, spd), (cl_p, cd_p) in zip(probe, pred, strict=False):
        cl_t, cd_t = naca0012_cl(angle, spd), naca0012_cd(angle, spd)
        print(f"{angle:>5.0f}{spd:>6.0f}{cl_p:>10.4f}{cl_t:>10.4f}{cd_p:>10.4f}{cd_t:>10.4f}")

    # 5. Export the surrogates to portable artifacts, one per format SurroForge offers:
    #    ONNX (framework-agnostic graph), a native PyTorch `.pt` checkpoint, and pickle.
    print("\n=== Export ===")
    # Best coefficient surrogate -> ONNX. Every candidate method (mlp/torch-mlp/polynomial)
    # supports ONNX, so this always succeeds — unless the optional `onnx` extra is missing.
    try:
        out = best_cl.export("onnx", export_dir / "coeffs_surrogate.onnx")
        print(f"  onnx  -> {out.name} ({out.stat().st_size} bytes)  [best: {best_cl.method}]")
    except ExportError as exc:
        print(f"  onnx  unavailable — {exc}")
    # Native PyTorch `.pt` of the torch-mlp coefficient surrogate, when the plugin is installed.
    # Only torch-backed methods offer `.pt`, so we export that run explicitly (it ties the best
    # here) rather than the best — this guarantees the native checkpoint is always demonstrated.
    torch_run = cl_runs.get("torch-mlp")
    if torch_run is not None:
        out = torch_run.export("pt", export_dir / "coeffs_surrogate.pt")
        print(f"  pt    -> {out.name} ({out.stat().st_size} bytes)  [torch-mlp, native PyTorch]")
    else:
        print("  pt    skipped — install ./plugins/surroforge-torch for the torch-mlp backend")
    # The field surrogate is reduced-field, which exports to pickle only (a POD basis +
    # inner model, not a single graph); export it that way for completeness.
    field_best = next(r for r in field_runs.values() if r.run_id == field_result["best"])
    field_out = field_best.export("pickle", export_dir / "field_surrogate.pkl")
    print(f"  field surrogate -> {field_out.name} (pickle; reduced-field is pickle-only)")
    if out_dir is not None:
        print(f"\nArtifacts written to {export_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", metavar="DIR", default=None,
        help="persist the trained-model artifacts (cl_surrogate.onnx/.pt, "
             "field_surrogate.pkl) to DIR; without it they go to a temp dir and are removed.",
    )
    args = parser.parse_args()
    # Scratch always lives in a temp dir (self-contained / CI-safe); --out only keeps the
    # exported artifacts, so a re-run never collides with an existing project.
    out_dir = Path(args.out) if args.out else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="surroforge-ex09-") as tmp:
        main(Path(tmp), out_dir)
