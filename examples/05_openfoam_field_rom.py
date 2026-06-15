"""OpenFOAM → Field ROM — ingest CFD case fields and surrogate them with POD.

A common real-world source of fields is an OpenFOAM parametric study: one case
directory per parameter combination, each holding a converged field (``p``, ``T``,
``U``, …) as a per-cell ``internalField`` list. SurroForge is *mesh-free* — it only
needs the field values as an ``(n_samples, n_dof)`` snapshot matrix — so bridging
OpenFOAM to the ROM is just "read each case's ``internalField`` into a row".

The reader lives in the package (:mod:`surroforge.data`); this script only synthesises a
tiny sweep of cases in genuine OpenFOAM ``volScalarField`` format so it is self-contained,
then ingests them with :func:`surroforge.data.read_openfoam_sweep` and trains the ordinary
``reduced-field`` method through the unchanged engine.

Run it:

    python examples/05_openfoam_field_rom.py

The default inner surrogate is polynomial, so this needs no optional backends. On real
cases you skip the synthetic writer below and call ``read_openfoam_sweep`` directly on your
run directory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from surroforge import Project, Trainer
from surroforge.data import read_openfoam_dimensioned_scalar, read_openfoam_sweep

# --------------------------------------------------------------------------- #
# Synthesise a tiny OpenFOAM sweep so the example is self-contained.
# (On real data you skip this and point `read_openfoam_sweep` at your run directory.)
# --------------------------------------------------------------------------- #

_FOAM_BANNER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | (synthetic — written by SurroForge example 05)  |
|   \\\\  /    A nd           |                                                 |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
"""


def _foam_file(class_: str, obj: str) -> str:
    return (
        _FOAM_BANNER
        + "FoamFile\n{\n"
        + "    version     2.0;\n    format      ascii;\n"
        + f"    class       {class_};\n    object      {obj};\n"
        + "}\n"
        + "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )


def _write_scalar_field(path: Path, values: np.ndarray, obj: str) -> None:
    lines = "\n".join(f"{v:.10g}" for v in values)
    path.write_text(
        _foam_file("volScalarField", obj)
        + "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        + f"internalField   nonuniform List<scalar>\n{values.size}\n(\n{lines}\n)\n;\n\n"
        + "boundaryField\n{\n    walls { type zeroGradient; }\n}\n"
    )


def _write_transport_properties(path: Path, nu: float) -> None:
    path.write_text(
        _foam_file("dictionary", "transportProperties")
        + "transportModel  Newtonian;\n"
        + f"nu              [0 2 -1 0 0 0 0] {nu:.10g};\n"
    )


def make_openfoam_sweep(
    root: Path, n_cases: int = 60, n_cells: int = 100, k_true: int = 3, seed: int = 0
) -> Path:
    """Write ``n_cases`` OpenFOAM cases of a low-rank pressure field + a DoE table.

    Each case parametrises the field by ``(u_in, aoa)`` and the resulting ``p``
    lies in a ``k_true``-dimensional subspace, so POD recovers it from ~k_true
    coefficients. Returns the cases directory.
    """
    rng = np.random.default_rng(seed)
    u_in = rng.uniform(1.0, 5.0, n_cases)
    aoa = rng.uniform(-10.0, 10.0, n_cases)  # angle of attack, degrees
    modes, _ = np.linalg.qr(rng.standard_normal((n_cells, k_true)))  # spatial modes
    # Three modal amplitudes, normalised to comparable energy then given a gentle
    # spectrum, so a 0.999 energy cut retains all `k_true` modes.
    amplitudes = np.column_stack([u_in, aoa**2, u_in * aoa])[:, :k_true]
    amplitudes /= amplitudes.std(axis=0)
    amplitudes *= 30.0 * np.array([1.0, 0.7, 0.5])[:k_true]  # pressure-scale, mild decay
    fields = amplitudes @ modes.T  # (n_cases, n_cells), intrinsic rank k_true

    cases = root / "cases"
    cases.mkdir(parents=True)
    for i in range(n_cases):
        case = cases / f"case_{i:04d}"
        (case / "constant").mkdir(parents=True)
        (case / "200").mkdir()  # a single converged time
        _write_transport_properties(case / "constant" / "transportProperties", nu=1.5e-5)
        _write_scalar_field(case / "200" / "p", fields[i], obj="p")

    names = [f"case_{i:04d}" for i in range(n_cases)]
    doe = pd.DataFrame({"case": names, "u_in": u_in, "aoa": aoa})
    doe.to_csv(cases / "samples.csv", index=False)
    return cases


def main(workdir: Path) -> None:
    cases = make_openfoam_sweep(workdir)
    print(f"Wrote {len(list(cases.glob('case_*')))} OpenFOAM cases under {cases.name}/")

    # Attach parameters per case: u_in/aoa from the DoE table, and nu recovered straight
    # from the case dictionary — showing parameters can come from the case itself.
    doe = pd.read_csv(cases / "samples.csv").set_index("case")

    def params_for(case_dir: Path) -> dict[str, float]:
        row = doe.loc[case_dir.name]
        nu = read_openfoam_dimensioned_scalar(
            case_dir / "constant" / "transportProperties", "nu"
        )
        return {"u_in": float(row["u_in"]), "aoa": float(row["aoa"]), "nu": nu}

    snapshots = read_openfoam_sweep(cases, field="p", params=params_for)
    field_cols = [c for c in snapshots.columns if c.startswith("p_")]
    csv = workdir / "snapshots.csv"
    snapshots.to_csv(csv, index=False)
    print(f"Ingested field 'p' as {len(field_cols)} DOFs across {len(snapshots)} snapshots")

    project = Project.init(workdir / "rom", name="openfoam-rom")
    project.add_dataset(csv, name="cfd")

    config = {
        "experiment": "openfoam-field-rom",
        "data": {
            "source": "cfd",
            "inputs": ["u_in", "aoa"],  # nu is constant here; add it if you sweep it
            "outputs": ["p_*"],  # the field as a column-group glob (Story R1.1)
            "split": {"test_fraction": 0.25, "seed": 1},
        },
        "model": {
            "method": "reduced-field",
            "reduction": {"basis": "pod", "energy": 0.999},
            "inner": {"method": "polynomial", "degree": 2},
        },
        "validation": {"cross_validation": 3, "metrics": ["rmse", "r2", "mae"]},
    }
    run = Trainer.from_dict(config).fit(project)

    test = run.metrics["test"]
    print(f"Run id:        {run.run_id}")
    print(f"Field width:   {run.metrics['n_outputs']} DOFs (intrinsic rank 3)")
    print(f"Held-out R²:   {test['r2']:.4f}   RMSE: {test['rmse']:.4g}")

    model = run.load_model()
    print(f"Modes kept:    {model.rank} (POD truncation at energy 0.999)")

    # Reconstruct the full field for the first few cases straight from parameters.
    sample = snapshots.head(3)
    pred = model.predict(sample[["u_in", "aoa"]].to_numpy())
    truth = sample[field_cols].to_numpy()
    rel = np.linalg.norm(pred - truth) / np.linalg.norm(truth)
    print(f"Reconstruction shape {pred.shape}, relative error {rel:.2e}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="surroforge-ex05-") as tmp:
        main(Path(tmp))
