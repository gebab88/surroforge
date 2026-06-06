"""Field ROM — reduce a high-dimensional field, then surrogate the coefficients.

Many simulations output a *field* (a value per mesh node / cell), not a single
scalar. The `reduced-field` method finds a low-dimensional POD basis from the
training snapshots, fits one inner surrogate per retained mode, and reconstructs
the full field on predict. Because it is an ordinary trainer that emits an
(n_samples, n_dof) array, it flows through the *unchanged* engine — split,
metrics, persistence, and reproduction all work without field-specific plumbing.

Run it:

    python examples/04_field_rom.py

The default inner surrogate is polynomial, so this needs no optional backends.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from surroforge import Project, Trainer


def make_low_rank_field(n_samples: int = 80, n_dof: int = 64, k_true: int = 3, seed: int = 0):
    """A field of intrinsic rank ``k_true``: smooth amplitudes × fixed spatial modes.

    Returns the 2 input parameters and the (n_samples, n_dof) field. The field
    lives on ``n_dof`` spatial locations but only varies along ``k_true`` modes,
    so POD should recover it with ~k_true coefficients.
    """
    rng = np.random.default_rng(seed)
    params = rng.uniform(-1.0, 1.0, (n_samples, 2))
    modes, _ = np.linalg.qr(rng.standard_normal((n_dof, k_true)))  # orthonormal spatial modes
    amplitudes = np.column_stack(
        [params[:, 0], params[:, 1] ** 2, params[:, 0] * params[:, 1]]
    )[:, :k_true] * np.array([5.0, 2.0, 1.0])[:k_true]
    return params, amplitudes @ modes.T


def main(workdir: Path) -> None:
    n_dof = 64
    params, field = make_low_rank_field(n_dof=n_dof)

    # Lay the field out as a column group f000..f063 alongside the inputs.
    columns = {"p0": params[:, 0], "p1": params[:, 1]}
    for j in range(n_dof):
        columns[f"f{j:03d}"] = field[:, j]
    csv = workdir / "snapshots.csv"
    pd.DataFrame(columns).to_csv(csv, index=False)

    project = Project.init(workdir / "rom", name="rom")
    project.add_dataset(csv, name="snaps")

    config = {
        "experiment": "field-rom",
        "data": {
            "source": "snaps",
            "inputs": ["p0", "p1"],
            "outputs": [f"f{j:03d}" for j in range(n_dof)],  # the field is a column group
            "split": {"test_fraction": 0.25, "seed": 1},
        },
        "model": {
            "method": "reduced-field",
            "reduction": {"basis": "pod", "energy": 0.999},  # keep 99.9% of the energy
            "inner": {"method": "polynomial", "degree": 2},   # one per retained mode
        },
        "validation": {"cross_validation": 3, "metrics": ["rmse", "r2", "mae"]},
    }
    run = Trainer.from_dict(config).fit(project)

    test = run.metrics["test"]
    print(f"Run id:        {run.run_id}")
    print(f"Field width:   {run.metrics['n_outputs']} DOFs (intrinsic rank 3)")
    print(f"Held-out R²:   {test['r2']:.4f}   RMSE: {test['rmse']:.4g}")

    # The persisted reduced-field model reloads and reconstructs the full field.
    model = run.load_model()
    print(f"Modes kept:    {model.rank} (POD truncation at energy 0.999)")

    pred = model.predict(params[:3])
    rel = np.linalg.norm(pred - field[:3]) / np.linalg.norm(field[:3])
    print(f"Reconstruction shape {pred.shape}, relative error {rel:.2e}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="surroforge-ex04-") as tmp:
        main(Path(tmp))
