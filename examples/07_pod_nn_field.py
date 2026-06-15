"""POD-NN — reduce a field with POD, then learn the modal coefficients with an MLP.

This is the field counterpart of `06_mlp_surrogate.py`. The field here is
*spatially* low-rank (POD collapses 64 DOFs to 3 modes), but each mode's
amplitude depends **nonlinearly** on the parameters (sin/cos). So the reduction
is easy, yet the per-mode coefficient map is the hard part — and that is exactly
where a neural-network inner surrogate beats a polynomial one.

It trains the same `reduced-field` reduction twice — once with `inner: mlp`,
once with `inner: polynomial` — and compares held-out field reconstruction. The
MLP inner wins, because a degree-2 polynomial cannot fit the oscillatory
coefficient curves. This is the spine theorem at work: `mlp` is an ordinary
`BaseTrainer`, so it drops into the unchanged `reduced-field` pipeline as an
inner with no special handling.

Run it:

    python examples/07_pod_nn_field.py

Core-only: both the POD reduction (numpy/scipy) and the MLP inner (scikit-learn)
ship with Core — no optional backend required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from surroforge import Project, Trainer
from surroforge.services import compare_runs


def make_nonlinear_field(n_samples: int = 200, n_dof: int = 64, seed: int = 0):
    """A rank-3 field whose modal amplitudes are *nonlinear* in the 2 parameters.

    The spatial structure is 3 fixed orthonormal modes (so POD recovers it with 3
    coefficients), but the amplitudes are sin/cos of the parameters — curves a
    degree-2 polynomial cannot represent, while an MLP can.
    """
    rng = np.random.default_rng(seed)
    params = rng.uniform(-1.0, 1.0, (n_samples, 2))
    modes, _ = np.linalg.qr(rng.standard_normal((n_dof, 3)))  # orthonormal spatial modes
    p0, p1 = params[:, 0], params[:, 1]
    amplitudes = np.column_stack(
        [np.sin(3.0 * p0), np.cos(3.0 * p1), p0 * np.sin(3.0 * p1)]
    ) * np.array([4.0, 2.0, 1.0])
    return params, amplitudes @ modes.T  # (n_samples, n_dof)


def main(workdir: Path) -> None:
    n_dof = 64
    params, field = make_nonlinear_field(n_dof=n_dof)

    # Lay the field out as a column group f000..f063 alongside the 2 inputs.
    columns = {"p0": params[:, 0], "p1": params[:, 1]}
    for j in range(n_dof):
        columns[f"f{j:03d}"] = field[:, j]
    csv = workdir / "snapshots.csv"
    pd.DataFrame(columns).to_csv(csv, index=False)

    project = Project.init(workdir / "pod-nn", name="pod-nn")
    project.add_dataset(csv, name="snaps")

    base_data = {
        "source": "snaps",
        "inputs": ["p0", "p1"],
        "outputs": [f"f{j:03d}" for j in range(n_dof)],  # the field is a column group
        "split": {"test_fraction": 0.25, "seed": 1},
    }
    reduction = {"basis": "pod", "energy": 0.999}  # same POD truncation for both

    # Same reduction, two inner surrogates: an MLP vs. a degree-2 polynomial.
    # tanh suits the smooth sinusoidal coefficient curves and converges cleanly with
    # lbfgs even on the small per-mode CV-fold subsets; max_iter is generous so no fold
    # trips a convergence warning.
    candidates = [
        (
            "pod-mlp",
            {
                "method": "mlp",
                "hidden_layer_sizes": [32, 32],
                "activation": "tanh",
                "max_iter": 5000,
            },
        ),
        ("pod-poly", {"method": "polynomial", "degree": 2}),
    ]
    runs = {}
    for experiment, inner in candidates:
        runs[experiment] = Trainer.from_dict(
            {
                "experiment": experiment,
                "data": base_data,
                "model": {"method": "reduced-field", "reduction": reduction, "inner": inner},
                "validation": {"cross_validation": 3, "metrics": ["rmse", "r2", "mae"]},
            }
        ).fit(project)

    # Both runs share method "reduced-field"; label each by its inner surrogate instead.
    inner_by_id = {runs["pod-mlp"].run_id: "mlp", runs["pod-poly"].run_id: "polynomial"}
    result = compare_runs(project, list(inner_by_id), metric="rmse")
    print(f"Field width: {runs['pod-mlp'].metrics['n_outputs']} DOFs, intrinsic rank 3\n")
    print(f"{'inner':<12}{'test R²':>10}{'test RMSE':>12}")
    for row in result["rows"]:
        marker = "  <- best" if row["run_id"] == result["best"] else ""
        print(
            f"{inner_by_id[row['run_id']]:<12}{row['test']['r2']:>10.4f}"
            f"{row['test']['rmse']:>12.4g}{marker}"
        )

    mlp_run = runs["pod-mlp"]
    poly_r2 = runs["pod-poly"].metrics["test"]["r2"]
    print(
        f"\nPOD reduces the field to {mlp_run.load_model().rank} modes for both. "
        f"The MLP inner reaches test R² {mlp_run.metrics['test']['r2']:.4f}"
    )
    print(
        f"vs. the polynomial inner's {poly_r2:.4f} — the network captures the oscillatory"
    )
    print("coefficient curves the quadratic inner cannot.")

    # The persisted POD-NN model reloads and reconstructs the full field.
    model = mlp_run.load_model()
    recon = model.predict(params[:3])
    rel = np.linalg.norm(recon - field[:3]) / np.linalg.norm(field[:3])
    print(f"\nReconstruction shape {recon.shape}, relative error {rel:.2e}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="surroforge-ex07-") as tmp:
        main(Path(tmp))
