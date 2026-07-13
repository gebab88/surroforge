"""Switch the neural-network backend with one config line — `mlp` <-> `torch-mlp`.

Both halves are Professional methods since the Wave-1 editions split: `mlp`
(scikit-learn) and `torch-mlp` (PyTorch) ship in the `surroforge_professional`
plugin — the PyTorch half additionally needs the `[torch]` extra
(`pip install 'surroforge_professional[torch]'`) — and require a license
(`SURROFORGE_PRO_LICENSE`). `torch-mlp` registers on the same
`surroforge.trainers` entry point. Because `torch-mlp` mirrors `mlp`'s
hyperparameter surface, moving between scikit-learn and PyTorch is a one-line
change — `model.method: mlp` -> `model.method: torch-mlp` — with the *same*
hyperparameter block. This example trains both on one dataset and compares them.

Run it:

    python examples/08_torch_mlp.py

The `mlp` half always runs (given the Professional license above). The
`torch-mlp` half additionally runs only if the `[torch]` extra is installed:

    pip install 'surroforge_professional[torch]'

Without it, the example trains `mlp` alone and prints how to enable the PyTorch
backend, so it still runs to completion (it is smoke-tested in CI, where PyTorch
is not installed). Both backends are deterministic (each carries its own
`random_state`) and ONNX-exportable; the reproduction and export flows are shown
in 03_reproduction.py and 06_mlp_surrogate.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from surroforge import Project, Trainer
from surroforge.plugins import available_trainers
from surroforge.services import compare_runs, predict_with_run

# The shared block — the ONLY thing that changes between backends is `method`.
NN_HYPERPARAMETERS = {
    "hidden_layer_sizes": [64, 64],
    "activation": "tanh",
    "solver": "lbfgs",
    "max_iter": 2000,
    "random_state": 0,
}

# True only when surroforge_professional[torch] is installed (its entry point is
# then registered). The example itself never imports torch — only the plugin does.
_HAS_TORCH = "torch-mlp" in available_trainers()


def make_nonlinear_samples(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """An oscillatory target both neural-network backends fit well."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.5, 1.5, n)
    x2 = rng.uniform(-1.5, 1.5, n)
    y = np.sin(3.0 * x1) + np.cos(3.0 * x2) + 0.4 * x1 * x2 + rng.normal(0, 0.02, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def main(workdir: Path) -> None:
    csv = workdir / "samples.csv"
    make_nonlinear_samples().to_csv(csv, index=False)
    project = Project.init(workdir / "switch", name="backend-switch")
    project.add_dataset(csv, name="wave")

    base_data = {
        "source": "wave",
        "inputs": ["x1", "x2"],
        "outputs": ["y"],
        "split": {"test_fraction": 0.25, "seed": 42},
    }
    validation = {"cross_validation": 5, "metrics": ["rmse", "r2", "mae"]}

    # Same hyperparameter block, two backends — the switch is the `method` field.
    candidates = [("sklearn (mlp)", {"method": "mlp", **NN_HYPERPARAMETERS})]
    if _HAS_TORCH:
        candidates.append(("pytorch (torch-mlp)", {"method": "torch-mlp", **NN_HYPERPARAMETERS}))

    runs = {}
    for experiment, model in candidates:
        runs[experiment] = Trainer.from_dict(
            {"experiment": experiment, "data": base_data, "model": model, "validation": validation}
        ).fit(project)

    if not _HAS_TORCH:
        print("(surroforge_professional[torch] not installed — running the scikit-learn")
        print("`mlp` backend only)")
        print("Enable the PyTorch backend with: pip install 'surroforge_professional[torch]'\n")

    # Both runs share the SAME hyperparameters; the method string is the only difference.
    label_by_id = {r.run_id: name for name, r in runs.items()}
    result = compare_runs(project, list(label_by_id), metric="rmse")
    print(f"{'backend':<22}{'method':<12}{'test R²':>10}{'test RMSE':>12}{'CV RMSE':>12}")
    for row in result["runs"]:
        marker = "  <- best" if row["run_id"] == result["best"] else ""
        print(
            f"{label_by_id[row['run_id']]:<22}{row['method']:<12}"
            f"{row['metrics']['r2']['test']:>10.4f}{row['metrics']['rmse']['test']:>12.4g}"
            f"{row['metrics']['rmse']['cv_mean']:>12.4g}{marker}"
        )

    if _HAS_TORCH:
        print("\nSame hidden_layer_sizes / activation / solver / random_state for both —")
        print("only `model.method` changed (mlp -> torch-mlp). Both fit the oscillatory target")
        print("to comparable held-out R² — the switch is transparent to the rest of the run.")
    else:
        print("\nInstall the plugin to see `torch-mlp` trained from the very same config block.")

    # Predict on fresh inputs with whichever backend won the comparison.
    best_run = next(r for r in runs.values() if r.run_id == result["best"])
    new_points = make_nonlinear_samples(n=5, seed=99).drop(columns=["y"])
    preds = predict_with_run(best_run, new_points)
    print(f"\nPredictions from the best run ({label_by_id[best_run.run_id]}) on 5 new points:")
    print(preds.to_string(index=False))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="surroforge-ex08-") as tmp:
        main(Path(tmp))
