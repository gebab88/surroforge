"""Neural-network surrogate — fit a strongly nonlinear response with `mlp`.

The `mlp` method is a multi-layer perceptron (scikit-learn's `MLPRegressor`, with
an input `StandardScaler` and the small-data-friendly `lbfgs` solver). It is a
genuine neural network but rides Core scikit-learn, so it needs no deep-learning
framework and no extra backend.

This example puts it head to head with a degree-2 polynomial on an oscillatory
target the polynomial cannot capture: the MLP wins decisively on held-out R².
It then exports the trained network to ONNX — `mlp` is the second ONNX-capable
method after `polynomial` (the rest are pickle-only).

Run it:

    python examples/06_mlp_surrogate.py

Self-contained: synthesises its own data in a temp dir and cleans up. Core-only
(scikit-learn). ONNX export uses the optional `[onnx]` extra and degrades
gracefully to pickle when it is absent.

Two notes the script doesn't belabour:
  * Being an ordinary trainer, `mlp` also serves as a `reduced-field` inner
    surrogate (POD-NN) — set `model.inner.method: mlp` (see 04_field_rom.py).
  * Like every iterative method it reproduces within its declared tolerance
    rather than bit-for-bit; the reproducibility flow is shown in 03_reproduction.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from surroforge import Project, Trainer
from surroforge.errors import ExportError
from surroforge.services import compare_runs, export_model, predict_with_run


def make_nonlinear_samples(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """An oscillatory target: a degree-2 polynomial can't fit sin/cos, an MLP can."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.5, 1.5, n)
    x2 = rng.uniform(-1.5, 1.5, n)
    y = np.sin(3.0 * x1) + np.cos(3.0 * x2) + 0.4 * x1 * x2 + rng.normal(0, 0.02, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def main(workdir: Path) -> None:
    # 1. Data + project + dataset (as in the quickstart).
    csv = workdir / "samples.csv"
    make_nonlinear_samples().to_csv(csv, index=False)
    project = Project.init(workdir / "mlp-study", name="mlp-study")
    project.add_dataset(csv, name="wave")

    base_data = {
        "source": "wave",
        "inputs": ["x1", "x2"],
        "outputs": ["y"],
        "split": {"test_fraction": 0.25, "seed": 42},
    }
    validation = {"cross_validation": 5, "metrics": ["rmse", "r2", "mae"]}

    # 2. Train the MLP and a degree-2 polynomial on the same data.
    candidates = [
        ("mlp", {"method": "mlp", "hidden_layer_sizes": [64, 64]}),
        ("poly-deg2", {"method": "polynomial", "degree": 2}),
    ]
    runs = {}
    for experiment, model in candidates:
        runs[experiment] = Trainer.from_dict(
            {"experiment": experiment, "data": base_data, "model": model, "validation": validation}
        ).fit(project)

    # 3. Compare. The MLP should win on the oscillatory target.
    result = compare_runs(project, [r.run_id for r in runs.values()], metric="rmse")
    print(f"{'method':<12}{'test RMSE':>12}{'test R²':>10}{'CV RMSE':>12}")
    for row in result["runs"]:
        marker = "  <- best" if row["run_id"] == result["best"] else ""
        print(
            f"{row['method']:<12}{row['metrics']['rmse']['test']:>12.4g}"
            f"{row['metrics']['r2']['test']:>10.4f}{row['metrics']['rmse']['cv_mean']:>12.4g}{marker}"
        )

    mlp_run = runs["mlp"]
    mlp_r2 = mlp_run.metrics["test"]["r2"]
    poly_r2 = runs["poly-deg2"].metrics["test"]["r2"]
    print(f"\nMLP test R² {mlp_r2:.4f} vs polynomial {poly_r2:.4f} — the network captures the")
    print("oscillation the quadratic surface cannot.")

    # 4. Predict on fresh inputs with the trained network.
    new_points = make_nonlinear_samples(n=5, seed=99).drop(columns=["y"])
    preds = predict_with_run(mlp_run, new_points)
    print("\nMLP predictions on 5 new points:")
    print(preds.to_string(index=False))

    # 5. Export the network to ONNX (mlp is ONNX-capable); fall back to pickle if
    #    the optional [onnx] extra is not installed.
    try:
        model_path = export_model(mlp_run, "onnx")
        print(f"\nExported MLP -> {model_path.name} (ONNX)")
    except ExportError as exc:
        model_path = export_model(mlp_run, "pickle")
        print(f"\nONNX export unavailable ({exc}); exported pickle -> {model_path.name}")
        print("Install the extra for ONNX: pip install 'surroforge[onnx]'")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="surroforge-ex06-") as tmp:
        main(Path(tmp))
