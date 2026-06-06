"""Train several methods on the same data and pick the best.

`compare_runs` tabulates the held-out test and cross-validation metrics across
runs and selects the winner by a chosen metric (lower-is-better for error
metrics). This mirrors `surroforge compare <run-ids>` on the CLI.

Run it:

    python examples/02_method_comparison.py

Polynomial ships with Core, so the example always compares at least two
polynomial degrees. If the optional `smt` backend is installed, Kriging and RBF
join the comparison automatically.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from surroforge import Project, Trainer
from surroforge.services import compare_runs

_HAS_SMT = importlib.util.find_spec("smt") is not None


def make_beam_samples(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    length = rng.uniform(1.0, 3.0, n)
    width = rng.uniform(0.5, 1.5, n)
    load = rng.uniform(10.0, 50.0, n)
    max_stress = 2.5 * length**2 + 1.3 * width * load + 0.7 * load + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {"length": length, "width": width, "load": load, "max_stress": max_stress}
    )


def candidates() -> list[tuple[str, dict]]:
    """(experiment-name, model-block) pairs — the methods we put head to head."""
    methods: list[tuple[str, dict]] = [
        ("poly-deg1", {"method": "polynomial", "degree": 1}),
        ("poly-deg2", {"method": "polynomial", "degree": 2}),
    ]
    if _HAS_SMT:
        methods += [
            ("kriging", {"method": "kriging", "corr": "squar_exp"}),
            ("rbf", {"method": "rbf", "d0": 1.0}),
        ]
    return methods


def main(workdir: Path) -> None:
    csv = workdir / "samples.csv"
    make_beam_samples().to_csv(csv, index=False)
    project = Project.init(workdir / "compare", name="compare")
    project.add_dataset(csv, name="beam")

    base_data = {
        "source": "beam",
        "inputs": ["length", "width", "load"],
        "outputs": ["max_stress"],
        "split": {"test_fraction": 0.2, "seed": 42},
    }

    run_ids = []
    for experiment, model in candidates():
        run = Trainer.from_dict(
            {
                "experiment": experiment,
                "data": base_data,
                "model": model,
                "validation": {"cross_validation": 5, "metrics": ["rmse", "r2", "mae"]},
            }
        ).fit(project)
        run_ids.append(run.run_id)

    if not _HAS_SMT:
        print("(smt not installed — comparing polynomial degrees only)\n")

    result = compare_runs(project, run_ids, metric="rmse")
    print(f"{'run':<10}{'method':<14}{'test RMSE':>12}{'test R²':>10}{'CV RMSE':>12}")
    for row in result["rows"]:
        marker = "  <- best" if row["run_id"] == result["best"] else ""
        print(
            f"{row['run_id']:<10}{row['method']:<14}"
            f"{row['test']['rmse']:>12.4g}{row['test']['r2']:>10.4f}"
            f"{row['cv_mean']['rmse']:>12.4g}{marker}"
        )
    print(f"\nBest by {result['metric']}: {result['best']}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="surroforge-ex02-") as tmp:
        main(Path(tmp))
