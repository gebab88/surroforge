"""Greenfield, end to end — the Python-API twin of the CLI quickstart.

From a raw data table to a *validated, reproducible* surrogate model,
driven entirely from Python (`Project` + `Trainer` + the `services` helpers). The
CLI walkthrough in docs/quickstart.md does the same steps from the shell.

Run it:

    python examples/01_quickstart.py

It is self-contained: it synthesises its own data in a temporary directory and
cleans up on exit, so it leaves nothing behind and needs no optional backends
(the polynomial method ships with Core).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from surroforge import Project, Trainer
from surroforge.errors import PluginError
from surroforge.services import (
    assert_reproducible,
    generate_report,
    predict_with_run,
    reproduce,
)


def make_beam_samples(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """A smooth, low-noise degree-2 target a polynomial surrogate fits well."""
    rng = np.random.default_rng(seed)
    length = rng.uniform(1.0, 3.0, n)
    width = rng.uniform(0.5, 1.5, n)
    load = rng.uniform(10.0, 50.0, n)
    max_stress = 2.5 * length**2 + 1.3 * width * load + 0.7 * load + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {"length": length, "width": width, "load": load, "max_stress": max_stress}
    )


def main(workdir: Path) -> None:
    # 1. Some data: each row is one simulation; columns are inputs and outputs.
    csv = workdir / "samples.csv"
    make_beam_samples().to_csv(csv, index=False)

    # 2. A project: an on-disk workspace for data, configs, runs, and reports.
    project = Project.init(workdir / "beam-study", name="beam-study")

    # 3. Register the dataset (validates the schema, hashes the content, copies
    #    the file into the project so it is self-contained).
    project.add_dataset(csv, name="beam")

    # 4. Describe the run. The same dict round-trips to/from YAML for the CLI.
    config = {
        "experiment": "beam-polynomial",
        "data": {
            "source": "beam",  # a registered dataset name, or a file path
            "inputs": ["length", "width", "load"],
            "outputs": ["max_stress"],
            "split": {"test_fraction": 0.2, "seed": 42},
        },
        "model": {"method": "polynomial", "degree": 2},
        "validation": {"cross_validation": 5, "metrics": ["rmse", "r2", "mae"]},
    }

    # 5. Train. `fit` resolves the dataset from the config's data.source.
    run = Trainer.from_dict(config).fit(project)
    test = run.metrics["test"]
    cv = run.metrics["cross_validation"]
    print(f"Run id:   {run.run_id}")
    print(f"Test R²:  {test['r2']:.4f}   RMSE: {test['rmse']:.4g}   MAE: {test['mae']:.4g}")
    print(f"5-fold CV RMSE: {cv['rmse']['mean']:.4g} ± {cv['rmse']['std']:.2g}")

    # 6. Predict on fresh inputs (same input columns, no target column needed).
    new_points = make_beam_samples(n=5, seed=99).drop(columns=["max_stress"])
    preds = predict_with_run(run, new_points)
    print("\nPredictions on 5 new points:")
    print(preds.to_string(index=False))

    # 7. The fitted model is already persisted in the run store — predict works
    #    from it directly. Standalone export (pickle/ONNX via `export_model` or
    #    `surroforge export`) is a Professional feature since the editions split.
    print(f"\nStored model artifact -> {run.model_path.name}")

    # 8. Report (Markdown, under the project's reports/ directory).
    report_path, _text = generate_report(project, run)
    print(f"Wrote report  -> {report_path.name}")

    # 9. Reproduce: because the run recorded its config, data hash, seed, and
    #    versions, re-running it must land within the method's declared tolerance.
    #    Verifying is a Professional feature (the recording happens on every run),
    #    so this last step degrades to a note when no licence is installed.
    try:
        result = reproduce(project, run.run_id)
    except PluginError as exc:
        print(f"\nReproduction skipped: {exc}")
    else:
        assert_reproducible(result)  # raises if it diverges beyond tolerance
        kind = "bit-identical" if result.bit_identical else f"within {result.max_abs_diff:.1e}"
        print(f"\nReproduction verdict: {result.verdict} ({kind})")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="surroforge-ex01-") as tmp:
        main(Path(tmp))
