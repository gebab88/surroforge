"""Reproduce a run and verify it — the reproducibility guarantee.

Every run records the inputs that determine its result: a hash of the config, a
hash of the data, the split seed, and the package versions. `reproduce` re-runs
the training from that record and compares a fingerprint of the out-of-sample
holdout predictions against the original. A deterministic method (closed-form
OLS) reproduces *bit-identically*; an iterative one reproduces *within its
declared tolerance*. This mirrors `surroforge reproduce <run-id>`.

Run it:

    python examples/03_reproduction.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from surroforge import Project, Trainer
from surroforge.errors import ReproducibilityError
from surroforge.services import assert_reproducible, reproduce


def make_beam_samples(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    length = rng.uniform(1.0, 3.0, n)
    width = rng.uniform(0.5, 1.5, n)
    load = rng.uniform(10.0, 50.0, n)
    max_stress = 2.5 * length**2 + 1.3 * width * load + 0.7 * load + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {"length": length, "width": width, "load": load, "max_stress": max_stress}
    )


def main(workdir: Path) -> None:
    csv = workdir / "samples.csv"
    make_beam_samples().to_csv(csv, index=False)
    project = Project.init(workdir / "study", name="reproduce")
    project.add_dataset(csv, name="beam")

    config = {
        "experiment": "beam-polynomial",
        "data": {
            "source": "beam",
            "inputs": ["length", "width", "load"],
            "outputs": ["max_stress"],
            "split": {"test_fraction": 0.2, "seed": 42},
        },
        "model": {"method": "polynomial", "degree": 2},
        "validation": {"cross_validation": 5, "metrics": ["rmse", "r2", "mae"]},
    }
    original = Trainer.from_dict(config).fit(project)
    print(f"Original run: {original.run_id}")
    print(f"  config hash: {original.config_hash[:16]}…")
    print(f"  data hash:   {original.data_hash[:16]}…")
    print(f"  declared reproduction tolerance: {original.lineage['reproduction_tolerance']}")

    # Re-run from the recorded lineage and compare.
    result = reproduce(project, original.run_id)
    print(f"\nVerdict:        {result.verdict}")
    print(f"bit-identical:  {result.bit_identical}")
    print(f"max abs diff:   {result.max_abs_diff:.2e}")
    print(f"data unchanged: {result.data_hash_match}")
    assert_reproducible(result)  # raises ReproducibilityError if it diverged
    print("\nassert_reproducible passed — the run is verified reproducible.")

    # The teeth: editing the underlying data is detected as a changed data hash.
    # (Here the dataset was copied into the project, so we show what a divergence
    # verdict looks like by asserting a tampered result is rejected.)
    from surroforge.services import ReproduceResult

    tampered = ReproduceResult(
        parent=None, reproduction=None, bit_identical=False,
        max_abs_diff=1.0, data_hash_match=True, verdict="diverged", tolerance=0.0,
    )
    try:
        assert_reproducible(tampered)
    except ReproducibilityError as exc:
        print(f"A diverged result is correctly rejected: {type(exc).__name__}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="surroforge-ex03-") as tmp:
        main(Path(tmp))
