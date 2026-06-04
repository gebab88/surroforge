"""Shared fixtures: a sample dataset and an initialised project."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surroforge.core.project import Project


def make_samples(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """A smooth, low-noise function that surrogate models fit well."""
    rng = np.random.default_rng(seed)
    length = rng.uniform(1.0, 3.0, n)
    width = rng.uniform(0.5, 1.5, n)
    load = rng.uniform(10.0, 50.0, n)
    max_stress = 2.5 * length**2 + 1.3 * width * load + 0.7 * load + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {"length": length, "width": width, "load": load, "max_stress": max_stress}
    )


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "samples.csv"
    make_samples().to_csv(path, index=False)
    return path


@pytest.fixture
def project(tmp_path, sample_csv):
    proj = Project.init(tmp_path / "proj", name="test-proj")
    proj.add_dataset(sample_csv, name="beam")
    return proj


@pytest.fixture
def trained_run(project):
    """A persisted polynomial run, ready for exporter/reporter tests."""
    from surroforge.core.config import parse_config
    from surroforge.engine import run_training

    return run_training(project, parse_config(poly_config("polynomial")))


def poly_config(method: str = "polynomial") -> dict:
    """A minimal valid config dict for the registered 'beam' dataset."""
    model: dict = {"method": method}
    if method == "polynomial":
        model["degree"] = 2
    return {
        "experiment": f"beam-{method}",
        "data": {
            "source": "beam",
            "inputs": ["length", "width", "load"],
            "outputs": ["max_stress"],
            "split": {"test_fraction": 0.2, "seed": 42},
        },
        "model": model,
        "validation": {"cross_validation": 5, "metrics": ["rmse", "r2", "mae"]},
    }
