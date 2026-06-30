from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import run_training


@pytest.fixture
def licensed(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "local-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))


def make_samples(n: int = 80, seed: int = 0) -> pd.DataFrame:
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
    proj = Project.init(tmp_path / "proj", name="professional-test")
    proj.add_dataset(sample_csv, name="beam")
    return proj


@pytest.fixture
def trained_run(project):
    config = parse_config(
        {
            "experiment": "beam-professional-html",
            "data": {
                "source": "beam",
                "inputs": ["length", "width", "load"],
                "outputs": ["max_stress"],
                "split": {"test_fraction": 0.2, "seed": 42},
            },
            "model": {"method": "polynomial", "degree": 2},
            "validation": {"cross_validation": 5, "metrics": ["rmse", "r2", "mae"]},
        }
    )
    return run_training(project, config)


@pytest.fixture
def multi_sample_csv(tmp_path):
    rng = np.random.default_rng(1)
    n = 80
    a = rng.uniform(0.0, 1.0, n)
    b = rng.uniform(0.0, 1.0, n)
    df = pd.DataFrame({"a": a, "b": b, "y1": 2 * a + b, "y2": a - 3 * b + 0.5})
    path = tmp_path / "multi.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def multi_run(tmp_path, multi_sample_csv):
    proj = Project.init(tmp_path / "mproj", name="multi-professional")
    proj.add_dataset(multi_sample_csv, name="multi")
    config = parse_config(
        {
            "experiment": "multi-output",
            "data": {
                "source": "multi",
                "inputs": ["a", "b"],
                "outputs": ["y1", "y2"],
                "split": {"test_fraction": 0.25, "seed": 7},
            },
            "model": {"method": "polynomial", "degree": 1},
            "validation": {"cross_validation": 4, "metrics": ["rmse", "r2"]},
        }
    )
    return run_training(proj, config)
