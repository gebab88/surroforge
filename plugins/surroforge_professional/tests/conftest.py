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


def blas_is_bit_reproducible() -> bool:
    """Whether numpy's BLAS/LAPACK is bit-reproducible run-to-run.

    Apple's Accelerate (vecLib) is not: its LAPACK shows ULP-level run-to-run
    variation, which is harmless for closed-form methods but pushes gradient-MLE
    Kriging to a materially different optimum on a *flat* likelihood (smooth
    targets drive the length-scales to the bound, so the optimum is barely
    identified) — diverging by anything from ~1e-5 up to a few percent. So
    Kriging's declared 1e-9 reproduction tolerance holds on OpenBLAS (Linux CI)
    but not on macOS/Accelerate. This returns ``False`` only for the known
    non-reproducible backend; every other BLAS keeps the guarantee asserted.
    """
    try:
        deps = np.__config__.show(mode="dicts").get("Build Dependencies", {})
        name = str(deps.get("blas", {}).get("name", "")).lower()
    except Exception:
        return True  # unknown backend: keep the bit-reproducibility assertions sharp
    return "accelerate" not in name


#: Skip a test that asserts bit-level reproducibility when the BLAS cannot deliver
#: it (Apple Accelerate). The guarantee stays asserted on OpenBLAS / CI.
requires_reproducible_blas = pytest.mark.skipif(
    not blas_is_bit_reproducible(),
    reason="bit-level reproduction needs a bit-reproducible BLAS; Apple Accelerate "
    "(vecLib) is not, so this holds on OpenBLAS/CI but not on macOS/Accelerate",
)


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
def second_run(project):
    config = parse_config(
        {
            "experiment": "beam-professional-html-2",
            "data": {
                "source": "beam",
                "inputs": ["length", "width", "load"],
                "outputs": ["max_stress"],
                "split": {"test_fraction": 0.2, "seed": 42},
            },
            "model": {"method": "polynomial", "degree": 1},
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
