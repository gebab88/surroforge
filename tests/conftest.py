"""Shared fixtures: a sample dataset and an initialised project."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from surroforge.core.project import Project

_SFLICENSE = Path(__file__).resolve().parents[1] / "tools" / "licensing" / "sflicense.py"


def _load_sflicense():
    spec = importlib.util.spec_from_file_location("sflicense", _SFLICENSE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def license_kit():
    """A throwaway signing keypair + mint helper for real-validation tests (M3).

    The public key is NOT trusted by the shipped package; grant fixtures
    monkeypatch it into ``license._TRUSTED_KEYS``.
    """
    pytest.importorskip("cryptography")
    sfl = _load_sflicense()
    seed_hex, pub_hex = sfl.generate_keypair()

    def mint(tier: str = "professional", **overrides) -> str:
        payload = {
            "v": 1,
            "id": "SF-TEST-LOCAL",
            "holder": "test-suite",
            "tier": tier,
            "issued": "2026-07-13",
            **overrides,
        }
        return sfl.mint(payload, seed_hex)

    return SimpleNamespace(mint=mint, pub_hex=pub_hex, seed_hex=seed_hex)


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


@pytest.fixture
def licensed_export(monkeypatch, license_kit):
    """Grant a real (test-signed) Professional licence for export tests (W2→M3).

    Skips cleanly on a Core-only checkout: ``license_kit`` skips without
    ``cryptography`` (a professional dependency), and without
    ``surroforge_professional`` there is no exporter to license.
    """
    pytest.importorskip("surroforge_professional")
    from surroforge_professional import license as _license

    monkeypatch.setattr(_license, "_TRUSTED_KEYS", (license_kit.pub_hex,), raising=False)
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", license_kit.mint())


@pytest.fixture
def granted_license(monkeypatch):
    """Bypass the Wave-3 feature gate with a fake always-yes authorizer.

    For tests of Core functionality that sits BEHIND the gate (reproduce math,
    field reproduce, lineage panels) — works on a Core-only checkout, unlike
    ``licensed_export`` (which needs the real professional authorizer). Never
    use it in tests that assert UNLICENSED behavior: the fake ignores env/HOME.
    """
    from surroforge.plugins import gate

    monkeypatch.setattr(gate, "_load_authorizers", lambda: ((lambda tier: True),))


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
