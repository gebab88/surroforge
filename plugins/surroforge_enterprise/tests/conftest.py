from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SFLICENSE = Path(__file__).resolve().parents[3] / "tools" / "licensing" / "sflicense.py"


def _load_sflicense():
    spec = importlib.util.spec_from_file_location("sflicense", _SFLICENSE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def license_kit():
    """Throwaway signing keypair + mint helper (M3) — mirrors the professional conftest.

    Duplicated because each plugin suite is its own pytest rootdir and cannot
    import a sibling suite's conftest.
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


@pytest.fixture
def licensed_enterprise(monkeypatch, tmp_path, license_kit):
    from surroforge_professional import license as _license

    monkeypatch.setattr(_license, "_TRUSTED_KEYS", (license_kit.pub_hex,))
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", license_kit.mint(tier="enterprise"))
    monkeypatch.setenv("HOME", str(tmp_path))
