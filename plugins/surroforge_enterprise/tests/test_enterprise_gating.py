"""autoencoder-field is an *enterprise*-tier method: a professional licence is not enough."""

from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.plugins import get_trainer


def test_professional_license_does_not_unlock_autoencoder(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "local-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(PluginError, match="'enterprise' license"):
        get_trainer("autoencoder-field")


def test_enterprise_license_unlocks_autoencoder(licensed_enterprise):
    cls = get_trainer("autoencoder-field")
    assert cls.name == "autoencoder-field"
    assert cls.required_tier == "enterprise"


def test_no_license_denies_autoencoder(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(PluginError, match="'enterprise' license"):
        get_trainer("autoencoder-field")
