from __future__ import annotations

import pytest


@pytest.fixture
def licensed_enterprise(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "enterprise-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))
