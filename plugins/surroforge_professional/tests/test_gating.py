"""Resolution-level gating of the Professional trainers (Wave 1)."""

from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.plugins import available_trainers, get_trainer


@pytest.fixture
def unlicensed(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.mark.parametrize("method", ["mlp", "random-forest", "gradient-boosting"])
def test_pro_method_denied_without_license(unlicensed, method):
    with pytest.raises(PluginError) as err:
        get_trainer(method)
    assert "'professional' license" in str(err.value)
    assert "SURROFORGE_PRO_LICENSE" in str(err.value)  # hint from the authorizer


@pytest.mark.parametrize("method", ["mlp", "random-forest", "gradient-boosting"])
def test_pro_method_resolves_with_license(licensed, method):
    cls = get_trainer(method)
    assert cls.name == method
    assert cls.required_tier == "professional"


def test_pro_methods_listed_via_entry_points(licensed):
    assert {"mlp", "random-forest", "gradient-boosting"} <= set(available_trainers())
