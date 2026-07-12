"""Resolution-level gating of the Professional trainers (Wave 1)."""

from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.plugins import available_exporters, available_trainers, get_exporter, get_trainer


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


# --- Exporters (Wave 2) --------------------------------------------------------


@pytest.mark.parametrize("fmt", ["pickle", "onnx", "pt"])
def test_exporter_denied_without_license(unlicensed, fmt):
    with pytest.raises(PluginError) as err:
        get_exporter(fmt)
    assert "'professional' license" in str(err.value)
    assert "SURROFORGE_PRO_LICENSE" in str(err.value)  # hint from the authorizer


@pytest.mark.parametrize("fmt", ["pickle", "onnx", "pt"])
def test_exporter_resolves_with_license(licensed, fmt):
    cls = get_exporter(fmt)
    assert cls.name == fmt
    assert cls.required_tier == "professional"


def test_exporters_listed_via_entry_points(licensed):
    assert {"pickle", "onnx", "pt"} <= set(available_exporters())


# --- Feature gate (Wave 3) -----------------------------------------------------


def test_reproduce_works_with_real_license(licensed, project, trained_run):
    from surroforge.services import assert_reproducible, reproduce

    result = reproduce(project, trained_run.run_id)
    assert_reproducible(result)  # must not raise


def test_reproduce_denied_by_real_authorizer_names_hint(unlicensed, project, trained_run):
    from surroforge.services import reproduce

    with pytest.raises(PluginError) as err:
        reproduce(project, trained_run.run_id)
    assert "'reproduce' requires a 'professional' license" in str(err.value)
    assert "SURROFORGE_PRO_LICENSE" in str(err.value)  # hint from the real authorizer
