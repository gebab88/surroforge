"""Plugin registry (PRD 4.11)."""

from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.plugins import (
    available_reporters,
    available_trainers,
    get_comparison_reporter,
    get_reporter,
    get_trainer,
)
from surroforge.trainers.base import BaseTrainer


def test_builtin_trainers_registered():
    assert set(available_trainers()) >= {"kriging", "rbf", "polynomial"}


def test_professional_trainers_not_in_core_builtin_table():
    # mlp/random-forest/gradient-boosting moved to surroforge_professional (Wave 1);
    # Core's _BUILTIN fallback table must not resurrect them for a source checkout.
    from surroforge.plugins.registry import _BUILTIN, TRAINER_GROUP

    assert not {"mlp", "random-forest", "gradient-boosting"} & set(_BUILTIN[TRAINER_GROUP])


def test_get_trainer_returns_subclass():
    cls = get_trainer("polynomial")
    assert issubclass(cls, BaseTrainer)


def test_unknown_trainer_raises():
    with pytest.raises(PluginError):
        get_trainer("does-not-exist")


def test_reporters_registered():
    assert "markdown" in available_reporters()
    assert get_reporter("markdown").name == "markdown"


def test_exporters_not_in_core_builtin_table():
    # pickle/onnx moved to surroforge_professional (Wave 2); Core's fallback table
    # must not resurrect them for a source checkout.
    from surroforge.plugins.registry import _BUILTIN, EXPORTER_GROUP

    assert _BUILTIN[EXPORTER_GROUP] == {}


def test_core_only_install_has_no_exporters(monkeypatch):
    # Without any edition installed there is no export format at all — and the
    # error is the editions-neutral "Unknown export format", not a licence pitch.
    from surroforge.plugins import registry

    monkeypatch.setattr(registry, "_entry_points_grouped", lambda group: {})
    assert registry.available_exporters() == []
    with pytest.raises(PluginError, match="Unknown export format"):
        registry.get_exporter("pickle")


def test_entry_point_scan_is_memoised():
    # The distribution scan is the dominant fixed cost of resolution; it must be
    # cached for the process (entry points cannot change at runtime).
    from surroforge.plugins import registry

    registry._scan_entry_points.cache_clear()
    first = registry._scan_entry_points(registry.TRAINER_GROUP)
    second = registry._scan_entry_points(registry.TRAINER_GROUP)
    assert first is second  # identical object => no re-scan
    registry._scan_entry_points.cache_clear()
    assert registry._scan_entry_points(registry.TRAINER_GROUP) is not first


def test_unknown_comparison_reporter_raises():
    with pytest.raises(PluginError) as err:
        get_comparison_reporter("nope")
    assert "comparison report format" in str(err.value)
    assert "nope" in str(err.value)


def test_comparison_reporter_resolves_from_entry_point(monkeypatch):
    from surroforge.plugins import registry

    class FakeComparisonReporter:
        name = "fake-html"
        extension = ".html"

    class FakeEP:
        name = "fake-html"
        value = "fake.module:FakeComparisonReporter"

        def load(self):
            return FakeComparisonReporter

    def fake_grouped(group):
        if group == registry.COMPARISON_REPORTER_GROUP:
            return {"fake-html": [FakeEP()]}
        return {}

    monkeypatch.setattr(registry, "_entry_points_grouped", fake_grouped)
    assert registry.get_comparison_reporter("fake-html") is FakeComparisonReporter
    assert "fake-html" in registry.available_comparison_reporters()


# --- required_tier seam (Wave 1) ---------------------------------------------


def test_required_tier_defaults_to_none():
    from surroforge.plugins.seam import resolve_required_tier

    class FreeTrainer:
        name = "free"

    assert resolve_required_tier(FreeTrainer) is None


def test_required_tier_reads_class_attribute():
    from surroforge.plugins.seam import resolve_required_tier

    class ProTrainer:
        name = "pro"
        required_tier = "professional"

    assert resolve_required_tier(ProTrainer) == "professional"


def test_required_tier_rejects_non_string():
    from surroforge.plugins.seam import resolve_required_tier

    class Broken:
        name = "broken"
        required_tier = 1

    with pytest.raises(PluginError, match="required_tier"):
        resolve_required_tier(Broken)


def test_required_tier_rejects_empty_string():
    from surroforge.plugins.seam import resolve_required_tier

    class Broken:
        name = "broken"
        required_tier = ""

    with pytest.raises(PluginError, match="required_tier"):
        resolve_required_tier(Broken)


def test_plugin_metadata_has_required_tier_default_none():
    from surroforge.plugins.seam import PluginMetadata

    meta = PluginMetadata(
        name="x", group="surroforge.trainers", api_version=1,
        source="builtin", ref="m:C",
    )
    assert meta.required_tier is None


def test_resolution_gates_tier_declaring_entry_point(monkeypatch):
    from surroforge.plugins import gate, registry

    class ProTrainer:
        name = "fake-pro"
        required_tier = "professional"

    class FakeEP:
        name = "fake-pro"
        value = "fake.module:ProTrainer"

        def load(self):
            return ProTrainer

    def fake_grouped(group):
        if group == registry.TRAINER_GROUP:
            return {"fake-pro": [FakeEP()]}
        return {}

    monkeypatch.setattr(registry, "_entry_points_grouped", fake_grouped)

    # Denied: no authorizer installed.
    monkeypatch.setattr(gate, "_load_authorizers", lambda: ())
    with pytest.raises(PluginError, match="'professional' license"):
        registry.get_trainer("fake-pro")

    # Granted: an authorizer approves the tier.
    monkeypatch.setattr(gate, "_load_authorizers", lambda: (lambda tier: True,))
    assert registry.get_trainer("fake-pro") is ProTrainer


def test_core_builtins_resolve_without_any_authorizer(monkeypatch):
    # Spec: a Core-only installation (no authorizer at all) must never be denied.
    from surroforge.plugins import gate, registry

    monkeypatch.setattr(gate, "_load_authorizers", lambda: ())
    # polynomial/reduced-field suffice as proof (kriging/rbf are equivalent but
    # import the optional SMT backend); free builtins carry required_tier=None.
    assert registry.get_trainer("polynomial") is not None
    assert registry.get_trainer("reduced-field") is not None
