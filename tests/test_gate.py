"""The tier-aware authorisation gate (Wave 1) — unit tests with fake authorizers."""

from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.plugins import gate
from surroforge.plugins.seam import PluginMetadata


def _meta(tier: str | None) -> PluginMetadata:
    return PluginMetadata(
        name="thing", group="surroforge.trainers", api_version=1,
        source="entry_point", ref="mod:Thing", required_tier=tier,
    )


def test_free_plugin_is_always_permitted(monkeypatch):
    monkeypatch.setattr(gate, "_load_authorizers", lambda: ())
    gate.check(_meta(None))  # must not raise


def test_tier_plugin_denied_without_any_authorizer(monkeypatch):
    monkeypatch.setattr(gate, "_load_authorizers", lambda: ())
    with pytest.raises(PluginError, match="no installed SurroForge edition"):
        gate.check(_meta("professional"))


def test_tier_plugin_permitted_by_granting_authorizer(monkeypatch):
    monkeypatch.setattr(
        gate, "_load_authorizers", lambda: (lambda tier: tier == "professional",)
    )
    gate.check(_meta("professional"))  # must not raise


def test_tier_plugin_denied_by_refusing_authorizer(monkeypatch):
    monkeypatch.setattr(gate, "_load_authorizers", lambda: (lambda tier: False,))
    with pytest.raises(PluginError, match="'enterprise' license"):
        gate.check(_meta("enterprise"))


def test_denying_authorizer_hint_is_appended(monkeypatch):
    def deny(tier: str) -> bool:
        return False

    deny.hint = "Set THE_LICENSE_VAR."
    monkeypatch.setattr(gate, "_load_authorizers", lambda: (deny,))
    with pytest.raises(PluginError, match="Set THE_LICENSE_VAR"):
        gate.check(_meta("professional"))


def test_error_names_plugin_and_tier(monkeypatch):
    monkeypatch.setattr(gate, "_load_authorizers", lambda: ())
    with pytest.raises(PluginError, match=r"'thing' requires a 'professional' license"):
        gate.check(_meta("professional"))
