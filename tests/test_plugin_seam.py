"""Core↔commercial plugin seam reservation (Story 5.5 / AR13).

These verify the *shape* is frozen: a plugin API-version check at resolution, a
deterministic name-collision policy, and a post-resolution gate hook that is a
Phase-1 no-op. No edition-named logic should exist in Core.
"""

from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.plugins import gate, registry
from surroforge.plugins.seam import (
    PLUGIN_API_VERSION,
    PluginMetadata,
    check_api_version,
    resolve_api_version,
)
from surroforge.trainers.polynomial import PolynomialTrainer


class _FakeEP:
    """Duck-typed importlib EntryPoint — only ``.value`` and ``.load()`` are used."""

    def __init__(self, value: str, cls: type) -> None:
        self.value = value
        self._cls = cls

    def load(self) -> type:
        return self._cls


class _FutureTrainer(PolynomialTrainer):
    api_version = 999  # built against an unsupported future contract


class _CurrentOverride(PolynomialTrainer):
    api_version = PLUGIN_API_VERSION


# --- API-version declaration -------------------------------------------------


def test_resolve_api_version_defaults_to_current():
    # An undeclared plugin is treated as current, so nothing breaks in Phase 1.
    assert resolve_api_version(PolynomialTrainer) == PLUGIN_API_VERSION


def test_resolve_api_version_reads_declared():
    assert resolve_api_version(_FutureTrainer) == 999


def test_resolve_api_version_rejects_non_integer():
    class _Bad(PolynomialTrainer):
        api_version = "1"  # type: ignore[assignment]

    with pytest.raises(PluginError):
        resolve_api_version(_Bad)


def test_check_api_version_accepts_current():
    meta = PluginMetadata("x", registry.TRAINER_GROUP, PLUGIN_API_VERSION, "entry_point", "p:X")
    check_api_version(meta)  # must not raise


def test_check_api_version_rejects_mismatch():
    meta = PluginMetadata("x", registry.TRAINER_GROUP, 999, "entry_point", "p:X")
    with pytest.raises(PluginError, match="API version 999"):
        check_api_version(meta)


# --- Gate is a documented no-op ----------------------------------------------


def test_gate_check_is_noop():
    meta = PluginMetadata("x", registry.TRAINER_GROUP, PLUGIN_API_VERSION, "builtin", "p:X")
    assert gate.check(meta) is None


# --- Registry wiring ---------------------------------------------------------


def test_registry_checks_api_version_at_resolution(monkeypatch):
    monkeypatch.setattr(
        registry,
        "_entry_points_grouped",
        lambda group: {"futuristic": [_FakeEP("pkg:Future", _FutureTrainer)]},
    )
    with pytest.raises(PluginError, match="API version 999"):
        registry.get_trainer("futuristic")


def test_registry_flags_ambiguous_name_collision(monkeypatch):
    monkeypatch.setattr(
        registry,
        "_entry_points_grouped",
        lambda group: {
            "dup": [_FakeEP("a:X", PolynomialTrainer), _FakeEP("b:Y", _CurrentOverride)]
        },
    )
    with pytest.raises(PluginError, match="Ambiguous"):
        registry.get_trainer("dup")


def test_registry_entry_point_overrides_builtin(monkeypatch):
    monkeypatch.setattr(
        registry,
        "_entry_points_grouped",
        lambda group: {"polynomial": [_FakeEP("pkg:Override", _CurrentOverride)]},
    )
    assert registry.get_trainer("polynomial") is _CurrentOverride


def test_duplicate_ref_is_not_a_collision(monkeypatch):
    # The same provider listed twice (e.g. an editable-install quirk) is de-duped
    # by ref, not treated as an ambiguous collision.
    monkeypatch.setattr(
        registry,
        "_entry_points_grouped",
        lambda group: {"polynomial": [_FakeEP("pkg:Override", _CurrentOverride)]},
    )
    assert registry.get_trainer("polynomial") is _CurrentOverride


def test_builtins_still_resolve():
    # The default resolution path (no monkeypatch) still returns the built-ins.
    assert registry.get_trainer("polynomial") is PolynomialTrainer
