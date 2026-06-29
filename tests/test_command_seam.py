from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.plugins import available_command_names, available_commands
from surroforge.plugins import registry


class _FakeEP:
    def __init__(self, name, value, obj):
        self.name = name
        self.value = value
        self._obj = obj

    def load(self):
        return self._obj


def test_available_commands_discovers_and_loads(monkeypatch):
    sentinel = object()
    grouped = {"doe": [_FakeEP("doe", "pkg.mod:app", sentinel)]}
    monkeypatch.setattr(
        registry, "_entry_points_grouped", lambda group: grouped if group == registry.COMMAND_GROUP else {}
    )

    assert available_command_names() == ["doe"]
    assert available_commands() == [("doe", sentinel)]


def test_available_commands_rejects_ambiguous(monkeypatch):
    grouped = {
        "doe": [_FakeEP("doe", "a:app", object()), _FakeEP("doe", "b:app", object())]
    }
    monkeypatch.setattr(
        registry, "_entry_points_grouped", lambda group: grouped if group == registry.COMMAND_GROUP else {}
    )

    with pytest.raises(PluginError, match="Ambiguous command 'doe'"):
        available_commands()


def test_available_commands_empty_by_default(monkeypatch):
    monkeypatch.setattr(registry, "_entry_points_grouped", lambda group: {})
    assert available_command_names() == []
    assert available_commands() == []
