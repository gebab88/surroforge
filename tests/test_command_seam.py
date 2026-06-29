from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from surroforge.cli.main import _mount_plugin_commands
from surroforge.errors import DataError, PluginError
from surroforge.plugins import available_command_names, available_commands, registry


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
        registry,
        "_entry_points_grouped",
        lambda group: grouped if group == registry.COMMAND_GROUP else {},
    )

    assert available_command_names() == ["doe"]
    assert available_commands() == [("doe", sentinel)]


def test_available_commands_rejects_ambiguous(monkeypatch):
    grouped = {
        "doe": [_FakeEP("doe", "a:app", object()), _FakeEP("doe", "b:app", object())]
    }
    monkeypatch.setattr(
        registry,
        "_entry_points_grouped",
        lambda group: grouped if group == registry.COMMAND_GROUP else {},
    )

    with pytest.raises(PluginError, match="Ambiguous command 'doe'"):
        available_commands()


def test_available_commands_empty_by_default(monkeypatch):
    monkeypatch.setattr(registry, "_entry_points_grouped", lambda group: {})
    assert available_command_names() == []
    assert available_commands() == []


def _build_plugin_app() -> typer.Typer:
    sub = typer.Typer(no_args_is_help=True)

    @sub.command("ok")
    def ok(name: str = typer.Argument(...)) -> None:
        typer.echo(f"hello {name}")

    @sub.command("boom")
    def boom() -> None:
        raise DataError("kaboom")

    return sub


def test_mount_adds_subcommand_and_passes_args():
    root = typer.Typer()
    _mount_plugin_commands(root, [("demo", _build_plugin_app())])

    result = CliRunner().invoke(root, ["demo", "ok", "World"])
    assert result.exit_code == 0
    assert "hello World" in result.stdout


def test_mount_wraps_callbacks_with_guard():
    root = typer.Typer()
    _mount_plugin_commands(root, [("demo", _build_plugin_app())])

    result = CliRunner().invoke(root, ["demo", "boom"])
    assert result.exit_code == 1
    assert "Error: kaboom" in result.stderr


def test_mount_handles_no_commands():
    root = typer.Typer()
    _mount_plugin_commands(root, [])  # must not raise
