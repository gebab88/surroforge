"""Persistence helpers: run ids, atomic I/O, environment capture (PRD 7.2)."""

from __future__ import annotations

import pytest

from surroforge.persistence import (
    capture_dependencies,
    capture_environment,
    format_run_id,
    parse_run_number,
    read_json,
    read_yaml,
    write_json,
    write_yaml,
)


@pytest.mark.parametrize(
    "number,expected",
    [(1, "run-001"), (42, "run-042"), (999, "run-999"), (1000, "run-1000")],
)
def test_format_run_id(number, expected):
    assert format_run_id(number) == expected


@pytest.mark.parametrize(
    "run_id,expected",
    [("run-001", 1), ("run-042", 42), ("run-1000", 1000), ("nope", None), ("run-x", None)],
)
def test_parse_run_number(run_id, expected):
    assert parse_run_number(run_id) == expected


def test_format_then_parse_roundtrip():
    for n in (1, 7, 250, 1234):
        assert parse_run_number(format_run_id(n)) == n


def test_json_roundtrip_and_no_temp_files(tmp_path):
    path = tmp_path / "sub" / "data.json"  # parent created on the fly
    payload = {"b": 2, "a": [1, 2, 3], "nested": {"x": True}}
    write_json(path, payload)
    assert read_json(path) == payload
    # Atomic write must not leave temp files behind.
    leftovers = [p for p in path.parent.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == []


def test_yaml_roundtrip(tmp_path):
    path = tmp_path / "meta.yaml"
    payload = {"name": "p", "datasets": [{"name": "d", "hash": "abc"}]}
    write_yaml(path, payload)
    assert read_yaml(path) == payload


def test_capture_dependencies_includes_core_libs():
    deps = capture_dependencies()
    joined = " ".join(deps)
    assert "numpy==" in joined
    assert "scikit-learn==" in joined


def test_capture_environment_fields():
    env = capture_environment()
    assert set(env) >= {"python_version", "surroforge_version", "platform", "dependencies"}
    assert isinstance(env["dependencies"], list)
    assert env["surroforge_version"]


def test_package_version_is_memoised():
    # md.version parses METADATA on each call; the installed version is fixed for
    # the process, so the lookup is cached (removes a parse from every run).
    from surroforge.persistence.store import _package_version

    _package_version.cache_clear()
    first = _package_version("numpy")
    assert first is _package_version("numpy")  # same cached object
    assert _package_version("definitely-not-an-installed-package") is None


def test_capture_dependencies_returns_fresh_list():
    # Caching is on the per-package lookup, not the list, so callers can safely
    # treat the returned list as their own.
    assert capture_dependencies() is not capture_dependencies()
