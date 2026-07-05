"""Plugin registry (PRD 4.11)."""

from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.plugins import (
    available_exporters,
    available_reporters,
    available_trainers,
    get_comparison_reporter,
    get_exporter,
    get_reporter,
    get_trainer,
)
from surroforge.trainers.base import BaseTrainer


def test_builtin_trainers_registered():
    assert set(available_trainers()) >= {"kriging", "rbf", "polynomial"}


def test_get_trainer_returns_subclass():
    cls = get_trainer("polynomial")
    assert issubclass(cls, BaseTrainer)


def test_unknown_trainer_raises():
    with pytest.raises(PluginError):
        get_trainer("does-not-exist")


def test_exporters_and_reporters_registered():
    assert set(available_exporters()) >= {"pickle", "onnx"}
    assert "markdown" in available_reporters()
    assert get_exporter("pickle").name == "pickle"
    assert get_reporter("markdown").name == "markdown"


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
