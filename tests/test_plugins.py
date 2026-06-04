"""Plugin registry (PRD 4.11)."""

from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.plugins import (
    available_exporters,
    available_reporters,
    available_trainers,
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
