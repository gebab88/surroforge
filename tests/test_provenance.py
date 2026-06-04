"""Pickle artifact provenance warning (Story 4.4 / AR12).

Loading a ``model.pkl`` unpickles it, which executes code embedded in the file.
SurroForge therefore warns once before loading any artifact it cannot confirm was
produced on this machine, while locally-produced artifacts load silently.
"""

from __future__ import annotations

import json
import platform
import warnings

import pytest

from surroforge.core.run import Run
from surroforge.persistence import is_locally_produced, machine_id

_FOREIGN_ID = "0" * 16


def _make_foreign(monkeypatch) -> None:
    """Make the current machine id differ from whatever a run recorded."""
    monkeypatch.setattr("surroforge.persistence.store.machine_id", lambda: _FOREIGN_ID)


def test_machine_id_is_stable_and_opaque():
    mid = machine_id()
    assert mid == machine_id()  # stable across calls
    assert len(mid) == 16 and all(c in "0123456789abcdef" for c in mid)
    # It is a hash, not the raw hostname — nothing identifying leaks into lineage.
    assert not platform.node() or platform.node() not in mid


def test_fresh_run_is_locally_produced(trained_run):
    # A run trained here records this machine's id, so provenance is confirmed.
    assert trained_run.lineage.get("machine_id") == machine_id()
    assert is_locally_produced(trained_run.lineage)


def test_local_artifact_loads_silently(trained_run):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trained_run.load_model()
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_foreign_artifact_warns_before_loading(trained_run, monkeypatch):
    _make_foreign(monkeypatch)
    with pytest.warns(UserWarning, match="not confirmed as produced on this machine"):
        model = trained_run.load_model()
    # AC: loading still proceeds — the warning is advisory, not a block.
    assert model is not None


def test_legacy_run_without_machine_id_warns(trained_run):
    # Simulate a pre-AR12 run: strip the provenance marker, then re-read fresh.
    lineage_file = trained_run.run_dir / "lineage.json"
    data = json.loads(lineage_file.read_text())
    data.pop("machine_id", None)
    lineage_file.write_text(json.dumps(data))

    fresh = Run(trained_run.run_dir)
    assert not is_locally_produced(fresh.lineage)  # cannot confirm -> foreign
    with pytest.warns(UserWarning, match="not confirmed"):
        fresh.load_model()


def test_foreign_warning_fires_once_per_artifact(trained_run, monkeypatch):
    # Two loads of the same foreign artifact warn exactly once (AC: "one-time"),
    # proving the explicit dedupe — not just the warnings module's once-filter,
    # which is defeated here by simplefilter("always").
    _make_foreign(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trained_run.load_model()
        trained_run.load_model()
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
