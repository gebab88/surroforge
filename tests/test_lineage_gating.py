"""Wave-3 feature gating: reproduce/lineage surfaces are Professional.

Core-side behavior tests. Unlicensed tests neutralize env + HOME (W2 pattern)
so a real dev-machine licence cannot defeat them; the licensed side uses the
``granted_license`` fake authorizer, so this file runs on a Core-only checkout.
"""

from __future__ import annotations

import pytest

from surroforge.core.config import parse_config
from surroforge.engine import run_training
from surroforge.errors import PluginError
from tests.conftest import poly_config


@pytest.fixture
def unlicensed(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-licence-home"))


def test_reproduce_service_denied_without_license(unlicensed, project):
    from surroforge.services import reproduce

    run = run_training(project, parse_config(poly_config("polynomial")))
    with pytest.raises(PluginError, match=r"'reproduce' requires a 'professional' license"):
        reproduce(project, run.run_id)


def test_reproduce_service_works_with_granted_license(granted_license, project):
    from surroforge.services import assert_reproducible, reproduce

    run = run_training(project, parse_config(poly_config("polynomial")))
    result = reproduce(project, run.run_id)
    assert_reproducible(result)  # must not raise
