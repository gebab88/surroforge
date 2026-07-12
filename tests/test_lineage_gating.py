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


def test_compare_runs_strips_lineage_without_license(unlicensed, project):
    from surroforge.services import compare_runs

    r1 = run_training(project, parse_config(poly_config("polynomial")))
    r2 = run_training(project, parse_config(poly_config("polynomial")))
    result = compare_runs(project, [r1.run_id, r2.run_id])
    assert "lineage_summary" not in result
    assert all("lineage" not in entry for entry in result["runs"])
    assert result["best"] is not None  # ranking/metrics stay free


def test_compare_runs_keeps_lineage_with_granted_license(granted_license, project):
    from surroforge.services import compare_runs

    r1 = run_training(project, parse_config(poly_config("polynomial")))
    r2 = run_training(project, parse_config(poly_config("polynomial")))
    result = compare_runs(project, [r1.run_id, r2.run_id])
    assert result["lineage_summary"]["same_data"] is True
    assert all("lineage" in entry for entry in result["runs"])
