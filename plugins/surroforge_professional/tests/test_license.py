from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge_professional.license import (
    LICENSE_ERROR,
    has_license,
    license_file,
    require_license,
)
from surroforge_professional.reports.html import HtmlReporter


def test_license_missing_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert has_license() is False
    with pytest.raises(PluginError, match="Professional license required"):
        require_license()


def test_env_license_permits_render(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "local-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert has_license() is True
    require_license()


def test_file_license_permits_render(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    path = license_file()
    path.parent.mkdir(parents=True)
    path.write_text("local-dev-license\n")

    assert has_license() is True
    require_license()


def test_reporter_checks_license_before_reading_run(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(PluginError, match="Professional license required"):
        HtmlReporter().render(object())


def test_missing_license_message_is_actionable(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(PluginError) as exc:
        require_license()

    assert str(exc.value) == LICENSE_ERROR
    assert "SURROFORGE_PRO_LICENSE" in str(exc.value)
    assert "~/.surroforge/professional.lic" in str(exc.value)


# --- tier-aware authorizer (Wave 1) -------------------------------------------


def test_plain_license_grants_professional_only(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "local-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))
    from surroforge_professional.license import authorize, granted_tier

    assert granted_tier() == "professional"
    assert authorize("professional") is True
    assert authorize("enterprise") is False


def test_enterprise_license_grants_both_tiers(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "Enterprise-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))
    from surroforge_professional.license import authorize, granted_tier

    assert granted_tier() == "enterprise"
    assert authorize("professional") is True
    assert authorize("enterprise") is True


def test_no_license_grants_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from surroforge_professional.license import authorize, granted_tier

    assert granted_tier() is None
    assert authorize("professional") is False


def test_unknown_required_tier_is_denied(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "enterprise")
    monkeypatch.setenv("HOME", str(tmp_path))
    from surroforge_professional.license import authorize

    assert authorize("galactic") is False


def test_license_file_content_selects_tier(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from surroforge_professional.license import granted_tier, license_file

    path = license_file()
    path.parent.mkdir(parents=True)
    path.write_text("ENTERPRISE site license\n")
    assert granted_tier() == "enterprise"


def test_authorize_carries_activation_hint():
    from surroforge_professional.license import LICENSE_ERROR, authorize

    assert authorize.hint == LICENSE_ERROR  # pyright: ignore[reportFunctionMemberAccess]


def test_require_license_enterprise_tier(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "local-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))
    from surroforge_professional.license import require_license

    require_license()  # professional default — passes
    with pytest.raises(PluginError):
        require_license("enterprise")


def test_authorizer_entry_point_is_registered():
    import importlib.metadata as md

    eps = {ep.name: ep.value for ep in md.entry_points(group="surroforge.authorizers")}
    assert eps.get("professional") == "surroforge_professional.license:authorize"
