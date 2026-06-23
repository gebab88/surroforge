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
