from __future__ import annotations

from surroforge.plugins import available_reporters, get_reporter
from surroforge_professional.reports.html import HtmlReporter


def test_html_reporter_registered():
    assert "html" in available_reporters()
    assert get_reporter("html") is HtmlReporter
