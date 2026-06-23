"""Static HTML report for SurroForge Professional."""

from __future__ import annotations

from surroforge.core.run import Run
from surroforge.reporters.base import BaseReporter
from surroforge_professional.license import require_license


class HtmlReporter(BaseReporter):
    name = "html"
    extension = ".html"

    def render(self, run: Run) -> str:
        require_license()
        return "<!doctype html>\n<title>SurroForge Report</title>\n"
