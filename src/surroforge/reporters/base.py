"""Reporter interface (PRD 4.9, 4.11)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from surroforge.core.run import Run


class BaseReporter(ABC):
    #: Registry name / ``--format`` value.
    name: str = "base"
    #: File extension (including the dot) for the rendered report.
    extension: str = ""

    @abstractmethod
    def render(self, run: Run) -> str:
        """Return the report for ``run`` as text."""
