"""Shippable conformance suite for trainer plugins (PRD 4.11, Story 5.3).

Plugin authors verify their trainer honours the Core contract before publishing::

    from surroforge.conformance import check_trainer
    from my_package import MyTrainer

    check_trainer(MyTrainer).raise_for_status()   # raises ConformanceError on a violation

The suite is framework-agnostic (no pytest dependency): :func:`check_trainer`
returns a structured :class:`ConformanceReport` you can print, inspect, or assert
on. Core runs the very same checks against its three built-in trainers in
``tests/test_conformance.py``.
"""

from surroforge.conformance.trainer import (
    CheckResult,
    ConformanceReport,
    check_trainer,
)
from surroforge.errors import ConformanceError

__all__ = ["CheckResult", "ConformanceError", "ConformanceReport", "check_trainer"]
