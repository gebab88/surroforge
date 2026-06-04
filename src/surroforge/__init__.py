"""SurroForge — surrogate models for FEM/CFD simulation data.

The high-level Python API (``Project``, ``Trainer``) mirrors the CLI and is the
entry point most users reach for. See the package README for the command-line
workflow.
"""

from __future__ import annotations

__version__ = "0.1.0"

from surroforge.api.highlevel import Trainer
from surroforge.core.project import Project

__all__ = ["Project", "Trainer", "__version__"]
