"""Exception hierarchy for SurroForge.

A single root (:class:`SurroForgeError`) lets the CLI catch every domain error
and render it as a clean message instead of a traceback, while library users can
still catch specific subclasses.
"""

from __future__ import annotations


class SurroForgeError(Exception):
    """Base class for all SurroForge errors."""


class ProjectError(SurroForgeError):
    """Project directory is missing, malformed, or already exists."""


class ConfigError(SurroForgeError):
    """A YAML training configuration is invalid."""


class DataError(SurroForgeError):
    """A dataset cannot be read, validated, or matched to a configuration."""


class TrainerError(SurroForgeError):
    """A surrogate method failed or was used outside its supported envelope."""


class PluginError(SurroForgeError):
    """A requested trainer, exporter, or reporter is not registered."""


class ExportError(SurroForgeError):
    """A model could not be exported to the requested format."""


class RunNotFoundError(SurroForgeError):
    """No run with the requested id exists in the project."""


class ReproducibilityError(SurroForgeError):
    """A reproduced run diverged from its parent beyond tolerance."""
