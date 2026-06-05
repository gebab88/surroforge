"""Core↔plugin seam contract — shape-only reservation (Story 5.5 / AR13).

Freezing the *shape* of how an external plugin (including a future commercial
edition) binds to Core, before the first external plugin exists, keeps Phase 2
additive rather than a breaking change. This module is deliberately
edition-agnostic: it carries NO professional/enterprise logic — only the plugin
API version, and the metadata handed to the authorisation gate. The resolution
precedence/collision policy lives in :mod:`surroforge.plugins.registry`; the
authorisation hook is :mod:`surroforge.plugins.gate`.
"""

from __future__ import annotations

from dataclasses import dataclass

from surroforge.errors import PluginError

#: The plugin API contract version Core implements. A plugin built against a
#: contract Core does not support is rejected at resolution. Bump only on a
#: breaking change to the trainer/exporter/reporter contracts.
PLUGIN_API_VERSION = 1

#: API versions Core can load. Phase 1 supports exactly the current one; keeping
#: it a set leaves room to accept older versions once back-compat shims exist.
SUPPORTED_API_VERSIONS = frozenset({PLUGIN_API_VERSION})


@dataclass(frozen=True)
class PluginMetadata:
    """Description of a resolved plugin, passed to :func:`surroforge.plugins.gate.check`."""

    name: str
    group: str
    api_version: int
    source: str  # "builtin" | "entry_point"
    ref: str  # the "module:attr" reference the class was loaded from


def resolve_api_version(cls: object) -> int:
    """The API version a plugin class declares, defaulting to the current one.

    A plugin opts into version checking by setting a class attribute
    ``api_version``. One that does not declare it is treated as current, so
    existing built-ins and undeclared third-party plugins keep working — the
    seam is reserved without forcing a migration in Phase 1.
    """
    version = getattr(cls, "api_version", PLUGIN_API_VERSION)
    if not isinstance(version, int) or isinstance(version, bool):
        raise PluginError(
            f"Plugin {getattr(cls, 'name', cls)!r} declares a non-integer "
            f"api_version {version!r}; it must be an int."
        )
    return version


def check_api_version(metadata: PluginMetadata) -> None:
    """Reject a plugin built against an unsupported API version (clean error)."""
    if metadata.api_version not in SUPPORTED_API_VERSIONS:
        raise PluginError(
            f"Plugin '{metadata.name}' targets plugin API version "
            f"{metadata.api_version}, but this SurroForge supports "
            f"{sorted(SUPPORTED_API_VERSIONS)}. Upgrade SurroForge or install a "
            "compatible version of the plugin."
        )
