"""Entry-point based plugin registry (PRD 4.11).

Built-in trainers/exporters/reporters and third-party plugins are discovered
through the same ``surroforge.*`` entry-point groups, so the core has no
special-case path. A built-in fallback table keeps things working even when the
package is run from a source checkout that was never ``pip install``-ed (no
entry-point metadata), which is exactly the situation during development.
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata as md
from typing import Any

from surroforge.errors import PluginError
from surroforge.plugins import gate
from surroforge.plugins.seam import (
    PluginMetadata,
    check_api_version,
    resolve_api_version,
)

TRAINER_GROUP = "surroforge.trainers"
EXPORTER_GROUP = "surroforge.exporters"
REPORTER_GROUP = "surroforge.reporters"
COMMAND_GROUP = "surroforge.commands"

_BUILTIN: dict[str, dict[str, str]] = {
    TRAINER_GROUP: {
        "polynomial": "surroforge.trainers.polynomial:PolynomialTrainer",
        "kriging": "surroforge.trainers.kriging:KrigingTrainer",
        "rbf": "surroforge.trainers.rbf:RBFTrainer",
        "mlp": "surroforge.trainers.mlp:MLPTrainer",
        "reduced-field": "surroforge.trainers.reduced_field:ReducedFieldTrainer",
    },
    EXPORTER_GROUP: {
        "pickle": "surroforge.exporters.pickle_exporter:PickleExporter",
        "onnx": "surroforge.exporters.onnx_exporter:ONNXExporter",
    },
    REPORTER_GROUP: {
        "markdown": "surroforge.reporters.markdown:MarkdownReporter",
    },
}


def _load_ref(ref: str) -> Any:
    module_name, _, attr = ref.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


@functools.cache
def _scan_entry_points(group: str) -> tuple[md.EntryPoint, ...]:
    """All entry points declared in ``group`` (memoised for the process).

    ``md.entry_points(group=...)`` walks every installed distribution's metadata,
    which profiling showed to be the single largest fixed cost of a training run.
    Installed entry points cannot change while the process runs, so the scan is
    cached. An immutable tuple is returned so the shared cached value is safe;
    grouping happens fresh in :func:`_entry_points_grouped`. (Tests inject fakes by
    replacing ``_entry_points_grouped`` itself, so they bypass this cache cleanly;
    use ``_scan_entry_points.cache_clear()`` if a process ever needs a re-scan.)
    """
    return tuple(md.entry_points(group=group))


def _entry_points_grouped(group: str) -> dict[str, list[md.EntryPoint]]:
    """Entry points in ``group``, grouped by name and de-duplicated by ref.

    Grouping (rather than the old ``{ep.name: ep}`` comprehension) surfaces a
    genuine name collision instead of silently keeping whichever provider happened
    to be iterated last — which made resolution depend on installation order.
    """
    grouped: dict[str, dict[str, md.EntryPoint]] = {}
    for ep in _scan_entry_points(group):
        grouped.setdefault(ep.name, {})[ep.value] = ep
    return {name: list(by_ref.values()) for name, by_ref in grouped.items()}


def _resolve(group: str, name: str, kind: str) -> Any:
    # Precedence policy (Story 5.5 / AR13): a single external entry point
    # overrides the built-in fallback; the built-in table is only consulted for
    # source checkouts that were never pip-installed. Two external providers for
    # the same name is an ambiguous collision we refuse to guess about.
    candidates = _entry_points_grouped(group).get(name, [])
    if len(candidates) > 1:
        raise PluginError(
            f"Ambiguous {kind} '{name}': it is registered by multiple plugins "
            f"({', '.join(sorted(ep.value for ep in candidates))}). Uninstall or "
            "rename one — SurroForge will not pick a provider for you."
        )
    if candidates:
        ep = candidates[0]
        cls, source, ref = ep.load(), "entry_point", ep.value
    elif name in _BUILTIN[group]:
        ref = _BUILTIN[group][name]
        cls, source = _load_ref(ref), "builtin"
    else:
        raise PluginError(f"Unknown {kind} '{name}'. Available: {sorted(_names(group))}")

    # Reserve the commercial seam: validate the declared plugin API version, then
    # run the authorisation gate (a documented Phase-1 no-op) after resolution.
    metadata = PluginMetadata(
        name=name,
        group=group,
        api_version=resolve_api_version(cls),
        source=source,
        ref=ref,
    )
    check_api_version(metadata)
    gate.check(metadata)
    return cls


def _names(group: str) -> set[str]:
    return set(_BUILTIN[group]) | set(_entry_points_grouped(group))


def get_trainer(name: str) -> type:
    return _resolve(TRAINER_GROUP, name, "method")


def get_exporter(name: str) -> type:
    return _resolve(EXPORTER_GROUP, name, "export format")


def get_reporter(name: str) -> type:
    return _resolve(REPORTER_GROUP, name, "report format")


def available_trainers() -> list[str]:
    return sorted(_names(TRAINER_GROUP))


def available_exporters() -> list[str]:
    return sorted(_names(EXPORTER_GROUP))


def available_reporters() -> list[str]:
    return sorted(_names(REPORTER_GROUP))


def available_command_names() -> list[str]:
    return sorted(_entry_points_grouped(COMMAND_GROUP))


def available_commands() -> list[tuple[str, Any]]:
    """Plugin-contributed CLI sub-apps registered under ``surroforge.commands``.

    Each entry point is expected to load a ``typer.Typer`` instance. Discovery is
    kept separate from :func:`_resolve` (which is trainer/exporter/reporter-shaped
    and consults the ``_BUILTIN`` table); Core ships no built-in plugin commands.
    """
    loaded: list[tuple[str, Any]] = []
    for name, candidates in sorted(_entry_points_grouped(COMMAND_GROUP).items()):
        if len(candidates) > 1:
            raise PluginError(
                f"Ambiguous command '{name}': it is registered by multiple plugins "
                f"({', '.join(sorted(ep.value for ep in candidates))}). Uninstall or "
                "rename one — SurroForge will not pick a provider for you."
            )
        loaded.append((name, candidates[0].load()))
    return loaded
