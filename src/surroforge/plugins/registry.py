"""Entry-point based plugin registry (PRD 4.11).

Built-in trainers/exporters/reporters and third-party plugins are discovered
through the same ``surroforge.*`` entry-point groups, so the core has no
special-case path. A built-in fallback table keeps things working even when the
package is run from a source checkout that was never ``pip install``-ed (no
entry-point metadata), which is exactly the situation during development.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
from typing import Any

from surroforge.errors import PluginError

TRAINER_GROUP = "surroforge.trainers"
EXPORTER_GROUP = "surroforge.exporters"
REPORTER_GROUP = "surroforge.reporters"

_BUILTIN: dict[str, dict[str, str]] = {
    TRAINER_GROUP: {
        "polynomial": "surroforge.trainers.polynomial:PolynomialTrainer",
        "kriging": "surroforge.trainers.kriging:KrigingTrainer",
        "rbf": "surroforge.trainers.rbf:RBFTrainer",
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


def _entry_points(group: str) -> dict[str, md.EntryPoint]:
    # entry_points(group=...) is available on Python 3.10+ (our minimum).
    return {ep.name: ep for ep in md.entry_points(group=group)}


def _resolve(group: str, name: str, kind: str) -> Any:
    # Installed entry points take precedence so third parties can extend or
    # override; the built-in table is the fallback for source checkouts.
    eps = _entry_points(group)
    if name in eps:
        return eps[name].load()
    if name in _BUILTIN[group]:
        return _load_ref(_BUILTIN[group][name])
    raise PluginError(
        f"Unknown {kind} '{name}'. Available: {sorted(_names(group))}"
    )


def _names(group: str) -> set[str]:
    return set(_BUILTIN[group]) | set(_entry_points(group))


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
