"""Plugin authorisation gate (Story 5.5 / AR13) — tier-aware since Wave 1.

Reserved call site, invoked by the registry immediately after a plugin is
resolved. It stays editions-neutral: Core never names an edition or an
activation mechanism. A plugin *declares* a required tier (opaque string data,
:func:`surroforge.plugins.seam.resolve_required_tier`); installed commercial
editions *register* authorizers under the ``surroforge.authorizers`` entry
point; this gate asks each authorizer whether the requirement is met. A
Core-only installation has no tier-declaring plugins, so the deny path never
fires there. NOTE: this module must not import the registry (the registry
imports this module), so it scans its entry-point group itself.
"""

from __future__ import annotations

import functools
import importlib.metadata as md
from collections.abc import Callable

from surroforge.errors import PluginError
from surroforge.plugins.seam import PluginMetadata

#: Entry-point group a commercial edition uses to register its authorizer:
#: a callable ``authorize(required_tier: str) -> bool``. It may carry a ``hint``
#: attribute (str) with an edition-specific activation hint that this gate
#: appends to its otherwise editions-neutral error message.
AUTHORIZER_GROUP = "surroforge.authorizers"


@functools.cache
def _load_authorizers() -> tuple[Callable[[str], bool], ...]:
    """All installed authorizers (memoised; installed dists cannot change at runtime)."""
    return tuple(ep.load() for ep in md.entry_points(group=AUTHORIZER_GROUP))


def check(metadata: PluginMetadata) -> None:
    """Authorise a resolved plugin: free plugins pass, tier-declaring ones are gated.

    The authorizer scan only runs for tier-declaring plugins, so the free
    (Core) resolution path pays no overhead.
    """
    if metadata.required_tier is None:
        return
    if any(authorize(metadata.required_tier) for authorize in _load_authorizers()):
        return
    raise PluginError(_denied_message(metadata.name, metadata.required_tier))


def require_license(feature: str, *, tier: str = "professional") -> None:
    """Gate a Core feature surface behind an edition tier (Wave 3).

    The resolution seam (:func:`check`) covers everything reached through the
    plugin registry; feature surfaces like ``reproduce`` or the lineage panels
    never resolve a plugin, so they call this directly — same authorizer group,
    same hint mechanics, same editions-neutral message, naming the FEATURE
    instead of a plugin.
    """
    if any(authorize(tier) for authorize in _load_authorizers()):
        return
    raise PluginError(_denied_message(feature, tier))


def _denied_message(name: str, tier: str) -> str:
    """The editions-neutral denial, shared by the seam gate and the feature gate.

    Core never names an activation mechanism; an installed edition's authorizer
    may carry a ``hint`` attribute that is appended verbatim. Wave-1 tests assert
    this wording — keep it byte-identical.
    """
    authorizers = _load_authorizers()
    hints = " ".join(h for a in authorizers if (h := getattr(a, "hint", "")))
    message = (
        f"'{name}' requires a '{tier}' license, "
        "and no installed SurroForge edition grants it."
    )
    return f"{message} {hints}".rstrip()
