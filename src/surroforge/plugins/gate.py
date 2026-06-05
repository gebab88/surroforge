"""Plugin authorisation gate — a Phase-1 no-op (Story 5.5 / AR13).

Reserved call site, invoked by the registry immediately after a plugin is
resolved, where a future licensed edition will authorise (or refuse) the plugin.
In Phase 1 this does nothing and permits everything; it contains NO edition-named
logic. Freezing this call site now is what lets Phase 2 add licensing without
touching Core's resolution path.
"""

from __future__ import annotations

from surroforge.plugins.seam import PluginMetadata


def check(metadata: PluginMetadata) -> None:
    """Authorise a resolved plugin. Phase-1 no-op: always permits.

    A Phase-2 edition replaces this body to gate on ``metadata`` (e.g. a licence
    check). It must stay a pure side-effect-free permit/deny on the passed
    metadata so the registry's contract is unchanged.
    """
    return None
