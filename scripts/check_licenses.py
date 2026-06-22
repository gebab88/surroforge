"""Fail if any SHIPPED dependency carries a copyleft (GPL/LGPL/AGPL) license (NFR8).

Reads ``pip-licenses --format=json`` from stdin (a scan of the whole environment)
and restricts it to the *install closure* of ``surroforge[all]`` before flagging
copyleft. Scoping matters: the local dev environment also holds build/test tooling
(pytest, twine, docutils, …) whose licenses are irrelevant to what a user receives
from ``pip install surroforge[all]``. The original GitHub Actions ``licenses`` job
got this scoping for free by installing only ``.[all]`` into a clean runner; here we
reproduce it by computing the closure from installed metadata.

Exits non-zero on the first copyleft hit. Used by the local CI runner (scripts/ci.sh).
"""

from __future__ import annotations

import json
import sys
from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def _req_active(req: Requirement, wanted_extras: set[str]) -> bool:
    """True if ``req`` is installed when its parent is pulled with ``wanted_extras``.

    Base requirements (no marker) are always active; an ``extra == "x"`` requirement
    is active only when ``x`` is among the parent's requested extras. Environment
    markers (python_version, sys_platform, …) are evaluated against the current host.
    """
    if req.marker is None:
        return True
    if req.marker.evaluate():  # base dep gated only by an environment marker
        return True
    return any(req.marker.evaluate({"extra": e}) for e in wanted_extras)


def closure(root: str, root_extras: set[str]) -> set[str]:
    """Canonical names installed by ``pip install root[root_extras]``, transitively.

    The root is pulled with ``root_extras``; transitive dependencies are pulled with
    only the extras they request of each other (``foo[bar]``), never the root's —
    matching how pip resolves an install.
    """
    extras_for: dict[str, set[str]] = {canonicalize_name(root): set(root_extras)}
    nodes: set[str] = set()
    queue: list[str] = [canonicalize_name(root)]
    while queue:
        name = queue.pop()
        nodes.add(name)
        try:
            reqs = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue  # not installed (an unselected optional dep) — nothing to walk
        wanted = extras_for.get(name, set())
        for raw in reqs:
            req = Requirement(raw)
            if not _req_active(req, wanted):
                continue
            dep = canonicalize_name(req.name)
            dep_extras = set(req.extras)
            if dep not in nodes or not dep_extras.issubset(extras_for.get(dep, set())):
                extras_for.setdefault(dep, set()).update(dep_extras)
                queue.append(dep)
    return nodes


def main() -> int:
    shipped = closure("surroforge", {"all"})
    data = json.load(sys.stdin)
    scanned = [d for d in data if canonicalize_name(d["Name"]) in shipped]
    bad = [d for d in scanned if "GPL" in d["License"].upper()]
    for d in bad:
        print(f"COPYLEFT: {d['Name']} {d['Version']} -> {d['License']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
