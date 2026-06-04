"""Architectural boundary guards (Tier-A invariants).

These run in the normal pytest suite so the boundaries are enforced even before a
CI pipeline exists. Two complementary checks:

1. AST scan — Core must never import a commercial sibling distribution
   (``surroforge_professional`` / ``surroforge_enterprise`` / any ``surroforge_*``).
   import-linter cannot catch this because the commercial package is not installed;
   this test defines the contract now, so a future violation goes red on the commit
   that introduces it.
2. import-linter contracts — the dependency-direction invariants in ``.importlinter``
   (core/engine/services never import the adapters; engine has no back-edge to
   services; errors is a leaf). Shelled out so a violation fails ``pytest`` locally.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "surroforge"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _imported_top_levels(source: str) -> set[str]:
    """Top-level package name of every import in a module's source."""
    tree = ast.parse(source)
    tops: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Only absolute imports carry a module; relative imports (level>0) stay
            # inside the package and cannot reach a sibling distribution.
            if node.level == 0 and node.module:
                tops.add(node.module.split(".")[0])
    return tops


def test_core_never_imports_a_commercial_distribution():
    """No module in the Core package may import a ``surroforge_*`` sibling dist.

    Commercial editions ship as separate distributions (e.g. surroforge-professional
    → import name ``surroforge_professional``). Core must run standalone and never
    reference them; commercial code attaches only through the plugin entry points.
    """
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        for top in _imported_top_levels(path.read_text()):
            # The Core package is exactly "surroforge"; "surroforge_<x>" is a
            # separate (commercial) distribution.
            if top.startswith("surroforge_"):
                rel = path.relative_to(_REPO_ROOT)
                offenders.append(f"{rel} imports '{top}'")
    assert not offenders, "Core imports a commercial distribution:\n" + "\n".join(offenders)


@pytest.mark.skipif(
    shutil.which("lint-imports") is None,
    reason="import-linter not installed (pip install 'surroforge[dev]')",
)
def test_import_linter_contracts_hold():
    """The .importlinter dependency-direction contracts must all be KEPT."""
    result = subprocess.run(
        ["lint-imports"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
