"""Smoke test: every example script under examples/ must run to completion.

The examples are living documentation of the public API — running them in CI
keeps them from drifting out of step with the code. Each script is
self-contained (it synthesises its own data in a temp dir), so this just shells
out to it and asserts a clean exit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = sorted((_REPO_ROOT / "examples").glob("[0-9]*.py"))


def test_examples_directory_is_present():
    # Guard: if the glob ever finds nothing, the parametrized test below would
    # silently pass by collecting zero cases. Fail loudly instead.
    assert _EXAMPLES, "no runnable example scripts found under examples/"


@pytest.mark.parametrize("script", _EXAMPLES, ids=lambda p: p.stem)
def test_example_runs_to_completion(script: Path):
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"{script.name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
