"""Smoke test: every example script under examples/ must run to completion.

The examples are living documentation of the public API — running them in CI
keeps them from drifting out of step with the code. Each script is
self-contained (it synthesises its own data in a temp dir), so this just shells
out to it and asserts a clean exit.
"""

from __future__ import annotations

import importlib.util
import os
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


_TOOLS_SFLICENSE = _REPO_ROOT / "tools" / "licensing" / "sflicense.py"


@pytest.fixture(scope="session")
def example_license_env(tmp_path_factory) -> dict[str, str]:
    """License env for the example subprocesses (M3: content validation).

    Examples that exercise Professional methods need a *valid* license in the
    child process. Monkeypatching does not cross the process boundary, so this
    mints a token from a throwaway keypair and injects a ``sitecustomize.py``
    via PYTHONPATH that makes the child trust that keypair. Empty dict (no-op)
    when the professional package is not installed. It is not a statement about
    what a fresh install can run out of the box.
    """
    try:
        import cryptography  # noqa: F401  (professional dependency)
        import surroforge_professional  # noqa: F401
    except ImportError:
        return {}
    spec = importlib.util.spec_from_file_location("sflicense", _TOOLS_SFLICENSE)
    assert spec is not None and spec.loader is not None
    sfl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sfl)
    seed_hex, pub_hex = sfl.generate_keypair()
    token = sfl.mint(
        {
            "v": 1,
            "id": "SF-TEST-EXAMPLES",
            "holder": "example-smoke-test",
            "tier": "professional",
            "issued": "2026-07-13",
        },
        seed_hex,
    )
    site_dir = tmp_path_factory.mktemp("license-site")
    (site_dir / "sitecustomize.py").write_text(
        "# Injected by tests/test_examples.py: trust the throwaway example-license key.\n"
        "try:\n"
        "    from surroforge_professional import license as _license\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        f"    _license._TRUSTED_KEYS = ({pub_hex!r},)\n"
    )
    pythonpath = os.pathsep.join(
        p for p in (str(site_dir), os.environ.get("PYTHONPATH", "")) if p
    )
    return {"SURROFORGE_PRO_LICENSE": token, "PYTHONPATH": pythonpath}


@pytest.mark.parametrize("script", _EXAMPLES, ids=lambda p: p.stem)
def test_example_runs_to_completion(script: Path, example_license_env):
    env = {**os.environ, **example_license_env}
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=_REPO_ROOT,
        env=env,
    )
    assert result.returncode == 0, (
        f"{script.name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
