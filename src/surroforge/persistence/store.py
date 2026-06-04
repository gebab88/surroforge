"""Low-level persistence: atomic JSON/YAML I/O, run ids, environment capture.

Writes are atomic (temp file + ``os.replace``) so a crash mid-write can never
leave a half-written ``lineage.json`` or project file behind.
"""

from __future__ import annotations

import json
import os
import platform
import re
import tempfile
from importlib import metadata as md
from pathlib import Path
from typing import Any

import yaml

# Libraries whose versions materially affect numeric reproducibility (PRD 5.2).
_TRACKED_PACKAGES = [
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "pydantic",
    "smt",
]

_RUN_ID_RE = re.compile(r"^run-(\d+)$")


def format_run_id(number: int) -> str:
    """Sequential, human-readable run id, e.g. ``run-001`` (PRD open question 4)."""
    return f"run-{number:03d}"


def parse_run_number(run_id: str) -> int | None:
    match = _RUN_ID_RE.match(run_id)
    return int(match.group(1)) if match else None


def write_json(path: str | Path, data: Any) -> None:
    _atomic_write(path, json.dumps(data, indent=2, sort_keys=False) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_yaml(path: str | Path, data: Any) -> None:
    _atomic_write(path, yaml.safe_dump(data, sort_keys=False))


def read_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text())


def _atomic_write(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def capture_dependencies() -> list[str]:
    """Pinned versions of reproducibility-critical libraries (PRD 7.2)."""
    pins: list[str] = []
    for name in _TRACKED_PACKAGES:
        try:
            pins.append(f"{name}=={md.version(name)}")
        except md.PackageNotFoundError:
            continue
    return pins


def capture_environment() -> dict[str, Any]:
    """Python/SurroForge versions and dependency pins for a run's lineage."""
    from surroforge import __version__

    return {
        "python_version": platform.python_version(),
        "surroforge_version": __version__,
        "platform": platform.platform(),
        "dependencies": capture_dependencies(),
    }
