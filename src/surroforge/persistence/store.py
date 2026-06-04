"""Low-level persistence: atomic JSON/YAML I/O, run ids, environment capture.

Writes are atomic (temp file + ``os.replace``) so a crash mid-write can never
leave a half-written ``lineage.json`` or project file behind.
"""

from __future__ import annotations

import getpass
import hashlib
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


def machine_id() -> str:
    """Opaque, stable identifier for the current machine + user.

    A short hash of hostname + username — NOT the raw values — so a run's
    lineage records *where it was produced* well enough to flag a foreign
    pickle (Story 4.4 / AR12) without leaking PII into a shareable run dir
    (NFR5). It only needs to be stable per machine and to differ across
    machines; it is never reversed or sent anywhere.
    """
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - getuser can fail in odd environments
        user = ""
    raw = f"{platform.node()}\x00{user}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def is_locally_produced(lineage: dict[str, Any]) -> bool:
    """Whether a run's recorded ``machine_id`` matches this machine.

    A missing id (a legacy run from before provenance capture) returns False:
    provenance cannot be confirmed, so callers should treat it as foreign.
    """
    recorded = lineage.get("machine_id")
    return bool(recorded) and recorded == machine_id()


def capture_environment() -> dict[str, Any]:
    """Python/SurroForge versions, dependency pins, and provenance for a run."""
    from surroforge import __version__

    return {
        "python_version": platform.python_version(),
        "surroforge_version": __version__,
        "platform": platform.platform(),
        "machine_id": machine_id(),
        "dependencies": capture_dependencies(),
    }
