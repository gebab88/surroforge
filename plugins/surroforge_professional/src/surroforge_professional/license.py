"""Local Professional license gate."""

from __future__ import annotations

import os
from pathlib import Path

from surroforge.errors import PluginError

LICENSE_ENV = "SURROFORGE_PRO_LICENSE"
LICENSE_FILE = Path("~/.surroforge/professional.lic")
LICENSE_ERROR = (
    "Professional license required. Set SURROFORGE_PRO_LICENSE "
    "or install ~/.surroforge/professional.lic."
)


def license_file() -> Path:
    return LICENSE_FILE.expanduser()


def has_license() -> bool:
    env_value = os.environ.get(LICENSE_ENV, "").strip()
    return bool(env_value) or license_file().is_file()


def require_license() -> None:
    if not has_license():
        raise PluginError(LICENSE_ERROR)
