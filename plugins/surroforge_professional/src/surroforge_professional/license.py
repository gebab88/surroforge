"""Local Professional/Enterprise license gate — tier-aware stub (Wave 1).

Real activation/validation is milestone M3. Until then the *presence* of a
license (env var or file) grants a tier and its *content* selects which one:
a value containing "enterprise" (case-insensitive) grants the enterprise tier,
any other non-empty value grants professional. ``authorize`` is registered on
Core's ``surroforge.authorizers`` entry point; ``has_license``/``require_license``
remain for in-package self-gates (reports, DoE) and keep their old behaviour.
"""

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

_TIER_ORDER = {"professional": 1, "enterprise": 2}


def license_file() -> Path:
    return LICENSE_FILE.expanduser()


def _license_value() -> str:
    env_value = os.environ.get(LICENSE_ENV, "").strip()
    if env_value:
        return env_value
    path = license_file()
    if path.is_file():
        return path.read_text().strip()
    return ""


def granted_tier() -> str | None:
    """The tier the local license grants: 'enterprise', 'professional', or None."""
    value = _license_value()
    if not value:
        return None
    return "enterprise" if "enterprise" in value.lower() else "professional"


def authorize(required_tier: str) -> bool:
    """Entry-point authorizer for Core's gate: does the local license meet the tier?

    Unknown requirements are denied (a newer plugin against an older edition
    must not be silently unlocked).
    """
    granted = granted_tier()
    if granted is None:
        return False
    return _TIER_ORDER[granted] >= _TIER_ORDER.get(required_tier, float("inf"))


#: Activation hint Core's editions-neutral gate appends to its error message.
authorize.hint = LICENSE_ERROR  # type: ignore[attr-defined]


def has_license() -> bool:
    return granted_tier() is not None


def require_license(required_tier: str = "professional") -> None:
    if not authorize(required_tier):
        raise PluginError(LICENSE_ERROR)
