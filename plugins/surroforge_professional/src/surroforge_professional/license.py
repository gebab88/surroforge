"""Local Professional/Enterprise license gate — signed offline licenses (M3).

A license is a single-line token ``SFL1.<b64url(payload)>.<b64url(signature)>``
whose Ed25519 signature is verified against the vendor keys in
``_TRUSTED_KEYS`` — content validation, fully offline (PRD §3.3). The payload
carries holder, tier, and an optional expiry date; issuance lives in
``tools/licensing`` (vendor side, never shipped). ``authorize`` is registered
on Core's ``surroforge.authorizers`` entry point and updates its ``hint``
attribute with the concrete finding (missing / invalid / expired / tier too
low); ``has_license``/``require_license`` remain for in-package self-gates.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path

from surroforge.errors import PluginError

LICENSE_ENV = "SURROFORGE_PRO_LICENSE"
LICENSE_FILE = Path("~/.surroforge/professional.lic")
LICENSE_ERROR = (
    "Professional license required. Set SURROFORGE_PRO_LICENSE "
    "or install ~/.surroforge/professional.lic."
)

_TOKEN_PREFIX = "SFL1"

#: Hex-encoded 32-byte Ed25519 public keys this build trusts. Rotation: append
#: the new key in a release and keep the old one until its licenses expire.
_TRUSTED_KEYS: tuple[str, ...] = (
    "cd4c914b640e865a83e2d3b9cf7778947d0a1be33e55dba19f490e78b4e18abe",
)

_TIER_ORDER = {"professional": 1, "enterprise": 2}


@dataclass(frozen=True)
class LicenseStatus:
    """Validation outcome of the local license (``error is None`` == valid)."""

    tier: str | None = None
    holder: str | None = None
    license_id: str | None = None
    issued: str | None = None
    expires: date | None = None
    source: str | None = None  # "env" | "file" | None
    error: str | None = None


def license_file() -> Path:
    return LICENSE_FILE.expanduser()


def _license_value() -> tuple[str, str | None]:
    env_value = os.environ.get(LICENSE_ENV, "").strip()
    if env_value:
        return env_value, "env"
    path = license_file()
    if path.is_file():
        return path.read_text().strip(), "file"
    return "", None


def _verify_signature(payload: bytes, signature: bytes) -> bool:
    # Lazy import keeps loading the authorizer entry point cheap.
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    for key_hex in _TRUSTED_KEYS:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(key_hex))
        try:
            key.verify(signature, payload)
        except InvalidSignature:
            continue
        return True
    return False


def validate_token(token: str) -> LicenseStatus:
    """Validate one token string (no env/file resolution; ``source`` stays None)."""
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != _TOKEN_PREFIX:
        return LicenseStatus(error="malformed token")
    try:
        payload_raw = base64.urlsafe_b64decode(parts[1].encode())
        signature = base64.urlsafe_b64decode(parts[2].encode())
    except (binascii.Error, ValueError):
        return LicenseStatus(error="malformed token")
    if not _verify_signature(payload_raw, signature):
        return LicenseStatus(error="signature check failed")
    try:
        payload = json.loads(payload_raw)
    except ValueError:
        return LicenseStatus(error="malformed token")
    if not isinstance(payload, dict) or payload.get("v") != 1:
        return LicenseStatus(error="unsupported schema version")
    tier = payload.get("tier")
    if tier not in _TIER_ORDER:
        return LicenseStatus(error=f"unknown tier {tier!r}")
    status = LicenseStatus(
        tier=tier,
        holder=payload.get("holder"),
        license_id=payload.get("id"),
        issued=payload.get("issued"),
    )
    expires_raw = payload.get("expires")
    if expires_raw is None:
        return status
    try:
        expires = date.fromisoformat(str(expires_raw))
    except ValueError:
        return LicenseStatus(error="malformed token")
    if datetime.now(timezone.utc).date() > expires:  # valid through the expiry day (UTC)
        return replace(status, expires=expires, error=f"expired {expires.isoformat()}")
    return replace(status, expires=expires)


def license_status() -> LicenseStatus:
    """Resolve (env wins over file, like the Wave-1 stub) and validate."""
    value, source = _license_value()
    if not value:
        return LicenseStatus(error="missing")
    return replace(validate_token(value), source=source)


def granted_tier() -> str | None:
    """The tier the local license grants: 'enterprise', 'professional', or None."""
    status = license_status()
    return status.tier if status.error is None else None


def _denied_hint(status: LicenseStatus) -> str:
    if status.error is None or status.error == "missing":
        return LICENSE_ERROR
    if status.error.startswith("expired "):
        expired_on = status.error.removeprefix("expired ")
        return f"Professional license expired on {expired_on}. Renew to restore access."
    return (
        f"Installed Professional license is invalid ({status.error}). "
        "Reinstall the license or contact your vendor."
    )


def authorize(required_tier: str) -> bool:
    """Entry-point authorizer for Core's gate: does the local license meet the tier?

    Side effect: updates ``authorize.hint`` with the concrete finding so the
    editions-neutral gate message ends in an actionable sentence. Unknown
    requirements are denied (a newer plugin against an older edition must not
    be silently unlocked).
    """
    status = license_status()
    if status.error is None and status.tier is not None:
        if _TIER_ORDER[status.tier] >= _TIER_ORDER.get(required_tier, float("inf")):
            authorize.hint = LICENSE_ERROR  # type: ignore[attr-defined]  # ground state
            return True
        authorize.hint = (  # type: ignore[attr-defined]
            f"Installed license grants '{status.tier}', "
            f"but this feature requires '{required_tier}'."
        )
        return False
    authorize.hint = _denied_hint(status)  # type: ignore[attr-defined]
    return False


#: Activation hint Core's editions-neutral gate appends to its error message;
#: updated per call with the concrete finding (see ``authorize``).
authorize.hint = LICENSE_ERROR  # type: ignore[attr-defined]


def has_license() -> bool:
    return granted_tier() is not None


def require_license(required_tier: str = "professional") -> None:
    if not authorize(required_tier):
        raise PluginError(authorize.hint)  # type: ignore[attr-defined]
