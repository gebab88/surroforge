"""Content validation of the signed offline license (M3).

Tokens are minted from a fixed TEST keypair and trusted via monkeypatched
``_TRUSTED_KEYS``; the production build trusts only the vendor key.
"""

from __future__ import annotations

import base64
import json
from datetime import date, datetime, timedelta, timezone

import pytest

pytest.importorskip("cryptography")

from surroforge.errors import PluginError
from surroforge_professional import license as lic

# Fixed test-only Ed25519 keypair (seed = b"surroforge-m3-test-keypair-0001!").
TEST_SEED_HEX = "737572726f666f7267652d6d332d746573742d6b6579706169722d3030303121"
TEST_PUB_HEX = "f96638d3d06b9b93ba477836a72d7477ca2d15730d6144f5a9c0a11f649386f0"

# Format lock, minted once from TEST_SEED_HEX (Ed25519 is deterministic): any
# refactor of the token layout that breaks this string is a breaking change.
GOLDEN_TOKEN = (
    "SFL1."
    "eyJob2xkZXIiOiJHb2xkZW4gVGVzdCIsImlkIjoiU0YtVEVTVC0wMDAxIiwiaXNzdWVkIjoi"
    "MjAyNi0wNy0xMyIsInRpZXIiOiJwcm9mZXNzaW9uYWwiLCJ2IjoxfQ==."
    "9VAgrVi4Fy64B24D88Lq-8XmuDL4dD7DxIV626CoFBRG7TDBv1rES4jzVUL3g_2XLK0rUdlv"
    "a8wTVBuaWTfnCA=="
)


def mint(payload: dict, seed_hex: str = TEST_SEED_HEX) -> str:
    """Standalone minting — deliberately independent of tools/licensing."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
    b64 = base64.urlsafe_b64encode
    return f"SFL1.{b64(raw).decode()}.{b64(key.sign(raw)).decode()}"


def payload(**overrides) -> dict:
    base = {
        "v": 1,
        "id": "SF-TEST-0001",
        "holder": "Test GmbH",
        "tier": "professional",
        "issued": "2026-07-13",
    }
    base.update(overrides)
    return base


@pytest.fixture
def trusted_test_key(monkeypatch):
    monkeypatch.setattr(lic, "_TRUSTED_KEYS", (TEST_PUB_HEX,))


@pytest.fixture
def no_license(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


# --- validity --------------------------------------------------------------------


def test_valid_token_grants_professional(no_license, trusted_test_key, monkeypatch):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload()))
    assert lic.granted_tier() == "professional"
    assert lic.has_license() is True
    lic.require_license()


def test_golden_token_validates(no_license, trusted_test_key, monkeypatch):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", GOLDEN_TOKEN)
    status = lic.license_status()
    assert status.error is None
    assert status.tier == "professional"
    assert status.holder == "Golden Test"
    assert status.license_id == "SF-TEST-0001"
    assert status.issued == "2026-07-13"
    assert status.expires is None
    assert status.source == "env"


def test_file_channel_and_env_precedence(no_license, trusted_test_key, monkeypatch):
    path = lic.license_file()
    path.parent.mkdir(parents=True)
    path.write_text(mint(payload(tier="enterprise")) + "\n")
    assert lic.granted_tier() == "enterprise"
    assert lic.license_status().source == "file"
    # env wins over the file, exactly like the stub did
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload()))
    assert lic.granted_tier() == "professional"
    assert lic.license_status().source == "env"


def test_empty_license_file_is_missing(no_license):
    path = lic.license_file()
    path.parent.mkdir(parents=True)
    path.write_text("\n")
    assert lic.has_license() is False
    assert lic.license_status().error == "missing"


# --- tiers -----------------------------------------------------------------------


def test_enterprise_token_grants_both_tiers(no_license, trusted_test_key, monkeypatch):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload(tier="enterprise")))
    assert lic.granted_tier() == "enterprise"
    assert lic.authorize("professional") is True
    assert lic.authorize("enterprise") is True


def test_professional_token_does_not_grant_enterprise(
    no_license, trusted_test_key, monkeypatch
):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload()))
    assert lic.authorize("enterprise") is False
    assert lic.authorize.hint == (  # pyright: ignore[reportFunctionMemberAccess]
        "Installed license grants 'professional', but this feature requires 'enterprise'."
    )


def test_unknown_required_tier_is_denied(no_license, trusted_test_key, monkeypatch):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload(tier="enterprise")))
    assert lic.authorize("galactic") is False


def test_unknown_granted_tier_is_invalid(no_license, trusted_test_key, monkeypatch):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload(tier="platinum")))
    assert lic.granted_tier() is None
    assert "unknown tier" in (lic.license_status().error or "")


def test_require_license_enterprise_tier(no_license, trusted_test_key, monkeypatch):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload()))
    lic.require_license()  # professional default — passes
    with pytest.raises(PluginError, match="requires 'enterprise'"):
        lic.require_license("enterprise")


# --- expiry ----------------------------------------------------------------------


def test_expired_token_is_denied_with_date(no_license, trusted_test_key, monkeypatch):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload(expires="2020-01-01")))
    assert lic.granted_tier() is None
    assert lic.authorize("professional") is False
    assert lic.authorize.hint == (  # pyright: ignore[reportFunctionMemberAccess]
        "Professional license expired on 2020-01-01. Renew to restore access."
    )
    with pytest.raises(PluginError, match="expired on 2020-01-01"):
        lic.require_license()


def test_expiry_day_itself_is_still_valid(no_license, trusted_test_key, monkeypatch):
    today = datetime.now(timezone.utc).date().isoformat()
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload(expires=today)))
    assert lic.granted_tier() == "professional"


def test_future_expiry_is_valid_and_reported(no_license, trusted_test_key, monkeypatch):
    future = (datetime.now(timezone.utc).date() + timedelta(days=30)).isoformat()
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload(expires=future)))
    status = lic.license_status()
    assert status.error is None
    assert status.expires == date.fromisoformat(future)


# --- invalid tokens --------------------------------------------------------------


def test_tampered_payload_is_rejected(no_license, trusted_test_key, monkeypatch):
    prefix, payload_b64, sig_b64 = mint(payload()).split(".")
    forged = json.loads(base64.urlsafe_b64decode(payload_b64))
    forged["tier"] = "enterprise"
    forged_b64 = base64.urlsafe_b64encode(
        json.dumps(forged, separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", f"{prefix}.{forged_b64}.{sig_b64}")
    assert lic.granted_tier() is None
    assert lic.license_status().error == "signature check failed"


def test_foreign_key_is_rejected(no_license, trusted_test_key, monkeypatch):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload(), "00" * 32))
    assert lic.granted_tier() is None
    assert lic.license_status().error == "signature check failed"


def test_wrong_length_signature_is_rejected(no_license, trusted_test_key, monkeypatch):
    # cryptography raises InvalidSignature for wrong-length signatures too; lock
    # that a structurally valid token with a short signature segment denies cleanly.
    prefix, payload_b64, _ = mint(payload()).split(".")
    short_sig = base64.urlsafe_b64encode(b"short").decode()
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", f"{prefix}.{payload_b64}.{short_sig}")
    assert lic.granted_tier() is None
    assert lic.license_status().error == "signature check failed"


def test_unsupported_schema_version(no_license, trusted_test_key, monkeypatch):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload(v=2)))
    assert lic.license_status().error == "unsupported schema version"


@pytest.mark.parametrize(
    "value",
    [
        "local-dev-license",  # the old stub value — Breaking on purpose (M3)
        "SFL1.only-two-parts",
        "SFL2.eyJ2IjoxfQ==.c2ln",
        "SFL1.!!notbase64!!.c2ln",
    ],
)
def test_malformed_values_are_invalid(no_license, trusted_test_key, monkeypatch, value):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", value)
    assert lic.granted_tier() is None
    assert lic.license_status().error == "malformed token"
    assert lic.authorize("professional") is False
    assert lic.authorize.hint == (  # pyright: ignore[reportFunctionMemberAccess]
        "Installed Professional license is invalid (malformed token). "
        "Reinstall the license or contact your vendor."
    )


# --- absence: frozen strings and ground state -------------------------------------


def test_license_missing_by_default(no_license):
    assert lic.has_license() is False
    with pytest.raises(PluginError, match="Professional license required"):
        lic.require_license()


def test_missing_license_message_is_frozen(no_license):
    with pytest.raises(PluginError) as exc:
        lic.require_license()
    assert str(exc.value) == lic.LICENSE_ERROR
    assert lic.LICENSE_ERROR == (
        "Professional license required. Set SURROFORGE_PRO_LICENSE "
        "or install ~/.surroforge/professional.lic."
    )


def test_authorize_hint_ground_state_is_the_frozen_error(no_license):
    # order-independent: an absence call resets the (mutable) hint attribute
    lic.authorize("professional")
    assert lic.authorize.hint == lic.LICENSE_ERROR  # pyright: ignore[reportFunctionMemberAccess]


def test_reporter_checks_license_before_reading_run(no_license):
    from surroforge_professional.reports.html import HtmlReporter

    with pytest.raises(PluginError, match="Professional license required"):
        HtmlReporter().render(object())


def test_authorizer_entry_point_is_registered():
    import importlib.metadata as md

    eps = {ep.name: ep.value for ep in md.entry_points(group="surroforge.authorizers")}
    assert eps.get("professional") == "surroforge_professional.license:authorize"
