"""`surroforge license status|install` (M3)."""

from __future__ import annotations

import base64
import json

import pytest

pytest.importorskip("cryptography")

from typer.testing import CliRunner

from surroforge_professional import license as lic
from surroforge_professional.commands.license import license_app

# Same fixed test keypair as test_license.py (suites cannot cross-import test modules).
TEST_SEED_HEX = "737572726f666f7267652d6d332d746573742d6b6579706169722d3030303121"
TEST_PUB_HEX = "f96638d3d06b9b93ba477836a72d7477ca2d15730d6144f5a9c0a11f649386f0"

runner = CliRunner()


def mint(payload: dict, seed_hex: str = TEST_SEED_HEX) -> str:
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


def test_status_with_valid_license(no_license, trusted_test_key, monkeypatch):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload()))
    result = runner.invoke(license_app, ["status"])
    assert result.exit_code == 0
    assert "Source:  env" in result.output
    assert "Holder:  Test GmbH" in result.output
    assert "Tier:    professional" in result.output
    assert "Expires: never" in result.output
    assert "Status:  valid" in result.output


def test_status_without_license_exits_nonzero(no_license):
    result = runner.invoke(license_app, ["status"])
    assert result.exit_code == 1
    assert "No license installed." in result.output
    assert "SURROFORGE_PRO_LICENSE" in result.output


def test_status_with_expired_license(no_license, trusted_test_key, monkeypatch):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload(expires="2020-01-01")))
    result = runner.invoke(license_app, ["status"])
    assert result.exit_code == 1
    assert "Expires: 2020-01-01" in result.output
    assert "Status:  expired" in result.output


def test_install_token_writes_file_and_grants(no_license, trusted_test_key):
    token = mint(payload())
    result = runner.invoke(license_app, ["install", token])
    assert result.exit_code == 0
    assert lic.license_file().read_text().strip() == token
    assert lic.granted_tier() == "professional"


def test_install_from_file_path(no_license, trusted_test_key, tmp_path):
    src = tmp_path / "customer.lic"
    src.write_text(mint(payload(tier="enterprise")) + "\n")
    result = runner.invoke(license_app, ["install", str(src)])
    assert result.exit_code == 0
    assert lic.granted_tier() == "enterprise"


def test_install_rejects_invalid_token(no_license, trusted_test_key):
    result = runner.invoke(license_app, ["install", "local-dev-license"])
    assert result.exit_code == 1
    assert not lic.license_file().exists()


def test_install_warns_when_env_takes_precedence(no_license, trusted_test_key, monkeypatch):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", mint(payload()))
    result = runner.invoke(license_app, ["install", mint(payload())])
    assert result.exit_code == 0
    assert "takes precedence" in result.output


def test_command_entry_point_is_registered():
    import importlib.metadata as md

    eps = {ep.name: ep.value for ep in md.entry_points(group="surroforge.commands")}
    assert eps.get("license") == "surroforge_professional.commands.license:license_app"


def test_install_accepts_token_longer_than_path_limit(no_license, trusted_test_key):
    # A single path component >255 bytes makes Path.is_file() raise ENAMETOOLONG;
    # the token must still install instead of crashing (it is a token, not a path).
    token = mint(payload(holder="H" * 300))
    assert len(token) > 255
    result = runner.invoke(license_app, ["install", token])
    assert result.exit_code == 0
    assert lic.license_file().read_text().strip() == token
