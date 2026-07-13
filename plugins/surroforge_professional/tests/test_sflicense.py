"""The vendor-side token library (tools/licensing) — format round-trips."""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

_TOOLS = Path(__file__).resolve().parents[3] / "tools" / "licensing"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sfl = _load("sflicense")

PAYLOAD = {
    "v": 1,
    "id": "SF-TEST-0001",
    "holder": "Test GmbH",
    "tier": "professional",
    "issued": "2026-07-13",
}


def test_mint_produces_verifiable_token():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    seed_hex, pub_hex = sfl.generate_keypair()
    token = sfl.mint(PAYLOAD, seed_hex)
    prefix, payload_b64, sig_b64 = token.split(".")
    assert prefix == "SFL1"
    # raises InvalidSignature if the signature does not cover the payload bytes
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex)).verify(
        base64.urlsafe_b64decode(sig_b64), base64.urlsafe_b64decode(payload_b64)
    )


def test_decode_round_trips_payload():
    seed_hex, _ = sfl.generate_keypair()
    assert sfl.decode(sfl.mint(PAYLOAD, seed_hex)) == PAYLOAD


def test_mint_is_deterministic():
    # Ed25519 signing is deterministic (RFC 8032): same seed + payload = same token.
    seed_hex, _ = sfl.generate_keypair()
    assert sfl.mint(PAYLOAD, seed_hex) == sfl.mint(PAYLOAD, seed_hex)


def test_decode_rejects_foreign_strings():
    with pytest.raises(ValueError):
        sfl.decode("local-dev-license")


# --- vendor CLI (issue.py) ------------------------------------------------------

issue = _load("issue")


def test_keygen_issue_inspect_round_trip(tmp_path, capsys):
    key_file = tmp_path / "signing.key"
    assert issue.main(["keygen", "--key", str(key_file)]) == 0
    assert key_file.stat().st_mode & 0o777 == 0o600
    pub_hex = capsys.readouterr().out.split()[-1]

    argv = ["issue", "--holder", "ACME GmbH", "--tier", "professional",
            "--expires", "2099-01-01", "--key", str(key_file)]
    assert issue.main(argv) == 0
    token = capsys.readouterr().out.strip()
    payload = sfl.decode(token)
    assert payload["holder"] == "ACME GmbH"
    assert payload["tier"] == "professional"
    assert payload["expires"] == "2099-01-01"
    assert payload["v"] == 1 and payload["id"].startswith("SF-")

    assert issue.main(["inspect", token, "--pub", pub_hex]) == 0
    assert "VALID" in capsys.readouterr().out


def test_keygen_refuses_to_overwrite(tmp_path, capsys):
    key_file = tmp_path / "signing.key"
    assert issue.main(["keygen", "--key", str(key_file)]) == 0
    capsys.readouterr()
    assert issue.main(["keygen", "--key", str(key_file)]) == 1
    assert issue.main(["keygen", "--key", str(key_file), "--force"]) == 0


def test_inspect_flags_wrong_key(tmp_path, capsys):
    key_file = tmp_path / "signing.key"
    assert issue.main(["keygen", "--key", str(key_file)]) == 0
    capsys.readouterr()
    assert issue.main(["issue", "--holder", "X", "--key", str(key_file)]) == 0
    token = capsys.readouterr().out.strip()
    _, wrong_pub = sfl.generate_keypair()
    assert issue.main(["inspect", token, "--pub", wrong_pub]) == 1


# --- cross-check: tool-minted tokens validate in the shipped package -------------


def test_tool_minted_token_validates_in_package(monkeypatch):
    lic = pytest.importorskip("surroforge_professional.license")
    seed_hex, pub_hex = sfl.generate_keypair()
    monkeypatch.setattr(lic, "_TRUSTED_KEYS", (pub_hex,))
    status = lic.validate_token(sfl.mint(PAYLOAD, seed_hex))
    assert status.error is None
    assert status.tier == "professional"
