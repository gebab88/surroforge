"""SurroForge license-token format: keygen and mint (vendor side).

Single source of truth for the SFL1 token layout, shared by the issuance CLI
(``issue.py``) and the test fixtures (which path-import this module to mint
throwaway licenses). The customer-side *verification* lives independently in
``surroforge_professional.license`` — it must stay self-contained because the
shipped package never sees this directory.

Token layout::

    SFL1.<base64url(payload-json)>.<base64url(ed25519-signature)>

The signature covers the raw payload-JSON bytes exactly as encoded in the
token, so the verify side needs no canonicalisation step.
"""

from __future__ import annotations

import base64
import json

TOKEN_PREFIX = "SFL1"


def generate_keypair() -> tuple[str, str]:
    """A fresh Ed25519 keypair as ``(private_seed_hex, public_key_hex)``."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return seed.hex(), pub.hex()


def mint(payload: dict, private_seed_hex: str) -> str:
    """Sign a payload dict into a single-line SFL1 license token."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_seed_hex))
    b64 = base64.urlsafe_b64encode
    return f"{TOKEN_PREFIX}.{b64(raw).decode()}.{b64(key.sign(raw)).decode()}"


def decode(token: str) -> dict:
    """The payload dict of a token — NO signature check (inspection helper)."""
    parts = token.strip().split(".")
    if len(parts) != 3 or parts[0] != TOKEN_PREFIX:
        raise ValueError("not an SFL1 license token")
    return json.loads(base64.urlsafe_b64decode(parts[1]))
