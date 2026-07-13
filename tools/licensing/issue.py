#!/usr/bin/env python3
"""SurroForge vendor licensing CLI: keygen, issue, inspect.

Vendor-side only — this directory is never packaged or shipped. Requires
``cryptography`` (present in the dev environment as a professional
dependency). The private signing key lives OUTSIDE the repository::

    ~/.surroforge-vendor/signing.key      (hex seed, chmod 600)

BACK THAT FILE UP: licenses are verified against the public key embedded in
``surroforge_professional/license.py`` (``_TRUSTED_KEYS``); losing the private
key means no new licenses can be issued for existing installations (recovery =
key rotation via a new professional release).

Usage::

    python tools/licensing/issue.py keygen [--key PATH] [--force]
    python tools/licensing/issue.py issue --holder "ACME GmbH"
        [--tier professional|enterprise] [--expires YYYY-MM-DD] [--id SF-...]
        [--key PATH]
    python tools/licensing/issue.py inspect TOKEN_OR_PATH [--pub HEX]
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling import, any cwd
import sflicense  # noqa: E402

DEFAULT_KEY = Path("~/.surroforge-vendor/signing.key")


def cmd_keygen(args: argparse.Namespace) -> int:
    key_path = Path(args.key).expanduser()
    if key_path.exists() and not args.force:
        print(f"Refusing to overwrite {key_path} (use --force).", file=sys.stderr)
        return 1
    seed_hex, pub_hex = sflicense.generate_keypair()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.touch(mode=0o600, exist_ok=True)
    key_path.chmod(0o600)
    key_path.write_text(seed_hex + "\n")
    print(f"Private signing key: {key_path}  (BACK THIS FILE UP)")
    print(f"Public key for _TRUSTED_KEYS: {pub_hex}")
    return 0


def cmd_issue(args: argparse.Namespace) -> int:
    key_path = Path(args.key).expanduser()
    if not key_path.is_file():
        print(f"Signing key not found: {key_path} (run keygen first).", file=sys.stderr)
        return 1
    payload = {
        "v": 1,
        "id": args.id or f"SF-{_dt.date.today():%Y%m%d}-{secrets.token_hex(2)}",
        "holder": args.holder,
        "tier": args.tier,
        "issued": _dt.date.today().isoformat(),
    }
    if args.expires:
        _dt.date.fromisoformat(args.expires)  # fail early on a bad date
        payload["expires"] = args.expires
    print(sflicense.mint(payload, key_path.read_text().strip()))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    candidate = Path(args.token).expanduser()
    token = candidate.read_text().strip() if candidate.is_file() else args.token.strip()
    try:
        payload = sflicense.decode(token)
    except ValueError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.pub:
        keys: tuple[str, ...] = (args.pub,)
    else:
        try:
            from surroforge_professional.license import _TRUSTED_KEYS as keys
        except ImportError:
            print("verdict: UNVERIFIED (professional package has no trusted keys; pass --pub)")
            return 1

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    _, payload_b64, sig_b64 = token.split(".")
    raw = base64.urlsafe_b64decode(payload_b64)
    sig = base64.urlsafe_b64decode(sig_b64)
    for key_hex in keys:
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(key_hex)).verify(sig, raw)
        except InvalidSignature:
            continue
        print("verdict: VALID signature (expiry not checked here — see the payload)")
        return 0
    print("verdict: INVALID signature", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="issue.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_keygen = sub.add_parser("keygen", help="generate the vendor signing keypair")
    p_keygen.add_argument("--key", default=str(DEFAULT_KEY))
    p_keygen.add_argument("--force", action="store_true")
    p_keygen.set_defaults(func=cmd_keygen)

    p_issue = sub.add_parser("issue", help="sign a customer license token")
    p_issue.add_argument("--holder", required=True)
    p_issue.add_argument("--tier", choices=("professional", "enterprise"),
                         default="professional")
    p_issue.add_argument("--expires", default=None, help="YYYY-MM-DD (omit = perpetual)")
    p_issue.add_argument("--id", default=None)
    p_issue.add_argument("--key", default=str(DEFAULT_KEY))
    p_issue.set_defaults(func=cmd_issue)

    p_inspect = sub.add_parser("inspect", help="decode + verify a token or .lic file")
    p_inspect.add_argument("token")
    p_inspect.add_argument("--pub", default=None, help="hex public key override")
    p_inspect.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
