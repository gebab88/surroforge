# M3: Echte Lizenz-Validierung — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Der Wave-1-Lizenz-Stub (prüft nur *Vorhandensein* eines Werts) wird durch echte Inhalts-Validierung ersetzt: eine Lizenz ist ein Ed25519-signierter `SFL1.…`-Token (Inhaber, Tier, optionales Ablaufdatum), offline geprüft gegen einen im Professional-Paket eingebetteten Vendor-Public-Key.

**Architecture:** Die gesamte Änderung liegt im Professional-Paket (`license.py`, neuer CLI-Befehl `surroforge license`), im neuen Vendor-Tooling `tools/licensing/` und in den Test-Fixtures — **`src/surroforge/` (Core) wird nicht angefasst**; das W1–W3-Gating hängt nur am Authorizer-Vertrag (`authorize(tier) -> bool` + `hint`-Attribut), der stabil bleibt. Spec: `docs/superpowers/specs/2026-07-13-editions-m3-license-design.md`.

**Tech Stack:** Python ≥3.10, `cryptography>=42` (Ed25519), Typer (CLI), pytest, argparse (Vendor-Tool).

## Global Constraints

- **Eingefrorener String** (Test asserted `==`), byte-identisch erhalten: `LICENSE_ERROR = "Professional license required. Set SURROFORGE_PRO_LICENSE or install ~/.surroforge/professional.lic."` (im Code als zweizeilige Klammer-Konkatenation wie heute).
- **W1-Gate-Meldung byte-identisch**: `'{name}' requires a '{tier}' license, and no installed SurroForge edition grants it. {hints}` (rstrip) — `src/surroforge/plugins/gate.py` wird NICHT geändert.
- **Stabile öffentliche API** von `surroforge_professional.license`: `authorize(required_tier) -> bool` mit `hint`-Attribut, `granted_tier() -> str | None`, `has_license() -> bool`, `require_license(required_tier="professional")`, `license_file() -> Path`, Konstanten `LICENSE_ENV`, `LICENSE_FILE`, `LICENSE_ERROR`.
- Alle Befehle über `.venv/bin/…` (System-Python ist 3.9; das venv hat 3.13).
- `git add` nur mit expliziten Pfaden; direkt auf `main` committen; **kein `Co-Authored-By:`-Trailer**.
- Kein Push (CLAUDE.md: erst Commits und geänderte Dateien zeigen).
- Der **private Produktions-Signierschlüssel** landet NIE im Repo — er lebt in `~/.surroforge-vendor/signing.key`.
- Zeilenlänge 100 (ruff, repo-weit); pyright `include = ["src", "plugins"]` — Plugin-Code muss pyright-sauber sein; `tools/` wird nicht gelintet, trotzdem sauber schreiben.
- Fixed-Test-Keypair (überall identisch; Seed = `b"surroforge-m3-test-keypair-0001!"`):
  - `TEST_SEED_HEX = "737572726f666f7267652d6d332d746573742d6b6579706169722d3030303121"`
  - `TEST_PUB_HEX = "f96638d3d06b9b93ba477836a72d7477ca2d15730d6144f5a9c0a11f649386f0"`

**Brücken-Eigenschaft (Task-Reihenfolge):** Der Stub akzeptiert jeden nicht-leeren Wert. Ein geprägter `SFL1.…`-Token ist nicht-leer → **Tasks 1–3 laufen unter dem Stub grün** und bleiben nach Task 4 (echte Validierung) grün. Nur Enterprise-Tier-Fixtures können nicht vorgezogen werden (der Stub erkennt „enterprise" per Substring, im Token ist das base64-kodiert) — sie kippen in Task 4 mit um.

---

### Task 1: `tools/licensing/sflicense.py` — Token-Format-Bibliothek

**Files:**
- Create: `tools/licensing/sflicense.py`
- Test: `plugins/surroforge_professional/tests/test_sflicense.py`

**Interfaces:**
- Consumes: nichts (reine Neuanlage; `cryptography` ist im venv installiert).
- Produces: `TOKEN_PREFIX = "SFL1"`, `generate_keypair() -> tuple[str, str]` (private_seed_hex, public_key_hex), `mint(payload: dict, private_seed_hex: str) -> str`, `decode(token: str) -> dict` (ohne Signaturprüfung; wirft `ValueError` bei Fremdformat). Tasks 2/3/4/6 path-importieren dieses Modul.

- [ ] **Step 1: Failing Test schreiben**

`plugins/surroforge_professional/tests/test_sflicense.py` (neu, komplett):

```python
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
```

- [ ] **Step 2: Test läuft rot**

Run: `.venv/bin/python -m pytest plugins/surroforge_professional/tests/test_sflicense.py -q`
Expected: FAIL/ERROR (FileNotFoundError — `tools/licensing/sflicense.py` existiert nicht)

- [ ] **Step 3: Implementierung**

`tools/licensing/sflicense.py` (neu, komplett):

```python
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
```

- [ ] **Step 4: Tests grün**

Run: `.venv/bin/python -m pytest plugins/surroforge_professional/tests/test_sflicense.py -q`
Expected: `4 passed`

- [ ] **Step 5: Lint + Commit**

```bash
.venv/bin/ruff check tools/licensing/sflicense.py plugins/surroforge_professional/tests/test_sflicense.py
git add tools/licensing/sflicense.py plugins/surroforge_professional/tests/test_sflicense.py
git commit -m "feat(editions): SFL1 license-token library (vendor side, M3)"
```

---

### Task 2: Grant-Fixtures prägen echte Tokens (Brücke unter dem Stub)

Alle Fixtures/Stellen, die eine Lizenz *gewähren*, stellen von Roh-Strings auf geprägte `SFL1`-Tokens um. Unter dem Stub (nicht-leerer Wert) bleiben sie grün; nach Task 4 validieren sie echt. `monkeypatch.setattr(..., raising=False)` überbrückt, dass `_TRUSTED_KEYS` erst in Task 4 existiert. **Unlizenziert-Tests (delenv + HOME) bleiben überall unangetastet.**

**Files:**
- Modify: `tests/conftest.py` (Fixture `licensed_export`, Zeile 76–84; neuer `license_kit`)
- Modify: `plugins/surroforge_professional/tests/conftest.py` (Fixture `licensed`, Zeile 12–15; neuer `license_kit`)
- Modify: `tests/test_examples.py` (Env-Injektion Zeile 29–41 → Fixture mit sitecustomize)

**Interfaces:**
- Consumes: `sflicense.generate_keypair()`, `sflicense.mint(payload, seed_hex)` aus Task 1 (path-import).
- Produces: Session-Fixture `license_kit` mit `.mint(tier="professional", **overrides) -> str`, `.pub_hex`, `.seed_hex` — in Task 4 (Enterprise-Conftest kopiert das Muster) und den bestehenden `licensed`/`licensed_export`-Konsumenten verwendet.

- [ ] **Step 1: `plugins/surroforge_professional/tests/conftest.py` — Kopf ergänzen und `licensed` ersetzen**

Nach den bestehenden Imports (Zeile 1–9) einfügen:

```python
import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SFLICENSE = Path(__file__).resolve().parents[3] / "tools" / "licensing" / "sflicense.py"


def _load_sflicense():
    spec = importlib.util.spec_from_file_location("sflicense", _SFLICENSE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def license_kit():
    """A throwaway signing keypair + mint helper for real-validation tests (M3).

    The public key is NOT trusted by the shipped package; grant fixtures
    monkeypatch it into ``license._TRUSTED_KEYS``.
    """
    pytest.importorskip("cryptography")
    sfl = _load_sflicense()
    seed_hex, pub_hex = sfl.generate_keypair()

    def mint(tier: str = "professional", **overrides) -> str:
        payload = {
            "v": 1,
            "id": "SF-TEST-LOCAL",
            "holder": "test-suite",
            "tier": tier,
            "issued": "2026-07-13",
            **overrides,
        }
        return sfl.mint(payload, seed_hex)

    return SimpleNamespace(mint=mint, pub_hex=pub_hex, seed_hex=seed_hex)
```

Das bestehende `licensed`-Fixture (Zeile 12–15) ersetzen durch:

```python
@pytest.fixture
def licensed(monkeypatch, tmp_path, license_kit):
    from surroforge_professional import license as _license

    # raising=False bridges the stub era (no _TRUSTED_KEYS yet) and M3.
    monkeypatch.setattr(_license, "_TRUSTED_KEYS", (license_kit.pub_hex,), raising=False)
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", license_kit.mint())
    monkeypatch.setenv("HOME", str(tmp_path))
```

- [ ] **Step 2: `tests/conftest.py` — gleiche Helfer, `licensed_export` ersetzen**

Nach den Imports (Zeile 1–9) denselben `_load_sflicense`-Block und dasselbe `license_kit`-Fixture wie in Step 1 einfügen — mit **einem** Pfad-Unterschied:

```python
_SFLICENSE = Path(__file__).resolve().parents[1] / "tools" / "licensing" / "sflicense.py"
```

(plus `import importlib.util`, `from pathlib import Path`, `from types import SimpleNamespace`). Das bestehende `licensed_export`-Fixture (Zeile 76–84) ersetzen durch:

```python
@pytest.fixture
def licensed_export(monkeypatch, license_kit):
    """Grant a real (test-signed) Professional licence for export tests (W2→M3).

    Skips cleanly on a Core-only checkout: ``license_kit`` skips without
    ``cryptography`` (a professional dependency), and without
    ``surroforge_professional`` there is no exporter to license.
    """
    pytest.importorskip("surroforge_professional")
    from surroforge_professional import license as _license

    monkeypatch.setattr(_license, "_TRUSTED_KEYS", (license_kit.pub_hex,), raising=False)
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", license_kit.mint())
```

- [ ] **Step 3: `tests/test_examples.py` — Subprozess-Injektion**

`import importlib.util` zu den Imports ergänzen. Den Kommentar+`env`-Bau (Zeile 30–34) aus `test_example_runs_to_completion` entfernen und ERSETZEN; neues Session-Fixture vor dem Test einfügen:

```python
_TOOLS_SFLICENSE = _REPO_ROOT / "tools" / "licensing" / "sflicense.py"


@pytest.fixture(scope="session")
def example_license_env(tmp_path_factory) -> dict[str, str]:
    """License env for the example subprocesses (M3: content validation).

    Examples that exercise Professional methods need a *valid* license in the
    child process. Monkeypatching does not cross the process boundary, so this
    mints a token from a throwaway keypair and injects a ``sitecustomize.py``
    via PYTHONPATH that makes the child trust that keypair. Empty dict (no-op)
    when the professional package is not installed. It is not a statement about
    what a fresh install can run out of the box.
    """
    try:
        import cryptography  # noqa: F401  (professional dependency)
        import surroforge_professional  # noqa: F401
    except ImportError:
        return {}
    spec = importlib.util.spec_from_file_location("sflicense", _TOOLS_SFLICENSE)
    assert spec is not None and spec.loader is not None
    sfl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sfl)
    seed_hex, pub_hex = sfl.generate_keypair()
    token = sfl.mint(
        {
            "v": 1,
            "id": "SF-TEST-EXAMPLES",
            "holder": "example-smoke-test",
            "tier": "professional",
            "issued": "2026-07-13",
        },
        seed_hex,
    )
    site_dir = tmp_path_factory.mktemp("license-site")
    (site_dir / "sitecustomize.py").write_text(
        "# Injected by tests/test_examples.py: trust the throwaway example-license key.\n"
        "try:\n"
        "    from surroforge_professional import license as _license\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        f"    _license._TRUSTED_KEYS = ({pub_hex!r},)\n"
    )
    pythonpath = os.pathsep.join(
        p for p in (str(site_dir), os.environ.get("PYTHONPATH", "")) if p
    )
    return {"SURROFORGE_PRO_LICENSE": token, "PYTHONPATH": pythonpath}


@pytest.mark.parametrize("script", _EXAMPLES, ids=lambda p: p.stem)
def test_example_runs_to_completion(script: Path, example_license_env):
    env = {**os.environ, **example_license_env}
```

(Der Rest des Tests — `subprocess.run(...)` und die Assertion — bleibt unverändert.)

- [ ] **Step 4: Alles grün unter dem Stub**

Run: `.venv/bin/python -m pytest plugins/surroforge_professional/tests -q && .venv/bin/python -m pytest tests -q`
Expected: beide PASS (Brücke: Token ist nicht-leer → Stub gewährt; `raising=False` toleriert das fehlende `_TRUSTED_KEYS`)

- [ ] **Step 5: Lint + Commit**

```bash
.venv/bin/ruff check tests/conftest.py tests/test_examples.py
git add tests/conftest.py tests/test_examples.py plugins/surroforge_professional/tests/conftest.py
git commit -m "test(editions): grant fixtures mint real SFL1 tokens (M3 bridge, green under the stub)"
```

---

### Task 3: `tools/licensing/issue.py` — Vendor-CLI (keygen / issue / inspect) + README

**Files:**
- Create: `tools/licensing/issue.py`
- Create: `tools/licensing/README.md`
- Test: `plugins/surroforge_professional/tests/test_sflicense.py` (Tests anhängen)

**Interfaces:**
- Consumes: `sflicense` (Sibling-Import; Task 1).
- Produces: `main(argv: list[str] | None = None) -> int` mit Subcommands `keygen`/`issue`/`inspect`; `DEFAULT_KEY = Path("~/.surroforge-vendor/signing.key")`. Task 4 nutzt `keygen` für den Produktionsschlüssel; Task 6 verlinkt das README.

- [ ] **Step 1: Failing Tests anhängen**

An `plugins/surroforge_professional/tests/test_sflicense.py` anhängen:

```python
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
```

- [ ] **Step 2: Tests laufen rot**

Run: `.venv/bin/python -m pytest plugins/surroforge_professional/tests/test_sflicense.py -q`
Expected: ERROR bei Collection (`issue.py` existiert nicht)

- [ ] **Step 3: Implementierung**

`tools/licensing/issue.py` (neu, komplett):

```python
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
```

`tools/licensing/README.md` (neu, komplett):

```markdown
# SurroForge licensing (vendor side)

Issues the signed `SFL1.…` license tokens that `surroforge_professional`
validates offline (M3). This directory is dev-tooling — it is never packaged.

## One-time setup

    .venv/bin/python tools/licensing/issue.py keygen

Writes the private signing key to `~/.surroforge-vendor/signing.key`
(hex seed, chmod 600) and prints the public key that is embedded in
`surroforge_professional/license.py` (`_TRUSTED_KEYS`).

**BACK UP THE PRIVATE KEY.** It never enters the repository. If it is lost, no
new licenses can be issued for existing installations — recovery means key
rotation: generate a new pair, append the new public key to `_TRUSTED_KEYS`,
and ship a new professional release.

## Issue a license

    .venv/bin/python tools/licensing/issue.py issue --holder "ACME GmbH" \
        --tier professional --expires 2027-07-13

Prints the token (one line). Omit `--expires` for a perpetual license. The
customer activates it with `surroforge license install <token>` (or the
`SURROFORGE_PRO_LICENSE` env var, which takes precedence).

## Inspect a token

    .venv/bin/python tools/licensing/issue.py inspect <token-or-.lic-file>

Decodes the payload and checks the signature against the installed package's
trusted keys (`--pub HEX` to override). Expiry is judged customer-side by
`surroforge license status`.
```

- [ ] **Step 4: Tests grün**

Run: `.venv/bin/python -m pytest plugins/surroforge_professional/tests/test_sflicense.py -q`
Expected: `7 passed`

- [ ] **Step 5: Lint + Commit**

```bash
.venv/bin/ruff check tools/licensing/issue.py plugins/surroforge_professional/tests/test_sflicense.py
git add tools/licensing/issue.py tools/licensing/README.md plugins/surroforge_professional/tests/test_sflicense.py
git commit -m "feat(editions): vendor licensing CLI — keygen/issue/inspect (M3)"
```

---

### Task 4: `license.py` — echte Inhalts-Validierung (der Umschalt-Task)

Der Kern von M3: `license.py` validiert Token-Inhalt statt Vorhandensein; Produktionsschlüssel wird erzeugt und eingebettet; `test_license.py` wird zur Validierungs-Suite; die Enterprise-Fixtures kippen mit um (der Stub konnte ihr Tier nicht aus dem Token lesen, siehe Brücken-Notiz oben).

**Files:**
- Modify: `plugins/surroforge_professional/src/surroforge_professional/license.py` (kompletter Neuinhalt)
- Modify: `plugins/surroforge_professional/pyproject.toml:14-19` (dependencies)
- Rewrite: `plugins/surroforge_professional/tests/test_license.py`
- Modify: `plugins/surroforge_enterprise/tests/conftest.py` (komplett) und `plugins/surroforge_enterprise/tests/test_enterprise_gating.py:11-15`
- Test (ergänzen): `plugins/surroforge_professional/tests/test_sflicense.py` (Kreuz-Test Tool↔Paket)

**Interfaces:**
- Consumes: `license_kit`-Muster aus Task 2; `issue.py keygen` aus Task 3.
- Produces (neu, paketintern — Task 5 nutzt sie): `LicenseStatus` (frozen dataclass: `tier`, `holder`, `license_id`, `issued: str | None`, `expires: date | None`, `source: str | None`, `error: str | None`; `error is None` ⇔ gültig), `license_status() -> LicenseStatus`, `validate_token(token: str) -> LicenseStatus`, `_TRUSTED_KEYS: tuple[str, ...]`. Öffentliche API unverändert (Global Constraints).

- [ ] **Step 1: `test_license.py` komplett neu schreiben (rot unter dem Stub)**

`plugins/surroforge_professional/tests/test_license.py` — gesamten Inhalt ersetzen:

```python
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
    assert lic.authorize.hint == (
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
    assert lic.authorize.hint == (
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
    assert lic.authorize.hint == (
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
    assert lic.authorize.hint == lic.LICENSE_ERROR


def test_reporter_checks_license_before_reading_run(no_license):
    from surroforge_professional.reports.html import HtmlReporter

    with pytest.raises(PluginError, match="Professional license required"):
        HtmlReporter().render(object())


def test_authorizer_entry_point_is_registered():
    import importlib.metadata as md

    eps = {ep.name: ep.value for ep in md.entry_points(group="surroforge.authorizers")}
    assert eps.get("professional") == "surroforge_professional.license:authorize"
```

Hinweis Zeilenumbrüche im `GOLDEN_TOKEN`: die vier String-Teile ergeben konkateniert exakt
`SFL1.eyJob2xkZXIiOiJHb2xkZW4gVGVzdCIsImlkIjoiU0YtVEVTVC0wMDAxIiwiaXNzdWVkIjoiMjAyNi0wNy0xMyIsInRpZXIiOiJwcm9mZXNzaW9uYWwiLCJ2IjoxfQ==.9VAgrVi4Fy64B24D88Lq-8XmuDL4dD7DxIV626CoFBRG7TDBv1rES4jzVUL3g_2XLK0rUdlva8wTVBuaWTfnCA==`
(Payload-Bytes: `{"holder":"Golden Test","id":"SF-TEST-0001","issued":"2026-07-13","tier":"professional","v":1}`).

- [ ] **Step 2: Rot verifizieren**

Run: `.venv/bin/python -m pytest plugins/surroforge_professional/tests/test_license.py -q`
Expected: FAIL (Stub kennt weder `license_status` noch `validate_token`; Stub-Grants akzeptieren `local-dev-license`)

- [ ] **Step 3: Produktionsschlüssel erzeugen (einmalig; NICHT ins Repo)**

```bash
test -f ~/.surroforge-vendor/signing.key && echo "key exists — wird wiederverwendet" \
  || .venv/bin/python tools/licensing/issue.py keygen
.venv/bin/python - <<'EOF'
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
seed = Path("~/.surroforge-vendor/signing.key").expanduser().read_text().strip()
key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed))
print("PROD_PUB_HEX:", key.public_key().public_bytes(
    serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex())
EOF
```

Expected: eine Zeile `PROD_PUB_HEX: <64 hex>` — diesen Wert in Step 4 in `_TRUSTED_KEYS` eintragen.

- [ ] **Step 4: `license.py` komplett ersetzen**

`plugins/surroforge_professional/src/surroforge_professional/license.py` — gesamten Inhalt ersetzen (`<PROD_PUB_HEX>` durch den Wert aus Step 3 ersetzen):

```python
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
    "<PROD_PUB_HEX>",
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
```

- [ ] **Step 5: `cryptography` als Pflicht-Dependency**

`plugins/surroforge_professional/pyproject.toml`, `dependencies` (Zeile 14–19) — `cryptography` ergänzen:

```toml
dependencies = [
    "surroforge",
    "cryptography>=42",
    "jinja2>=3.1",
    "numpy>=1.24",
    "scikit-learn>=1.3",
]
```

Dann Metadaten auffrischen: `.venv/bin/pip install -q -e ./plugins/surroforge_professional`

- [ ] **Step 6: Enterprise-Fixtures mit umkippen**

`plugins/surroforge_enterprise/tests/conftest.py` — gesamten Inhalt ersetzen:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SFLICENSE = Path(__file__).resolve().parents[3] / "tools" / "licensing" / "sflicense.py"


def _load_sflicense():
    spec = importlib.util.spec_from_file_location("sflicense", _SFLICENSE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def license_kit():
    """Throwaway signing keypair + mint helper (M3) — mirrors the professional conftest.

    Duplicated because each plugin suite is its own pytest rootdir and cannot
    import a sibling suite's conftest.
    """
    pytest.importorskip("cryptography")
    sfl = _load_sflicense()
    seed_hex, pub_hex = sfl.generate_keypair()

    def mint(tier: str = "professional", **overrides) -> str:
        payload = {
            "v": 1,
            "id": "SF-TEST-LOCAL",
            "holder": "test-suite",
            "tier": tier,
            "issued": "2026-07-13",
            **overrides,
        }
        return sfl.mint(payload, seed_hex)

    return SimpleNamespace(mint=mint, pub_hex=pub_hex, seed_hex=seed_hex)


@pytest.fixture
def licensed_enterprise(monkeypatch, tmp_path, license_kit):
    from surroforge_professional import license as _license

    monkeypatch.setattr(_license, "_TRUSTED_KEYS", (license_kit.pub_hex,))
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", license_kit.mint(tier="enterprise"))
    monkeypatch.setenv("HOME", str(tmp_path))
```

`plugins/surroforge_enterprise/tests/test_enterprise_gating.py` — den ersten Test (Zeile 11–15) ersetzen:

```python
def test_professional_license_does_not_unlock_autoencoder(monkeypatch, tmp_path, license_kit):
    from surroforge_professional import license as _license

    monkeypatch.setattr(_license, "_TRUSTED_KEYS", (license_kit.pub_hex,))
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", license_kit.mint())
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(PluginError, match="'enterprise' license"):
        get_trainer("autoencoder-field")
```

- [ ] **Step 7: Kreuz-Test Tool↔Paket anhängen**

An `plugins/surroforge_professional/tests/test_sflicense.py` anhängen:

```python
# --- cross-check: tool-minted tokens validate in the shipped package -------------


def test_tool_minted_token_validates_in_package(monkeypatch):
    lic = pytest.importorskip("surroforge_professional.license")
    seed_hex, pub_hex = sfl.generate_keypair()
    monkeypatch.setattr(lic, "_TRUSTED_KEYS", (pub_hex,))
    status = lic.validate_token(sfl.mint(PAYLOAD, seed_hex))
    assert status.error is None
    assert status.tier == "professional"
```

- [ ] **Step 8: Alle Suiten grün**

Run: `.venv/bin/python -m pytest plugins/surroforge_professional/tests plugins/surroforge_enterprise/tests -q && .venv/bin/python -m pytest tests -q`
Expected: alle PASS (inkl. `test_gating.py` — Absenz-Hints unverändert `LICENSE_ERROR`; `licensed`-Fixture prägt gültige Tokens; Beispiele laufen mit sitecustomize-Injektion)

- [ ] **Step 9: Live-E2E mit dem echten Produktionsschlüssel**

```bash
DEV_LICENSE="$(.venv/bin/python tools/licensing/issue.py issue --holder 'Georg Babij (dev)')"
SURROFORGE_PRO_LICENSE="$DEV_LICENSE" .venv/bin/python examples/03_reproduction.py
env -u SURROFORGE_PRO_LICENSE HOME=$(mktemp -d) .venv/bin/python examples/01_quickstart.py
```

Expected: Beispiel 03 läuft mit der echt signierten Dev-Lizenz durch (Reproduktions-Verdikt gedruckt); Beispiel 01 läuft OHNE Lizenz durch und druckt `Reproduction skipped: …` — beweist beide Seiten des echten Authorizers end-to-end.

- [ ] **Step 10: Lint/Typen + Commit**

```bash
.venv/bin/ruff check plugins/surroforge_professional plugins/surroforge_enterprise
.venv/bin/mypy src
.venv/bin/pyright
git add plugins/surroforge_professional/src/surroforge_professional/license.py \
        plugins/surroforge_professional/pyproject.toml \
        plugins/surroforge_professional/tests/test_license.py \
        plugins/surroforge_professional/tests/test_sflicense.py \
        plugins/surroforge_enterprise/tests/conftest.py \
        plugins/surroforge_enterprise/tests/test_enterprise_gating.py
git commit -m "feat(editions)!: license check validates signed SFL1 tokens (M3)"
```

---

### Task 5: CLI `surroforge license` — status / install

**Files:**
- Create: `plugins/surroforge_professional/src/surroforge_professional/commands/license.py`
- Modify: `plugins/surroforge_professional/pyproject.toml:41-42` (Entry-Point-Gruppe `surroforge.commands`)
- Test: `plugins/surroforge_professional/tests/test_license_cli.py`

**Interfaces:**
- Consumes: `LicenseStatus`, `license_status()`, `validate_token()`, `license_file()`, `LICENSE_ENV`, `LICENSE_ERROR` aus Task 4.
- Produces: `license_app` (Typer-App), gemountet als `surroforge license` über die bestehende Commands-Mechanik (`_mount_plugin_commands`, `src/surroforge/cli/main.py:705`). Der Befehl ist bewusst NICHT lizenzpflichtig.

- [ ] **Step 1: Failing Tests schreiben**

`plugins/surroforge_professional/tests/test_license_cli.py` (neu, komplett):

```python
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
```

- [ ] **Step 2: Rot verifizieren**

Run: `.venv/bin/python -m pytest plugins/surroforge_professional/tests/test_license_cli.py -q`
Expected: ERROR bei Collection (`surroforge_professional.commands.license` existiert nicht)

- [ ] **Step 3: Implementierung**

`plugins/surroforge_professional/src/surroforge_professional/commands/license.py` (neu, komplett):

```python
"""`surroforge license` — inspect and install the local license (M3).

Deliberately NOT license-gated: it must work before a valid license exists.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from surroforge_professional.license import (
    LICENSE_ENV,
    LICENSE_ERROR,
    LicenseStatus,
    license_file,
    license_status,
    validate_token,
)

license_app = typer.Typer(
    help="Inspect and install the SurroForge license.",
    no_args_is_help=True,
)


@license_app.callback()
def _license_callback() -> None:
    """Inspect and install the SurroForge license."""


def _verdict(status: LicenseStatus) -> str:
    if status.error is None:
        return "valid"
    if status.error.startswith("expired "):
        return "expired"
    return f"invalid ({status.error})"


@license_app.command("status")
def status() -> None:
    """Show the installed license and its validity (exit 0 only when valid)."""
    st = license_status()
    if st.error == "missing":
        typer.echo("No license installed.")
        typer.echo(LICENSE_ERROR)
        raise typer.Exit(1)
    typer.echo(f"Source:  {st.source}")
    typer.echo(f"Holder:  {st.holder or '—'}")
    typer.echo(f"Tier:    {st.tier or '—'}")
    typer.echo(f"ID:      {st.license_id or '—'}")
    typer.echo(f"Issued:  {st.issued or '—'}")
    typer.echo(f"Expires: {st.expires.isoformat() if st.expires else 'never'}")
    typer.echo(f"Status:  {_verdict(st)}")
    if st.error is not None:
        raise typer.Exit(1)


@license_app.command("install")
def install(
    source: str = typer.Argument(
        ..., help="License token, or path to a file containing it."
    ),
) -> None:
    """Validate a license token and install it to ~/.surroforge/professional.lic."""
    candidate = Path(source).expanduser()
    token = candidate.read_text().strip() if candidate.is_file() else source.strip()
    st = validate_token(token)
    if st.error is not None:
        typer.echo(f"Error: license not installed — {st.error}.")
        raise typer.Exit(1)
    target = license_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(token + "\n")
    expires = st.expires.isoformat() if st.expires else "never"
    typer.echo(f"Installed {st.tier} license for {st.holder} (expires: {expires}) -> {target}")
    if os.environ.get(LICENSE_ENV, "").strip():
        typer.echo(f"Note: {LICENSE_ENV} is set and takes precedence over the installed file.")
```

`plugins/surroforge_professional/pyproject.toml` — Entry-Point ergänzen (Gruppe existiert, Zeile 41–42):

```toml
[project.entry-points."surroforge.commands"]
doe = "surroforge_professional.commands.doe:doe_app"
license = "surroforge_professional.commands.license:license_app"
```

Dann: `.venv/bin/pip install -q -e ./plugins/surroforge_professional`

- [ ] **Step 4: Tests grün + Mount-Smoke**

Run: `.venv/bin/python -m pytest plugins/surroforge_professional/tests/test_license_cli.py -q`
Expected: `8 passed`

Run: `.venv/bin/surroforge license --help && env -u SURROFORGE_PRO_LICENSE HOME=$(mktemp -d) .venv/bin/surroforge license status; echo "exit=$?"`
Expected: Hilfetext mit `status`/`install`; danach `No license installed.` + Hint, `exit=1`

- [ ] **Step 5: Lint + Commit**

```bash
.venv/bin/ruff check plugins/surroforge_professional
.venv/bin/pyright
git add plugins/surroforge_professional/src/surroforge_professional/commands/license.py \
        plugins/surroforge_professional/pyproject.toml \
        plugins/surroforge_professional/tests/test_license_cli.py
git commit -m "feat(editions): surroforge license status/install (M3)"
```

---

### Task 6: Doku-Sweep, CHANGELOG, MEMORY, Full-CI

**Files:**
- Modify: `CHANGELOG.md` (neuer Bullet unter `### Changed — Editionen (Breaking)`, vor dem `mlp`-Bullet, Zeile 12)
- Modify: `plugins/surroforge_professional/README.md:13-26`
- Modify: `docs/index.md:50-51`
- Modify: `docs/reference/cli.md` (neuer Abschnitt nach `## Reproduce`)
- Modify: `examples/03_reproduction.py:10-12`, `examples/06_mlp_surrogate.py:6-7`, `examples/08_torch_mlp.py:6-7`
- Modify: `MEMORY.md:69-72` (+ Offene-Punkte-Abschnitt)

**Interfaces:**
- Consumes: Verhalten/Namen aus Tasks 4–5 (`surroforge license status|install`, `tools/licensing/issue.py`). Keine Produces (reine Doku).

- [ ] **Step 1: CHANGELOG.md**

Unter `### Changed — Editionen (Breaking)` als ERSTEN Bullet (vor `- **\`mlp\`, …`) einfügen:

```markdown
- **Lizenzprüfung validiert Inhalte (M3):** eine Lizenz ist jetzt ein Ed25519-signierter
  `SFL1.…`-Token (Inhaber, Tier, optionales Ablaufdatum), komplett offline geprüft gegen den
  im Professional-Paket eingebetteten Vendor-Schlüssel (`cryptography` wird Pflicht-Dependency).
  Stub-Werte wie `SURROFORGE_PRO_LICENSE=dev` schalten nichts mehr frei; die Kanäle bleiben
  (Env-Variable oder `~/.surroforge/professional.lic`, Env gewinnt). Neu: `surroforge license
  status` und `surroforge license install <token-oder-datei>`; abgelaufene/ungültige Lizenzen
  nennen den Grund in der Gate-Meldung. Vendor-Tooling: `tools/licensing/` (keygen/issue/inspect).
```

- [ ] **Step 2: professional-README — License-Gate-Abschnitt ersetzen**

`plugins/surroforge_professional/README.md`, Abschnitt `## License Gate` (Zeile 13–26) ersetzen durch:

```markdown
## License Gate

Professional features need a signed license token (M3: content validation,
offline against the vendor key). Obtain a token, then either

```bash
surroforge license install <token-or-file>   # writes ~/.surroforge/professional.lic
surroforge license status                    # verify: holder, tier, expiry
```

or set it for the session (takes precedence over the installed file):

```bash
export SURROFORGE_PRO_LICENSE='SFL1.…'
```

For development, issue yourself a license: `python tools/licensing/issue.py issue
--holder "dev"` (one-time `keygen` first — see `tools/licensing/README.md`).
```

- [ ] **Step 3: docs/index.md:50-51 ersetzen**

Alt:

```markdown
Licensed methods need `SURROFORGE_PRO_LICENSE` (or `~/.surroforge/professional.lic`);
an enterprise licence unlocks the professional tier as well.
```

Neu:

```markdown
Licensed methods need a signed license key — install it with `surroforge license install`
(or set `SURROFORGE_PRO_LICENSE`); an enterprise licence unlocks the professional tier as well.
```

- [ ] **Step 4: docs/reference/cli.md — Abschnitt ergänzen**

Nach dem Abschnitt `## Reproduce` (vor `## Misc`) einfügen (Stil an die bestehenden Abschnitte anpassen — vorher kurz einen Nachbarabschnitt lesen):

```markdown
## License (Professional)

Available once the `surroforge_professional` plugin is installed.

| Command | Purpose |
| --- | --- |
| `surroforge license status` | Show holder, tier, and expiry of the installed license; exit 0 only when valid. |
| `surroforge license install <token-or-file>` | Validate a license token and write it to `~/.surroforge/professional.lic`. |

`SURROFORGE_PRO_LICENSE` takes precedence over the installed file. Expired or
tampered licenses are rejected with the concrete reason.
```

- [ ] **Step 5: Beispiel-Docstrings (Stub-Wording entfernen)**

`examples/03_reproduction.py` Zeile 10–12, alt:

```
Since the editions split, `reproduce` is a Professional feature: set
`SURROFORGE_PRO_LICENSE` (any value activates the stub licence) before running.
The recording itself happens on every run, licensed or not.
```

neu:

```
Since the editions split, `reproduce` is a Professional feature: set
`SURROFORGE_PRO_LICENSE` to your license key before running. The recording
itself happens on every run, licensed or not.
```

`examples/06_mlp_surrogate.py` Zeile 6–7: `… install the` / `` `surroforge_professional` plugin and set a license, e.g. `SURROFORGE_PRO_LICENSE=dev`. `` → `` `surroforge_professional` plugin and set `SURROFORGE_PRO_LICENSE` to a license key. ``

`examples/08_torch_mlp.py` Zeile 6–7: `(e.g. \`SURROFORGE_PRO_LICENSE=dev\`)` → `` (`SURROFORGE_PRO_LICENSE`) `` — Satz bleibt sonst unverändert. (07 und 09 nennen die Variable ohne Stub-Wert-Beispiel → unverändert.)

- [ ] **Step 6: MEMORY.md**

Zeile 69–72 (Bullet „Professional-HTML ist Plugin-only") — den Klammerteil zur Lizenzprüfung aktualisieren: aus „nutzt in v1 nur eine lokale, tier-aware Lizenzprüfung (`SURROFORGE_PRO_LICENSE` oder Datei; Wert mit ‚enterprise' schaltet die Enterprise-Stufe frei), keine Online-Aktivierung" wird „nutzt die lokale, tier-aware Lizenzprüfung (M3: Ed25519-signierte `SFL1`-Tokens, offline gegen den eingebetteten Vendor-Key; Env oder `~/.surroforge/professional.lic`), keine Online-Aktivierung". Im Offene-Punkte-Abschnitt „echte Lizenz-Aktivierung (M3)" als erledigt austragen (Rest des Abschnitts unverändert lassen).

- [ ] **Step 7: Full-CI**

Run: `scripts/ci.sh`
Expected: alle Pflicht-Gates PASS (ruff, mypy, pyright, import-linter, pytest, reproducibility, beide Plugin-Suiten, profile); docs-Gate PASS (mkdocs strict — cli.md-Änderung prüft sich hier)

- [ ] **Step 8: Commit**

```bash
git add CHANGELOG.md plugins/surroforge_professional/README.md docs/index.md \
        docs/reference/cli.md examples/03_reproduction.py examples/06_mlp_surrogate.py \
        examples/08_torch_mlp.py MEMORY.md
git commit -m "docs(editions): M3 license validation — activation docs, CHANGELOG, memory"
```

---

## Plan-Selbstreview (durchgeführt)

- **Spec-Abdeckung:** Token-Format §1 → Task 1/4; Validierung §2 → Task 4; Meldungen §3 → Task 4 (Hint-Tests); CLI §4 → Task 5; Issuance §5 → Task 3; Test-Strategie → Tasks 1–5 (Golden, Kreuz-Test, sitecustomize, Fixtures); Migration/Doku → Task 6. PRD bleibt unverändert (Spec).
- **Brücke:** Tasks 2/3 sind unter dem Stub grün (nicht-leerer Token; `raising=False`); Enterprise-Flip bewusst erst in Task 4 (Substring-Erkennung des Stubs sieht das b64-kodierte Tier nicht).
- **Typ-Konsistenz:** `license_kit.mint(tier=..., **overrides)`, `LicenseStatus`-Felder (`license_id`, nicht `id`), `validate_token`/`license_status` in Task 4 definiert und in Task 5 konsumiert; `TEST_SEED_HEX`/`TEST_PUB_HEX`/`GOLDEN_TOKEN` überall identisch.
