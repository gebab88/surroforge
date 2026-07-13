# M3: Echte Lizenz-Validierung (signierte Offline-Lizenz) — Design

**Datum:** 2026-07-13
**Status:** Design (abgesegnet, bereit für Plan)
**Bezug:** PRD §3.3 (Lizenzaktivierung), §9.2 („Lizenz-Aktivierungs-Mechanik"); W1-Spec
(Authorizer-Seam `surroforge.authorizers`, Hint-Mechanik); Roadmap-Must M3. Ersetzt den
Wave-1-Stub in `plugins/surroforge_professional/src/surroforge_professional/license.py`
(prüft heute nur *Vorhandensein*; „enterprise"-Substring wählt das Tier).

## Ziel

Die Lizenzprüfung validiert den **Inhalt** statt des Vorhandenseins: eine Lizenz ist ein
Ed25519-signierter Token (Inhaber, Tier, optionales Ablaufdatum), geprüft komplett
offline gegen einen im Professional-Paket eingebetteten Public Key. Alle Konsumenten des
Stubs bleiben unverändert — `authorize(tier)` + `hint`, `granted_tier()`,
`has_license()`, `require_license()`, `LICENSE_ENV`/`LICENSE_FILE` und der eingefrorene
String `LICENSE_ERROR`. **Kein Core-Change:** `src/surroforge/` bleibt unangetastet;
das W1–W3-Gating hängt nur am Authorizer-Vertrag.

## Entscheidungen (aus dem Brainstorming, locked)

| Frage | Entscheidung |
|---|---|
| Trust-Modell | **Signierte Offline-Lizenz** (Ed25519); kein Lizenzserver |
| Ablauf | **Optionales `expires`-Feld**; fehlt → unbefristet; überschritten → harte Sperre mit klarer Meldung (keine Grace-Period) |
| Krypto | **Abhängigkeit `cryptography`** im Professional-Paket (kein Vendoring) |
| Issuance-Tooling | **`tools/licensing/` im Repo**; privater Signierschlüssel NIE im Repo (lokal `~/.surroforge-vendor/`) |
| Token-Format | Eigenes Minimalformat `SFL1.…` — bewusst **kein JWT/JOSE** (keine Algorithmus-Aushandlung, kein `alg:none`-Potenzial, keine Parsing-Abhängigkeit) |
| Maschinen-Binding | **Nein** (Floating/Site = Phase 3; `v`-Feld hält das Format erweiterbar) |
| Online-Aktivierung | **Nein** (bleibt in PRD §3.3 als späterer zweiter Weg stehen) |
| Enterprise-Kodierung | `tier`-Feld im signierten Payload; Enterprise bringt weiterhin **keinen eigenen Authorizer** mit |

## Architektur

### §1 Token-Format

Eine Lizenz ist ein einzeiliger Token — derselbe String funktioniert in beiden
bestehenden Kanälen (Env `SURROFORGE_PRO_LICENSE` gewinnt über Datei
`~/.surroforge/professional.lic`, wie heute):

```
SFL1.<base64url(payload-json)>.<base64url(ed25519-signatur, 64 bytes)>
```

- Signiert werden die **rohen Payload-JSON-Bytes** (exakt die Bytes, die
  base64-kodiert im Token stehen) — keine Kanonisierungsfallen.
- base64url mit Padding (`base64.urlsafe_b64encode` unverändert); beim Lesen wird der
  Wert wie bisher ge`strip()`t. Das b64url-Alphabet enthält keine Punkte → Split auf
  `.` ist eindeutig.
- Payload-Felder: `v` (int, muss `1` sein), `id` (str, z. B. `SF-20260713-a1b2`),
  `holder` (str), `tier` (`"professional"` | `"enterprise"`), `issued` (ISO-Datum),
  `expires` (**optional**, ISO-Datum). Unbekannte Zusatzfelder werden ignoriert
  (Vorwärtskompatibilität).
- Ablauf-Semantik: gültig **bis einschließlich** `expires`, verglichen gegen das
  UTC-Tagesdatum.

### §2 Validierung (`license.py`, Professional-Paket)

Öffentliche Oberfläche unverändert; neue Interna:

- `_TRUSTED_KEYS: tuple[str, ...]` — hex-kodierte 32-Byte-Ed25519-Public-Keys.
  Ships mit **einem** Produktionsschlüssel; das Tupel erlaubt spätere
  Schlüsselrotation und Test-Injektion (monkeypatch).
- `LicenseStatus` (frozen dataclass): `tier`, `holder`, `license_id`, `issued`,
  `expires`, `source` (`"env"` | `"file"` | `None`), `error` (`None` ⇔ gültig).
  `license_status() -> LicenseStatus` ist die paketinterne Auskunfts-API — Grundlage
  für `granted_tier()` und den `license`-CLI-Befehl.
- Prüfkette in `_validate(value)`: (1) Prefix `SFL1.` + drei Segmente + base64-Decode —
  sonst `malformed`; (2) Ed25519-Verify gegen jeden Trusted Key (`cryptography`,
  **lazy import** erst im Verify-Pfad, nicht beim Entry-Point-Laden) — sonst
  `bad-signature`; (3) Payload-JSON: `v != 1` → `unsupported-version`, unbekanntes
  `tier` → `unknown-tier` (künftige Tiers entsperren alte Installationen nicht),
  `expires` überschritten → `expired`.
- `granted_tier()` liefert `status.tier` nur bei `error is None`, sonst `None` —
  `authorize()`/`has_license()`/`require_license()` bauen wie heute darauf auf;
  `_TIER_ORDER`-Vergleich unverändert (unbekannte *Anforderungs*-Tiers bleiben deny).
- **Kein Ergebnis-Caching:** jede Prüfung liest Env/Datei frisch (Stub-Verhalten,
  hält Tests einfach; eine Signaturprüfung pro Gate-Check ist billig).
- Fehlende `cryptography`-Installation ist ein kaputtes Environment (harte Dependency)
  → ImportError propagiert; kein stiller Deny.
- `pyproject.toml` (professional): `cryptography>=42` zu `dependencies`.

### §3 Meldungen: lagegenaue Hint, eingefrorene Strings bleiben

Die W1-Gate-Meldung (`'{name}' requires a '{tier}' license, …`) bleibt byte-identisch;
nur der Hint-Teil wird befundabhängig. `authorize()` setzt vor **jedem** Return
`authorize.hint` (mutierendes Funktionsattribut — das Gate liest es nach dem Aufruf;
Grundzustand beim Import ist `LICENSE_ERROR`):

| Befund | Hint |
|---|---|
| keine Lizenz vorhanden | `LICENSE_ERROR` (**eingefroren, byte-identisch**; Frozen-Test bleibt) |
| Format/Signatur ungültig | `Installed Professional license is invalid ({reason}). Reinstall the license or contact your vendor.` — reason z. B. `signature check failed`, `malformed token`, `unsupported schema version`, `unknown tier '…'` |
| abgelaufen | `Professional license expired on {YYYY-MM-DD}. Renew to restore access.` |
| Tier zu niedrig | `Installed license grants '{granted}', but this feature requires '{required}'.` |

`require_license()` (paketinterne Self-Gates) wirft `PluginError` mit derselben
lagegenauen Meldung; im Absenz-Fall weiterhin exakt `LICENSE_ERROR`. Die bestehenden
Real-Authorizer-Tests, die `SURROFORGE_PRO_LICENSE` im Fehlertext asserten, sind
Absenz-Fälle → bleiben grün.

### §4 CLI: `surroforge license` (Professional-Paket)

Neues Modul `surroforge_professional/commands/license.py` (Typer-App), Entry-Point
`license` in der bestehenden Gruppe `surroforge.commands` (Muster: `doe`). Der Befehl
selbst ist **nicht** lizenzpflichtig — er muss ohne gültige Lizenz nutzbar sein.

- `surroforge license status` — zeigt Quelle (env/file/none), Inhaber, Tier, ID,
  Ausstellungs-/Ablaufdatum (`never` wenn unbefristet) und Verdikt
  (`valid` / `expired` / `invalid` / `none`). **Exit 0 nur bei gültig**, sonst 1 —
  skriptbar. Ohne Lizenz: Hinweis mit `LICENSE_ERROR`-Text.
- `surroforge license install <datei-oder-token>` — Argument ist ein Dateipfad
  (existierende Datei wird gelesen) oder der Token selbst. **Erst validieren**, dann
  nach `license_file()` schreiben (`~/.surroforge/` anlegen); Bestätigung mit
  Inhaber/Tier/Ablauf. Ungültige Token werden mit Grund abgelehnt (exit 1, Datei
  unangetastet). Ist die Env-Var gesetzt, warnt der Befehl, dass sie über die
  installierte Datei gewinnt.
- Kein `activate`-Online-Befehl. Core-only-Installationen haben den Befehl nicht
  (Plugin-Modell); die Aktivierungs-Hints nennen weiterhin Env-Var und Dateipfad.

### §5 Issuance-Tooling: `tools/licensing/`

Dev-seitig, wird nicht paketiert; nutzt `cryptography` aus dem Dev-Environment.

- `sflicense.py` — Format-Bibliothek und **einzige Signier-Quelle**:
  `generate_keypair()`, `mint(payload, private_key_hex) -> token`,
  `decode(token)`; von `issue.py` und den Test-Fixtures (path-import) genutzt.
- `issue.py` — argparse-CLI mit drei Subcommands:
  - `keygen` — Schlüsselpaar erzeugen; privater Schlüssel (hex-Seed) nach
    `~/.surroforge-vendor/signing.key` (Mode 0600, verweigert Überschreiben ohne
    `--force`); druckt den Public-Key-Hex zum Einbetten in `_TRUSTED_KEYS`.
  - `issue --holder … [--tier professional|enterprise] [--expires YYYY-MM-DD]
    [--id …] [--key PFAD]` — druckt den Token (Default-Tier professional, Default-ID
    `SF-<datum>-<hex4>`, `issued` = heute).
  - `inspect <token-oder-datei> [--pub HEX]` — dekodiert, verifiziert (Default: gegen
    die eingebetteten `_TRUSTED_KEYS` des installierten Pakets), druckt Payload +
    Verdikt.
- `README.md` — Ablauf (keygen einmalig → issue pro Kunde → Kunde installiert) und
  **Backup-Pflicht**: verlorener privater Schlüssel = keine neuen Lizenzen, die
  bestehende Installationen akzeptieren; Rotation nur über neues Release
  (`_TRUSTED_KEYS`-Tupel erweitern).
- Produktionsschlüsselpaar wird **einmalig bei der Implementierung** per `keygen`
  erzeugt; der Public Key wird in `license.py` eingebettet, der private bleibt in
  `~/.surroforge-vendor/`.

## Test-Strategie

- **`test_license.py` wird zur Validierungs-Suite umgeschrieben:** Rundreise
  mint→verify; manipuliertes Payload (Byte geflippt) → `bad-signature`; fremder
  Schlüssel → deny; `expired` (+ Datum in der Meldung); unbefristet gültig;
  enterprise-Token deckt professional-Anforderung; professional-Token ≠
  enterprise-Anforderung (+ Tier-Hint); unknown tier; `v != 1`; Kaputt-Tokens (kein
  Prefix, fehlende Segmente, bad base64, leere Datei); Env-vor-Datei-Präzedenz;
  Datei-Kanal via HOME-Umleitung. Der Frozen-Test auf `LICENSE_ERROR` bleibt
  unverändert bestehen.
- **Golden-Token-Test:** fest verdrahtetes Test-Schlüsselpaar + eingefrorener
  Token-String validiert → Formatstabilität über Refactorings.
- **Kreuz-Test Tool↔Paket:** `tools/licensing/sflicense.mint` → `license.py`
  validiert (wichtigster Integrationspunkt; testet das Tool gleich mit).
- **Fixtures auf geprägte Tokens umgestellt** (session-scoped Test-Keypair, Minten
  via path-importiertem `sflicense.py`, monkeypatch `_TRUSTED_KEYS`, HOME-Umleitung
  wie bisher): `licensed` (professional-conftest), `licensed_enterprise`
  (enterprise-conftest, tier=enterprise), `licensed_export` (Core-`tests/conftest.py`).
- **`test_examples.py` (Subprozesse!):** Monkeypatch reicht nicht über die
  Prozessgrenze. Lösung: per `PYTHONPATH` injizierte `sitecustomize.py` im tmp-Dir,
  die `surroforge_professional.license._TRUSTED_KEYS` auf den Test-Public-Key setzt
  (no-op, wenn professional nicht installiert ist), plus Env-Token vom Test-Key.
  Invasivster Testteil — klar kommentiert.
- **Real-Authorizer-Gating-Tests** (`test_gating.py`, `test_enterprise_gating.py`):
  Grant-Fälle auf geprägte Tokens umgestellt; Absenz-Fälle (delenv) unverändert.
- **Neue CLI-Tests:** `status` (valid/expired/none, Exit-Codes), `install`
  (Token-Argument und Datei-Argument, ungültig abgelehnt, Env-Präzedenz-Warnung).
- **Core-Suite: keine Verhaltensänderung** — `granted_license`-Fake-Authorizer
  umgeht `license.py` komplett; Unlizenziert-Neutralisierung (delenv + HOME) ergibt
  weiter den Absenz-Fall.

## Migration & Downstream

- **Breaking (CHANGELOG, Editionen):** Stub-Werte (`SURROFORGE_PRO_LICENSE=dev`,
  „enterprise"-Substrings) schalten nichts mehr frei — eine Lizenz ist jetzt ein vom
  Vendor signierter Key. Dev-Workflow: mit `tools/licensing` einmalig eine echte
  Dev-Lizenz prägen (Env-Profil oder `license install`).
- **Doku:** professional-README (Aktivierung: Key erhalten → `license install` oder
  Env), `docs/index.md`-Editions-Zeile, `docs/reference/cli.md` (neuer
  `license`-Befehl), Wording-Check der Stellen „Set SURROFORGE_PRO_LICENSE" („… to
  your license key"), Docstrings der Beispiele 03/06/07/08, MEMORY.md (M3 abhaken).
- **PRD:** keine Änderung nötig — §3.3 („Lizenzdatei oder Online-Aktivierung") ist
  mit dem Datei-Weg erfüllt; Online bleibt als späterer zweiter Weg stehen.

## Außerhalb von M3

- Online-Aktivierung, Floating-/Site-Lizenzen, Lizenzserver (Phase 3).
- Maschinen-Binding, Grace-Period, Offline-Toleranz-Fenster (Enterprise, §3.3).
- Schlüsselrotations-*Prozess* (Mechanik via `_TRUSTED_KEYS`-Tupel ist vorbereitet).
- Anti-Tamper/Code-Obfuskation (siehe Risiken).
- Pro-Befehl-Sichtbarkeit in Core-only-Installationen (Could-Karte, separat).

## Risiken / Ehrlichkeits-Notizen

- **Produktgrenze, keine Sicherheitsgrenze:** Python-Code ist patchbar; die Signatur
  verhindert Key-*Fälschung*, nicht Code-Manipulation beim Kunden — bewusst
  akzeptiert, konsistent zur W1–W3-Haltung.
- **Schlüsselverlust:** ohne Backup des privaten Schlüssels können keine neuen
  Lizenzen für bestehende Installationen ausgestellt werden (Tool-README macht die
  Backup-Pflicht explizit).
- **Uhr-Manipulation:** `expires` prüft die Systemuhr; auf air-gapped Maschinen
  verstellbar — akzeptiert, kein Anti-Tamper.
- **Dynamische Hint = mutierender Zustand** am Funktionsobjekt: dokumentiert,
  CLI-Nutzung ist single-threaded; Grundzustand ist `LICENSE_ERROR`, das Gate liest
  den Wert unmittelbar nach dem Aufruf.
- **Env gewinnt über frisch installierte Datei** — UX-Falle, die `license install`
  mit einer Warnung entschärft.
- **Hint-Texte ändern sich in Nicht-Absenz-Fällen** (invalid/expired/Tier) — falls
  ein bestehender Test einen solchen Fall wörtlich assertet, wird er im Plan
  inventarisiert und angepasst; die Absenz-Meldung und das W1-Template sind
  byte-stabil.
