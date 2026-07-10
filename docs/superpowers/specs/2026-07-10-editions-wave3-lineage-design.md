# Welle 3: reproduce/Lineage-Oberflächen → Professional (Feature-Gating) — Design

**Datum:** 2026-07-10
**Status:** Design (abgesegnet, bereit für Plan)
**Bezug:** Plan `docs/superpowers/plans/2026-07-08-editions-code-migration.md` (Abschnitt
„Welle 3 · Repro/Lineage-Oberfläche gaten"); W1-Spec (Seam/Authorizer),
W2-Spec (Export-Gating, `licensed_export`-Muster); PRD §3.2 („Reproduzierbarkeit &
Lineage → Professional"), §4.5, §5.2, §7.1/§7.2

## Ziel

Die Reproduzierbarkeits-/Lineage-**Oberflächen** verlangen eine Professional-Lizenz:
`reproduce` (hart), das Herkunfts-Panel in `info`, der Herkunftsblock im Markdown-Report
und der Lineage-Teil von `compare` (alle sanft). Die **Erfassung** (lineage.json pro Run)
bleibt unveränderte Core-Infrastruktur — sie trägt freie Features (predict:
Provenance-Warnung, reduced-field-`basis_file`) und sichert den Upgrade-Pfad (alte
Core-Runs sind nach Lizenzkauf reproduzierbar). Anders als W1/W2 laufen diese Befehle
nicht durch den Plugin-Seam; hier kommt erstmals das im Migrationsplan skizzierte
Core-Feature-Gate zum Einsatz. Keine Extraktion (gelockte Plan-Entscheidung: Lineage
hat kein schützenswertes IP — SHA-256 + sortiertes JSON).

## Entscheidungen (aus dem Brainstorming, locked)

| Frage | Entscheidung |
|---|---|
| lineage.json | **Weiter schreiben** (Core-interne Infra); PRD §7.1/§7.2/§5.2 werden umformuliert |
| Gate-Mechanik | Core-Feature-Gate `gate.require_license(feature, tier="professional")` — Authorizer + Hint wie W1; harte Oberflächen rufen direkt, sanfte fangen `PluginError` |
| `reproduce` | **hart**, Gate in `services.reproduce` (deckt CLI + Python-API in einem Punkt) |
| `info`-Schnitt | Identität frei (experiment, method, created_at, data_source, onnx_artifact, Metriken); Herkunft Pro (config_hash, data_hash, seed, prediction_hash, python_version, surroforge_version, parent_run, dependencies); ohne Lizenz Hinweiszeile statt Block |
| Markdown-Report | vierte Oberfläche, **gleicher Feld-Schnitt** wie `info`; kursiver Hinweis-Absatz statt Herkunftsblock |
| `compare` | **Service-Ebene**: `compare_runs` lässt ohne Lizenz `lineage` (pro Run) und `lineage_summary` weg; CLI rendert konditional + Hinweiszeile |
| `rom inspect` | bleibt frei (ROM-Diagnostik liest lineage intern, zeigt keine Provenance) |
| Test-Strategie | Core-Mathematik-Tests hinter dem Gate: leichtes `granted_license`-Fixture (Fake-Authorizer-Monkeypatch, kein Skip auf Core-only-Checkouts) |
| W2-Follow-up | Regressionstest „unlizenziert trainieren → unlizenziert predicten" wird in W3 mitgeliefert |

## Architektur

### §1 Feature-Gate (Core, `src/surroforge/plugins/gate.py`)

```python
def require_license(feature: str, *, tier: str = "professional") -> None:
    """Gate a Core feature surface behind an edition tier (Wave 3).

    Same authorizer group + hint mechanics as the resolution seam; the
    editions-neutral message names the FEATURE instead of a plugin.
    """
    if any(authorize(tier) for authorize in _load_authorizers()):
        return
    raise PluginError(_denied_message(feature, tier))
```

- Die Meldungs-Konstruktion (Template + Hint-Sammlung) wird aus `check()` in einen
  gemeinsamen Helfer `_denied_message(name, tier)` gezogen — **eine** Quelle für beide
  Gate-Formen; `check()`-Verhalten bleibt byte-identisch (W1-Tests asserten die Meldung).
- Meldung: `'reproduce' requires a 'professional' license, and no installed SurroForge
  edition grants it. Set SURROFORGE_PRO_LICENSE or install ~/.surroforge/professional.lic.`
  (Hint nur bei installiertem professional — editions-neutral wie W1).
- Kein neuer Fehlertyp: `PluginError` (der `_guard`-Dekorator rendert ihn als `Error: …`,
  exit 1; W1/W2-Präzedenz). Kein `is_authorized`-Zweitprimitiv — sanfte Oberflächen
  fangen den `PluginError` (YAGNI).
- Performance: `_load_authorizers()` ist gecacht; das Gate feuert nur auf den vier
  Oberflächen, nie auf train/predict-Pfaden.

### §2 `reproduce` — hart

`src/surroforge/services.py`, `reproduce()` (Zeile ~163): erste Anweisung
`gate.require_license("reproduce")`. Import `from surroforge.plugins import gate` auf
Modulebene (services importiert plugins bereits).

- Deckt CLI (`surroforge reproduce`, cli/main.py:574-596) **und** Python-API ab —
  kein zweites Gate im CLI nötig.
- `assert_reproducible` bleibt ungegatet: es konsumiert nur ein `ReproduceResult`,
  das es ohne Lizenz nie gibt.
- `run_training(parent_run=…)` (interner Mechanismus) bleibt frei — das Feature ist
  `reproduce`, nicht das Trainieren.
- Unlizenziert: `Error: 'reproduce' requires a 'professional' license, …` — exit 1.

### §3 `info` + Markdown-Report — sanft, gemeinsamer Schnitt

**Feld-Schnitt (gilt für beide Oberflächen):**
- **Frei:** experiment, method, created_at, data_source, onnx_artifact, komplette
  Metrik-Zusammenfassung (+ im Report: alles Nicht-Provenance, inkl. Feld-/ROM-Teile).
- **Professional:** config_hash, data_hash, seed, prediction_hash, python_version,
  surroforge_version, parent_run, dependencies.

**CLI `info`** (cli/main.py:599-615): Identitätsfelder + onnx_artifact drucken; dann

```python
try:
    gate.require_license("lineage")
except PluginError as exc:
    _info(f"  (provenance hidden: {exc})")
else:
    …Herkunftsfelder + dependencies wie heute…
```

danach wie bisher die Metrik-Zusammenfassung. Die Hinweiszeile transportiert die
Gate-Meldung wörtlich (inkl. Authorizer-Hint) — kein zweiter Meldungs-Mechanismus.

**Markdown-Report** (`src/surroforge/reporters/markdown.py`): der Kontext-Bau prüft das
Gate einmal; ohne Lizenz entfallen die Provenance-Einträge (data_hash, config_hash,
seed, prediction_hash, parent_run, dependencies) aus dem Template-Kontext, und an ihrer
Stelle rendert ein kursiver Absatz `*Provenance omitted: <Gate-Meldung>*`. Das Template
bekommt dafür einen bedingten Block (Jinja-`if`), die freien Teile bleiben byte-gleich.

### §4 `compare` — Service-Ebene

`services.compare_runs` (Zeile ~389): nach `build_comparison(...)`

```python
try:
    gate.require_license("lineage")
except PluginError:
    for entry in result["runs"]:
        entry.pop("lineage", None)
    result.pop("lineage_summary", None)
```

- `build_comparison` (comparison.py) bleibt pur/ungegatet — Bau ist Erfassung, nicht
  Anzeige; das Gate sitzt am Service-Austritt.
- CLI `compare` (cli/main.py:366-…, `_render_lineage`-Aufruf Zeile ~415): rendert das
  Lineage-Panel nur noch, wenn `lineage_summary` im Ergebnis ist; sonst Hinweiszeile
  analog `info` (eigener `require_license`-try/except für die Meldung).
- Professional-HTML-Comparison-Report (liest `entry["lineage"]`/`lineage_summary`):
  läuft nur lizenziert (Reporter selbst ist seam-/self-gegatet) → sieht die Keys
  unverändert. Kein Code-Change dort.
- Python-API-Konsequenz (gewollt): unlizenzierte `compare_runs`-Nutzer bekommen
  Ranking/Metriken/Config-Diff, aber keine Lineage-Keys.

### §5 Python-API-Grenze (Ehrlichkeit)

`Run.lineage` (Property) und die abgeleiteten Accessors (`data_hash`, `config_hash`,
`seed`, `holdout_fingerprint`, …) bleiben **frei**: sie sind interne Infrastruktur
(predict, Reporter, reproduce selbst) und die Datei liegt ohnehin lesbar im
Run-Verzeichnis. Gegatet sind die **Feature-Oberflächen** (reproduce, Anzeige-Panels,
Vergleichs-Aggregation), nicht der Datenzugriff — Produkt-, keine Sicherheitsgrenze,
konsistent zum akzeptierten Fork-Vorbehalt des Migrationsplans und zum Lizenz-Stub
bis M3. python-api.md dokumentiert die Grenze entsprechend.

## Test-Strategie

**Neues Fixture** `granted_license` in `tests/conftest.py` (Fake-Authorizer, W1-Muster
aus `test_plugins.py`):

```python
@pytest.fixture
def granted_license(monkeypatch):
    """Bypass the feature gate with a fake authorizer (works on Core-only checkouts).

    For tests of Core functionality that sits BEHIND the gate (reproduce math,
    canonicalisation, field reproduce). Licence-integration tests use the real
    professional authorizer instead (licensed_export pattern / plugin suite).
    """
    from surroforge.plugins import gate

    monkeypatch.setattr(gate, "_load_authorizers", lambda: ((lambda tier: True),))
```

- **Reproduce-Mathematik-Tests** (11 Dateien: test_reproduce, test_reproduce_field,
  test_reproduce_scalar_identity, test_canonicalize, test_probe_coverage,
  test_reduced_field, test_field_ingest, test_openfoam, test_reduction_config,
  test_cli, test_scenarios) bekommen das Fixture. Rein-reproduce-Module dürfen es als
  modul-lokales `autouse` einbinden; **Misch-Module (test_cli, test_scenarios, …)
  injizieren es NUR in die reproduce-Tests** — ein autouse-Fake-Authorizer würde dort
  die W2/W3-Unlizenziert-Tests kippen (der Fake ignoriert Env/HOME).
- **Neue Gating-Tests (Core):** `require_license` erlaubt bei zustimmendem Authorizer,
  verweigert ohne (Meldung inkl. Feature-Name; Monkeypatch `()` wie W1);
  `services.reproduce` unlizenziert → PluginError; CLI: `reproduce` unlizenziert →
  exit 1 + Gate-Meldung; `info` unlizenziert → Identität + Metriken sichtbar,
  `(provenance hidden:` da, kein `data_hash`; lizenziert → alle Felder; Markdown-Report
  unlizenziert → kein Hash im Text, Hinweis-Absatz da; `compare` unlizenziert → kein
  Lineage-Panel, Hinweiszeile; `compare_runs` (API) unlizenziert → Keys fehlen,
  lizenziert → Keys da. Unlizenziert-Tests neutralisieren Env+HOME (W2-Muster).
- **Gating-Tests professional** (Integration mit echtem Authorizer): reproduce/info
  mit `SURROFORGE_PRO_LICENSE` → voll funktionsfähig (Muster test_gating.py).
- **W2-Follow-up:** `test_train_then_predict_without_license` — unlizenziert
  trainieren, unlizenziert predicten, beides grün (zweites Schloss auf der
  Run-Store-Invariante; Env+HOME neutralisiert).
- Bestehende professional-Tests (HTML-Comparison-Report) laufen lizenziert → unberührt.

## Downstream (in W3 mitgenommen)

- **PRD:** §7.1 (Zeile ~362) „`lineage.json` … erst in Professional geschrieben" →
  „wird von Core als interne Infrastruktur geschrieben; nutzbar (reproduce, Anzeige,
  Export) erst in Professional". §7.2-Überschrift/Intro analog. §5.2 (Zeile ~261)
  „jedoch ohne Herkunfts-Hashes und ohne reproduce" → „ohne nutzbare Herkunfts-Ansicht
  und ohne reproduce". §4.5 bleibt (beschreibt bereits die Professional-Sicht).
- **Beispiel** `examples/03_reproduction.py`: Docstring-Hinweis (reproduce = Professional,
  `SURROFORGE_PRO_LICENSE`); läuft unter test_examples eh mit injizierter Lizenz.
- **Doku:** `docs/scenarios/reproduction.md` (Professional-Hinweis prominent),
  `docs/quickstart.md` §6 (info-Felder) + §10 (reproduce), `docs/index.md`
  („Reproducibility is built in" → Professional-Zusatz), README („Reproducibility"-
  Abschnitt), `docs/reference/cli.md` (info/reproduce/compare), `docs/reference/
  python-api.md` (reproduce/assert_reproducible/compare_runs-Zeilen + §5-Grenze).
- **CHANGELOG** (Breaking-Eintrag unter Editionen), **MEMORY.md** (Architektur +
  offene Punkte: W3 abhaken).

## Außerhalb von Welle 3

- Echte Lizenz-Validierung/Aktivierung → **M3** (das Feature-Gate nutzt den Stub).
- PDF-Audit-Reports mit Lineage-Anhang → Enterprise, Phase 3.
- Kein Gating von `rom inspect`, `validate`, `list`, `predict`, `report`-Metrikteil.

## Risiken / Ehrlichkeits-Notizen

- **Produktgrenze, keine Sicherheitsgrenze:** lineage.json liegt lesbar im
  Run-Verzeichnis, `Run.lineage` bleibt frei, Core ist Apache (Self-Gate forkbar) —
  bewusst akzeptiert (gelockte Migrationsplan-Entscheidung).
- **Meldungs-Refactor in `check()`:** W1-Tests asserten die Seam-Meldung wörtlich —
  `_denied_message` muss byte-identisch formatieren (Regression durch bestehende
  W1-Gating-Tests abgedeckt).
- **Fake-Authorizer-Fixture** umgeht bewusst Env/HOME — es lizenziert *funktional*,
  testet nie den Lizenz-Stub selbst; Stub-Integration bleibt in den
  professional-Gating-Tests. Verwechslungsgefahr in Misch-Modulen ist in der
  Test-Strategie explizit adressiert (kein autouse dort).
- **Vier Oberflächen, ein Feature-Name:** `info`/Report/`compare` teilen den
  Feature-Namen `"lineage"`, `reproduce` heißt `"reproduce"` — zwei Namen, damit die
  Meldung dem Nutzer den tatsächlich verweigerten Befehl nennt.
