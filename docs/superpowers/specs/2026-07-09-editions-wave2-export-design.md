# Welle 2: Modell-Export → Professional (Export-Gating) — Design

**Datum:** 2026-07-09
**Status:** Design (abgesegnet, bereit für Plan)
**Bezug:** Plan `docs/superpowers/plans/2026-07-08-editions-code-migration.md` (Abschnitt
„Welle 2 · Export → Professional"); W1-Spec
`docs/superpowers/specs/2026-07-08-editions-wave1-methods-design.md`; PRD §3.2
(Edition-Matrix: „Export (ONNX, Pickle) → Professional"), §4.4 (Export), §4.11 (Plugin-System)

## Ziel

Die Exporter `pickle` und `onnx` (heute Core-Builtins) werden **Professional**-Features,
durchgesetzt über den in W1 gebauten tier-aware Seam (`required_tier` + Authorizer). Der
`pt`-Exporter (schon in professional, bisher ungegatet) wird ebenfalls gegatet. **Train und
Predict bleiben vollständig frei**: Der Run-Store (`model.pkl` schreiben/lesen) wird vorher
von den Exportern entkoppelt und bleibt Core-Infrastruktur.

## Entscheidungen (aus dem Brainstorming, locked)

| Frage | Entscheidung |
|---|---|
| Enforcement | Nur der **W1-Seam** — alle Nutzer-Exportpfade laufen durch `get_exporter`; **keine Self-Gates** in W2 (`gate.require_license` erst W3) |
| Store-Entkopplung | Interner Model-Store `persistence/model_io.py` (Core); Engine/Run nutzen ihn statt der Exporter |
| Exporter-Heimat | `pickle_exporter.py` + `onnx_exporter.py` per `git mv` nach `surroforge_professional/exporters/`; `pt_exporter.py` zieht in dasselbe Unterpaket |
| Gating | alle drei Exporter: `required_tier = "professional"` |
| `train --onnx` ohne Lizenz | **Sanft überspringen**: bestehendes try/except greift, Log-Zeile „ONNX export skipped: …", Run läuft durch |
| `_BUILTIN[EXPORTER_GROUP]` | wird `{}` (Präzedenz: `COMPARISON_REPORTER_GROUP` ist schon leer) |
| Root-`pyproject.toml` | die zwei Exporter-Entry-Points entfernen (Kollisionsgefahr, W1-mlp-Präzedenz) |
| `BaseExporter` | bleibt Core (`exporters/base.py`) — Plugin-Schnittstelle, editions-neutral |

## Befund (Prüfpunkt aus dem Migrationsplan, beantwortet)

- **Kopplung ist real:** `engine.py:33–34` importiert `ONNXExporter`/`PickleExporter` direkt;
  `engine.py:233` schreibt das Predict-Artefakt `model.pkl` über `PickleExporter`;
  `run.py:262` lädt es über `load_pickle`. Blindes Verschieben bräche Train/Predict → §1.
- **Alle Nutzer-Exportpfade** (CLI `export`, `Run.export`, `services.export_model`) laufen
  durch `get_exporter` → der W1-Seam deckt W2 vollständig; keine Self-Gates nötig.
- **Einzige Seam-Umgehung:** der `train --onnx`-Block (`engine.py:283–288`, Direktimport,
  best-effort try/except) → §3.
- Konformanz-Suite (`conformance/trainer.py`) pickelt direkt via `pickle.dumps` — kein
  Exporter-Bezug. `.importlinter` nennt `surroforge.exporters` nur im errors-Leaf-Vertrag;
  das Paket bleibt (mit `base.py`) bestehen — Verträge unverändert gültig.

## Architektur

### §1 Interner Model-Store (Core)

Neu `src/surroforge/persistence/model_io.py`:

```python
def dump_model(trainer: BaseTrainer, path: Path) -> Path:
    # heutige PickleExporter-Logik: pickle.dump(trainer, protocol=HIGHEST_PROTOCOL)

def load_model(path: str | Path) -> BaseTrainer:
    # heutige load_pickle-Logik
```

- Beide über `surroforge.persistence.__init__` re-exportiert.
- `engine.py:233`: `PickleExporter().export(...)` → `dump_model(final_model, run_dir / MODEL_PICKLE)`
  (das ungenutzte `n_features` entfällt); Importe `engine.py:33–34` werden gestrichen.
- `run.py:262` (`Run.load_model`): lazy Import `from surroforge.persistence.model_io import
  load_model` statt `load_pickle`; `_warn_if_foreign_artifact` unverändert davor.
- **Damit ist Train/Predict exporter-frei und ungegatet** — der Run-Store ist
  Persistenz-Infrastruktur, kein Export-Feature.

### §2 Exporter → surroforge_professional

```
plugins/surroforge_professional/src/surroforge_professional/exporters/   # NEU (Unterpaket)
├── pickle_exporter.py   # git mv aus src/surroforge/exporters/; required_tier = "professional"
├── onnx_exporter.py     # git mv aus src/surroforge/exporters/; required_tier = "professional"
└── pt_exporter.py       # git mv aus Paket-Root;                required_tier = "professional"
```

- `PickleExporter.export` delegiert an `surroforge.persistence.model_io.dump_model` —
  **Artefakt-Identität**: `surroforge export --format pickle` schreibt byte-gleich dasselbe
  Format wie der Run-Store.
- Die Modul-Funktion `load_pickle` **entfällt** (einzige Nutzer waren `run.py` und die
  mitziehenden Tests; Ersatz ist `persistence.model_io.load_model`).
- `ONNXExporter` unverändert dünn (delegiert an `trainer.to_onnx`); `PtExporter` unverändert
  (delegiert an `to_pt`, torch-frei auf Modul-Ebene — der Seam läuft daher vor jedem
  torch-Import).
- In Core bleibt nur `src/surroforge/exporters/base.py` (+ `__init__.py`).

`plugins/surroforge_professional/pyproject.toml`:
- `[project.entry-points."surroforge.exporters"]`:
  `pickle = "surroforge_professional.exporters.pickle_exporter:PickleExporter"`,
  `onnx = "surroforge_professional.exporters.onnx_exporter:ONNXExporter"`,
  `pt`-Ref auf `surroforge_professional.exporters.pt_exporter:PtExporter` aktualisieren.
- `[onnx]`-Extra um `skl2onnx>=1.16` erweitern (mlp/random-forest `to_onnx` brauchen es;
  bisher kam es implizit über Cores `[onnx]`-Extra).

Root-`pyproject.toml`:
- `[project.entry-points."surroforge.exporters"]`-Block (Zeilen 80–82) **entfernen**.
- Cores `[onnx]`-Extra **bleibt**: es trägt das Konvertierungs-Backend (`skl2onnx`/`onnx`/
  `onnxruntime`) für `polynomial.to_onnx`, das Apache-Core bleibt (s. Risiken).

`src/surroforge/plugins/registry.py`:
- `_BUILTIN[EXPORTER_GROUP] = {}` mit Kommentar analog zur Comparison-Reporter-Zeile
  („Core ships no exporters; pickle/onnx/pt come with surroforge_professional").

### §3 `train --onnx` durch den Seam (sanft)

`engine.py:282–288`: der Direktaufruf `ONNXExporter().export(...)` wird durch
`get_exporter("onnx")().export(...)` ersetzt — **innerhalb** des bestehenden
try/except-Blocks („ONNX is best-effort; never fail a run over it"):

- Lizenz vorhanden → Export wie heute, `onnx_exported: true`.
- professional installiert, keine Lizenz → `gate.check` wirft `PluginError` → Log
  `ONNX export skipped: 'onnx' requires a 'professional' license, and no installed
  SurroForge edition grants it. Set SURROFORGE_PRO_LICENSE or install
  ~/.surroforge/professional.lic.` — Run läuft durch, `onnx_exported: false`.
- professional nicht installiert → `PluginError` („Unknown export format 'onnx'…") →
  dieselbe Skip-Mechanik. Lineage-Schema unverändert.

`cli/main.py:345` (Post-Train-Hinweis „ONNX export available — run 'surroforge export …'"):
nur noch anzeigen, wenn der Exporter tatsächlich auflösbar ist — `get_exporter("onnx")` in
try/except `PluginError`; bei Fehlschlag entfällt der Hinweis (kein Hinweis auf einen
Befehl, der sofort scheitern würde).

### §4 Fehler-UX auf der Export-Oberfläche

CLI `export`, `Run.export` und `services.export_model` bleiben code-seitig unverändert
(`get_exporter` gated bereits). Ergebnismatrix:

| Situation | Verhalten |
|---|---|
| professional + gültige Lizenz | Export wie bisher (Deklarations-Gate `supports_export` unverändert dahinter) |
| professional installiert, keine/zu niedrige Lizenz | `PluginError` mit W1-Meldung + Aktivierungs-Hint (CLI: `Error: …`, exit 1) |
| professional nicht installiert | `PluginError: Unknown export format 'onnx'. Available: []` — Core bleibt editions-neutral (kennt keine Editionen); die Doku trägt die Erklärung |

Text-Nachzieher:
- `cli/main.py:525` Hilfetext `--format`: „Export format: onnx or pickle." → Editions-Hinweis
  („pickle/onnx need a Professional license; pt additionally the [torch] extra").
- `run.py:268–277` `Run.export`-Docstring: „``pickle`` and ``onnx`` are built in" → alle
  Formate kommen aus `surroforge_professional` (lizenzpflichtig; `pt` mit `[torch]`).

## Test-Strategie

**Core (neu):**
- `model_io`-Roundtrip: `dump_model`/`load_model` mit gefittetem Polynom-Trainer.
- `train --onnx` ohne Lizenz: Run wird grün, Log enthält „ONNX export skipped:",
  `onnx_exported is False`, `has_onnx() is False`. (Env/Lizenzdatei neutralisieren wie in
  den W1-Gating-Tests; die Assertion nur auf den Skip-Präfix stützen, damit sie sowohl mit
  professional [Gate-Meldung] als auch ohne [Unknown-format] besteht.)
- Core-only-Auflösung: mit leeren Entry-Points (Monkeypatch `_entry_points_grouped`, wie
  bestehende Registry-Tests) ist `available_exporters() == []` und `get_exporter("pickle")`
  wirft „Unknown export format".
- `tests/test_plugins.py:43–45` entsprechend umbauen (Erwartung „pickle/onnx builtin" fällt).

**Umzug nach professional:**
- `tests/test_exporters.py` → `plugins/surroforge_professional/tests/` (mit `licensed`-Fixture);
  `load_pickle`-Aufrufe auf `model_io.load_model` bzw. Exporter-Roundtrip umstellen.
- Neue Gating-Tests (Muster W1): `get_exporter("onnx"/"pickle"/"pt")` ohne Lizenz →
  `PluginError` mit Tier-Meldung; mit `SURROFORGE_PRO_LICENSE` → Klasse. (`pt` ist ohne
  torch auflösbar — Gate feuert vor jedem torch-Import.)

**Anpassung bestehender Core-Tests mit Export-Schritten:**
- Neues Fixture `licensed_export` in `tests/conftest.py`:
  `pytest.importorskip("surroforge_professional")` + `monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "test")`
  — auf einem Core-only-Checkout skippen diese Tests sauber, im Dev-Venv laufen sie voll.
- Nutzer: `tests/test_cli.py:87` (export onnx), `tests/test_scenarios.py:34`
  (Greenfield-Export), `tests/test_training.py:95` (onnx at train).
  `tests/test_training.py:86` (opt-in-Default) braucht **keine** Lizenz (exportiert nicht).

## Downstream (in W2 mitgenommen)

- **Beispiele:**
  - `01_quickstart.py:87`: Export-Schritt Core-tauglich machen — auf `run.model_path`
    verweisen (Run-Store) + Hinweis, dass `export_model`/CLI-Export Professional ist.
    Beispiel 01 bleibt damit ohne Lizenz lauffähig.
  - `06_mlp_surrogate.py:19` („export uses the optional `[onnx]` extra"): auf
    `surroforge_professional[onnx]` umschreiben (Beispiel verlangt eh schon eine Lizenz
    für mlp; der pickle-Fallback ist damit ebenfalls gedeckt).
- **Doku:** `docs/quickstart.md` §8 (Zeilen 119–122: „always available" / „needs onnx
  extra" → Professional-Hinweis); `README.md` (Quickstart-Export-Zeilen + Kommentar der
  `[onnx]`-Extra-Zeile: Konvertierungs-Backend, Export selbst ist Professional);
  `docs/reference/cli.md` (Zeilen 36–38 train `--onnx` + `export`-Abschnitt ab Zeile 59);
  `docs/reference/python-api.md` (Zeilen 60/78/91: `fit(export_onnx=…)`, `has_onnx`,
  `export_model`). PRD ist seit W1 korrekt (§3.2: Export = Pro).
- **MEMORY.md:** Architektur-Zeile `exporters/` + „Offene Punkte" (W2 abhaken).
- **CHANGELOG:** Breaking-Eintrag (Export lizenzpflichtig; `load_pickle` → `model_io`).
- **Dev-Umgebung:** beide editierbaren Installs auffrischen (`pip install -e .` +
  `-e plugins/surroforge_professional`), damit die Entry-Point-Metadata die
  Exporter-Verschiebung sieht; `scripts/ci.sh` unverändert (globbt `plugins/*/tests`).

## Außerhalb von Welle 2

- reproduce/Lineage-Oberflächen-Gating + `gate.require_license(feature)` → **W3**.
- Echte Lizenz-Validierung/Aktivierung → **M3**.

## Risiken / Ehrlichkeits-Notizen

- **Produktgrenze, keine Sicherheitsgrenze:** `model.pkl` liegt im Run-Verzeichnis des
  Nutzers (Store bleibt bewusst frei), `polynomial.to_onnx` bleibt Apache-Core und ist
  aus Python direkt aufrufbar, und die Konformanz-Suite pickelt direkt. Das Gate schützt
  die **Bequemlichkeits-Oberfläche** (CLI/API-Export) — konsistent mit dem Lizenz-Stub
  bis M3. Bewusst akzeptiert.
- **Fehler-UX ohne professional** („Unknown export format 'onnx'. Available: []") nennt
  keine Editionen — Preis der Editions-Neutralität von Core; Doku/README tragen die
  Erklärung. Bewusst akzeptiert (W1-Prinzip).
- **Artefakt-Identität** Run-Store ↔ pickle-Export ist über die gemeinsame
  `dump_model`-Implementierung strukturell gesichert (kein Format-Drift möglich).
- **Entry-Point-Kollision** bei veralteten Installs (Root-Paket alt, professional neu):
  durch das Entfernen der Root-EPs + Editable-Reinstall im selben Schritt adressiert;
  die W1-`_resolve`-Kollisionsmeldung fängt den Restfall laut ab.
