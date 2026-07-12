# SurroForge — Projekt-Memory

Kontextdokument für Entwickler (und KI-Assistenten). Hält fest, was nicht aus dem
Code allein ersichtlich ist: Architektur, Befehle, Konventionen und Entscheidungen.

## Was ist SurroForge?

CLI-Werkzeug (mit spiegelgleicher Python-API) zum Erstellen, Validieren und
Verwalten von **Surrogate-Modellen** aus tabellarischen FEM/CFD-Simulationsdaten.
Open Core unter Apache 2.0. Maßgeblich ist `PRD.md`; **Phase 1 (MVP)** ist umgesetzt,
und der aktuelle Stand geht bereits darüber hinaus: Field-ROM/OpenFOAM-Erweiterungen
sowie die Editionen-Aufteilung (Welle 1): NN-/Ensemble-Methoden (`mlp`, `random-forest`,
`gradient-boosting`, `torch-mlp`) sind lizenzpflichtig in `surroforge_professional`,
`autoencoder-field` ist Enterprise-tier in `surroforge_enterprise`.

**Kernversprechen:** vom Solver-Output zum validierten, reproduzierbaren
Surrogate-Modell — Daten, Konfiguration und Modelle sauber versioniert.

## Umgebung & Befehle

- **venv:** `.venv/` (Python 3.13, gebaut aus `/opt/local/bin/python3`).
  Das System-`/usr/bin/python3` ist 3.9 und erfüllt die Anforderung `>=3.10` nicht.
- **Installieren:** `.venv/bin/pip install -e '.[all,dev]'`
- **CLI:** `.venv/bin/surroforge --help`
- **Tests Core:** `.venv/bin/python -m pytest tests` (aktuell 430 gesammelt)
- **Tests Plugins:** `.venv/bin/python -m pytest plugins/surroforge_professional/tests`
  und `.venv/bin/python -m pytest plugins/surroforge_enterprise/tests`
- **Lint/Typen:** `.venv/bin/ruff check src tests` · `.venv/bin/mypy src`

Vor Release die Gates aus `RELEASING.md` frisch ausführen.

## Architektur (src-Layout)

```
src/surroforge/
├── cli/main.py      # Typer-CLI; alle Befehle aus PRD 4.2; _guard rendert Fehler sauber
├── core/            # config (Pydantic), project, run, dataset
├── data/            # loader (CSV/Parquet/HDF5), OpenFOAM/mesh readers, hashing
├── trainers/        # base + polynomial, kriging/rbf, reduced-field (mlp/RF/GB: professional)
├── validators/      # metrics (RMSE/R²/MAE + field metrics), cross_validation
├── exporters/       # nur base (Schnittstelle); pickle/onnx/pt sind Professional (Welle 2)
├── reporters/       # base, markdown; html kommt über surroforge_professional
├── persistence/     # store: atomare I/O, Run-IDs, Environment-Capture; model_io: Run-Store-Pickle
├── plugins/         # registry/gate/seam: Entry-Points + Builtin-Fallback
├── api/highlevel.py # Project, Trainer (PRD 4.10)
├── engine.py        # run_training: Orchestrator (Daten→Split→CV→Final-Modell→Persistenz)
└── services.py      # reproduce / compare / predict / export / report
```

Datenfluss eines Laufs: `engine.run_training()` löst die Datenquelle auf, splittet
Train/Test, berechnet Holdout- und CV-Metriken, fittet das **finale Modell auf allen
Daten** (das gespeicherte Artefakt) und schreibt `runs/<id>/` mit `config.snapshot.yaml`,
`model.pkl`, optional `model.onnx`, `metrics.json`, `lineage.json`, `log.txt`.

## Wichtige Designentscheidungen (nicht offensichtlich aus dem Code)

- **Run-IDs sequenziell** `run-001` (PRD offene Frage 4 so entschieden).
- **Plugin-Registry:** Trainer/Exporter/Reporter werden über Entry-Point-Gruppen
  (`surroforge.trainers|exporters|reporters`) entdeckt. Eine Builtin-Fallback-Tabelle
  sorgt dafür, dass ein Source-Checkout ohne `pip install` ebenfalls funktioniert.
  Externe/kommerzielle Pakete nutzen denselben Mechanismus (`surroforge_professional`,
  `surroforge_enterprise`); lizenzpflichtige Trainer deklarieren `required_tier`, den
  die Registry über die `surroforge.authorizers`-Entry-Points durchsetzt.
- **Optionale Backends lazy:** SMT (Kriging/RBF), ONNX, meshio und schwere Plugin-Backends
  werden erst zur Laufzeit bzw. über Extras/Plugins geladen; fehlende Backends scheitern
  beim `fit`/`export`/Importpfad mit klarer Meldung.
- **Kriging/RBF sind single-output** (SMT-Limitierung). Multi-Output ist über `mlp`,
  `torch-mlp` und Feld-ROM-Pfade abgedeckt, nicht über SMT.
- **Professional-HTML ist Plugin-only:** Core bleibt Markdown-only. Der HTML-Reporter lebt
  in `plugins/surroforge_professional/`, registriert `html` per Entry Point und nutzt in v1
  nur eine lokale, tier-aware Lizenzprüfung (`SURROFORGE_PRO_LICENSE` oder Datei; Wert mit
  „enterprise" schaltet die Enterprise-Stufe frei), keine Online-Aktivierung.
- **Reproduzierbarkeit:** Jeder Lauf speichert einen *Prediction-Fingerprint*
  (SHA-256 über Vorhersagen auf den Trainingsinputs). Seit Welle 3 sind die
  Oberflächen (reproduce, Herkunfts-Anzeige in info/Report/compare) Professional;
  die Erfassung bleibt Core. `reproduce` vergleicht ihn
  → Polynom reproduziert bit-identisch. Daten-Hash: bei registrierten Datensätzen
  der gespeicherte Metadaten-Hash; bei Pfad-Quellen aus der Datei neu berechnet.
- **Fehler-UX:** Domänenfehler erben von `SurroForgeError`; der `_guard`-Dekorator
  in der CLI rendert sie als `Error: …` (kein Traceback), exit 1.

## Konventionen

- Konfiguration ist **eine** YAML-Datei = vollständige Spezifikation eines Laufs
  (PRD 4.3), validiert mit Pydantic. Pflichtfelder: `experiment`, `data.source`,
  `data.inputs`, `data.outputs`, `model.method`, `validation.cross_validation`.
- `model.<extra>` = methodenspezifische Hyperparameter (Pydantic `extra="allow"`).
- `validation.cross_validation: 5` ist Kurzform für `{folds: 5}`.
- CLI-/API-Ausgaben primär Englisch; Code-Kommentare/Docstrings erklären das *Warum*
  und referenzieren PRD-Abschnitte.
- Tests liegen unter `tests/` und importieren Helfer via `from tests.conftest import …`
  (`pythonpath=["."]` in `pyproject.toml`).

## Teststruktur

Core-Tests liegen unter `tests/` und decken Konfiguration, Daten, Training, Export,
Reports, Reproduktion, Plugins, CLI/API, Field-ROM, OpenFOAM, Conformance und die drei
PRD-Akzeptanzszenarien (Greenfield, Methodenvergleich, Reproduktion) ab.
Plugin-Tests liegen plugin-lokal unter `plugins/*/tests/`. `tests/conftest.py` liefert
Fixtures: `sample_csv`, `project` (mit Datensatz „beam“), `trained_run`, sowie
`make_samples()`/`poly_config()`.

## Offene Punkte / Nächste Schritte

- Git ist initialisiert; `main` läuft gegen `origin/main`.
- Implementierungspläne liegen als getrackte Artefakte unter `docs/superpowers/plans/`
  (`2026-06-23-professional-html-reports.md`, `2026-06-25-autoencoder-field-plugin.md`).
- Phase 2 ist weit fortgeschritten (`surroforge_professional` mit mlp/RF/GB/torch-mlp,
  `surroforge_enterprise` mit autoencoder-field, OpenFOAM/Field-ROM, DoE, erweiterte
  Run-Vergleiche), aber nicht vollständig: offen sind u. a. skalierbare GP,
  echte Lizenz-Aktivierung (M3) und Support-Infrastruktur.
- Release-Kandidat für die aktuellen `[Unreleased]`-Features prüfen: Version/Changelog
  schneiden, Release-Gates aus `RELEASING.md` laufen lassen, dann taggen/publishen.
- Produktvalidierung/Interviews/Pilotkunden sind für die aktuelle Engineering-Queue
  nur **Nice to have**; nicht als blockierende Vorarbeit für Release oder Phase-2-Features
  behandeln.
