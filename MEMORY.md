# SurroForge — Projekt-Memory

Kontextdokument für Entwickler (und KI-Assistenten). Hält fest, was nicht aus dem
Code allein ersichtlich ist: Architektur, Befehle, Konventionen und Entscheidungen.

## Was ist SurroForge?

CLI-Werkzeug (mit spiegelgleicher Python-API) zum Erstellen, Validieren und
Verwalten von **Surrogate-Modellen** aus tabellarischen FEM/CFD-Simulationsdaten.
Open Core unter Apache 2.0. Maßgeblich ist `PRD.md`; implementiert ist **Phase 1
(MVP)** aus PRD-Abschnitt 9.1.

**Kernversprechen:** vom Solver-Output zum validierten, reproduzierbaren
Surrogate-Modell — Daten, Konfiguration und Modelle sauber versioniert.

## Umgebung & Befehle

- **venv:** `.venv/` (Python 3.13, gebaut aus `/opt/local/bin/python3`).
  Das System-`/usr/bin/python3` ist 3.9 und erfüllt die Anforderung `>=3.10` nicht.
- **Installieren:** `.venv/bin/pip install -e '.[all,dev]'`
- **CLI:** `.venv/bin/surroforge --help`
- **Tests:** `.venv/bin/python -m pytest` (116 Tests)
- **Lint/Typen:** `.venv/bin/ruff check src tests` · `.venv/bin/mypy src`

Status: ruff sauber, mypy ohne Befunde, 116 Tests grün.

## Architektur (src-Layout)

```
src/surroforge/
├── cli/main.py      # Typer-CLI; alle Befehle aus PRD 4.2; _guard rendert Fehler sauber
├── core/            # config (Pydantic), project, run, dataset
├── data/            # loader (CSV/Parquet/HDF5), hashing (SHA-256)
├── trainers/        # base + polynomial (sklearn), kriging/rbf (SMT)
├── validators/      # metrics (RMSE/R²/MAE), cross_validation (k-fold)
├── exporters/       # base, pickle (immer), onnx
├── reporters/       # base, markdown (Jinja2)
├── persistence/     # store: atomare I/O, Run-IDs, Environment-Capture
├── plugins/         # registry: Entry-Points + Builtin-Fallback
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
  Die kommerziellen Editionen sollen später denselben Mechanismus nutzen.
- **Optionale Backends lazy:** SMT (Kriging/RBF) und ONNX werden erst zur Laufzeit
  importiert; die Methoden erscheinen in der Registry auch ohne installiertes SMT
  und scheitern erst beim `fit`/`export` mit klarer Meldung.
- **Kriging/RBF sind im MVP single-output** (SMT-Limitierung). Multi-Output ist
  laut PRD ohnehin Phase 2 (Professional).
- **Reproduzierbarkeit:** Jeder Lauf speichert einen *Prediction-Fingerprint*
  (SHA-256 über Vorhersagen auf den Trainingsinputs). `reproduce` vergleicht ihn
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

Pro Modul eine Datei (`test_config`, `test_data`, `test_metrics`,
`test_cross_validation`, `test_trainers`, `test_exporters`, `test_reporters`,
`test_persistence`, `test_project`, `test_training`, `test_reproduce`,
`test_plugins`, `test_api`, `test_cli`) plus `test_scenarios.py` für die drei
PRD-Akzeptanzszenarien (Greenfield, Methodenvergleich, Reproduktion).
`tests/conftest.py` liefert Fixtures: `sample_csv`, `project` (mit Datensatz „beam"),
`trained_run`, sowie `make_samples()`/`poly_config()`.

## Offene Punkte / Out of Scope (MVP)

- Noch **kein Git-Repo** initialisiert.
- Phase 2/3 (GP/RF/Boosting/NN, HTML/PDF-Reports, Active Learning, Multi-Fidelity,
  DoE, Lizenzmechanik) sind bewusst nicht enthalten — siehe PRD 9.2/9.3.
- ONNX-Export nur für Methoden mit `supports_onnx=True` (aktuell Polynom);
  SMT-Modelle exportieren als Pickle.
