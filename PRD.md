# Product Requirements Document — SurroForge

**Version:** 0.1
**Datum:** Mai 2026
**Status:** Konzeptphase — Grundlage für die MVP-Entwicklung

---

## 1. Überblick

### 1.1 Produkt

**SurroForge** ist ein Command-Line-Werkzeug zur Erstellung, Validierung und Verwaltung von Surrogate-Modellen für FEM- und CFD-Simulationsdaten im klassischen Maschinenbau, in der Luftfahrt- und Raumfahrt, der Fahrzeugtechnik, Medizintechnik, etc. .

### 1.2 Vision

Surrogate-Modelle werden zum selbstverständlichen Werkzeug jedes Berechnungsingenieurs — so alltäglich wie ein Mesh-Generator oder ein Post-Processor.

### 1.3 Mission

Wir bauen ein Open-Core-Werkzeug, mit dem Ingenieure aus Simulationsdaten reproduzierbare Surrogate-Modelle trainieren, vergleichen und teilen können — skriptbar, versionierbar und ohne Vendor-Lock-in.

### 1.4 Kernversprechen

Vom Solver-Output zum validierten Surrogate-Modell in unter einer Stunde — mit allen Daten, Konfigurationen und Modellen sauber versioniert und nach Monaten oder Jahren reproduzierbar.

### 1.5 MVP-Ansatz

**MVP** steht für *Minimum Viable Product* — die kleinste auslieferbare Version von SurroForge, die einem Berechnungsingenieur echten Nutzen bringt und an realen Nutzern getestet werden kann. Statt von Anfang an alle Methoden, Editionen und Komfortfunktionen zu bauen, liefert das MVP **nur den durchgängigen Kern-Workflow** und lernt aus echter Nutzung, bevor wir in die Breite investieren.

**Das Leitprinzip:** Baue das Kleinste, das veröffentlicht werden kann, um etwas Wichtiges zu lernen — dann iteriere.

**Was das MVP von SurroForge leistet** (vollständig in Abschnitt 9.1 spezifiziert):

- Ein durchgängiger Pfad vom registrierten Datensatz bis zum validierten, exportierten Surrogate-Modell — über CLI und Python-API
- Drei bewusst etablierte Methoden (Kriging, RBF, polynomiale Regression) statt einer breiten Methodenpalette
- Reproduzierbarkeit und Run-Persistenz von Tag eins, weil dies das Kernversprechen (Abschnitt 1.4) und Hauptdifferenzierungsmerkmal ist — nicht ein späteres Add-on
- Markdown-Reports und ONNX-Export als minimale, aber vollständige Ausgabekette
- Plugin-Schnittstellen, damit der spätere Ausbau (Pro/Enterprise) additiv erfolgt und den Kern nicht umbaut

**Was das MVP bewusst weglässt:** erweiterte Methoden (GP, RF, Boosting, NN), interaktive HTML-/PDF-Reports, Active Learning, Multi-Fidelity, DoE-Generierung sowie die gesamte Lizenz- und Monetarisierungs-Mechanik. Diese gehören in Phase 2 und 3 (Abschnitte 9.2, 9.3) und werden erst gebaut, wenn das MVP seine kritischen Annahmen (Abschnitt 11.1) bestätigt hat.

**Warum dieser Zuschnitt:** Das MVP soll die teuerste Unsicherheit zuerst beantworten — ob der Open-Source-Kern messbare Adoption in der CAE-Community erzeugt (Annahme 2) und ob Reproduzierbarkeit als Differenzierungsmerkmal trägt. Erst wenn das belegt ist, rechtfertigt sich die Investition in kommerzielle Editionen. Die Definition of Done für Phase 1 (drei externe Testnutzer durchlaufen die drei Standard-Szenarien erfolgreich) ist genau dieser Lerntest.

---

## 2. Zielnutzer

### 2.1 Primäre Persona

**Methoden-Engineer** in einem CAE-Team mittelständischer Maschinenbau-Unternehmen. Python-affin, arbeitet in der Kommandozeile, kennt SMT und scikit-learn, will Workflows standardisieren.

### 2.2 Sekundäre Persona

**Doktorand oder wissenschaftliche Hilfskraft** in einer Forschungsgruppe mit Engineering-Fokus. Lernt Surrogate-Methoden, braucht niedrige Einstiegshürden, wirkt als Multiplikator in die Industrie.

### 2.3 Käuferpersona

**Leitung von Simulationsabteilung** Ingenieurbüro oder KMU oder Konzern. Entscheidet über die Beschaffung der kommerziellen Editionen. Nutzt das Tool nicht selbst, sieht aber Reports und Audit-Outputs.

---

## 3. Geschäftsmodell und Editionen

### 3.1 Modell

Open Core. Der Produktkern ist Open Source unter Apache 2.0, kommerzielle Erweiterungen werden als separate Module unter proprietärer Lizenz angeboten.

### 3.2 Edition-Matrix

| Funktion | Core | Professional | Enterprise |
|---|---|---|---|
| Basis-Methoden (Kriging, RBF, Polynom) | ✓ | ✓ | ✓ |
| Cross-Validation, Standard-Metriken | ✓ | ✓ | ✓ |
| ONNX-Export, Markdown-Reports | ✓ | ✓ | ✓ |
| Run-Persistenz, Reproduzierbarkeit | ✓ | ✓ | ✓ |
| Python-API und Plugin-Schnittstellen | ✓ | ✓ | ✓ |
| Erweiterte Methoden (GP, RF, Gradient Boosting, NN) | — | ✓ | ✓ |
| HTML-Reports mit interaktiven Visualisierungen | — | ✓ | ✓ |
| Multi-Output-Modelle, erweiterte Vergleiche | — | ✓ | ✓ |
| Active Learning, Multi-Fidelity | — | — | ✓ |
| Audit-Reports (PDF) mit Lineage-Tracking | — | — | ✓ |
| Floating-Lizenzen, Site-License | — | — | ✓ |
| SLA-Support, Customer Success | — | — | ✓ |

### 3.3 Lizenzaktivierung

Core funktioniert ohne Aktivierung. Professional und Enterprise werden über Lizenzdatei oder Online-Aktivierung freigeschaltet. Offline-Toleranz für Enterprise: bis zu 30 Tage.

---

## 4. Funktionale Anforderungen

### 4.1 Kern-Workflow

Ein typischer Nutzerablauf umfasst:

1. Projekt initialisieren
2. Datensatz registrieren (mit Validierung)
3. Surrogate-Modell trainieren nach YAML-Konfiguration
4. Modell validieren (Cross-Validation)
5. Mehrere Trainingsläufe vergleichen
6. Bestes Modell exportieren oder Report generieren
7. Modell für Vorhersagen oder externe Anwendungen nutzen

### 4.2 CLI-Befehle (MVP)

| Befehl | Zweck |
|---|---|
| `surroforge init <name>` | Projektverzeichnis anlegen |
| `surroforge data add <file>` | Datensatz registrieren, Schema validieren |
| `surroforge data list` | Registrierte Datensätze auflisten |
| `surroforge train --config <yaml>` | Surrogate trainieren |
| `surroforge validate <run-id>` | Modell validieren, Metriken anzeigen |
| `surroforge compare --runs <ids>` | Trainingsläufe gegenüberstellen |
| `surroforge predict <run-id> --input <file>` | Vorhersagen mit trainiertem Modell |
| `surroforge export <run-id> --format <fmt>` | Modell exportieren |
| `surroforge report <run-id>` | Report generieren |
| `surroforge list` | Alle Runs im Projekt auflisten |
| `surroforge reproduce <run-id>` | Run identisch nachvollziehen |
| `surroforge info <run-id>` | Detailinfos zu einem Run |

Globale Optionen: `--verbose / -v`, `--quiet / -q`, `--config-dir`, `--help`.

### 4.3 Konfigurationsformat

Trainingsläufe werden über YAML-Dateien definiert. Eine YAML-Datei ist die vollständige und einzige Spezifikation eines Trainingslaufs.

Pflichtfelder:

- `experiment`: Identifikator des Experiments
- `data.source`: Pfad zur Datenquelle
- `data.inputs`: Liste der Input-Spaltennamen
- `data.outputs`: Liste der Output-Spaltennamen
- `model.method`: Methodenname
- `validation.cross_validation`: Validierungsstrategie

Optionale Felder mit Defaults:

- `data.split.method` (Default: random)
- `data.split.test_fraction` (Default: 0.2)
- `data.split.seed` (Default: 42)
- `model.<methodenspezifische Hyperparameter>`
- `validation.metrics` (Default: `[rmse, r2, mae]`)

### 4.4 Datenformate

**Eingabe:** CSV, Parquet, HDF5. Tabellarisch, mit benannten Spalten für Inputs und Outputs.

**Ausgabe:**
- Modell als Pickle und ONNX
- Metadaten als JSON
- Reports als Markdown (Core), HTML (Pro), PDF-Audit (Enterprise)
- Konfigurations-Snapshot als YAML

### 4.5 Run-Persistenz und Reproduzierbarkeit

Jeder Trainingslauf erzeugt einen Run mit:

- Eindeutiger ID (z. B. `run-001` oder UUID)
- Snapshot der vollständigen Konfiguration
- Daten-Hash (SHA-256 über die verwendeten Daten)
- Random Seed
- Python-Version und Bibliotheksversionen
- Trainings-Metriken (Train, Test, Cross-Validation)
- Modell-Artefakt

`surroforge reproduce <run-id>` muss bitidentische Modelle erzeugen, sofern Python- und Bibliotheksversionen passen.

### 4.6 Methoden (MVP — Core)

- **Kriging** (über SMT)
- **Radial Basis Functions** (über SMT)
- **Polynomielle Regression** (über scikit-learn)

Cross-Validation: k-fold mit konfigurierbarem k. Standardmetriken: RMSE, R², MAE.

### 4.7 Methoden (Phase 2 — Professional)

- **Gaussian Processes** (über GPyTorch, mit skalierbaren Varianten)
- **Random Forest** (über scikit-learn, mit Hyperparameter-Tuning)
- **Gradient Boosting** (über XGBoost oder scikit-learn)
- **Neuronale Netze** als optionales Backend (über PyTorch)

### 4.8 Methoden (Phase 3 — Enterprise)

- **Active Learning** mit Acquisition-Funktionen (über BoTorch)
- **Multi-Fidelity-Modelle** (Co-Kriging, hierarchische Modelle)

### 4.9 Reports

**Markdown-Report (Core):**
- Daten-Übersicht
- Modell-Spezifikation
- Validierungs-Metriken
- Konfigurationsdetails

**HTML-Report (Professional):**
- Alles aus Markdown
- Interaktive Plots (Residuen, Prediction vs. Actual, partielle Abhängigkeiten)
- Vergleichs-Tabellen

**PDF-Audit-Report (Enterprise):**
- Alles aus HTML als statisches PDF
- Vollständiges Lineage-Tracking (Daten-Provenance, Konfigurations-Historie)
- Konformitäts-Anhang für V&V-Dokumentation

### 4.10 Python-API

Vollständig parallel zur CLI. Alle Operationen sind über Python aufrufbar:

```python
from surroforge import Project, Trainer

project = Project.open("path/to/project")
trainer = Trainer.from_config("configs/kriging.yaml")
run = trainer.fit(project.dataset("my_data"))
print(run.metrics)
```

Die API folgt denselben Konzepten und Begriffen wie die CLI.

### 4.11 Plugin-System

Externe Methoden, Validierungsstrategien, Exporter und Reporter können als Python-Pakete bereitgestellt werden und werden via Entry-Points registriert.

Plugin-Punkte (MVP):
- `surroforge.trainers` — neue Methoden
- `surroforge.exporters` — zusätzliche Export-Formate
- `surroforge.reporters` — zusätzliche Report-Templates

Kommerzielle Editionen (Professional, Enterprise) werden technisch als interne Plugins mit Lizenz-Check implementiert. Drittanbieter dürfen eigene Plugins entwickeln; Stabilitätsgarantie ab Version 1.0.

---

## 5. Nicht-funktionale Anforderungen

### 5.1 Performance

- Training eines Kriging-Modells auf 200 Punkten mit 5 Inputs: unter 5 Sekunden
- Training eines Random-Forest auf 10.000 Punkten: unter 30 Sekunden
- Cross-Validation mit k=5 darf das Training maximal um Faktor 6 verlangsamen
- Daten-Hashing bei `data add`: unter 5 Sekunden für Datensätze bis 100 MB

### 5.2 Reproduzierbarkeit

- Jeder Run ist vollständig reproduzierbar bei identischer Python- und Bibliotheksversion
- Konfiguration, Daten-Hash, Seed und Versionen werden automatisch erfasst — kein Opt-Out

### 5.3 Plattform-Support

- Primär: Linux (Ubuntu 22.04+), macOS (12+), Windows 10/11
- Python 3.10 oder höher
- Distribution über PyPI (`pip install surroforge`) und optional über conda-forge

### 5.4 Stabilität und Versionierung

- Semantic Versioning ab Version 1.0
- CLI- und API-Verträge sind ab Version 1.0 backward-kompatibel innerhalb einer Major-Version
- Deprecation-Hinweise mindestens eine Minor-Version vor Entfernung

### 5.5 Sicherheit

- Keine ausgehende Netzwerkverbindung in der Core-Edition (außer expliziter Update-Check, opt-in)
- Lizenzprüfung in Pro/Enterprise erfolgt lokal oder über vom Nutzer konfigurierte Endpunkte
- Keine Telemetrie ohne Opt-In; falls aktiviert, transparent und abschaltbar

### 5.6 Sprache

- CLI- und API-Ausgaben primär Englisch
- Deutsche Lokalisierung optional konfigurierbar (Phase 2)
- Dokumentation primär Englisch, deutsche Kurzversion gepflegt

---

## 6. Technische Architektur

### 6.1 Tech-Stack

| Komponente | Wahl |
|---|---|
| Sprache | Python 3.10+ |
| CLI-Framework | Typer (basiert auf Click) |
| Konfiguration | YAML (PyYAML), validiert mit Pydantic |
| Surrogate-Backends | SMT, scikit-learn, GPyTorch, BoTorch (Pro/Enterprise) |
| Datenhaltung | Pandas, PyArrow, h5py |
| Hashing/Persistenz | SHA-256, JSON, Pickle |
| ONNX-Export | skl2onnx, onnxruntime |
| Reports | Markdown direkt; HTML via Jinja2 + Plotly; PDF via WeasyPrint |
| Testing | pytest, hypothesis |
| Linting/Formatting | ruff, mypy |
| Dokumentation | MkDocs Material |

### 6.2 Modulstruktur (grob)

```
surroforge/
├── cli/              # Typer-Befehle
├── core/             # Projekt, Run, Dataset, Konfiguration
├── data/             # Datenimport, Validierung, Hashing
├── trainers/         # Basis-Methoden (Core)
├── validators/       # Cross-Validation, Metriken
├── exporters/        # ONNX und Standard-Formate
├── reporters/        # Markdown-Report
├── persistence/      # Run-Speicherung, Lineage
├── plugins/          # Plugin-Registry und Entry-Points
└── api/              # Hochlevel-Python-API
```

Kommerzielle Module liegen in separaten Paketen:
```
surroforge-professional/  # Erweiterte Methoden, HTML-Reports
surroforge-enterprise/    # Active Learning, Multi-Fidelity, Audit-PDF
```

### 6.3 Abhängigkeiten und Lizenzen

Core-Abhängigkeiten ausschließlich permissiv lizenziert (Apache, MIT, BSD). Keine GPL- oder LGPL-Abhängigkeiten im Core, um die Kommerzialisierungsstrategie nicht zu gefährden.

### 6.4 Lizenzierung (Code)

- Core: Apache 2.0
- Professional und Enterprise: proprietär, separate Lizenz
- Drittanbieter-Plugins: freie Lizenzwahl

---

## 7. Datenmodell

### 7.1 Projektstruktur

```
mein-projekt/
├── surroforge.yaml          # Projekt-Metadaten
├── data/                    # Eingangsdaten (versioniert oder ignored)
├── configs/                 # YAML-Konfigurationen pro Experiment
├── runs/                    # Trainingsergebnisse
│   ├── run-001/
│   │   ├── config.snapshot.yaml
│   │   ├── model.onnx
│   │   ├── model.pkl
│   │   ├── metrics.json
│   │   ├── lineage.json
│   │   └── log.txt
│   └── run-002/
└── reports/                 # Generierte Reports
```

### 7.2 Run-Metadaten (lineage.json)

Pflichtfelder pro Run:
- `run_id`
- `created_at` (ISO 8601)
- `config_hash`
- `data_hash`
- `seed`
- `python_version`
- `surroforge_version`
- `dependencies` (Liste mit Versionen)
- `parent_run` (für Reproduktionen)

### 7.3 Projekt-Metadaten (surroforge.yaml)

- Projektname
- Schema-Version
- Erstellungsdatum
- Optionale Beschreibung
- Liste registrierter Datensätze mit Hash und Schema

---

## 8. Out of Scope

Folgendes wird bewusst nicht gebaut:

- **Solver-Anbindung:** Keine direkten FEM/CFD-Aufrufe (ANSYS, OpenFOAM, Abaqus). Nutzer importieren Ergebnisse als Tabelle.
- **Mesh-Generierung und Pre-Processing:** Liegt außerhalb des Scopes.
- **Optimierung, Sensitivitätsanalyse, UQ:** Werden von anderen Werkzeugen (pymoo, SALib, chaospy) übernommen, die SurroForge-Modelle konsumieren können.
- **Eigene GUI:** Auch langfristig nicht primäres Interface. Optional ein Lokal-Dashboard für Visualisierung (Phase 3+).
- **Cloud-Plattform:** Lokales Werkzeug. Keine Multi-User-, Hosting- oder SaaS-Funktionalität.
- **DoE-Generierung:** Erst ab Phase 2 als ergänzendes Submodul (`surroforge doe ...`), kein MVP-Bestandteil.

---

## 9. Roadmap

### 9.1 Phase 1 — MVP (Monate 1–4)

**Ziel:** Erste öffentliche Version des Open-Source-Kerns auf PyPI und GitHub.

Inhalte:
- CLI-Skeleton mit allen Befehlen aus Abschnitt 4.2
- Drei Methoden: Kriging, RBF, polynomiale Regression
- YAML-Konfiguration mit Pydantic-Validierung
- Cross-Validation und Standard-Metriken
- Run-Persistenz mit Hashing
- ONNX-Export
- Markdown-Reports
- Vollständige Python-API
- Plugin-Schnittstellen
- Dokumentation mit Tutorials und API-Referenz

**Definition of Done:** Drei externe Testnutzer durchlaufen die drei Standard-Szenarien (Greenfield, Methoden-Vergleich, Reproduktion) erfolgreich.

### 9.2 Phase 2 — Professional (Monate 5–9)

**Ziel:** Erste kommerzielle Edition. Beginn der Monetarisierung.

Inhalte:
- Erweiterte Methoden: Gaussian Processes, Random Forest, Gradient Boosting, NN
- HTML-Reports mit interaktiven Visualisierungen
- Multi-Output-Modelle
- Erweiterte Run-Vergleiche
- Lizenz-Aktivierungs-Mechanik
- E-Mail-Support-Infrastruktur
- DoE-Modul als ergänzendes Submodul

**Definition of Done:** Drei zahlende Pilotkunden, dokumentierte Konversion vom Core.

### 9.3 Phase 3 — Enterprise (Monate 10–18)

**Ziel:** Enterprise-fähiges Produkt. Audit-Funktionen, größere Vertragsvolumen.

Inhalte:
- Active Learning mit Acquisition-Funktionen
- Multi-Fidelity-Modelle
- PDF-Audit-Reports mit Lineage-Tracking
- Floating-Lizenz-Server
- Compliance-Datenblatt und IT-Freigabe-Material
- Customer-Success- und Schulungsangebot

**Definition of Done:** Erste Enterprise-Kunde mit Site-License oder mehreren Floating-Lizenzen.

---

## 10. Erfolgskriterien

### 10.1 Pro Phase

**Phase 1:**
- 500+ PyPI-Downloads im ersten Monat nach Release
- 50+ GitHub-Stars in 3 Monaten
- Mindestens 3 externe Issues oder Pull Requests
- 100+ aktive Installationen (Telemetrie opt-in) in 6 Monaten

**Phase 2:**
- 3+ zahlende Pilotkunden
- Konversionsrate Core → Professional über 1 %
- Erste Konferenz- oder Community-Erwähnung

**Phase 3:**
- 1+ Enterprise-Kunde mit Vertragsvolumen über 20.000 €
- Audit-Reports von echten Kunden in V&V-Kontexten genutzt
- Etabliertes Customer-Success-Modell

### 10.2 Übergreifend

- Time-to-first-trained-model unter einer Stunde von Installation
- Reproduktions-Erfolg in über 95 % der Fälle (gemessen in CI gegen frühere Releases)
- Dokumentation hält über 80 % der Erstnutzer ohne externe Unterstützung

---

## 11. Annahmen und Risiken

### 11.1 Kritische Annahmen

1. Der Schmerz inkonsistenter Workflows ist in der Zielgruppe real und teuer genug.
2. Der Open-Source-Kern erzeugt messbare Adoption in der CAE-Community.
3. Mindestens 1–3 % der aktiven Open-Source-Nutzer konvertieren in 12 Monaten zur Professional-Edition.
4. CAE-Leitungen akzeptieren Open Core als kommerziell glaubwürdiges Modell.
5. Der adressierbare Markt im DACH-Maschinenbau ist groß genug für 100+ zahlende Kunden.

Diese Annahmen werden vor und während der Entwicklung durch Interviews und frühe Releases validiert.

### 11.2 Technische Risiken

- **Plugin-Trennlinie:** Saubere Trennung zwischen Core- und kommerziellen Modulen ist architektonisch anspruchsvoll. Mitigationsstrategie: Plugin-Schnittstellen früh definieren, alle erweiterten Methoden konsistent als Plugins implementieren.
- **Reproduzierbarkeit über Python-/Bibliotheksversionen:** Numerische Unterschiede zwischen NumPy- oder SciPy-Versionen können Bit-Identität verhindern. Mitigationsstrategie: Versionspinning im Run-Snapshot, tolerante Reproduzierbarkeits-Tests (z. B. relative Toleranz).
- **Performance bei großen Datensätzen:** Gaussian Processes skalieren schlecht über 10.000 Punkten. Mitigationsstrategie: Skalierbare GP-Varianten in Pro-Edition (Sparse GP, GPyTorch), klare Dokumentation der Grenzen.

### 11.3 Geschäftliche Risiken

- **Konversionsrate niedriger als geplant:** Mitigationsstrategie: konservative Geldplanung, Pilot-Programme früh starten, Akademie-Lizenzen für Multiplikator-Effekt.
- **Konkurrenz durch Enterprise-Suiten oder neue ML-Engineering-Startups:** Mitigationsstrategie: scharfe Positionierung über Reproduzierbarkeit und Open Core, nicht über Methoden-Breite.

---

## 12. Offene Fragen

Diese werden vor MVP-Release entschieden:

1. **CLI-Sprache:** Englisch als Default fix, deutsche Lokalisierung ab welcher Phase?
2. **Sichtbarkeit von Pro/Enterprise-Befehlen im Core-Help:** Sichtbar mit Hinweis oder vollständig ausgeblendet?
3. **Plugin-API-Stabilität:** Ab welcher Version garantieren wir Plugin-API-Stabilität?
4. **Run-ID-Format:** Sequenziell (`run-001`) oder UUID? Implikation für Sortierung und Lesbarkeit.
5. **Konfiguration: nur YAML oder auch TOML?** YAML ist im Engineering verbreiteter, TOML bei modernen Python-Tools üblicher.
6. **Daten-Versionierung:** Wie integriert sich SurroForge mit DVC oder ähnlichen Werkzeugen?

---

## 13. Nächste Schritte

1. Annahmen aus Abschnitt 11.1 in 5–8 Interviews validieren
2. Architektur-Spike: Plugin-System und Lizenz-Mechanik prototypisch bauen
3. MVP-Backlog aus Phase 1 in konkrete Tickets übersetzen
4. CI/CD-Pipeline aufsetzen (GitHub Actions, PyPI-Release-Workflow)
5. Dokumentations-Skeleton mit MkDocs Material aufsetzen
6. Erste Release-Kandidatin nach 4 Monaten
