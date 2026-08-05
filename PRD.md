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

Wir bauen ein Open-Core-Werkzeug, mit dem Ingenieure aus Simulationsdaten Surrogate-Modelle trainieren, vergleichen und teilen können — skriptbar, offen und ohne Vendor-Lock-in. Die kommerziellen Editionen ergänzen die für Engineering-Nachweise entscheidenden Ebenen: Reproduzierbarkeit und Lineage, Modell-Deployment, KI-Agenten-Steuerung und erweiterte Methoden.

### 1.4 Kernversprechen

Vom Solver-Output zum validierten Surrogate-Modell in unter einer Stunde. Die kommerziellen Editionen machen daraus einen nachweisbaren Prozess: Daten, Konfigurationen und Modelle sauber versioniert und nach Monaten oder Jahren bit-genau reproduzierbar — die Auditierbarkeit, die Engineering-Freigaben verlangen.

### 1.5 MVP-Ansatz

**MVP** steht für *Minimum Viable Product* — die kleinste auslieferbare Version von SurroForge, die einem Berechnungsingenieur echten Nutzen bringt und an realen Nutzern getestet werden kann. Statt von Anfang an alle Methoden, Editionen und Komfortfunktionen zu bauen, liefert das MVP **nur den durchgängigen Kern-Workflow** und lernt aus echter Nutzung, bevor wir in die Breite investieren.

**Das Leitprinzip:** Baue das Kleinste, das veröffentlicht werden kann, um etwas Wichtiges zu lernen — dann iteriere.

**Was das MVP von SurroForge leistet** (vollständig in Abschnitt 9.1 spezifiziert):

- Ein durchgängiger Pfad vom registrierten Datensatz bis zum trainierten, validierten Surrogate-Modell — über CLI und Python-API
- Drei bewusst etablierte Methoden (Kriging, RBF, polynomiale Regression) statt einer breiten Methodenpalette
- Ein minimaler Run-Store (Modelle speichern, auflisten, Vorhersagen erzeugen) — die *nachweisbare* Reproduzierbarkeit (Herkunfts-Hashes, `reproduce`, Lineage) ist die erste kommerzielle Ebene (Abschnitt 3.2), nicht Teil des freien Kerns
- Markdown-Reports als Ausgabe; portabler Modell-Export (ONNX/Pickle) gehört zu Professional
- Plugin-Schnittstellen, damit der spätere Ausbau (Pro/Enterprise) additiv erfolgt und den Kern nicht umbaut

**Was das MVP bewusst weglässt:** erweiterte Methoden (skalierbare GP, RF, Boosting, NN), Modell-Export, Reproduzierbarkeit/Lineage und `reproduce`, interaktive HTML-/PDF-Reports, Active Learning, Multi-Fidelity, DoE-Generierung, KI-Agenten-Steuerung (maschinenlesbare JSON-Ausgabe in Phase 2, MCP-Server in Phase 3) sowie die gesamte Lizenz- und Monetarisierungs-Mechanik. Diese gehören in Phase 2 und 3 (Abschnitte 9.2, 9.3) und werden erst gebaut, wenn das MVP seine kritischen Annahmen (Abschnitt 11.1) bestätigt hat.

**Warum dieser Zuschnitt:** Das MVP soll die teuerste Unsicherheit zuerst beantworten — ob der freie Open-Source-Kern messbare Adoption in der CAE-Community erzeugt (Annahme 2) und als Funnel taugt. Erst wenn das belegt ist, rechtfertigt sich die Investition in die kommerziellen Ebenen (Reproduzierbarkeit, Deployment, Methoden-Breite), auf denen die Konversion beruht. Die Definition of Done für Phase 1 (drei externe Testnutzer durchlaufen die drei Standard-Szenarien erfolgreich) ist genau dieser Lerntest.

---

## 2. Zielnutzer

### 2.1 Primäre Persona

**Methoden-Engineer** in einem CAE-Team mittelständischer Maschinenbau-Unternehmen. Python-affin, arbeitet in der Kommandozeile, kennt SMT und scikit-learn, will Workflows standardisieren. Delegiert wiederkehrende Workflow-Schritte zunehmend an KI-Coding-Agenten und erwartet, dass sich seine Werkzeuge von diesen steuern lassen.

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
| Cross-Validation, Standard-Metriken (RMSE/R²/MAE) | ✓ | ✓ | ✓ |
| Train / Validate / Predict, Terminal-Vergleich, Multi-Output | ✓ | ✓ | ✓ |
| Feld-ROM (reduced-field POD, Core-Inner-Surrogat) | ✓ | ✓ | ✓ |
| Markdown-Reports | ✓ | ✓ | ✓ |
| OpenFOAM-Ingestion, minimaler Run-Store | ✓ | ✓ | ✓ |
| Python-API und Plugin-Schnittstellen | ✓ | ✓ | ✓ |
| Erweiterte Methoden (Random Forest, Gradient Boosting) | — | ✓ | ✓ |
| Neuronale Netze (sklearn-MLP, PyTorch) | — | ✓ | ✓ |
| Skalierbare Gaussian Processes (Sparse GP, GPyTorch) | — | ✓ | ✓ |
| Feld-ROM mit NN-/erweitertem Inner-Surrogat | — | ✓ | ✓ |
| Reproduzierbarkeit & Lineage (`reproduce`, Herkunfts-Hashes, Lineage-Ansicht/-Export) | — | ✓ | ✓ |
| Modell-Export (ONNX, Pickle) | — | ✓ | ✓ |
| HTML-Reports (interaktiv), DoE/Sampling, erweiterte Vergleiche | — | ✓ | ✓ |
| Maschinenlesbare CLI-Ausgabe (`--json`) für skriptbare Agenten | — | ✓ | ✓ |
| Lizenz-Aktivierung (Datei/Online) | — | ✓ | ✓ |
| KI-Agenten-Steuerung über MCP-Server (4.13) | — | — | ✓ |
| Autoencoder-Feld-Surrogate (nichtlineare Reduktion) | — | — | ✓ |
| Active Learning, Multi-Fidelity | — | — | ✓ |
| PDF-Audit-Reports mit Lineage/Compliance-Anhang | — | — | ✓ |
| Floating-Lizenzen, Site-License | — | — | ✓ |
| SLA-Support, Customer Success | — | — | ✓ |

**Trennprinzip:** Das freie Core ist ein vollständig nutzbarer Einstieg (drei etablierte Methoden, Training, Vorhersage, Vergleich, Markdown-Report) und dient als Adoptions-Funnel. Der kommerzielle Wert liegt in vier Ebenen: *Methoden-Breite* (RF/Boosting/NN/skalierbare GP), *Vertrauen* (Reproduzierbarkeit + Lineage), *Deployment/Integration* (Modell-Export, maschinenlesbare Ausgabe; die native Agenten-Anbindung über MCP ist Enterprise) und *Präsentation/Workflow* (HTML-Reports, DoE). Neuronale Netze werden automatisch über die Plugin-Registry gegatet: jeder Aufruf eines NN-Trainers — auch als Inner-Surrogat eines Feld-ROM — löst den Lizenz-Check aus — stufen-bewusst: das Autoencoder-Feld-Surrogat verlangt die Enterprise-Stufe.

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

Globale Optionen: `--verbose / -v`, `--quiet / -q`, `--config-dir`, `--json` (maschinenlesbare Ausgabe, Professional — siehe 4.12), `--help`.

Mit den kommerziellen Editionen kommen weitere Befehlsgruppen über die
Plugin-Mechanik hinzu: `surroforge license …` und `surroforge doe …`
(Professional) sowie `surroforge mcp serve` (Agenten-Server, Enterprise —
siehe 4.13).

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
- Modell-Export als Pickle und ONNX (Professional)
- Metadaten als JSON
- Maschinenlesbare CLI-Ausgabe als JSON auf stdout (`--json`, Professional — siehe 4.12)
- Reports als Markdown (Core), HTML (Pro), PDF-Audit (Enterprise)
- Konfigurations-Snapshot als YAML

### 4.5 Run-Persistenz und Reproduzierbarkeit

Jeder Trainingslauf erzeugt einen Run. Der **freie Kern** speichert:

- Eindeutiger ID (z. B. `run-001` oder UUID)
- Snapshot der vollständigen Konfiguration
- Trainings-Metriken (Train, Test, Cross-Validation)
- Modell-Artefakt (für spätere Vorhersagen über `predict`)

**Professional** macht die vollständige Lineage und die Reproduzierbarkeits-Garantie nutzbar:

- Daten-Hash (SHA-256 über die verwendeten Daten), Random Seed, Python- und Bibliotheksversionen — von Core bei jedem Run erfasst, einsehbar erst mit Lizenz
- Vollständige Herkunfts-Ansicht (`info`, Lineage-Spalten in `compare`) und Lineage-Export
- `surroforge reproduce <run-id>` erzeugt bitidentische Modelle, sofern Python- und Bibliotheksversionen passen

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

- **Autoencoder-Feld-Surrogate** (nichtlineare Feld-Reduktion, über PyTorch)
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

### 4.12 KI-Agenten-Steuerung (Professional / Enterprise)

SurroForge ist durch KI-Agenten steuerbar: ein Agent (z. B. Claude oder Claude Code) fährt den vollständigen Kern-Workflow selbst — Datensatz registrieren, trainieren, validieren, vergleichen, vorhersagen, berichten. Dafür stehen zwei Oberflächen auf zwei Editionsstufen bereit:

**1. Maschinenlesbarer Ausgabemodus (Professional).** Ein globales `--json`-Flag lässt jeden Befehl statt Terminal-Text ein stabiles JSON-Objekt auf stdout ausgeben, kombiniert mit deterministischen Exit-Codes. Für skript- oder shell-basierte Agenten, die die CLI direkt aufrufen. Das ist die Einstiegsstufe der Agenten-Steuerung: sie setzt voraus, dass der Agent eine Shell hat und Aufrufe selbst orchestriert.

**2. MCP-Server (Enterprise).** Stellt die SurroForge-Verben als Tools des Model Context Protocol (MCP) über stdio bereit, sodass MCP-fähige Agenten sie nativ aufrufen — mit entdeckbarem Tool-Katalog, Schema pro Parameter und Freigaben pro Tool statt pauschalem Shell-Zugriff. Ausgeliefert im `surroforge_enterprise`-Plugin und über die bestehende Plugin-Mechanik registriert (`surroforge mcp serve`); die `mcp`-SDK ist ein optionales Extra (`pip install 'surroforge_enterprise[agent]'`). Vollständig spezifiziert in **4.13**.

**Warum der MCP-Server Enterprise ist:** Die native Agenten-Anbindung ist genau die Ebene, auf der ein CAE-Team seine Workflows automatisiert und damit organisatorisch abhängig wird — dieselbe Käufer-Logik wie bei Audit-Reports und Floating-Lizenzen. Der `--json`-Modus bleibt in Professional, damit auch dort ein automatisierbarer Pfad existiert.

**Architektur.** Beide Oberflächen setzen ausschließlich auf der bestehenden Services-/Engine-/Python-API-Schicht auf (4.10, 4.11) und bauen keine eigene Domänenlogik. Damit erben sie das Lizenz-Gate automatisch: gegatete Verben (`reproduce`, `export`, HTML-`compare`, Provenance in `info`) verlangen weiterhin eine Lizenz und liefern eine strukturierte „requires license"-Antwort (`error_type` + `message`, abgeleitet aus der bestehenden `PluginError`-Hierarchie). Freie Core-Verben (`train`, `predict`, `validate`, Terminal-`compare`, `list`, `data …`, `rom inspect`, Markdown-`report`) bleiben inhaltlich unverändert — die Agent-Schicht kapselt sie nur maschinenlesbar.

**Abgedeckte Verben:** `init`, `data add`/`add-openfoam`/`list`, `train`, `validate`, `compare`, `predict`, `export`, `report`, `list`, `reproduce`, `info`, `rom inspect`.

Stabilität und Sicherheit des Agent-Vertrags: siehe 5.5 (lokal, ohne ausgehende Netzwerkverbindung) und 5.7 (versioniertes, rückwärtskompatibles Ausgabeschema).

### 4.13 MCP-Server (Enterprise)

Der MCP-Server ist die native Agenten-Oberfläche von SurroForge: er macht die
Kern-Verben als Tools des **Model Context Protocol** verfügbar, sodass ein Agent
(Claude Code, Claude Desktop, jeder MCP-fähige Client) den vollständigen Workflow
ausführt, ohne die CLI über eine Shell zu bedienen und ohne Terminal-Ausgaben zu
parsen.

**Warum nicht nur `--json`?** Das JSON-Flag bedient Agenten, die ohnehin eine Shell
haben. MCP liefert darüber hinaus das, was ein Shell-Aufruf nicht kann: eine
typisierte, vom Client **entdeckbare** Tool-Liste mit JSON-Schema pro Parameter,
Freigabe-Dialoge pro Tool statt pauschalem Shell-Zugriff, Fortschritts-Meldungen
für lange Trainings und Run-Artefakte als adressierbare Resources. Beide
Oberflächen bleiben nebeneinander bestehen; sie teilen sich Services-Schicht,
Ausgabeschema und Lizenz-Gate.

#### 4.13.1 Start, Registrierung, Installation

- Registriert über den bestehenden `surroforge.commands`-Entry-Point — hier im
  `surroforge_enterprise`-Plugin, das damit erstmals nicht nur Trainer, sondern auch
  eine Befehlsgruppe beisteuert. Auffindbar auf demselben Weg wie `surroforge doe`
  und `surroforge license` aus dem Professional-Plugin.
- Start: `surroforge mcp serve [--project <pfad>]`. Ohne `--project` gilt das über
  `Project.discover()` gefundene Projekt des Arbeitsverzeichnisses.
- Transport: **stdio**. Ein Server-Prozess bedient genau einen Client.
- Lizenz: Der Start verlangt eine gültige **Enterprise**-Lizenz. Ohne sie bricht
  `surroforge mcp serve` sofort mit einer klaren Meldung ab, statt einen Server mit
  halbem Tool-Katalog anzubieten — der Gate-Punkt liegt am Prozessstart, nicht pro
  Tool-Aufruf (siehe 4.13.5).
- Abhängigkeit: das `mcp`-Python-SDK als optionales Extra
  (`pip install 'surroforge_enterprise[agent]'`). Fehlt es, scheitert der Befehl
  mit derselben klaren Meldung wie andere fehlende Backends (`PluginError`), nicht
  mit einem `ImportError`-Traceback.
- Client-Konfiguration erfolgt über den Standard-Mechanismus des jeweiligen Clients
  (Kommando + Argumente + Arbeitsverzeichnis); SurroForge liefert dafür ein
  dokumentiertes Snippet, keine eigene Registry.

#### 4.13.2 Tool-Katalog

Ein Tool pro CLI-Verb, Namensraum-präfixiert (`surroforge_…`), damit die Namen im
globalen Tool-Katalog eines Agenten eindeutig bleiben. Jedes Tool ist eine dünne
Hülle um die in 4.10/4.11 genannten Schichten — keine eigene Domänenlogik:

| MCP-Tool | Wesentliche Parameter | Dahinterliegende Funktion | Edition des Verbs |
|---|---|---|---|
| `surroforge_init` | `path`, `name`, `description` | `Project.init` | Core |
| `surroforge_data_add` | `file`, `name` | `Project.add_dataset` | Core |
| `surroforge_data_add_openfoam` | `samples_dir`, `field`, `name` | OpenFOAM-Ingestion | Core |
| `surroforge_data_list` | — | `Project.datasets` | Core |
| `surroforge_train` | `config` (Pfad **oder** Inline-Mapping) | `engine.run_training` | Core |
| `surroforge_validate` | `run_id` | `Run.metrics` | Core |
| `surroforge_list_runs` | — | `Project.runs` | Core |
| `surroforge_info` | `run_id` | `Run`-Metadaten | Core (Herkunftsfelder: Pro) |
| `surroforge_compare` | `run_ids`, `metric` | `services.compare_runs` | Core |
| `surroforge_predict` | `run_id`, `input_file` **oder** `rows`, `field` | `services.predict_from_file` / `predict_with_run` | Core |
| `surroforge_report` | `run_id`, `format`, `output` | `services.generate_report` | Core (`markdown`), Pro (`html`) |
| `surroforge_comparison_report` | `run_ids`, `metric`, `format` | `services.generate_comparison_report` | Professional |
| `surroforge_export` | `run_id`, `format`, `output` | `services.export_model` | Professional |
| `surroforge_reproduce` | `run_id`, `tolerance` | `services.reproduce` + `assert_reproducible` | Professional |
| `surroforge_rom_inspect` | `run_id` | `services.inspect_rom_run` | Core |

Die letzte Spalte nennt die Edition des **darunterliegenden Verbs**, nicht die
Zugangshürde zum Tool: Wer den Server startet, hat eine Enterprise-Lizenz, und weil
die Stufen geordnet sind (Enterprise schließt Professional ein), ist der gesamte
Katalog nach dem Start nutzbar. Die Spalte bleibt trotzdem relevant — sie dokumentiert,
welche Verben ein Agent auf einer reinen Core- oder Professional-Installation über
CLI/API erreichen kann, und ist die Grundlage, falls später ein eingeschränkter
Server für niedrigere Stufen angeboten wird (offene Frage 8).

Der Katalog ist **geschlossen**: es gibt kein generisches „führe Kommando aus"-,
„lies beliebige Datei"- oder „evaluiere Python"-Tool. Was ein Agent über MCP tun
kann, steht vollständig in dieser Tabelle.

Jedes Tool trägt die MCP-Annotationen `readOnlyHint` bzw. `destructiveHint`, damit
Clients lesende Aufrufe (`list`, `info`, `validate`, `compare`, `rom inspect`)
automatisch freigeben und schreibende (`init`, `data add`, `train`, `export`,
`report`) einzeln bestätigen lassen können.

#### 4.13.3 Resources

Run-Artefakte werden zusätzlich als MCP-Resources exponiert, damit ein Agent sie
lesen kann, ohne ein Tool aufzurufen oder Dateipfade zu raten:

- `surroforge://runs/<run-id>/config` — Konfigurations-Snapshot (YAML)
- `surroforge://runs/<run-id>/metrics` — Metriken (JSON)
- `surroforge://runs/<run-id>/lineage` — Herkunft (JSON)
- `surroforge://runs/<run-id>/report` — zuletzt erzeugter Report
- `surroforge://project` — Projekt-Metadaten inkl. registrierter Datensätze

#### 4.13.4 Lange Läufe

Ein Training kann Minuten dauern und damit das Timeout eines Clients überschreiten.
Anforderungen: `surroforge_train` sendet MCP-**Progress-Notifications** (Phase:
Datenauflösung → Split → Cross-Validation-Fold *k* → finales Modell → Persistenz)
und gibt die Run-ID zurück. Bricht der Client ab, wird der Lauf abgebrochen und
hinterlässt dank atomarer Run-Persistenz **kein** halbfertiges Run-Verzeichnis —
der Agent sieht in `surroforge_list_runs` entweder einen vollständigen Run oder
gar keinen.

#### 4.13.5 Fehler- und Lizenz-Vertrag

Domänenfehler werden als strukturierte Tool-Fehler zurückgegeben, nie als
Traceback: `error_type` (der Klassenname aus der bestehenden
`SurroForgeError`-Hierarchie — `ConfigError`, `DataError`, `TrainerError`,
`PluginError`, `RunNotFoundError`, `ReproducibilityError`, …) plus `message`. Damit
kann ein Agent zwischen „Konfiguration reparieren", „Lizenz fehlt" und „Run
existiert nicht" unterscheiden, statt Text zu interpretieren.

**Lizenzgatung liegt am Prozessstart, nicht am einzelnen Tool.** Der Server ist
selbst ein Enterprise-Feature: ohne gültige Enterprise-Lizenz startet er nicht und
meldet das mit `error_type: "PluginError"` plus Hinweis auf
`surroforge license install`. Läuft er, ist der volle Katalog freigeschaltet, denn
die Enterprise-Stufe schließt die Professional-Stufe ein — ein Agent stößt also
mitten im Workflow nicht mehr auf eine Lizenzwand, was gerade für autonom
arbeitende Agenten der entscheidende Unterschied ist. Die Gatung der Services-Schicht
bleibt darunter unverändert bestehen und greift weiterhin, wenn dieselben Verben
über CLI oder Python-API ohne passende Lizenz aufgerufen werden. Ein Downgrade auf
stille Teilergebnisse findet in keinem Fall statt.

#### 4.13.6 Sicherheit

- stdio-only, lokal, keine ausgehende Netzwerkverbindung (siehe 5.5).
- Alle Pfad-Parameter werden gegen die Projektwurzel aufgelöst; Zugriffe außerhalb
  werden abgelehnt, sofern nicht explizit über `--allow-path` freigegeben. Damit ist
  der Wirkungsradius eines fehlgeleiteten Agenten auf ein Projektverzeichnis
  begrenzt.
- Kein Tool löscht Runs, Datensätze oder Projekte.

#### 4.13.7 Versionierung und Abnahme

Die Tool-Signaturen sind Teil des Agent-Ausgabevertrags aus 5.7: versioniert, ab
1.0 innerhalb einer Major-Version rückwärtskompatibel, Änderungen mit
Deprecation-Vorlauf und CI-Prüfung gegen frühere Releases. Der Server meldet seine
Schema-Version in der MCP-Initialisierung.

**Definition of Done:** Ein MCP-Client fährt mit gültiger Enterprise-Lizenz das
Greenfield-Szenario (Projekt anlegen → Datensatz registrieren → trainieren →
validieren → Report) sowie den Methodenvergleich ausschließlich über MCP-Tools
durch, ohne Shell-Zugriff; ohne Lizenz verweigert `surroforge mcp serve` den Start
mit strukturierter `PluginError`-Meldung.

---

## 5. Nicht-funktionale Anforderungen

### 5.1 Performance

- Training eines Kriging-Modells auf 200 Punkten mit 5 Inputs: unter 5 Sekunden
- Training eines Random-Forest auf 10.000 Punkten: unter 30 Sekunden
- Cross-Validation mit k=5 darf das Training maximal um Faktor 6 verlangsamen
- Datensatz-Integritäts-Hashing bei `data add`: unter 5 Sekunden für Datensätze bis 100 MB (Core; die Verknüpfung zur reproduzierbaren Run-Lineage ist Professional)

### 5.2 Reproduzierbarkeit (Professional)

- In Professional ist jeder Run vollständig reproduzierbar bei identischer Python- und Bibliotheksversion
- Konfiguration, Daten-Hash, Seed und Versionen werden bei jedem Run automatisch erfasst — kein Opt-Out; nutzbar wird diese Herkunft in Professional
- Der freie Kern speichert Konfiguration und Modell für Vorhersagen; Herkunfts-Daten werden intern erfasst, sind ohne Lizenz aber weder einsehbar noch per `reproduce` nutzbar

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
- Der MCP-Server (Enterprise, 4.13) läuft lokal über stdio und baut selbst keine ausgehende Netzwerkverbindung auf; sein Tool-Katalog ist geschlossen (kein generisches Shell-, Datei- oder Eval-Tool) und alle Pfade sind auf die Projektwurzel begrenzt
- Lizenzprüfung in Pro/Enterprise erfolgt lokal oder über vom Nutzer konfigurierte Endpunkte
- Keine Telemetrie ohne Opt-In; falls aktiviert, transparent und abschaltbar

### 5.6 Sprache

- CLI- und API-Ausgaben primär Englisch
- Deutsche Lokalisierung optional konfigurierbar (Phase 2)
- Dokumentation primär Englisch, deutsche Kurzversion gepflegt

### 5.7 Stabilität des Agent-Ausgabevertrags

- Das maschinenlesbare Ausgabeschema (`--json`, 4.12) und die MCP-Tool-Signaturen (4.13) sind versioniert und ab Version 1.0 innerhalb einer Major-Version rückwärtskompatibel — analog zum CLI-/API-Vertrag (5.4) —, damit Agenten und Skripte sich verlässlich darauf abstützen können
- Schema-Änderungen folgen demselben Deprecation-Pfad wie CLI/API (mindestens eine Minor-Version Vorlauf) und werden in CI gegen frühere Releases geprüft

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
| Agenten-Anbindung | MCP-Python-SDK über stdio (Enterprise, optionales Extra `agent`) |
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
├── exporters/        # ONNX/Pickle-Export (Professional)
├── reporters/        # Markdown-Report
├── persistence/      # Run-Speicherung (Core); Lineage/reproduce (Professional)
├── plugins/          # Plugin-Registry und Entry-Points
└── api/              # Hochlevel-Python-API
```

Kommerzielle Module liegen in separaten Paketen:
```
surroforge_professional/  # Erw. Methoden (RF/Boosting/NN), Export, Repro/Lineage, HTML-Reports, DoE
surroforge_enterprise/    # Autoencoder-Feld-Surrogat, Active Learning, Multi-Fidelity, Audit-PDF,
                          # mcp/ (Agenten-Server, 4.13)
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

Jeder Run enthält `config.snapshot.yaml`, `model.pkl` (interner Store für `predict`), `metrics.json`, `log.txt` und `lineage.json` — die Herkunfts-**Erfassung** ist Core-interne Infrastruktur (sie trägt `predict` und den Upgrade-Pfad: früher trainierte Runs bleiben nach Lizenzkauf reproduzierbar). **Nutzbar** wird die Herkunft erst in Professional (`reproduce`, Anzeige in `info`/Report/`compare`); `model.onnx` (Export) wird erst in Professional geschrieben.

### 7.2 Run-Metadaten (lineage.json; Erfassung Core-intern, Nutzung Professional)

`lineage.json` wird von Core für jeden Run geschrieben (interne Infrastruktur); die Professional-Oberflächen (`reproduce`, `info`, Report, `compare`) machen sie nutzbar. Pflichtfelder pro Run:
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
- CLI-Skeleton mit den Core-Befehlen aus Abschnitt 4.2 (Pro/Enterprise-Befehle wie `export`/`reproduce` erscheinen erst mit der jeweiligen Edition)
- Drei Methoden: Kriging, RBF, polynomiale Regression
- YAML-Konfiguration mit Pydantic-Validierung
- Cross-Validation und Standard-Metriken
- Minimaler Run-Store: Speichern, Auflisten, Vorhersagen (ohne Herkunfts-Hashes/`reproduce` — die kommen mit Professional, Phase 2)
- Markdown-Reports
- Vollständige Python-API
- Plugin-Schnittstellen
- Dokumentation mit Tutorials und API-Referenz

**Definition of Done:** Drei externe Testnutzer durchlaufen die drei Standard-Szenarien (Greenfield, Methoden-Vergleich, Vorhersage auf neuen Daten) erfolgreich. (Das Reproduktions-Szenario wandert mit `reproduce` nach Phase 2.)

### 9.2 Phase 2 — Professional (Monate 5–9)

**Ziel:** Erste kommerzielle Edition. Beginn der Monetarisierung.

Inhalte:
- Erweiterte Methoden: skalierbare Gaussian Processes, Random Forest, Gradient Boosting, neuronale Netze (sklearn-MLP, PyTorch)
- Reproduzierbarkeit & Lineage: Herkunfts-Hashing, `reproduce`, Lineage-Ansicht (`info`/`compare`) und Lineage-Export
- Modell-Export (ONNX, Pickle)
- HTML-Reports mit interaktiven Visualisierungen
- Erweiterte Run-Vergleiche
- Lizenz-Aktivierungs-Mechanik
- E-Mail-Support-Infrastruktur
- DoE-Modul als ergänzendes Submodul
- KI-Agenten-Steuerung, Stufe 1: maschinenlesbarer JSON-Modus (`--json`, setzt auf die Services-/API-Schicht auf, erbt das Lizenz-Gate); der MCP-Server folgt in Phase 3

**Definition of Done:** Drei zahlende Pilotkunden, dokumentierte Konversion vom Core; das Reproduktions-Szenario (bit-identischer Nachvollzug) läuft in Professional durch.

### 9.3 Phase 3 — Enterprise (Monate 10–18)

**Ziel:** Enterprise-fähiges Produkt. Audit-Funktionen, größere Vertragsvolumen.

Inhalte:
- Autoencoder-Feld-Surrogat
- MCP-Server für KI-Agenten-Steuerung (4.13): Tool-Katalog, Resources, Progress-Notifications, Enterprise-Gate am Prozessstart
- Active Learning mit Acquisition-Funktionen
- Multi-Fidelity-Modelle
- PDF-Audit-Reports mit Lineage-Tracking
- Floating-Lizenz-Server
- Compliance-Datenblatt und IT-Freigabe-Material
- Customer-Success- und Schulungsangebot

**Definition of Done:** Erste Enterprise-Kunde mit Site-License oder mehreren Floating-Lizenzen; ein MCP-Client fährt Greenfield- und Vergleichs-Szenario ohne Shell-Zugriff durch (4.13.7).

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
- **Stabilität des Agent-Vertrags:** Ein maschinenlesbares JSON-/MCP-Schema (4.12, 4.13) wird von Agenten und Skripten als Vertrag konsumiert; nachträgliche Änderungen brechen Integrationen. Mitigationsstrategie: versioniertes, in CI getestetes Ausgabeschema mit Deprecation-Pfad (5.7).

### 11.3 Geschäftliche Risiken

- **Konversionsrate niedriger als geplant:** Mitigationsstrategie: konservative Geldplanung, Pilot-Programme früh starten, Akademie-Lizenzen für Multiplikator-Effekt.
- **Konkurrenz durch Enterprise-Suiten oder neue ML-Engineering-Startups:** Mitigationsstrategie: freier, offener Kern als Adoptions-Funnel plus scharf abgegrenzte kommerzielle Ebenen — Reproduzierbarkeit/Auditierbarkeit, Modell-Deployment und Methoden-Breite. Open Core senkt die Einstiegshürde; die bezahlten Ebenen liefern den nachweisbaren Engineering-Wert.

---

## 12. Offene Fragen

Diese werden vor MVP-Release entschieden:

1. **CLI-Sprache:** Englisch als Default fix, deutsche Lokalisierung ab welcher Phase?
2. **Sichtbarkeit von Pro/Enterprise-Befehlen im Core-Help:** Sichtbar mit Hinweis oder vollständig ausgeblendet?
3. **Plugin-API-Stabilität:** Ab welcher Version garantieren wir Plugin-API-Stabilität?
4. **Run-ID-Format:** Sequenziell (`run-001`) oder UUID? Implikation für Sortierung und Lesbarkeit.
5. **Konfiguration: nur YAML oder auch TOML?** YAML ist im Engineering verbreiteter, TOML bei modernen Python-Tools üblicher.
6. **Daten-Versionierung:** Wie integriert sich SurroForge mit DVC oder ähnlichen Werkzeugen?
7. **MCP-Tool-Granularität:** Ein Tool pro CLI-Verb (4.13.2, aktuelle Festlegung) oder zusätzlich gröbere Workflow-Tools („trainiere und vergleiche drei Methoden")? Feine Tools sind vorhersagbar und leichter zu gaten, kosten den Agenten aber mehr Runden.
8. **Eingeschränkter MCP-Server unterhalb von Enterprise?** Der volle Server ist Enterprise (3.2, 4.13). Offen bleibt, ob ein reduzierter Server mit ausschließlich Core-Verben als Adoptions-Funnel sinnvoll ist — er senkt die Einstiegshürde für Agenten-Nutzer, entwertet aber das Enterprise-Argument und verdoppelt die Vertragsfläche (5.7).

---

## 13. Nächste Schritte

1. Annahmen aus Abschnitt 11.1 in 5–8 Interviews validieren
2. Architektur-Spike: Plugin-System und Lizenz-Mechanik prototypisch bauen
3. MVP-Backlog aus Phase 1 in konkrete Tickets übersetzen
4. CI/CD-Pipeline aufsetzen (GitHub Actions, PyPI-Release-Workflow)
5. Dokumentations-Skeleton mit MkDocs Material aufsetzen
6. Erste Release-Kandidatin nach 4 Monaten
