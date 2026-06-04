# Tech-Spike-Plan — SurroForge

**Version:** 0.1
**Zeitrahmen:** 2–4 Wochen vor MVP-Baubeginn
**Zweck:** Die kritischen technischen Unbekannten ausräumen, bevor Produktcode entsteht

---

## Was ein Spike ist (und was nicht)

Ein Spike ist ein zeitlich strikt begrenztes Experiment, das genau eine Frage beantwortet. Das Ergebnis ist Wissen, nicht wiederverwendbarer Code. Jeder Spike hat eine Leitfrage, ein Erfolgskriterium, eine Timebox und definierte mögliche Ausgänge mit ihren Konsequenzen. Wenn die Timebox abläuft, wird entschieden — nicht verlängert.

Wichtig: Diese fünf Spikes werden in der angegebenen Reihenfolge bearbeitet, weil spätere auf den Ergebnissen früherer aufbauen. Spike 1 ist bewusst zuerst, weil er das größte unterschätzte Risiko adressiert.

---

## Spike 1 — ONNX-Export von Kriging und Gaussian Processes

**Priorität: kritisch. Timebox: 4 Tage.**

### Leitfrage
Lassen sich Kriging-Modelle (SMT) und Gaussian Processes (GPyTorch) verlustfrei oder verlustarm nach ONNX exportieren — und wenn nein, welches Modell-Austauschformat tritt an die Stelle?

### Warum riskant
Das PRD verspricht ONNX-Export als Core-Feature. Aber `skl2onnx` deckt nur scikit-learn-Modelle ab. SMT-Kriging und GPyTorch-GPs haben **keinen** etablierten ONNX-Export-Pfad. Kriging-Vorhersagen involvieren Kovarianzmatrizen und Kernel-Auswertungen, die sich nicht trivial in den ONNX-Operatorsatz abbilden lassen. Das ist das größte versteckte Risiko im gesamten PRD: Ein Kernversprechen („interoperabler Export") könnte für die wichtigsten Methoden technisch nicht haltbar sein.

### Experiment
1. Polynomiale Regression (scikit-learn) nach ONNX exportieren via skl2onnx, mit onnxruntime laden, Vorhersagen gegen Original vergleichen. (Dürfte funktionieren — als Referenz.)
2. SMT-Kriging-Modell: Versuch eines ONNX-Exports. Falls kein Pfad existiert, prüfen, ob ein manueller Export der Vorhersagefunktion (gespeicherte Trainingspunkte + Kernel-Parameter + eigener Inferenz-Code) als Alternative trägt.
3. GPyTorch-GP: torch.onnx.export auf die Vorhersagefunktion testen.

### Erfolgskriterium
Für jede der drei Methoden ist klar, ob ONNX-Export funktioniert, mit welcher numerischen Abweichung, oder welches Alternativformat genutzt wird.

### Mögliche Ausgänge und Konsequenzen
- **ONNX funktioniert für alle:** Ideal. PRD bleibt unverändert.
- **ONNX nur für sklearn-basierte Methoden:** Wahrscheinlichster Ausgang. Konsequenz: PRD anpassen — ONNX für sklearn-Methoden, ein dokumentiertes natives Format (Pickle + JSON-Metadaten + eigener Inferenz-Loader) für Kriging/GP. Das Versprechen „interoperabler Export" wird differenziert formuliert.
- **Export generell schwierig:** Konsequenz: Export-Strategie grundlegend überdenken. Eventuell eigenes, gut dokumentiertes Format als Primärweg, ONNX als optionales Extra.

---

## Spike 2 — Plugin-System und Lizenz-Gating

**Priorität: kritisch. Timebox: 5 Tage.**

### Leitfrage
Lässt sich die Trennung zwischen Open-Source-Core und kommerziellen Modulen sauber über Pythons Entry-Point-Mechanismus umsetzen — inklusive Lizenzprüfung beim Laden eines kommerziellen Plugins?

### Warum riskant
Das gesamte Open-Core-Geschäftsmodell steht und fällt mit dieser Trennlinie. Wenn die Architektur unsauber ist, verschwimmt die Grenze zwischen Free und Paid, und das Modell wird unhaltbar. Zusätzlich: In Python ist jeder Lizenz-Check grundsätzlich umgehbar (Code ist lesbar). Die Frage ist nicht „wie mache ich es unknackbar", sondern „wie mache ich es sauber, wartbar und für ehrliche Kunden reibungslos".

### Experiment
1. Core-Paket mit einem Trainer-Interface (abstrakte Basisklasse) und Entry-Point-Registry (`importlib.metadata.entry_points`) bauen.
2. Ein separates Dummy-„Professional"-Paket bauen, das einen zusätzlichen Trainer über Entry-Point registriert.
3. Lizenz-Gate implementieren: Beim Laden des Pro-Trainers wird eine Lizenzdatei geprüft (Signatur via `cryptography`). Ohne gültige Lizenz: klare Fehlermeldung, kein Absturz.
4. Prüfen, wie sich nicht-installierte Pro-Pakete im Core-Help verhalten (graceful degradation).

### Erfolgskriterium
Core läuft eigenständig. Pro-Trainer erscheint nur bei installiertem Pro-Paket und gültiger Lizenz. Fehlende Lizenz produziert eine hilfreiche Meldung. Die Architektur fühlt sich nicht „verbogen" an.

### Mögliche Ausgänge und Konsequenzen
- **Entry-Points + Signatur-Lizenz funktionieren sauber:** Wahrscheinlichster Ausgang. Architektur-Entscheidung steht.
- **Lizenz-Gating reibungsvoll (Offline-Fälle, Mehrbenutzer):** Konsequenz: Evaluation einer fertigen Lizenz-Bibliothek (Keygen, Cryptlex, LicenseSpring) statt Eigenbau.
- **Trennlinie architektonisch unklar:** Konsequenz: Vor MVP-Bau ein sauberes Interface-Design erarbeiten — wichtiger als jedes Feature.

### Hinweis zur Sicherheit
Pythons Lesbarkeit bedeutet: Der ehrliche Kunde wird durch Lizenzen geführt, der entschlossene Angreifer nicht abgehalten. Für echten Schutz kommerzieller Module: Cython-Kompilierung als spätere Option erwägen — aber nicht im MVP, das ist verfrühte Optimierung.

---

## Spike 3 — Reproduzierbarkeit über Versionen

**Priorität: hoch. Timebox: 3 Tage.**

### Leitfrage
Ist bitidentische Reproduzierbarkeit über die drei Core-Methoden realistisch erreichbar — oder muss das nicht-verhandelbare Reproduzierbarkeits-Versprechen auf „relative Toleranz" umformuliert werden?

### Warum riskant
Das PRD setzt Reproduzierbarkeit als Pflicht ohne Opt-Out. Aber numerische Reproduzierbarkeit ist tückisch: BLAS-Threading, nicht-deterministische Optimizer (besonders bei GP-Training), Plattformunterschiede und Float-Akkumulationsreihenfolgen können Bit-Identität verhindern — selbst bei identischen Seeds.

### Experiment
1. Polynom und RBF zweimal mit identischem Seed trainieren, Modell-Parameter bitweise vergleichen.
2. Kriging zweimal trainieren (SMT nutzt interne Optimierung der Kernel-Parameter) — prüfen, ob das deterministisch ist.
3. Threading-Einfluss testen: `OMP_NUM_THREADS=1` vs. Standard.
4. Falls bit-identisch unmöglich: definieren, welche relative Toleranz akzeptabel ist und wie der Reproduktions-Test sie misst.

### Erfolgskriterium
Für jede Core-Methode steht fest, ob Bit-Identität erreichbar ist, unter welchen Bedingungen, und welche Toleranz sonst gilt.

### Mögliche Ausgänge und Konsequenzen
- **Bit-Identität bei festem Threading erreichbar:** PRD bleibt, ergänzt um die Bedingung „bei deterministischem Threading".
- **Nur relative Toleranz erreichbar (wahrscheinlich bei Kriging):** Konsequenz: PRD umformulieren von „bitidentisch" zu „reproduzierbar innerhalb dokumentierter Toleranz". Das ist ehrlicher und immer noch ein starkes Feature.
- **Reproduzierbarkeit fundamental fragil:** Unwahrscheinlich für die Core-Methoden, aber falls doch: Reproduzierbarkeits-Versprechen als Differenzierer überdenken.

---

## Spike 4 — Daten-Hashing-Strategie

**Priorität: mittel. Timebox: 2 Tage.**

### Leitfrage
Wie wird ein stabiler, format-unabhängiger Hash über einen Datensatz berechnet, sodass dieselben Daten in CSV und Parquet denselben Hash ergeben?

### Warum riskant
Run-Persistenz und Reproduzierbarkeit (Spikes 2 und 3) hängen am Daten-Hash. Aber ein naiver Hash über die Rohdatei scheitert: dieselben Daten als CSV und als Parquet ergeben unterschiedliche Bytes. Float-Parsing-Varianz (CSV speichert Text, Parquet binär) kann sogar bei reiner CSV-Nutzung zu Hash-Unterschieden führen.

### Experiment
1. Denselben Datensatz als CSV und Parquet speichern, beide laden, Hash über die geladenen Werte (nicht die Datei) berechnen — vergleichen.
2. Float-Präzisions-Problem prüfen: Führt CSV-Roundtrip zu abweichenden Werten? Falls ja: Kanonisierungsstrategie (definierte Rundung oder Parquet-Byte-Repräsentation als Referenz) testen.
3. Spaltenreihenfolge und Index-Einfluss prüfen.

### Erfolgskriterium
Eine klare, dokumentierte Hashing-Strategie, die für dieselben Daten unabhängig vom Eingabeformat denselben Hash liefert.

### Mögliche Ausgänge und Konsequenzen
- **Hash über kanonisierte In-Memory-Repräsentation funktioniert:** Wahrscheinlichster Ausgang. Strategie steht.
- **Float-Varianz erzwingt Rundung:** Konsequenz: Definierte Präzision (z. B. float64 mit fester Rundung) im Schema festlegen, dokumentieren.

---

## Spike 5 — End-to-End-Durchstich (Walking Skeleton)

**Priorität: hoch. Timebox: 4 Tage.**

### Leitfrage
Lässt sich das 15-Minuten-Versprechen mit einem minimalen, durchgängigen Pfad einlösen — von `init` über `train` bis `report` — und wo sind die Reibungspunkte?

### Warum riskant
Die ersten vier Spikes klären Einzelrisiken. Dieser verbindet sie zum dünnsten möglichen Gesamtpfad und validiert das zentrale Nutzerversprechen. Er deckt Integrationsprobleme auf, die in isolierten Spikes unsichtbar bleiben.

### Experiment
Ein „Walking Skeleton": die dünnste durchgehende Implementierung, die einen echten Workflow erlaubt.
1. `surroforge init` legt Projektstruktur an.
2. `surroforge data add` registriert eine CSV mit Hashing (aus Spike 4).
3. `surroforge train` trainiert ein Polynom-Modell nach YAML, persistiert einen Run.
4. `surroforge report` erzeugt einen Markdown-Report.
5. Zeit messen: Wie lange braucht ein Erstnutzer realistisch von Installation bis Report?

### Erfolgskriterium
Ein durchgängiger, lauffähiger Pfad existiert. Die Reibungspunkte (Konfig-Syntax, Fehlermeldungen, Doku-Bedarf) sind benannt. Die 15-Minuten-Behauptung ist gestützt oder widerlegt.

### Mögliche Ausgänge und Konsequenzen
- **Durchstich gelingt flüssig:** Starkes Signal. MVP-Bau kann mit Zuversicht starten.
- **Durchstich gelingt, aber mit Reibung:** Normal. Die Reibungspunkte fließen priorisiert in den MVP-Backlog.
- **Durchstich stockt an unerwarteter Stelle:** Wertvollste Erkenntnis. Besser jetzt als nach drei Monaten Bau.

---

## Zeitplan-Übersicht

| Spike | Thema | Timebox | Abhängig von |
|---|---|---|---|
| 1 | ONNX-Export Kriging/GP | 4 Tage | — |
| 2 | Plugin-System + Lizenz | 5 Tage | — |
| 3 | Reproduzierbarkeit | 3 Tage | — |
| 4 | Daten-Hashing | 2 Tage | — |
| 5 | End-to-End-Durchstich | 4 Tage | 2, 4 |

Gesamt: ca. 18 Arbeitstage, also 3,5–4 Wochen Solo-Arbeit mit Puffer. Bei paralleler Bearbeitung (falls mehr als eine Person) deutlich kürzer.

---

## Entscheidungs-Checkpoint nach den Spikes

Nach Abschluss aller fünf Spikes wird in einer kurzen Review entschieden:

1. Welche PRD-Versprechen müssen umformuliert werden? (Wahrscheinlich: ONNX-Export und Reproduzierbarkeit.)
2. Steht die Architektur-Grundentscheidung für Plugin und Lizenz?
3. Ist das 15-Minuten-Versprechen realistisch?
4. Gibt es ein Risiko, das so groß ist, dass die Produktstrategie angepasst werden muss?

Erst wenn diese vier Fragen beantwortet sind, beginnt der eigentliche MVP-Bau auf solidem Fundament.

---

## Was die Spikes bewusst NICHT klären

- Performance-Optimierung (verfrüht — erst messen, wenn das Skelett steht)
- Vollständige Methodenpalette (nur eine Methode im Durchstich nötig)
- HTML/PDF-Reports (Phase 2/3, nicht MVP-kritisch)
- Cross-Plattform-Tests (Linux reicht für die Spikes; Windows/macOS später)
- Lizenz-Server-Infrastruktur (Lizenzdatei genügt für den Spike)
