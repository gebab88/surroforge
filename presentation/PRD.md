# PRD — Präsentation SurroForge

**Version:** 0.1
**Datum:** Mai 2026
**Zweck:** Spezifikation für die Erstellung einer Pitch-Präsentation des Projektes SurroForge

---

## 1. Zielsetzung der Präsentation

Die Präsentation stellt SurroForge als Projektidee vor: ein CLI-Werkzeug für Surrogate-Modellierung für FEM und CFD, basierend auf einem Open-Core-Geschäftsmodell. Sie soll Verständnis für den adressierten Schmerz, die Lösung, das Geschäftsmodell und den aktuellen Validierungsstand in Optimierungs-Workflows (MDAO) schaffen — und zur Diskussion oder Entscheidung anregen.

---

## 2. Publikum

**Primär:** Bewertungs- oder Diskussionsrunde (Coach, Jury, Mentor, Workshop-Gruppe oder vergleichbare Stakeholder), die das Projekt strategisch beurteilen soll.

**Sekundär:** Beratungs- oder Pilotkontakte, an die die Folien zur Information weitergegeben werden können.

Vorausgesetztes Vorwissen: technische Grundbegriffe (FEM, CFD, Simulation) bekannt; Surrogate-Modelle und ML-Methoden nicht zwingend vertraut. Geschäftsmodell- und Validierungs-Vokabular sollte verständlich sein.

---

## 3. Kernbotschaften

Drei zentrale Aussagen, die das Publikum mitnehmen soll:

1. **In Optimisierungsabteilungen (MDAO) gibt es einen konkreten, teuren Schmerz** — verlorene Tage durch kostspielige High-fidelity-Simulationen, ineffiziente Surrogate-Modelle und Wissensengpässe.
2. **SurroForge adressiert diesen Schmerz fokussiert** mit einem CLI-Tool, das Standardisierung, Reproduzierbarkeit und Methoden-Vergleich in einem schlanken Werkzeug bündelt — ohne GUI-Ballast.
3. **Das Open-Core-Modell ist eine bewusste strategische Wahl**, offener Kern und geschlossenes Herz die Adoption (über den freien Kern) und Monetarisierung (über professionelle und Enterprise-Editionen) verbindet.

---

## 4. Foliengliederung

8 Folien, ca. 8–10 Minuten Vortragszeit. Pro Folie bleibt rund eine Minute, was eine straffe, fokussierte Erzählung erzwingt — kein Raum für Nebenschauplätze.

### Folie 1 — Titel
- Produktname SurroForge
- Untertitel: „Effiziente ML-Surrogate-Modelle für FEM und CFD"

### Folie 2 — Problem und Anlass kombiniert
- Ingenieure verlieren Zeit in MDAO-Workflows durch schwache Modelle (zu langsam & zu ungenau) 
- Das Verbinden einzelner Software-Tools sowie Einbindung von Datenquellen kostet Zeit und führt zu Fehlern
- Methodenwissen lebt in Köpfen einzelner Schlüsselpersonen
- Warum jetzt: AI-Hype, methodisches Ökosystem (SMT, GPyTorch) ist reif, jedoch fehlt das Standardwerkzeug darüber 
- Baue das Bild "The ideal plane" ein. Es liegt im Directory

### Folie 3 — Die Lösung
- SurroForge in einem Satz: CLI für effiziente ML-Surrogate-Modellierung
- Von Input-Daten aus FEM/CFD/Experiment zum validierten ML-Surrogate-Modell in unter einer Stunde
- Fokus auf schnelle und reproduzierbare ML-Surrogate-Modelle, die sich in bestehende Workflows integrieren lassen
- Optional: ein konkretes CLI-Snippet als visueller Anker

### Folie 4 — Zielnutzer und Käufer
- Methodeningenieur als primärer Nutzer mit seinem zentralen Schmerz: Zeitverlust durch ineffiziente Surrogate-Modelle und fragmentierte Tools
- Leitung der MDAO-Abteilung als Käuferpersona: Wissenssicherung und Effizienzsteigerung
- Verbindung: gleiche Krankheit, unterschiedliche Symptome

### Folie 5 — Geschäftsmodell: Open-Core
- Drei Editionen: Core (Open Source), Professional, Enterprise
- Konversionslogik vom freien Kern zur kommerziellen Edition
- Bewusste Beschränkung: keine GUI und kein Optimierungssolver 
- GUI-Add-on und Optimierungssolver als Erweiterungen zukaufbar, jedoch Kern bleibt Erstellung von ML-Surrogate in CLI
- Strategischer Vorteil: Vertrauen + Monetarisierung

### Folie 6 — Markt und Positionierung
- DACH-Region als primärer Markt: Klassischer Maschinenbau, Automotive, Luftfahrt (Bitte füge Emojis hinzu, um die Branchen zu visualisieren)
- Positionierung zwischen Enterprise-Suiten (optiSLang, HEEDS) und Open-Source-Bibliotheken (SMT)

### Folie 7 — Status, Annahmen und Roadmap
- Konzeptphase abgeschlossen, fünf kritische Annahmen definiert
- Interview-Validierung in Planung, erste Pilotkunden identifiziert
- Nächste Schritte: Validierung, Prototyping, frühe Kundenakquise

### Folie 8 — Abschluss und Diskussion
- Drei zentrale Aussagen als Takeaways: Nenne es „Drei Gründe, warum SurroForge relevant ist“
- Kontakt / nächste Schritte

---

## 5. Tonalität und Stil

- **Ehrlich statt hochglänzend.** Kein Overpromising, keine Buzzword-Inflation. Konkrete Zahlen und ehrliche Unsicherheiten wirken bei diesem Publikum stärker als Marketing-Sprache.
- **Engineering-nah.** Konzepte aus dem CAE-Alltag (DoE, Validierung, Cross-Validation) dürfen vorkommen — werden aber kurz erklärt, falls das Publikum methodisch breiter aufgestellt ist.
- **Konkret statt abstrakt.** Wo immer möglich, konkrete Beispiele:  Tag eine Methodeningenieurs, eine YAML-Konfiguration, ein CLI-Vergleichs-Output.
- **Pragmatisch.** Das Projekt ist in der Validierungsphase, nicht im Verkauf — das soll spürbar sein.

---

## 6. Visuelles Design

- **Farbpalette:** Eine zurückhaltende, technisch-professionelle Palette. Dunkles Blau und Anthrazit als Primärfarbe, mit einer warmen Akzentfarbe (Terracotta und Bernstein) für Hervorhebungen. Bewusst nicht das übliche helle Tech-Blau.
- **Schrift:** Klare Sans-Serif für den Body (z. B. Inter, Source Sans, Calibri), eine Serif-Schrift für Titel zur Differenzierung (z. B. Source Serif, Georgia) — optional, falls verfügbar.
- **Visuelles Motiv:** Ein wiederkehrendes Element, idealerweise ein abstrakter Datenfluss oder ein Kurven-/Funktionssymbol, das auf die Surrogate-Idee anspielt.
- **Code- und Konfigurations-Snippets:** Mit Monospace-Schrift in einer dezent abgesetzten Box. Sparsam einsetzen — pro Folie maximal ein Snippet, und nur dort, wo es einen Punkt schärfer macht als Prosa.
- **Diagramme:** Mindestens auf zwei bis drei Folien (Problem, Lösung, Geschäftsmodell), um die Sprache zu unterbrechen. Klar gezeichnet, nicht überladen.

---

## 7. Format und Umfang

- **Folienanzahl:** 8 Folien, mit Toleranz auf 6 nach unten
- **Vortragsdauer:** 8 Minuten Vortrag, plus 5 Minuten Diskussion
- **Format:** LaTeX Beamer class, querformatig 16:9
- **Sprache:** Deutsch
- **Notizen:** keine Notizen
- **Anlass und Setting bestätigen:** Pitch-Termin

---

## 8. Erfolgskriterien

Die Präsentation ist erfolgreich, wenn nach dem Vortrag:

- Das Publikum den adressierten Schmerz im MDAO in eigenen Worten beschreiben kann
- Open-Core als bewusste strategische Entscheidung verstanden wird, nicht als Verlegenheitslösung
- Die Diskussion sich um inhaltliche Schärfung dreht, nicht um Grundverständnis
- Mindestens eine konkrete Rückmeldung zur Validierungsstrategie oder zum Geschäftsmodell kommt

---

## 9. Was bewusst nicht in die Präsentation gehört

- Detaillierte technische Architektur (gehört in eine separate technische Spezifikation)
- Vollständige Persona-Beschreibungen (eine verdichtete Variante reicht)
- Finanzplanung und konkrete Umsatzziele (zu früh in der Validierung)
- Lange Methoden-Erklärungen zu Kriging, RBF, Gaussian Processes
- Vergleichsmatrizen gegenüber jedem einzelnen Wettbewerber

---