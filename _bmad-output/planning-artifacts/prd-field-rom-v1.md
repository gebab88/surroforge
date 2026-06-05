# Product Requirements Document — Field ROM v1 (SurroForge)

**Version:** 0.1
**Datum:** Juni 2026
**Status:** Scoping — ableitbar in Epics/Stories (BMAD `create-prd`-Nachfolge)
**Bezug:** Produkt-PRD [`PRD.md`](../../PRD.md) · Brainstorming [`brainstorming-session-2026-06-04-231637.md`](../brainstorming/brainstorming-session-2026-06-04-231637.md) (101 Ideen, ADR-ROM-001) · Architektur [`architecture.md`](architecture.md)

> Dies ist ein **Feature-PRD für eine dünne Scheibe**, kein neues Produkt-PRD. Es erbt Positionierung, Personas, Geschäftsmodell und NFRs aus `PRD.md` und spezifiziert nur die Deltas für „Field ROM v1". Idee-IDs (`#NN`) verweisen auf das Brainstorming.

---

## 1. Überblick

### 1.1 Produkt-Erweiterung

SurroForge bildet heute Parameter → **skalare/vektorielle** Outputs ab (`BaseTrainer`: Kriging, RBF, Polynom). **Field ROM** erweitert das auf **feldwertige Outputs** über einem Mesh (Verschiebung, Spannung, Temperatur, Eigenfrequenzen): die Felder werden auf eine kleine **modale Basis** reduziert (v1: POD/Snapshot-Moden), ein Surrogate bildet Parameter → Modalkoeffizienten ab, und das volle Feld wird bei Bedarf rekonstruiert.

### 1.2 Problem / Motivation

Der Methoden-Engineer (PRD §2.1) hat heute pro Knoten eine Spalte — Tausende bis Millionen Output-Größen, die SurroForge als unkorrelierte Skalare behandeln müsste. Das ist weder trainierbar noch reproduzierbar sinnvoll. Felder leben auf einer niedrigdimensionalen Mannigfaltigkeit; modale Reduktion macht sie erst handhabbar und schließt den Loop zurück in ParaView/Post-Processoren.

### 1.3 Kernversprechen (ROM-spezifisch)

Vom Snapshot-Satz zum **validierten, reproduzierbaren Feld-Surrogate** — mit demselben versionierten Run-/Lineage-/Reproduce-Vertrag wie für Skalare, ohne dass der bestehende Skalar-Pfad bricht.

### 1.4 Leitprinzip — minimale Angriffsfläche

Die zentrale Erkenntnis der Session (das **Spine-Theorem**):

> **#36 (Basis im Trainer versteckt) + ein feld-primärer Fingerprint (#43→#64)** bilden ein Rückgrat, an dem die übrigen ~100 Ideen zu *optionalen* Erweiterungen eines ansonsten unveränderten Cores werden.

Konkret: Field ROM v1 liefert **einen neuen Trainer hinter dem bestehenden `BaseTrainer`-Vertrag** plus **eine Evolution des Reproduzierbarkeits-Fingerprints**. Export, Vergleich, Konformität, Reporting, Registry sehen einen ganz normalen Trainer mit `(n, n_field_dof)`-Output. Das ist der gesamte Architektur-Eingriff.

### 1.5 v1-Zuschnitt

**In v1 (die dünne Scheibe, Plan A der Session):**

1. `ReducedFieldTrainer(BaseTrainer)` (#36) — POD-Energieschwelle (#2/#7) + ein innerer Surrogate **pro Mode** (#11).
2. Snapshot-Matrix-Ingestion (#19) — `(n_samples × n_dof)` aus Parquet/HDF5/CSV, **mesh-frei** (#53).
3. `model.reduction:`-Konfigblock (#29) mit Rang- oder Energieziel, Rang **als aufgelöst** im Snapshot (#34).
4. **Feld-primärer Fingerprint** mit Kanonisierung (#16 + #64) — das **Feld-Verdikt ist Grundwahrheit**, Koeffizientenraum die schnelle/Audit-Linse (ADR-ROM-001).
5. Zwei-Quellen-Fehler (#25: Trunkierung vs. Surrogate) + Feld-Write-Back (#21).
6. `surroforge[rom]`-Extra (#31) + Rekonstruktions-Konformitätsprobe (#57).

**Bewusst NICHT in v1** (→ §11): Solver-native/Mesh-Reader (#17/#18/#20), gelernte/Greedy/Krylov-Basen (#3–6), Multi-Output-Vektor-Surrogate (#12), Living-HTML-Report (#45), Kalman-Predict (#87), Inverse-Run (#54), Embedded-Twin (#68–70), Basis-Hub (#99), das **physik-übergreifende Zwei-Verdikt** (#73, nur Bitwise-Äquivalent in v1), Coordinate-Channel (#63, Forschungsspur).

---

## 2. Zielnutzer (Delta zu PRD §2)

Dieselben Personas. ROM-spezifischer Schmerz:

- **Methoden-Engineer (primär):** hat Feld-Ergebnisse aus DOE-Läufen, will ein deploybar kleines Feld-Surrogate mit nachvollziehbarem Fehler — und muss die Reduktion vor Kollegen verteidigen.
- **Käuferpersona (Sim-Leitung):** sieht im Audit, *wie viele Moden* / *welche Energie* das Modell nutzt und ob das ausgelieferte Feld noch dem Sign-off entspricht. Die Auditierbarkeit der Reduktion (`rom inspect`, Zwei-Quellen-Fehler) adressiert genau sie.

---

## 3. Scope-Festlegungen

### 3.1 Edition-Einordnung — **Core** (Entscheidung, mit Begründung)

Field ROM v1 landet im **Core (Apache 2.0)** als optionales Extra `surroforge[rom]`, analog zu `[smt]`/`[onnx]`. Begründung:

- Der Fingerprint-Umbau berührt zwangsläufig den `reproduce()`-Vertrag — dieser **muss** im Core liegen (er ist das Hauptdifferenzierungsmerkmal, PRD §1.4).
- Das Spine-Theorem macht v1 zu „einem Trainer + Fingerprint-Evolution" — keine neue kommerzielle Subsystem-Grenze nötig.
- Die kommerziell interessanten ROM-Flächen (Basis-Hub #99, Embedded-Twin #68–70, Inverse-Design #54/#56, Kalman-Predict #87) sind **Phase-2-Differenzierer** und docken später als Plugins an denselben Seam an (#36 erlaubt einen Closed-Source-Reducer als gewöhnlichen Entry-Point).

→ Die Edition-Matrix (PRD §3.2) erhält eine Zeile „Field ROM (Basis-Reduktion, POD)" in der Core-Spalte. Endgültige Bestätigung: **Offene Frage OQ-1** (§9).

### 3.2 Out of Scope (v1)

Erbt PRD §8 (keine Solver-Anbindung, keine Mesh-Generierung). Zusätzlich v1-spezifisch ausgeschlossen: alles unter §1.5 „Bewusst NICHT in v1" und die Forschungs-/Risiko-Tier-Ideen (#63/#66/#13/#96, OT-Konformität #67).

---

## 4. Funktionale Anforderungen

### 4.1 Kern-Workflow (Delta zum Skalar-Workflow PRD §4.1)

Identisch bis auf: (2) registrierter Datensatz ist eine **Feld-Snapshot-Matrix**; (3) Training erzeugt zusätzlich eine **Basis** + **Koeffizienten-Surrogates**; (6/7) `predict`/`export` liefern standardmäßig **kompakte Koeffizienten**, das volle Feld auf `--field` (#35). Es kommen **keine neuen Top-Level-CLI-Verben** hinzu außer `rom inspect`.

### 4.2 Feld-Ingestion — Snapshot-Matrix (#19, #53)

- **Eingabe:** tabellarisch wie heute (CSV/Parquet/HDF5), aber die Output-Größe ist eine **Spaltengruppe** = ein Feldvektor über `n_dof` DOFs. Eine Zeile = ein DOE-Punkt: `(inputs…, dof_0, dof_1, …, dof_{n-1})`.
- **Mesh-frei (v1-Invariante):** „die Output-Spalten bewegen sich gemeinsam"; es werden **keine Koordinaten/Konnektivität** gelesen. Die Konformitätsprobe (§4.9) **erzwingt**, dass der Reducer keine Geometrie berührt — Mesh-frei ist eine getestete Invariante, kein Feature (#53).
- **Feld-Deklaration:** `data.outputs` akzeptiert ein **Glob/Präfix** (z. B. `u_*`), das geordnet zur Feldspaltenliste expandiert; die Anzahl ist `n_dof`. Genaues Schema → **OQ-2**.

### 4.3 Konfiguration — `model.reduction:` (#29, #34)

```yaml
experiment: bracket-thermal-field
data:
  source: snapshots             # registrierter (n_samples × n_dof)-Datensatz
  inputs: [load, thickness, conductivity]
  outputs: [temp_*]             # Feld-Spaltengruppe (expandiert, geordnet)
  split: { test_fraction: 0.2, seed: 42 }
model:
  method: reduced-field         # der ReducedFieldTrainer-Entry-Point
  reduction:
    basis: pod                  # v1: POD aus Snapshots
    energy: 0.999               # ODER rank: 12 ; ODER rank: adaptive
  inner:
    method: kriging             # innerer Surrogate je Mode (#11)
    corr: squar_exp
validation:
  cross_validation: 5
  metrics: [energy_l2, truncation_error, rmse]   # metric-as-key (#41, #25)
```

- `reduction.energy` **oder** `reduction.rank`; `rank: adaptive` lässt die Engine den Rang bei Fit auflösen und schreibt den **aufgelösten** Rang in den **Config-Snapshot** (#34) — der Snapshot, nicht das Eingangs-YAML, ist die kanonische, reproduzierbare Spezifikation.
- `model.inner` ist der bestehende Methodenblock, durchgereicht an die bestehende `get_trainer`-Auflösung — die drei eingebauten Methoden sind unverändert als innere Surrogates nutzbar.

### 4.4 `ReducedFieldTrainer` (#36, #11, #2/#7)

- Erbt `BaseTrainer`, implementiert `fit(x, Y)` / `predict(x) -> (n, n_dof)` hinter dem **unveränderten** Vertrag. Downstream (Export, `compare`, Konformität, ONNX-Pfadprüfung) sieht einen gewöhnlichen `(n, n_dof)`-Trainer.
- `fit`: POD/SVD über die Trainings-Snapshots → Basis `V (n_dof × r)` (energie-/rang-bestimmt, sign/order-kanonisiert #16); projiziert Snapshots → Koeffizienten `(n × r)`; trainiert **r unabhängige innere Surrogates** (`model.inner`), einen je Mode (#11). Reuse des gesamten bestehenden Trainer-Stacks unverändert.
- `predict`: innere Surrogates → Koeffizienten → optional Rekonstruktion `Ŷ = C·Vᵀ`. Standard-Rückgabe an `predict`-Aufrufer: **Koeffizienten**; volles Feld nur bei `--field` (#35, §4.7).
- `export_formats = {pickle}` in v1 (ONNX für Feld-ROM → Phase 2). `reproduction_tolerance`: feld-energie-relativ (§4.5), nicht der flache `1e-9`-Skalarwert.

### 4.5 Reproduzierbarkeit / Fingerprint — die zentrale Änderung (ADR-ROM-001)

> Dies ist die **risiko- und hebelstärkste** Anforderung. Sie muss **vor** dem Einfrieren des v1-Vertrags gespikt sein (Plan B, §12). Die Session hat das Design via ADR + First-Principles + Pre-mortem auf Entscheidungsreife gebracht; v1 implementiert dessen Minimalkern.

Heutiger Fingerprint (Skalar): SHA-256 über die ersten ≤512 Zeilen der out-of-sample-Holdout-Vorhersagen, verglichen via `reproduce()` gegen flaches `1e-9`. Für Felder bricht das vierfach: Millionen-Knoten (512-Cap sampelt eine willkürliche Scheibe), Knoten-Reihenfolge, SVD-Vorzeichen/Ordnungs-Gauge (falsches *diverged*), ein einzelnes lineares Verdikt kann „gleiche Physik, anderer Solver" nicht ausdrücken.

**v1-Design (gestaffelt, aus ADR-ROM-001):**

1. **Fundament — Kanonisierung (#16):** Vorzeichen (max-abs-Komponente) und Ordnung (Energie) der Moden vor allem Weiteren fixieren. Ohne das false-diverged jede Option.
2. **Definition / Grundwahrheit — beobachtbare Feld-Äquivalenz** (First-Principles-Umkehr): Das Verdikt rekonstruiert das out-of-sample-Holdout-**Feld** an **gepinnten, gauge-aware Probe-Knoten**, kanonisiert, hasht float64 und vergleicht in **feld-energie-relativer** Toleranz. Das rekonstruierte Feld ist das, worauf der Nutzer sich verlässt — nicht die (basis-relativen, privaten) Koeffizienten.
3. **Beschleuniger + Diagnose — Koeffizientenraum** (#43): kanonisierte out-of-sample-Koeffizienten + **gebundener Basis-Hash** als schnelle, per-Mode-auditierbare Linse (welcher Mode bewegte sich) — **nicht** als Quelle der Wahrheit.
4. **Kontinuität automatisch:** Ein Skalar-Run hat die Identitätsbasis ⇒ das Feld *ist* die Vorhersage ⇒ der Byte-Vergleich ist **bit-für-bit das heutige Verhalten**. Der Skalar-Pfad bleibt unberührt — by construction.

**Pre-mortem-Deltas (verbindlich für v1):**

- **Degenerierte/wiederholte Moden** (symmetrische Bauteile → wiederholte Singulärwerte → willkürlicher entarteter Unterraum): Kanonisierung allein würde false-diverged. Mitigation: nahezu-gleiche Singulärwert-Cluster erkennen und den **Unterraum via Hauptwinkel** vergleichen — bzw. auf das gauge-invariante **Feld-Verdikt** zurückfallen. Bestätigt, dass die feld-primäre Definition **tragend** ist, nicht nur sauberer.
- **Basis-Hash ist Diagnose, kein Gate:** ein LAPACK/SciPy-Upgrade ändert Basis-Bytes; würde man darauf gaten, läse jede Reproduktion *diverged* trotz identischem Feld. Der Basis-Hash wird zu **Metadaten** demotiert („Basis geändert, Feld reproduziert ⇒ benignes Re-Gauge").
- **Toleranz in die gehashte Spezifikation falten:** Aufweiten der Toleranz = ein *anderer* Run, kein lockererer Vergleich. (Per-Mode-Bänder #72 → Phase 2.)
- **Probe-Abdeckung in Lineage erfassen** und bei out-of-probe-Energie warnen; bevorzugt eine **globale energiegewichtete Feldnorm** als Definition, Probes nur als Schnellpfad.
- **Zwei-Verdikt (physik vs. bitwise, #73): in v1 NUR das Bitwise-Äquivalent** (= heutiger Vertrag, auf das Feld angewandt). Das *physik-übergreifende* Verdikt (cross-solver) ist die neue Vertragsfläche und landet erst, wenn cross-solver relevant wird (Phase 2). Verdikt-Präzedenz-Policy wird dann ausgeliefert.

### 4.6 Validierung & Metriken (#25, #41, #40)

- **Zwei-Quellen-Fehler (#25):** Report trennt **Trunkierungsfehler** (Projektionsboden, den die Basis nie unterbietet) von **Surrogate-Fehler** — sagt dem Nutzer, ob *Moden* oder *Daten* fehlen. Umgesetzt als Zwei-Pass-Report (#42: Koeffizienten vs. rekonstruiertes Feld in einer Tabelle).
- **Metric-as-key (#41):** `energy_l2`, `truncation_error`, `rmse@mode_k` werden als gewöhnliche `AVAILABLE_METRICS`-Callables registriert ⇒ fließen ohne neue Verdrahtung durch CV, Best-Run-Auswahl und Report.
- **Honest-Basis-CV (#40):** die Basis wird **innerhalb jeder CV-Fold** aus *nur* den Trainings-Snapshots dieser Fold neu abgeleitet — über den bestehenden Per-Fold-Fresh-Factory-Vertrag (`cross_validate(factory, …)`). Globale Basis würde Test-Snapshots in die Basis leaken und R² aufblähen; dies macht Basis-Varianz zu einer erstklassigen Zahl.

### 4.7 Rekonstruktion & Query (#21, #22, #35)

- **Lazy-Koeffizienten-Storage (#22):** Run-Artefakt = Basis + Koeffizienten-Mapper; Rekonstruktion on demand. Artefakt bleibt KB-groß auch bei Millionen-Knoten-Meshes (#47).
- **Reconstruct-on-predict (#35):** `predict` liefert standardmäßig kompakte Koeffizienten; volles Feld nur bei `--field` (KB vs. GB pro Query).
- **Feld-Write-Back (#21):** `predict --field` rekonstruiert und schreibt ein volles Feld (v1: Parquet/HDF5/CSV-Spaltengruppe; VTK/Exodus-Writer → Phase 2 mit den Mesh-Readern).

### 4.8 CLI / API — Delta

| Element | Delta |
|---|---|
| `surroforge rom inspect <run-id>` | **Neu** (#30): Energie-Spektrum, gewählter (aufgelöster) Rang, Energie je Mode, Probe-Abdeckung. Macht die Reduktion auditierbar. |
| `surroforge predict … [--field]` | `--field`-Flag (#35): Koeffizienten (default) vs. volles Feld. |
| `surroforge train` | unverändert (Reduktion ist Config, nicht Flag). |
| Python-API | `Trainer.from_config(...).fit(...)` unverändert; ein Feld-Run unterscheidet sich nur in der Config. |

### 4.9 Konformität (#57)

Die auslieferbare Konformitäts-Suite (`check_trainer` → `ConformanceReport`) erhält eine **Rekonstruktions-Probe**: ein `coeffs → field → coeffs`-Round-Trip innerhalb `DEFAULT_TOLERANCE` (beweist, dass die Basis ein treuer Projektor ist — kein Vorzeichen-/Normalisierungs-Bug) **und** ein **No-Mesh-Check** (#53: der Reducer berührt keine Koordinaten/Konnektivität). Wiederverwendet die bestehende `record()`/Toleranz-Plumbing wie `fingerprint_participation`. Ein kommerzieller ROM-Plugin muss das bestehen, bevor er Konformität beansprucht.

### 4.10 Packaging (#31)

`surroforge[rom]`-Extra: numpy/scipy (bereits Core-Deps; v1 braucht **keine neue schwere Abhängigkeit** — SVD aus scipy). meshio/pyvista **deferred** auf Phase 2 (mit Mesh-Readern). Lazy-Imports, gleiches Muster wie `[smt]`/`[onnx]`.

---

## 5. Nicht-funktionale Anforderungen (Deltas zu PRD §5)

- **Reproduzierbarkeit (§5.2):** ROM-Runs sind über das **Feld-Verdikt** reproduzierbar; der Skalar-Pfad bleibt **byte-identisch** (Identitätsbasis-Äquivalenz) — als Regressions-Guard getestet (analog zum bestehenden Profiling-Gate-Denken).
- **SVD-Determinismus:** SVD-Vorzeichen/Reihenfolge sind über BLAS-Thread-Zahlen nicht garantiert stabil — bereits ein Core-Anliegen (gepinnte Threads in CI). Kanonisierung (#16) + Feld-Verdikt fangen das ab; **muss** in CI unter gepinnten Threads getestet werden.
- **Performance (§5.1):** POD-SVD über `(n_samples × n_dof)` mit moderatem `n_samples` ist günstig; der Kostentreiber sind die `r` inneren Fits × `(folds+2)` — derselbe O(·)-Charakter wie heute, nur ×r. Richtwert: ein v1-Greenfield-Feld-Run (≤ a few hundred Snapshots, r ≤ ~30) im Sekundenbereich.
- **Artefaktgröße:** Run-Dir **kleiner als die Rohdaten** (#47, Lazy-Koeffizienten).

---

## 6. Architektur-Andockung (an die bestehenden Seams)

Die zentrale Aussage: **wie wenig sich ändert.** Downstream sieht einen normalen Trainer.

| Seam / Datei | Berührung in v1 |
|---|---|
| `trainers/` | **+1 Datei** `reduced_field.py` (`ReducedFieldTrainer`); Entry-Point `surroforge.trainers`. |
| `trainers/base.py` | Keine Vertragsänderung. (`reproduction_tolerance` ist bereits feldfähig deklariert — Story 5.4.) |
| `engine.py` | Fingerprint-Funktion verallgemeinert (§4.5): feld-primär statt 512-Byte-Slice; Identitätsbasis ⇒ heutige Bytes. Auflösung des `reduction`-Blocks + Snapshot-Schreiben des aufgelösten Rangs (#34). |
| `services.reproduce` / `_verdict` | Feld-Verdikt (Bitwise-Stufe) + Basis-Hash als Diagnose-Metadatum; Toleranz aus gehashter Spec. |
| `core/config.py` | `model.reduction`-Block + `model.inner`-Block (Pydantic). |
| `validators/metrics.py` | `energy_l2`, `truncation_error` als `AVAILABLE_METRICS`-Callables (#41). |
| `validators/cross_validation.py` | **Unverändert** — Honest-Basis-CV nutzt den bestehenden Per-Fold-Factory-Vertrag (#40). |
| `conformance/trainer.py` | **+** Rekonstruktions-Round-Trip + No-Mesh-Check (#57). |
| `persistence` / `lineage.json` | Neue Felder (§7); `basis.npz` (content-addressed) + Koeffizienten im Run-Dir (#32/#47). |
| `cli/main.py` | **+** `rom inspect`; `predict --field`. |
| `exporters` / `reporters` / `plugins` | **Unverändert** in v1 (ONNX-Feld-Export, HTML-Living-Report → Phase 2). |

---

## 7. Datenmodell-Deltas (zu PRD §7.2)

`lineage.json` erhält (ROM-Runs):

- `reduction`: aufgelöste Spec (`basis`, `rank` [resolved], `energy_captured`) — Teil der gehashten Spezifikation.
- `basis_hash`: kanonisierter Basis-Hash — **diagnostisch, kein Gate** (Pre-mortem-Delta).
- `holdout_fingerprint`: nun das **feld-primäre** Verdikt-Token (Skalar-Run: unverändert).
- `reproduction_tolerance`: feld-energie-relativ, in die gehashte Spec gefaltet.
- `probe_coverage_energy`: Anteil der Feldenergie, den die Probe-Knoten abdecken (Pre-mortem-Delta).

Run-Dir: `basis.npz` (über Runs per Hash dedupliziert, #32) + kompakte Koeffizienten statt voller Feld-Snapshots (#47).

---

## 8. Risiken (Delta zu PRD §11.2)

- **Fingerprint ist der kritische Pfad** — höchster Hebel *und* höchstes Risiko; Plan B (§12) de-riskt ihn vor dem v1-Vertrag.
- **Degenerierte Moden** (symmetrische Teile) → Unterraum-Vergleich/Feld-Verdikt-Fallback (§4.5). *Der häufigste Fall (Symmetrie) ist genau der gefährliche.*
- **SVD-Determinismus über BLAS-Threads** → Kanonisierung + gepinnte CI-Threads.
- **Basis-Hash-Falle** (Gate statt Diagnose) → demotiert (§4.5).

---

## 9. Offene Fragen

- **OQ-1 — Edition:** Field ROM v1 in Core bestätigen (Default, §3.1) oder als Pro-Differenzierer? *Empfehlung: Core* (Fingerprint muss ohnehin Core sein).
- **OQ-2 — Feld-Schema:** Wie wird eine Feld-Spaltengruppe deklariert — Glob/Präfix in `data.outputs` (Default), separates Feld-Manifest, oder Feld-Dataset-Typ bei `data add`?
- **OQ-3 — Probe-Wahl:** v1 mit einfachem deterministischem, energiegewichtetem Probe-Set; Q-DEIM/QR-Picker (#49/#84) jetzt oder Phase 2?
- **OQ-4 — Verdikt-Toleranz-Herleitung:** konkrete feld-energie-relative Default-Toleranz + wie sie in den Spec-Hash eingeht.
- **OQ-5 — Forschungsspur:** Coordinate-Channel (#63) als paralleler Spike *vs.* modale Reduktion bewerten (Brainstorm-Follow-up)?

---

## 10. Erfolgskriterien (v1 Definition of Done)

- Ein **Greenfield-Feld-ROM-Run** läuft durch: `train` → `validate`/CV → `report` (Zwei-Quellen-Fehler) → `reproduce` liefert ein **Feld-Verdikt**.
- **Alle drei eingebauten inneren Trainer** (Kriging/RBF/Polynom) bestehen die bestehende Konformitäts-Suite als innerer Surrogate **unverändert**, und der `ReducedFieldTrainer` besteht die Rekonstruktions- + No-Mesh-Probe.
- **Skalar-Pfad byte-identisch** zur Vor-ROM-Baseline (Regressions-Guard in CI).
- `surroforge rom inspect` zeigt Energie-Spektrum, aufgelösten Rang, Probe-Abdeckung.
- `surroforge[rom]` installiert ohne neue schwere Abhängigkeit; Mesh-frei ist eine getestete Invariante.

---

## 11. Abgrenzung — Phase 2 (Differenzierer) & Forschung

| Tier | Inhalte (Brainstorm) |
|---|---|
| 🔵 **Phase-2-Differenzierer** | Living-Field-Report #45 · Kalman-Predict #87 · Pose-Space-Korrektur #82 · Basis-Hub #99 · Inverse-Run #54 / Goal-Field #56 · Embedded-Twin #68/#70 · Reanalysis #91 · Mesh-Reader #17/#18 + VTK-Write-Back · physik-übergreifendes Zwei-Verdikt #73 + Per-Mode-Bänder #72. |
| 🔴 **Forschung / riskant** | Coordinate-Channel #63 · Self-Basis #66 · Grassmann #13 · Balanced-Truncation #96 · Degeneracy-Handling (über v1-Fallback hinaus) · OT-Konformität #67. |

---

## 12. Umsetzungsplan (Scheibe → Stories)

**Reihenfolge (de-risk first):**

1. **Plan B — Fingerprint-Spike (zuerst):** feld-primäres Verdikt + Koeffizienten-Schnellpfad prototypisieren; Äquivalenz unter gepinnter Basis beweisen; **Degeneracy-Test** (symmetrisches Teil → wiederholte Singulärwerte) bestätigt Unterraum-/Feld-Fallback; Verdikt-Präzedenz + decision-relevante Toleranz festlegen; Toleranz in Spec-Hash falten; Basis-Hash zu Diagnose demotieren. **Output:** aktualisiertes ADR + `reproduce()`-Vertragsvorschlag → wird zu Stories.
2. **Plan A — v1-Scheibe:** `ReducedFieldTrainer` (#36/#11/#2/#7) → besteht bestehende Konformität unverändert · Snapshot-Ingestion (#19) · `model.reduction`-Block (#29/#34) · feld-primärer Fingerprint (#16/#64) · Zwei-Quellen-Fehler (#25) + Write-Back (#21) · `[rom]`-Extra (#31) + Konformitätsprobe (#57).

**Vorgeschlagene Epics:** *Epic R1 — Reduktions-Engine & Trainer* · *Epic R2 — Feld-Reproduzierbarkeit (Fingerprint)* · *Epic R3 — Feld-Validierung & Report* · *Epic R4 — Ingestion, Query & Packaging*. Format wie [`epics.md`](epics.md); AR-/NFR-Mapping analog Phase 1.

---

## 13. Bezüge

- Produkt-PRD: [`PRD.md`](../../PRD.md) (§1.4 Kernversprechen, §3.2 Edition-Matrix, §4.5/§5.2 Reproduzierbarkeit, §8 Out of Scope)
- Brainstorming + ADR-ROM-001: [`brainstorming-session-2026-06-04-231637.md`](../brainstorming/brainstorming-session-2026-06-04-231637.md)
- Architektur (Seams, Plugin-Authoring, Reproduzierbarkeits-Patterns): [`architecture.md`](architecture.md)
