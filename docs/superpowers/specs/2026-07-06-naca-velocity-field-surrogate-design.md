# NACA 0012: Geschwindigkeitsfeld-Surrogat (U) — Design

**Datum:** 2026-07-06
**Status:** Vom Nutzer abschnittsweise freigegeben (Abschnitte 1–4)
**Kontext:** `examples/naca0012_openfoam/` — der Real-CFD-Begleiter zu Example 09.

## Ziel

Zusätzlich zum Druckfeld-Surrogat `(AoA, |U∞|) → p` ein Surrogat für das
**volle Geschwindigkeits-Vektorfeld** `(AoA, |U∞|) → (Ux, Uy, Uz)` je Zelle trainieren
und zusammen mit dem Druckfeld in **eine** ParaView-lesbare VTKHDF-Zeitreihe schreiben.
Gewählter Ansatz: **A — symmetrische Erweiterung** (ein Dataset, eine VTKHDF-Datei);
Alternative B (getrenntes Dataset `sweep-u` + eigene Datei) wurde verworfen, weil sie
die `same_data`-Lineage bricht und die Vektor-Erweiterung des Writers trotzdem braucht.

Kein neues CFD nötig: die 110 konvergierten `U`-Felder (ASCII `List<vector>`,
10 720 Zellen) liegen bereits unter `run/case_*/<t_conv>/U`.

## Nutzerentscheidungen

1. **Zielgröße:** voller Vektor (Ux, Uy, Uz) — 32 160 DOFs. Uz ≈ 0 (2-D-Fall) wird
   trivial mitgelernt, keine Sonderbehandlung. (Alternativen „nur Ux/Uy" und
   „nur |U|" verworfen: Re-Padding-Aufwand bzw. Verlust der Richtungsinformation.)
2. **Ansatz A** — ein Dataset, eine VTKHDF-Datei (siehe oben).

## 1. Ingest & Datenfluss (`ingest_and_train.py`)

- Zweiter Aufruf `read_openfoam_sweep(run_dir, field="U", params=<gleicher Callback>)`.
  Der Core-Reader beherrscht `List<vector>` bereits: cell-major-Spalten
  `U_0000_x, U_0000_y, U_0000_z, U_0001_x, …` (openfoam.py, `_KIND_COMPONENTS`).
- **Alignment-Guard:** Beide Frames sind nach Case-Namen sortiert; vor dem Merge wird
  geprüft, dass die `aoa`/`speed`-Spalten beider Frames exakt übereinstimmen. Ein Case,
  der `p` hat, aber `U` nicht (oder umgekehrt), würde die Zeilen sonst still
  verschieben → harter `RuntimeError` mit Case-Kontext.
- Die `U_*`-Spalten werden per `pd.concat(axis=1)` an die bestehende Snapshot-Tabelle
  angehängt. Es bleibt bei **einem** Dataset `sweep` (eine CSV, ein `data_hash` →
  `same_data`-Lineage über alle Runs des Vergleichs).
- Ingest-Meldung: `Ingested 110 real cases: 10720 pressure DOFs + 32160 velocity DOFs + Cl + Cd`.
- Größe: CSV wächst von ~15 MB auf ~60 MB (110 × 42 884 Spalten) — akzeptiert.

## 2. Training & Vergleich

Dritter Trainingsblock nach exakt dem p-Feld-Muster:

```
=== Field surrogate: (AoA, speed) -> velocity field (POD + inner) ===
```

- Experimente `velocity-{mlp,torch-mlp,polynomial}`; Modell `reduced-field` mit
  `{"basis": "pod", "energy": 0.999}`; innere Kandidaten `_candidates([48, 48])`
  (torch-mlp nur, wenn das Plugin installiert ist).
- `data.outputs: ["U_*"]` — die fnmatch-Wildcard expandiert in Frame-Spaltenreihenfolge
  (loader.py `expand_field_columns`), die cell-major-x/y/z-Ordnung bleibt erhalten.
- Gleiche Validierung (3-fach CV, `rmse/r2/mae`, 0.25-Holdout, Seed 1), gleicher
  Vergleichs-Report (`_report`, Metrik RMSE, „← best"-Markierung).

## 3. VTKHDF-Generalisierung (`surrogate_vtk.py`)

Signaturwechsel von „ein Skalarfeld" auf „benannte Felder" — sauberer Bruch ohne
Kompatibilitätsschicht (Konsumenten: nur `ingest_and_train.py` und
`tests/test_naca_vtk_export.py`):

```python
def write_surrogate_vtkhdf(
    run_dir: Path,
    output: Path,
    samples: pd.DataFrame,
    fields: Mapping[str, np.ndarray],   # name -> (n_cases, n_cells * n_components)
) -> Path
```

- Komponentenzahl je Feld abgeleitet als `matrix.shape[1] // n_cells` gegen die
  Zellzahl des VTU-Templates; muss exakt aufgehen und 1 oder 3 sein — sonst
  benannter `ValueError` (Feldname + Shapes).
- Skalarfelder wie bisher flach; Vektorfelder als `(n_steps·n_cells, 3)`-Dataset
  unter `CellData/<name>` (ParaView liest die Komponentenzahl aus der zweiten Achse;
  Magnitude/Glyphs/Stromlinien funktionieren). Vektor-Reshape aus der cell-major-Reihe
  `[x0 y0 z0 x1 …]` → `reshape(-1, 3)`.
- `Steps/CellDataOffsets/<name>` zählt für Skalar **und** Vektor in Zellen (wie bisher).
- Ergebnis: weiterhin **eine** Datei `surrogate_fields.vtkhdf` mit `p` und `U` pro
  Zeitschritt; `samples.csv` daneben unverändert.

## 4. Export, Tests, Verifikation, README

**Export:** bestes U-Surrogat (RMSE-Vergleich) → `velocity_surrogate.pkl`
(`reduced-field` ist pickle-only). Bestehende Artefakte unverändert. VTKHDF-Schritt
wertet **beide** besten Feld-Surrogate auf allen Cases aus:
`surrogate p+U fields -> surrogate_vtk/surrogate_fields.vtkhdf`.

**Tests (`tests/test_naca_vtk_export.py`):** auf die `fields`-Mapping-Signatur
umstellen; Vektorfall ergänzen (minimal-VTU mit 3 Zellen, `{"p": (2,3), "U": (2,9)}` →
`CellData/U`-Shape `(6,3)`, Werte, `CellDataOffsets/U == [0,3]`); Fehlerfall
„DOF-Zahl nicht durch Zellzahl teilbar". Läuft ohne OpenFOAM in CI.

**Verifikation:** `ingest_and_train.py run --out ex09_real` einmal komplett neu
ausführen (überschreibt `ex09_real/naca` wie dokumentiert); VTKHDF stichprobenartig
mit h5py prüfen (Shapes, Offsets, Endlichkeit).

**README (`examples/naca0012_openfoam/README.md`):**
- „What it does" + Export-Beschreibung erwähnen das U-Surrogat (drittes Surrogat).
- „Verified"-Fit-Tabelle bekommt eine Zeile `Field (AoA, U) → U (POD-NN)` mit den
  echten neuen Zahlen aus dem Verifikationslauf.
- Ingest-Zahl aktualisieren (10 720 p-DOFs + 32 160 U-DOFs + Cl + Cd).
- On-Disk-Layout ergänzt `velocity_surrogate.pkl`; VTKHDF-Beschreibung: `p` **und** `U`.

## Nicht-Ziele

- Kein neuer CFD-Lauf, keine Änderung an `run_sweep.py`.
- Keine Änderung am Core (`read_openfoam_sweep` kann Vektoren bereits).
- Keine Magnitude-/Komponenten-Filterung, kein zweites Dataset.
