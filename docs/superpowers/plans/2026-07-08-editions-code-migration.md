# Editions-Code-Migration: Methoden, Export & Lineage → Professional

**Datum:** 2026-07-08
**Status:** geplant (durchgesprochen, noch nicht umgesetzt)
**Bezug:** PRD §3.2 (Edition-Matrix, umgeschrieben in `50156ca`), §4.11 (Plugin-System), §6 (Architektur)

## Kontext

Das PRD schreibt den Open-Core-Zuschnitt vor; der Code tut es noch nicht. Heute liegen
`mlp`, `random_forest`, `gradient_boosting` als Builtins in `src/surroforge/trainers/`,
die Exporter (`onnx`, `pickle`) in `src/surroforge/exporters/`, und Reproduzierbarkeit/
Lineage (`reproduce`, Herkunfts-Hashes) ist Core-Infrastruktur. Nach dem neuen Zuschnitt
gehören all das in **Professional**. Diese Migration zieht den Code nach.

## Zwei Rahmenbedingungen

1. **Der Code-Umzug ist Pflicht, nicht nur Gating.** Core ist Apache-2.0 — dort liegender
   Code ist frei forkbar, egal welches Gate davorsteht. Bezahlte *Methoden* müssen physisch
   ins proprietäre `surroforge-professional` wandern. (Lineage ist die Ausnahme, s. Welle 3:
   dort gibt es kein schützenswertes IP — SHA-256 + sortiertes JSON —, daher gaten wir nur
   die Oberfläche statt zu extrahieren.)
2. **Gating funktioniert schon mit dem heutigen Lizenz-Stub.** `has_license()`
   (`surroforge_professional/license.py`) prüft *Vorhandensein* (Env `SURROFORGE_PRO_LICENSE`
   oder `~/.surroforge/professional.lic`). Die Migration ist damit **nicht** von M3 (echte
   Aktivierung) blockiert. M3 validiert später den *Inhalt* der Lizenz — komplementär, aber
   getrennt. Lokal schaltet `SURROFORGE_PRO_LICENSE=dev` alles frei.

## Entscheidungen (locked)

| Frage | Entscheidung | Begründung |
|---|---|---|
| **Enforcement** | **Hybrid** — Seam für auflösbare Plugins (Trainer/Exporter), Self-Gate für Befehle | Der Seam (`gate.check`) feuert bereits bei jeder `_resolve`-Auflösung (`registry.py:118`) und deckt Trainer/Exporter zentral ab — auch `reduced_field` mit NN-Inner. `reproduce`/`info`/`compare` sind aber keine Plugin-Auflösungen, also brauchen sie Self-Gate. |
| **Lineage** | **Oberfläche gaten**, nicht extrahieren | Erfassung (Hashes) ist billig, load-bearing für Store & `compare` und tief verwoben (~18 Dateien: engine 32×, services 34×, run 30×). Extraktion brächte großes Risiko am Byte-Identitäts-Vertrag für **null** zusätzlichen Schutz (kein IP-Moat). |
| **Phasing** | **W1 → W2 → W3** | Methoden zuerst (schneller, hochwertiger Gewinn, geringes Risiko), Export danach, Lineage zuletzt. Jede Welle einzeln testbar und committbar. |

## Enforcement-Architektur (vereinheitlichter Authorizer)

Weil `export`/`reproduce`/`info`/`compare` in Core's `cli/main.py` **bleiben** (Oberfläche
gaten), darf ihr Self-Gate nicht `surroforge_professional.license` importieren — Core hängt
nicht von Pro ab. Deshalb bekommt `gate.py` einen editions-neutralen Authorizer-Hook, den
*Seam und Self-Gates gemeinsam* nutzen:

```python
# src/surroforge/plugins/gate.py  (Core, editions-neutral)
_authorizer = None                      # von einer kommerziellen Edition beim Import gesetzt

def register_authorizer(fn):            # fn() -> bool  ("darf lizenzpflichtiges laufen?")
    global _authorizer
    _authorizer = fn

def _authorized() -> bool:
    return _authorizer is None or _authorizer()   # kein Authorizer installiert => Core-only, alles erlaubt

def check(metadata):                    # Resolution-Seam (Trainer/Exporter)
    if metadata.requires_license and not _authorized():
        raise PluginError("Professional license required …")

def require_license(feature: str):      # Command-Self-Gate (reproduce/export/…)
    if not _authorized():
        raise PluginError(f"'{feature}' requires a Professional license …")
```

- `PluginMetadata` bekommt ein neutrales Feld `requires_license: bool`, in `_resolve` aus
  einem Klassen-Attribut gelesen (analog zu `resolve_api_version`). `gate.py` nennt nie
  „Professional" — es fragt nur „verlangt dieses Plugin eine Lizenz?".
- Die Pro-Installation registriert den Authorizer **einmal** beim Import:
  `gate.register_authorizer(has_license)`.
- **Offener Sub-Punkt:** `surroforge-torch` und `surroforge-autoencoder` sind eigene Pakete.
  Wo lebt `has_license`/der Authorizer, damit alle drei Pro-Pakete ihn teilen? Optionen:
  (a) ein kleines gemeinsames `surroforge-license`-Paket, von dem alle Pro-Pakete abhängen;
  (b) `has_license` bleibt in `surroforge-professional`, und torch/autoencoder hängen davon ab.
  → in Welle 1 entscheiden.

## Welle 1 · Methoden → Professional

**Ziel:** `mlp`, `random_forest`, `gradient_boosting` sind Pro; Core behält nur
Kriging/RBF/Polynom (+ `reduced_field` mit Nicht-NN-Inner).

Schritte:
- Trainer-Module nach `plugins/surroforge-professional/src/surroforge_professional/trainers/`
  verschieben; `requires_license = True` als Klassen-Attribut setzen.
- Aus `_BUILTIN[TRAINER_GROUP]` in `src/surroforge/plugins/registry.py` streichen
  (mlp, gradient-boosting, random-forest).
- Im Pro-`pyproject.toml` ergänzen: `[project.entry-points."surroforge.trainers"]` mit
  `mlp`, `random-forest`, `gradient-boosting` (gleiches Muster wie die vorhandenen
  `surroforge.reporters`/`.commands`-Einträge).
- Seam scharfstellen: `requires_license` in `PluginMetadata` + `seam.resolve_requires_license`;
  `_resolve` füllt das Feld; `gate.check` wertet es aus.
- Authorizer: Sub-Punkt oben entscheiden, dann `register_authorizer(has_license)` beim Import
  des/der Pro-Pakete(s).
- `surroforge-torch` / `surroforge-autoencoder`: `requires_license = True` an den Trainer-Klassen.

Effekt: `reduced_field` mit `inner: mlp` gated automatisch (`get_trainer("mlp")` →
`_resolve` → `gate.check`).

**Breaking Changes / Tests:**
- `from surroforge.trainers.{mlp,random_forest,gradient_boosting} import …` verschwindet aus Core.
- Core-Tests, die diese Methoden trainieren, ins Pro-Paket verschieben oder mit einer
  Lizenz-Fixture (`monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "test")`) versehen.
- NACA-Beispiel: mlp/torch-Runs werden Pro; der Polynom-Pfad bleibt der Core-Weg (bereits so geplant).
- `docs/superpowers/plans/2026-06-29-gradient-boosting.md`, `2026-06-30-random-forest.md`
  beschreiben diese als Core-Features — historisch, nicht anfassen.

## Welle 2 · Export → Professional

**Ziel:** `onnx`/`pickle`-Export sind Pro; Core kann weiter `predict`.

Schritte:
- **Prüfpunkt zuerst:** hängt der interne Run-Store (`src/surroforge/persistence/store.py`)
  am Pickle-*Exporter*-Plugin? Der Store muss das Modell für `predict` serialisieren —
  diese interne Serialisierung bleibt Core und muss vom nutzerseitigen Exporter entkoppelt sein.
- `onnx_exporter.py` / `pickle_exporter.py` nach Pro; aus `_BUILTIN[EXPORTER_GROUP]` streichen;
  als `[project.entry-points."surroforge.exporters"]` registrieren → laufen über denselben Seam.
- `export`-Befehl in `cli/main.py`: `gate.require_license("export")` oben (Self-Gate).
- `train --onnx` (Export beim Training) ebenso gaten.

## Welle 3 · Repro/Lineage-Oberfläche gaten

**Ziel:** Reproduzierbarkeit/Lineage nur in Pro nutzbar; Erfassung bleibt interne Core-Infra.

Schritte (reine Self-Gates, keine Extraktion):
- `reproduce`-Befehl (`cli/main.py`) + `services.reproduce`: `gate.require_license("reproduce")`.
- `info`: Lineage-Panel nur mit Lizenz zeigen (Basis-Infos — Methode/Metriken — bleiben frei).
- `compare`: Lineage-Spalten nur mit Lizenz.
- `lineage.json`-Export/Anzeige gaten; die interne Erfassung (Hashes) läuft unverändert weiter.

**Akzeptierte Konsequenz:** Da Core Apache ist und die Erfassung dort bleibt, kann ein Fork
das Self-Gate entfernen. Reproduzierbarkeit-als-bezahlt hat keinen Code-Moat — bewusst so
entschieden.

## Downstream-Doku (separate Aufräum-Runde, nicht Teil dieser Migration)

Diese Nutzer-Doku beschreibt noch das alte Modell und sollte nach der Migration nachgezogen werden:
- `docs/scenarios/reproduction.md` — Reproduktions-Szenario (jetzt Phase 2 / Pro)
- `docs/reference/cli.md` — dokumentiert `export`/`reproduce` als Core-Befehle
- `docs/quickstart.md`, `docs/index.md` — erwähnen ggf. Reproduzierbarkeit als Core-Versprechen

## Bezug zur Release-Sequenz

Diese Migration ist Teil der **M3-Lizenzarbeit** (siehe Release-Sequenz). Sie kann vor der
echten Aktivierung (M3) laufen, weil der Stub-`has_license` fürs Gating reicht. Der
API-Freeze (M4a, Plugin-API-Version) sollte **vor** dieser Migration stehen, da sie
`PluginMetadata` um `requires_license` erweitert — ein additives, abwärtskompatibles Feld,
aber es gehört in die eingefrorene Contract-Version.

## Offene Punkte

- Wo lebt der geteilte Authorizer/`has_license` für alle drei Pro-Pakete? (Welle 1)
- Entkopplung interner Store ↔ Pickle-Exporter — Aufwand erst in Welle 2 messbar.
- `PluginMetadata.requires_license` additiv einführen, ohne die eingefrorene Plugin-API-Version zu brechen.
