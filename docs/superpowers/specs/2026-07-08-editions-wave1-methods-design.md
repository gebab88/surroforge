# Welle 1: NN- & Ensemble-Methoden → Editionen (tier-aware) — Design

**Datum:** 2026-07-08
**Status:** Design (abgesegnet, bereit für Plan)
**Bezug:** Plan `docs/superpowers/plans/2026-07-08-editions-code-migration.md`; PRD §3.2 (Edition-Matrix), §4.11 (Plugin-System), §6 (Architektur)

> **Verfeinerung ggü. dem Plan-Dokument:** Der dort skizzierte `requires_license`-**Bool**
> wird hier zu einem `required_tier`-**String** (`"professional"` | `"enterprise"` | `None`),
> weil autoencoder als Enterprise-Stufe eine geordnete Tier-Prüfung verlangt.

## Ziel

`mlp`, `random_forest`, `gradient_boosting` (heute Core-Builtins) und `torch-mlp`
(heute eigenes Paket) werden **Professional**-Methoden; `autoencoder-field` wird
**Enterprise**. Core behält nur Kriging/RBF/Polynom (+ `reduced-field`). Durchsetzung
zentral über den Plugin-Seam, **stufen-bewusst** (Professional vs Enterprise). Welle 1
wird am Stück umgesetzt.

## Entscheidungen (aus dem Brainstorming, locked)

| Frage | Entscheidung |
|---|---|
| Enforcement | Hybrid; in W1 nur der **Resolution-Seam** (Befehle folgen in W2/W3) |
| Lizenz-Modell | **Tier-aware jetzt**: `professional` < `enterprise`, enterprise ⊇ professional |
| Methoden-Heimat | `mlp`/`random_forest`/`gradient_boosting` **und** `torch-mlp` → `surroforge-professional` |
| Autoencoder | → **neues** `surroforge-enterprise`-Paket |
| torch-Paket | **auflösen**, `torch-mlp` (+ `pt`-Exporter) in professional; `torch` als optionales Extra `[torch]` |
| autoencoder-Paket | **auflösen**, Code nach enterprise |
| Authorizer-Heimat | in `surroforge_professional`; via `surroforge.authorizers`-Entry-Point; enterprise nutzt denselben (hängt von professional ab) |
| Lizenz-Status | beide kommerziellen Ziel-Pakete (`surroforge_professional`, `surroforge_enterprise`) `Proprietary` |
| Namensgebung | **Distributions- = Import-Name** (Unterstriche): `surroforge_professional`, `surroforge_enterprise`. Pip normalisiert per PEP 503 (Bindestrich-Schreibweise löst identisch auf). Verzeichnis-Rename `git mv plugins/surroforge-professional plugins/surroforge_professional` gehört zu W1. |

## Rahmenbedingung

Gating funktioniert schon mit dem heutigen Lizenz-**Stub** (Vorhandensein von Env/Datei);
in W1 wird der Stub um eine **Tier-Auslesung** erweitert. Echte Aktivierung/Validierung
bleibt M3. Nicht von M3 blockiert.

## Architektur

### Tier-Modell

Geordnete Stufen `professional (1) < enterprise (2)`. Eine gewährte Stufe erfüllt jede
gleich-hohe oder niedrigere Anforderung: eine **Enterprise-Lizenz schaltet Professional-
Features mit frei**, eine **Professional-Lizenz schaltet Enterprise-Features NICHT frei**.

### Core-Seam (bleibt editions-neutral)

`src/surroforge/plugins/seam.py`
- `PluginMetadata` bekommt `required_tier: str | None = None`.
- Neue `resolve_required_tier(cls) -> str | None`: liest Klassen-Attribut `required_tier`
  (default `None`), validiert `str | None` (Core kennt die konkreten Tier-Namen NICHT —
  sie sind Daten, die durchgereicht werden), analog zu `resolve_api_version`.
- `PLUGIN_API_VERSION` bleibt `1`: `required_tier` ist additiv und abwärtskompatibel
  (undeklariert → `None`), kein Bump.

`src/surroforge/plugins/registry.py`
- `_resolve` füllt `required_tier=resolve_required_tier(cls)` in die Metadata (vor `gate.check`).
- `_BUILTIN[TRAINER_GROUP]`: `mlp`, `gradient-boosting`, `random-forest` **entfernen**
  (Core-Builtins = `polynomial`, `kriging`, `rbf`, `reduced-field`).
- `ep.load()`-Fehler bei fehlendem Extra sauber behandeln — siehe Fehlerbehandlung.

`src/surroforge/plugins/gate.py` (editions-neutral)
- Authorizer-Discovery: neue Entry-Point-Gruppe `surroforge.authorizers`, gescannt/gecacht
  (wie die übrigen Gruppen). Jeder Authorizer ist ein Callable `authorize(required_tier: str) -> bool`;
  optional trägt er ein Attribut `hint: str` mit editions-eigenem Aktivierungs-Hinweis.
- `check(metadata)`:
  ```python
  if metadata.required_tier is None:
      return                                   # frei (Core)
  authorizers = _load_authorizers()            # cached
  if any(a(metadata.required_tier) for a in authorizers):
      return
  hints = " ".join(h for a in authorizers if (h := getattr(a, "hint", "")))
  raise PluginError(
      f"'{metadata.name}' requires a '{metadata.required_tier}' license, and no "
      f"installed SurroForge edition grants it. {hints}".rstrip()
  )
  ```
- **Editions-Neutralität der Meldung:** Core nennt selbst KEINE Aktivierungs-Mechanik
  (kein Env-Name, kein Lizenzpfad) — das wäre Professional-Wissen im Core. Der actionable
  Hinweis kommt vom installierten Authorizer: professional setzt
  `authorize.hint = "Set SURROFORGE_PRO_LICENSE or install ~/.surroforge/professional.lic."`.
  Ohne installierte Edition bleibt die generische Meldung („no installed SurroForge edition …").
- **Semantik:** kein Authorizer installiert + `required_tier` gesetzt → `PluginError` (deny).
  Core-only hat gar keine `required_tier`-Plugins → `check` verweigert dort nie.
- **Performance:** der Authorizer-Scan feuert nur, wenn `required_tier` gesetzt ist —
  Core-Builtins (`None`) kurzschließen sofort, kein Overhead auf dem freien Pfad.

### Authorizer in surroforge-professional (tier-aware)

`plugins/surroforge-professional/src/surroforge_professional/license.py`
- `granted_tier() -> "professional" | "enterprise" | None` — Stub: liest
  `SURROFORGE_PRO_LICENSE` (Env) bzw. `~/.surroforge/professional.lic`; Wert enthält
  (case-insensitiv) „enterprise" → `"enterprise"`, sonst vorhanden → `"professional"`,
  leer/fehlt → `None`. (M3 härtet das zu echter Validierung.)
- `_TIER_ORDER = {"professional": 1, "enterprise": 2}`.
- `authorize(required_tier: str) -> bool`: `g = granted_tier(); return g is not None and
  _TIER_ORDER[g] >= _TIER_ORDER.get(required_tier, 10**9)` (unbekannte Anforderung → deny).
- Rückwärtskompatibel für bestehende Self-Gates: `has_license() -> bool = granted_tier() is not None`;
  `require_license(tier: str = "professional") -> None` raised, wenn nicht `authorize(tier)`.
  Bestehende Aufrufer (`html`, `comparison`, `doe`) rufen `require_license()` ohne Argument
  → Professional-Tier → unverändertes Verhalten.
- `authorize.hint = "Set SURROFORGE_PRO_LICENSE or install ~/.surroforge/professional.lic."`
  (der editions-eigene Aktivierungs-Hinweis, den Core's Gate-Meldung anhängt — s. o.).
- Registriert als `[project.entry-points."surroforge.authorizers"]`:
  `professional = "surroforge_professional.license:authorize"`.

### Paket-Topologie (Ziel)

```
plugins/
├── surroforge_professional/            # Proprietary (git mv aus surroforge-professional)
│   └── src/surroforge_professional/
│       ├── trainers/                   # NEU
│       │   ├── mlp.py                   #   required_tier = "professional"  (aus Core)
│       │   ├── random_forest.py         #   required_tier = "professional"  (aus Core)
│       │   ├── gradient_boosting.py     #   required_tier = "professional"  (aus Core)
│       │   └── torch_mlp.py             #   required_tier = "professional"  (aus surroforge-torch)
│       ├── pt_exporter.py               # aus surroforge-torch (Gating erst W2)
│       ├── license.py                   # granted_tier / authorize / has_license
│       ├── commands/ doe/ reports/      # unverändert
│       └── ...
└── surroforge_enterprise/              # NEU, Proprietary
    └── src/surroforge_enterprise/
        └── trainers/
            └── autoencoder_field.py     #   required_tier = "enterprise"   (aus surroforge-autoencoder)

# entfernt: plugins/surroforge-torch/, plugins/surroforge-autoencoder/
```

`surroforge_professional/pyproject.toml`
- `name = "surroforge_professional"` (Distributions- = Import-Name).
- `[project.entry-points."surroforge.trainers"]`: `mlp`, `random-forest`, `gradient-boosting`,
  `torch-mlp`.
- `[project.entry-points."surroforge.exporters"]`: `pt` (aus torch; ungegatet bis W2).
- `[project.entry-points."surroforge.authorizers"]`: `professional`.
- `dependencies`: `scikit-learn` explizit ergänzen.
- `[project.optional-dependencies]`: `torch = ["torch>=2.0"]`.

`surroforge_enterprise/pyproject.toml` (neu)
- `name = "surroforge_enterprise"`, `license = "Proprietary"`,
  `dependencies = ["surroforge", "surroforge_professional", "numpy>=1.24"]`.
- `[project.optional-dependencies]`: `torch = ["torch>=2.0"]`.
- `[project.entry-points."surroforge.trainers"]`: `autoencoder-field = "surroforge_enterprise.trainers.autoencoder_field:AutoencoderFieldTrainer"`.

## Datenfluss (Beispiel)

`surroforge train --config ...` mit `model.method: mlp`:
1. Engine ruft `get_trainer("mlp")` → `registry._resolve(TRAINER_GROUP, "mlp", ...)`.
2. Entry-Point (professional) wird geladen; Metadata mit `required_tier="professional"`.
3. `gate.check(metadata)` → `authorize("professional")` → `granted_tier()`.
   - Keine/leere Lizenz → `PluginError` („requires a professional license").
   - `SURROFORGE_PRO_LICENSE=dev` → `professional` → erlaubt.
4. `reduced-field` mit `inner: mlp` läuft über denselben `get_trainer("mlp")` → automatisch gegatet.
5. `autoencoder-field`: `required_tier="enterprise"` → eine Professional-Lizenz wird abgelehnt,
   `SURROFORGE_PRO_LICENSE=enterprise` erlaubt.

## Fehlerbehandlung

- **Fehlende Lizenz/zu niedrige Stufe:** `PluginError` mit klarer Meldung inkl. der geforderten
  Stufe und dem Aktivierungs-Hinweis (Env/Datei).
- **Fehlendes `[torch]`-Extra:** `torch` wird in `torch_mlp.py` **lazy** importiert (nicht auf
  Modul-Ebene), damit `ep.load()` in `_resolve` gelingt und der Seam vor dem Import läuft; beim
  Instanzieren/Fit gibt ein fehlender torch-Import einen klaren `PluginError`:
  „torch-mlp requires the torch extra: pip install 'surroforge_professional[torch]'". Analog
  `autoencoder-field` in enterprise.

## Test-Strategie

- **Seam/Gate (Core, unit):** `resolve_required_tier` liest/validiert das Attribut;
  `gate.check` erlaubt bei `None`, verweigert bei gesetztem Tier ohne Authorizer, erlaubt mit
  zustimmendem Authorizer; Fake-Authorizer injizieren (wie bestehende Tests
  `_entry_points_grouped`/Authorizer-Loader ersetzen).
- **Tier-Logik (professional, unit):** `authorize("professional")`/`authorize("enterprise")`
  gegen `granted_tier()` ∈ {None, professional, enterprise}: Pro-Lizenz erfüllt professional,
  **nicht** enterprise; Enterprise-Lizenz erfüllt beide.
- **Integration:** `get_trainer("mlp")` ohne Lizenz → `PluginError`; mit `SURROFORGE_PRO_LICENSE`
  → Klasse; `get_trainer("autoencoder-field")` mit Pro-Lizenz → `PluginError`, mit Enterprise → ok.
- **Migrierte Trainer-Tests:** die bisherigen Core-Tests für mlp/RF/GB nach professional (mit
  Lizenz-Fixture `monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "test")`); torch/autoencoder-Tests
  mit in die Zielpakete.
- **Konformanz-Suite:** `tests/test_conformance.py` importiert die drei Trainer direkt und
  parametrisiert über sie — diese Params in Core streichen (bleiben: polynomial/kriging/rbf/
  reduced-field); professional fährt dieselbe geteilte Suite (`surroforge.conformance`) gegen
  seine vier Trainer, enterprise gegen autoencoder-field (wie es torch/autoencoder heute schon tun).
- **Core-only unberührt:** `available_trainers()` in Core enthält die drei nicht mehr; ein Test,
  der Core-only-Auflösung von kriging/rbf/polynomial/reduced-field ohne jeden Authorizer bestätigt.
- **`[torch]`-Extra fehlt:** klarer `PluginError` beim Instanzieren von `torch-mlp`.

## Downstream (in W1 mitgenommen)

- **PRD §3.2 + Editions-Artefakt:** Autoencoder von *Professional* → *Enterprise* verschieben
  (revidiert `50156ca`). Zeile „Neuronale Netze (sklearn-MLP, PyTorch, Autoencoder)" → Autoencoder
  in die Enterprise-Sektion.
- **Beispiele 06–09** — alle vier berühren migrierende Methoden:
  - `06_mlp_surrogate.py`: Prämisse im Docstring („mlp … rides Core scikit-learn, in Core")
    wird falsch → Text auf Professional umschreiben + Lizenz-Hinweis.
  - `07_pod_nn_field.py`: `inner: mlp` braucht Lizenz; der Polynom-Zweig bleibt Core →
    Hinweis ergänzen.
  - `08_torch_mlp.py`: nennt das aufgelöste Paket `surroforge-torch` beim Namen → auf
    `surroforge_professional[torch]` umschreiben + Lizenz-Hinweis.
  - `09_naca0012_openfoam/`: mlp/torch-Runs brauchen jetzt eine Lizenz; Polynom bleibt der
    Core-Pfad. Docstring/README-Hinweis („für die NN-Kandidaten
    `pip install 'surroforge_professional[torch]'` + `SURROFORGE_PRO_LICENSE`").
  - Begleitend: `docs/examples.md` + `examples/README.md` (Beschreibungen von 06–08).
- **Dev-Umgebung/CI:** `scripts/ci.sh` globbt `plugins/*/tests` generisch — enterprise wird
  automatisch erfasst, die aufgelösten Pakete fallen weg. Editierbare Installs im Venv
  umstellen (surroforge-torch/-autoencoder deinstallieren, enterprise installieren);
  Setup-Hinweise (README/LOCAL_SETUP), die `pip install -e plugins/surroforge-torch` o. ä.
  nennen, anpassen.
- **CHANGELOG.**

## Außerhalb von Welle 1

- Export-Gating (`onnx`/`pickle`/`pt`) und `gate.require_license(feature)` für Befehle → W2.
- reproduce/Lineage-Gating → W3.
- Echte Lizenz-Validierung/Aktivierung → M3.
- Nutzer-Doku-Sweep (`docs/scenarios/reproduction.md`, `docs/reference/cli.md`) → separate Runde.

## Risiken / offene Punkte

- **Enterprise-Paket startet Phase-3-Scaffolding früh** (Roadmap: „Won't this cycle") — bewusst,
  vorerst nur als Code-Heimat für autoencoder.
- **torch als schwere Abhängigkeit:** durch `[torch]`-Extra isoliert; Entry-Point muss ohne das
  Extra sauber failen (lazy Import).
- **Stub-Tier-Auslesung** ist bewusst simpel (Env-/Datei-Inhalt); M3 ersetzt sie durch echte
  Validierung. Bis dahin ist die Durchsetzung nur so stark wie „Datei/Variable vorhanden".
- **`required_tier` im eingefrorenen Plugin-Contract (API v1):** additiv/abwärtskompatibel —
  dokumentieren, kein Version-Bump.
