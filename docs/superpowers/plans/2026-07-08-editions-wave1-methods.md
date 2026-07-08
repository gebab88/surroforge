# Welle 1: NN- & Ensemble-Methoden → Editionen (tier-aware) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `mlp`/`random-forest`/`gradient-boosting` + `torch-mlp` werden lizenzpflichtige Professional-Methoden, `autoencoder-field` wird Enterprise; durchgesetzt über einen tier-aware Plugin-Seam in Core.

**Architecture:** Core bleibt editions-neutral: Plugins *deklarieren* `required_tier` (Klassen-Attribut, für Core opake String-Daten), installierte Editionen *registrieren* Authorizer über die neue Entry-Point-Gruppe `surroforge.authorizers`, und `gate.check` (feuert bereits bei jeder Auflösung, `registry.py:118`) fragt sie. `surroforge_professional` absorbiert die vier Pro-Trainer (torch via `[torch]`-Extra + Guard-Modul) und stellt den tier-aware Authorizer; das neue `surroforge_enterprise` beherbergt `autoencoder-field`.

**Tech Stack:** Python 3.10+, hatchling (src-Layout), Entry-Points via `importlib.metadata`, pytest, scikit-learn/SMT/PyTorch.

**Spec:** `docs/superpowers/specs/2026-07-08-editions-wave1-methods-design.md` — bei Widerspruch gewinnt die Spec.

> **Verfeinerung ggü. der Spec (Mechanik, nicht Verhalten):** Die Spec verlangt, dass
> `ep.load()` ohne das `[torch]`-Extra gelingt („lazy import"). Beide Torch-Trainer
> definieren aber `nn.Module`-Subklassen auf Modul-Ebene — ein echter Lazy-Import ist
> ohne großen Umbau unmöglich. Der Plan nutzt stattdessen ein **Guard-Modul-Paar**:
> `trainers/torch_mlp.py` (Entry-Point-Ziel; try/except-Import) + `trainers/_torch_impl.py`
> (die unveränderte Implementierung). Ohne torch liefert das Guard-Modul eine Stub-Klasse,
> deren `__init__` den in der Spec geforderten `PluginError` mit Install-Hinweis wirft.
> Spec-Intention (Auflösung gelingt, Fehler erst beim Instanzieren) bleibt exakt erhalten.

## Global Constraints

- Git-Identität: `Georg Babij <g.babij@gmail.com>` (repo-lokal gesetzt); Commits direkt auf `main`; **kein `Co-Authored-By`-Trailer**; **niemals pushen** (User zeigt-dann-pusht).
- Conventional-Commits-Stil wie im Log (`feat:`/`refactor:`/`docs:`/`test:`).
- `git add` immer mit expliziten Pfaden — nie `git add -A`/`.` (gitignorierte `examples/09_naca0012_openfoam/{run,ex09_real}/` sind GB-groß).
- Zeilenlänge 100 (ruff), `from __future__ import annotations` in jedem neuen Modul.
- Tier-Namen exakt: `"professional"`, `"enterprise"`; Entry-Point-Gruppe exakt `surroforge.authorizers`.
- Distributions- = Import-Name: `surroforge_professional`, `surroforge_enterprise` (Unterstriche, PEP 503 normalisiert).
- Der exakte String `LICENSE_ERROR` = `"Professional license required. Set SURROFORGE_PRO_LICENSE or install ~/.surroforge/professional.lic."` darf sich NICHT ändern (Test asserted `==`).
- Test-Umgebung: das Dev-Venv (`./.venv`) hat alle Plugins editable installiert; Tests laufen mit `python -m pytest` aus der Repo-Wurzel.

---

### Task 1: Core-Seam — `required_tier` in `PluginMetadata`

**Files:**
- Modify: `src/surroforge/plugins/seam.py` (Dataclass + neue Resolver-Funktion)
- Test: `tests/test_plugins.py` (Tests anhängen)

**Interfaces:**
- Produces: `PluginMetadata.required_tier: str | None = None` (neues frozen-Dataclass-Feld, letztes Feld, Default `None`); `resolve_required_tier(cls: object) -> str | None` — liest Klassen-Attribut `required_tier`, `None` wenn undeklariert, `PluginError` bei Nicht-String/Leerstring. Task 2 und 3 konsumieren beides.

- [ ] **Step 1: Failing Tests schreiben** — ans Ende von `tests/test_plugins.py` anhängen:

```python
# --- required_tier seam (Wave 1) ---------------------------------------------


def test_required_tier_defaults_to_none():
    from surroforge.plugins.seam import resolve_required_tier

    class FreeTrainer:
        name = "free"

    assert resolve_required_tier(FreeTrainer) is None


def test_required_tier_reads_class_attribute():
    from surroforge.plugins.seam import resolve_required_tier

    class ProTrainer:
        name = "pro"
        required_tier = "professional"

    assert resolve_required_tier(ProTrainer) == "professional"


def test_required_tier_rejects_non_string():
    from surroforge.plugins.seam import resolve_required_tier

    class Broken:
        name = "broken"
        required_tier = 1

    with pytest.raises(PluginError, match="required_tier"):
        resolve_required_tier(Broken)


def test_required_tier_rejects_empty_string():
    from surroforge.plugins.seam import resolve_required_tier

    class Broken:
        name = "broken"
        required_tier = ""

    with pytest.raises(PluginError, match="required_tier"):
        resolve_required_tier(Broken)


def test_plugin_metadata_has_required_tier_default_none():
    from surroforge.plugins.seam import PluginMetadata

    meta = PluginMetadata(
        name="x", group="surroforge.trainers", api_version=1,
        source="builtin", ref="m:C",
    )
    assert meta.required_tier is None
```

(`pytest` und `PluginError` sind in `tests/test_plugins.py` bereits importiert.)

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `python -m pytest tests/test_plugins.py -k required_tier -q`
Expected: FAIL / ERROR mit `ImportError: cannot import name 'resolve_required_tier'`

- [ ] **Step 3: Implementieren** — in `src/surroforge/plugins/seam.py`:

Dataclass erweitern (Feld ans Ende, Default `None`):

```python
@dataclass(frozen=True)
class PluginMetadata:
    """Description of a resolved plugin, passed to :func:`surroforge.plugins.gate.check`."""

    name: str
    group: str
    api_version: int
    source: str  # "builtin" | "entry_point"
    ref: str  # the "module:attr" reference the class was loaded from
    required_tier: str | None = None  # licence tier the plugin declares; None = free
```

Neue Funktion (unter `resolve_api_version`, gleicher Stil):

```python
def resolve_required_tier(cls: object) -> str | None:
    """The licence tier a plugin class declares, or ``None`` for a free plugin.

    Tier names are opaque data to Core — a commercial edition's authorizer
    (:mod:`surroforge.plugins.gate`) interprets them. Declaring anything but a
    non-empty string is a plugin bug and fails loudly. The field is additive and
    backwards-compatible (undeclared → free), so ``PLUGIN_API_VERSION`` stays 1.
    """
    tier = getattr(cls, "required_tier", None)
    if tier is None:
        return None
    if not isinstance(tier, str) or not tier:
        raise PluginError(
            f"Plugin {getattr(cls, 'name', cls)!r} declares an invalid "
            f"required_tier {tier!r}; it must be a non-empty string."
        )
    return tier
```

- [ ] **Step 4: Tests grün**

Run: `python -m pytest tests/test_plugins.py -q`
Expected: alle PASS (auch die Bestandstests — das neue Feld hat einen Default).

- [ ] **Step 5: Commit**

```bash
git add src/surroforge/plugins/seam.py tests/test_plugins.py
git commit -m "feat(plugins): add required_tier to the plugin seam (Wave 1)"
```

---

### Task 2: Core-Gate — Authorizer-Discovery + Tier-Prüfung + Hint

**Files:**
- Modify: `src/surroforge/plugins/gate.py` (kompletter Ersatz des No-op-Bodys)
- Test: `tests/test_gate.py` (neu)

**Interfaces:**
- Consumes: `PluginMetadata.required_tier` (Task 1).
- Produces: `gate.AUTHORIZER_GROUP = "surroforge.authorizers"`; `gate._load_authorizers() -> tuple[Callable[[str], bool], ...]` (functools.cache; Tests patchen dieses Attribut); `gate.check(metadata)` — erlaubt bei `required_tier is None`, sonst fragt Authorizer, `PluginError` mit angehängten `hint`-Attributen bei Ablehnung. WICHTIG: `gate.py` darf `registry.py` NICHT importieren (Zirkularität — registry importiert gate); es scannt Entry-Points selbst via `importlib.metadata`.

- [ ] **Step 1: Failing Tests schreiben** — `tests/test_gate.py` (neu):

```python
"""The tier-aware authorisation gate (Wave 1) — unit tests with fake authorizers."""

from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.plugins import gate
from surroforge.plugins.seam import PluginMetadata


def _meta(tier: str | None) -> PluginMetadata:
    return PluginMetadata(
        name="thing", group="surroforge.trainers", api_version=1,
        source="entry_point", ref="mod:Thing", required_tier=tier,
    )


def test_free_plugin_is_always_permitted(monkeypatch):
    monkeypatch.setattr(gate, "_load_authorizers", lambda: ())
    gate.check(_meta(None))  # must not raise


def test_tier_plugin_denied_without_any_authorizer(monkeypatch):
    monkeypatch.setattr(gate, "_load_authorizers", lambda: ())
    with pytest.raises(PluginError, match="no installed SurroForge edition"):
        gate.check(_meta("professional"))


def test_tier_plugin_permitted_by_granting_authorizer(monkeypatch):
    monkeypatch.setattr(
        gate, "_load_authorizers", lambda: (lambda tier: tier == "professional",)
    )
    gate.check(_meta("professional"))  # must not raise


def test_tier_plugin_denied_by_refusing_authorizer(monkeypatch):
    monkeypatch.setattr(gate, "_load_authorizers", lambda: (lambda tier: False,))
    with pytest.raises(PluginError, match="'enterprise' license"):
        gate.check(_meta("enterprise"))


def test_denying_authorizer_hint_is_appended(monkeypatch):
    def deny(tier: str) -> bool:
        return False

    deny.hint = "Set THE_LICENSE_VAR."
    monkeypatch.setattr(gate, "_load_authorizers", lambda: (deny,))
    with pytest.raises(PluginError, match="Set THE_LICENSE_VAR"):
        gate.check(_meta("professional"))


def test_error_names_plugin_and_tier(monkeypatch):
    monkeypatch.setattr(gate, "_load_authorizers", lambda: ())
    with pytest.raises(PluginError, match=r"'thing' requires a 'professional' license"):
        gate.check(_meta("professional"))
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `python -m pytest tests/test_gate.py -q`
Expected: FAILs — `gate.check` ist noch der No-op (kein Raise), `_load_authorizers` existiert nicht.

- [ ] **Step 3: Implementieren** — `src/surroforge/plugins/gate.py` komplett ersetzen durch:

```python
"""Plugin authorisation gate (Story 5.5 / AR13) — tier-aware since Wave 1.

Reserved call site, invoked by the registry immediately after a plugin is
resolved. It stays editions-neutral: Core never names an edition or an
activation mechanism. A plugin *declares* a required tier (opaque string data,
:func:`surroforge.plugins.seam.resolve_required_tier`); installed commercial
editions *register* authorizers under the ``surroforge.authorizers`` entry
point; this gate asks each authorizer whether the requirement is met. A
Core-only installation has no tier-declaring plugins, so the deny path never
fires there. NOTE: this module must not import the registry (the registry
imports this module), so it scans its entry-point group itself.
"""

from __future__ import annotations

import functools
import importlib.metadata as md
from collections.abc import Callable

from surroforge.errors import PluginError
from surroforge.plugins.seam import PluginMetadata

#: Entry-point group a commercial edition uses to register its authorizer:
#: a callable ``authorize(required_tier: str) -> bool``. It may carry a ``hint``
#: attribute (str) with an edition-specific activation hint that this gate
#: appends to its otherwise editions-neutral error message.
AUTHORIZER_GROUP = "surroforge.authorizers"


@functools.cache
def _load_authorizers() -> tuple[Callable[[str], bool], ...]:
    """All installed authorizers (memoised; installed dists cannot change at runtime)."""
    return tuple(ep.load() for ep in md.entry_points(group=AUTHORIZER_GROUP))


def check(metadata: PluginMetadata) -> None:
    """Authorise a resolved plugin: free plugins pass, tier-declaring ones are gated.

    The authorizer scan only runs for tier-declaring plugins, so the free
    (Core) resolution path pays no overhead.
    """
    if metadata.required_tier is None:
        return
    authorizers = _load_authorizers()
    if any(authorize(metadata.required_tier) for authorize in authorizers):
        return
    hints = " ".join(h for a in authorizers if (h := getattr(a, "hint", "")))
    message = (
        f"'{metadata.name}' requires a '{metadata.required_tier}' license, "
        "and no installed SurroForge edition grants it."
    )
    raise PluginError(f"{message} {hints}".rstrip())
```

- [ ] **Step 4: Tests grün**

Run: `python -m pytest tests/test_gate.py tests/test_plugins.py -q`
Expected: alle PASS. (Die Bestandssuite bleibt grün: kein heutiges Plugin deklariert `required_tier`, also verweigert `check` nirgends.)

- [ ] **Step 5: Commit**

```bash
git add src/surroforge/plugins/gate.py tests/test_gate.py
git commit -m "feat(plugins): tier-aware authorisation gate with authorizer entry points"
```

---

### Task 3: Registry — `required_tier` in die Metadata füllen

**Files:**
- Modify: `src/surroforge/plugins/registry.py` (Import + eine Zeile in `_resolve`)
- Test: `tests/test_plugins.py` (Integrationstest anhängen)

**Interfaces:**
- Consumes: `resolve_required_tier` (Task 1), `gate._load_authorizers`-Patchbarkeit (Task 2).
- Produces: `_resolve` konstruiert `PluginMetadata(..., required_tier=resolve_required_tier(cls))` — damit gated JEDE Auflösung (Trainer, Exporter, Reporter) deklarierende Plugins.

- [ ] **Step 1: Failing Test schreiben** — ans Ende von `tests/test_plugins.py`:

```python
def test_resolution_gates_tier_declaring_entry_point(monkeypatch):
    from surroforge.plugins import gate, registry

    class ProTrainer:
        name = "fake-pro"
        required_tier = "professional"

    class FakeEP:
        name = "fake-pro"
        value = "fake.module:ProTrainer"

        def load(self):
            return ProTrainer

    def fake_grouped(group):
        if group == registry.TRAINER_GROUP:
            return {"fake-pro": [FakeEP()]}
        return {}

    monkeypatch.setattr(registry, "_entry_points_grouped", fake_grouped)

    # Denied: no authorizer installed.
    monkeypatch.setattr(gate, "_load_authorizers", lambda: ())
    with pytest.raises(PluginError, match="'professional' license"):
        registry.get_trainer("fake-pro")

    # Granted: an authorizer approves the tier.
    monkeypatch.setattr(gate, "_load_authorizers", lambda: (lambda tier: True,))
    assert registry.get_trainer("fake-pro") is ProTrainer


def test_core_builtins_resolve_without_any_authorizer(monkeypatch):
    # Spec: a Core-only installation (no authorizer at all) must never be denied.
    from surroforge.plugins import gate, registry

    monkeypatch.setattr(gate, "_load_authorizers", lambda: ())
    # polynomial/reduced-field suffice as proof (kriging/rbf are equivalent but
    # import the optional SMT backend); free builtins carry required_tier=None.
    assert registry.get_trainer("polynomial") is not None
    assert registry.get_trainer("reduced-field") is not None
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `python -m pytest tests/test_plugins.py::test_resolution_gates_tier_declaring_entry_point -q`
Expected: FAIL — der erste `get_trainer`-Aufruf raised nicht (Metadata trägt noch `required_tier=None`-Default).

- [ ] **Step 3: Implementieren** — in `src/surroforge/plugins/registry.py`:

Import erweitern:

```python
from surroforge.plugins.seam import (
    PluginMetadata,
    check_api_version,
    resolve_api_version,
    resolve_required_tier,
)
```

In `_resolve` die Metadata-Konstruktion ergänzen:

```python
    metadata = PluginMetadata(
        name=name,
        group=group,
        api_version=resolve_api_version(cls),
        source=source,
        ref=ref,
        required_tier=resolve_required_tier(cls),
    )
```

- [ ] **Step 4: Tests grün**

Run: `python -m pytest tests/test_plugins.py tests/test_gate.py -q`
Expected: alle PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surroforge/plugins/registry.py tests/test_plugins.py
git commit -m "feat(plugins): resolve and enforce required_tier at plugin resolution"
```

---

### Task 4: Verzeichnis-Rename + Distributionsname `surroforge_professional`

**Files:**
- Rename: `plugins/surroforge-professional/` → `plugins/surroforge_professional/` (git mv)
- Modify: `plugins/surroforge_professional/pyproject.toml` (name, version, description)
- Modify: `plugins/surroforge_professional/src/surroforge_professional/__init__.py` (`__version__`)

**Interfaces:**
- Produces: den kanonischen Paketpfad `plugins/surroforge_professional/`, Distributionsname `surroforge_professional`, Version `0.2.0`. Alle Folgetasks nutzen diesen Pfad. Importname war schon `surroforge_professional` — Python-Importe ändern sich NICHT.

- [ ] **Step 1: Rename**

```bash
git mv plugins/surroforge-professional plugins/surroforge_professional
```

- [ ] **Step 2: pyproject anpassen** — in `plugins/surroforge_professional/pyproject.toml`:

```toml
# alt:
name = "surroforge-professional"
version = "0.1.0"
description = "Professional report formats for SurroForge."
# neu:
name = "surroforge_professional"
version = "0.2.0"
description = "Professional edition for SurroForge: licensed methods, HTML reports, DoE."
```

(Die `readme = { text = ... }`-Zeile analog auf den neuen description-Text setzen.)

In `plugins/surroforge_professional/src/surroforge_professional/__init__.py`: `__version__ = "0.1.0"` → `__version__ = "0.2.0"`.

- [ ] **Step 3: Editable-Install erneuern + Tests**

```bash
./.venv/bin/pip install -e plugins/surroforge_professional
python -m pytest plugins/surroforge_professional/tests -q
```

Expected: Install ersetzt die alte Distribution in place (PEP 503: `surroforge-professional` ≡ `surroforge_professional` — gleicher normalisierter Name, kein Uninstall nötig); alle Professional-Tests PASS.

- [ ] **Step 4: Commit**

```bash
git add -A plugins/surroforge-professional plugins/surroforge_professional
git commit -m "refactor(professional): rename distribution to surroforge_professional (dist = import name)"
```

(`git add -A` ist hier auf die zwei expliziten Pfade beschränkt — nötig, damit git die Löschseite des mv erfasst.)

---

### Task 5: Tier-aware Lizenz + Authorizer-Entry-Point

**Files:**
- Modify: `plugins/surroforge_professional/src/surroforge_professional/license.py` (kompletter Ersatz)
- Modify: `plugins/surroforge_professional/pyproject.toml` (Authorizer-Entry-Point)
- Test: `plugins/surroforge_professional/tests/test_license.py` (Tests anhängen)

**Interfaces:**
- Consumes: `gate.AUTHORIZER_GROUP`-Vertrag (Task 2): Callable `(str) -> bool` mit optionalem `hint`.
- Produces: `granted_tier() -> str | None`, `authorize(required_tier: str) -> bool` (mit `authorize.hint`), `has_license() -> bool`, `require_license(required_tier: str = "professional") -> None`, Konstanten `LICENSE_ENV`, `LICENSE_FILE`, `LICENSE_ERROR` (String unverändert!), `license_file()`. Bestehende Self-Gates (`reports/html.py`, `reports/comparison.py`, `commands/doe.py`) rufen `require_license()` ohne Argument — Signatur bleibt kompatibel.

- [ ] **Step 1: Failing Tests schreiben** — ans Ende von `plugins/surroforge_professional/tests/test_license.py`:

```python
# --- tier-aware authorizer (Wave 1) -------------------------------------------


def test_plain_license_grants_professional_only(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "local-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))
    from surroforge_professional.license import authorize, granted_tier

    assert granted_tier() == "professional"
    assert authorize("professional") is True
    assert authorize("enterprise") is False


def test_enterprise_license_grants_both_tiers(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "Enterprise-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))
    from surroforge_professional.license import authorize, granted_tier

    assert granted_tier() == "enterprise"
    assert authorize("professional") is True
    assert authorize("enterprise") is True


def test_no_license_grants_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from surroforge_professional.license import authorize, granted_tier

    assert granted_tier() is None
    assert authorize("professional") is False


def test_unknown_required_tier_is_denied(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "enterprise")
    monkeypatch.setenv("HOME", str(tmp_path))
    from surroforge_professional.license import authorize

    assert authorize("galactic") is False


def test_license_file_content_selects_tier(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from surroforge_professional.license import granted_tier, license_file

    path = license_file()
    path.parent.mkdir(parents=True)
    path.write_text("ENTERPRISE site license\n")
    assert granted_tier() == "enterprise"


def test_authorize_carries_activation_hint():
    from surroforge_professional.license import LICENSE_ERROR, authorize

    assert authorize.hint == LICENSE_ERROR


def test_require_license_enterprise_tier(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "local-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))
    from surroforge_professional.license import require_license

    require_license()  # professional default — passes
    with pytest.raises(PluginError):
        require_license("enterprise")


def test_authorizer_entry_point_is_registered():
    import importlib.metadata as md

    eps = {ep.name: ep.value for ep in md.entry_points(group="surroforge.authorizers")}
    assert eps.get("professional") == "surroforge_professional.license:authorize"
```

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `python -m pytest plugins/surroforge_professional/tests/test_license.py -q`
Expected: die neuen Tests FAILen (`granted_tier`/`authorize` existieren nicht; Entry-Point fehlt); die 5 Bestandstests PASSen weiter.

- [ ] **Step 3: Implementieren** — `license.py` komplett ersetzen durch:

```python
"""Local Professional/Enterprise license gate — tier-aware stub (Wave 1).

Real activation/validation is milestone M3. Until then the *presence* of a
license (env var or file) grants a tier and its *content* selects which one:
a value containing "enterprise" (case-insensitive) grants the enterprise tier,
any other non-empty value grants professional. ``authorize`` is registered on
Core's ``surroforge.authorizers`` entry point; ``has_license``/``require_license``
remain for in-package self-gates (reports, DoE) and keep their old behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path

from surroforge.errors import PluginError

LICENSE_ENV = "SURROFORGE_PRO_LICENSE"
LICENSE_FILE = Path("~/.surroforge/professional.lic")
LICENSE_ERROR = (
    "Professional license required. Set SURROFORGE_PRO_LICENSE "
    "or install ~/.surroforge/professional.lic."
)

_TIER_ORDER = {"professional": 1, "enterprise": 2}


def license_file() -> Path:
    return LICENSE_FILE.expanduser()


def _license_value() -> str:
    env_value = os.environ.get(LICENSE_ENV, "").strip()
    if env_value:
        return env_value
    path = license_file()
    if path.is_file():
        return path.read_text().strip()
    return ""


def granted_tier() -> str | None:
    """The tier the local license grants: 'enterprise', 'professional', or None."""
    value = _license_value()
    if not value:
        return None
    return "enterprise" if "enterprise" in value.lower() else "professional"


def authorize(required_tier: str) -> bool:
    """Entry-point authorizer for Core's gate: does the local license meet the tier?

    Unknown requirements are denied (a newer plugin against an older edition
    must not be silently unlocked).
    """
    granted = granted_tier()
    if granted is None:
        return False
    return _TIER_ORDER[granted] >= _TIER_ORDER.get(required_tier, float("inf"))


#: Activation hint Core's editions-neutral gate appends to its error message.
authorize.hint = LICENSE_ERROR  # type: ignore[attr-defined]


def has_license() -> bool:
    return granted_tier() is not None


def require_license(required_tier: str = "professional") -> None:
    if not authorize(required_tier):
        raise PluginError(LICENSE_ERROR)
```

In `plugins/surroforge_professional/pyproject.toml` neuen Entry-Point-Block ergänzen (neben den bestehenden):

```toml
[project.entry-points."surroforge.authorizers"]
professional = "surroforge_professional.license:authorize"
```

- [ ] **Step 4: Reinstall (Entry-Point!) + Tests grün**

```bash
./.venv/bin/pip install -e plugins/surroforge_professional
python -m pytest plugins/surroforge_professional/tests/test_license.py tests/test_gate.py -q
```

Expected: alle PASS (Entry-Point-Registrierung braucht den Reinstall).

- [ ] **Step 5: Commit**

```bash
git add plugins/surroforge_professional/src/surroforge_professional/license.py \
        plugins/surroforge_professional/pyproject.toml \
        plugins/surroforge_professional/tests/test_license.py
git commit -m "feat(professional): tier-aware licence stub + surroforge.authorizers entry point"
```

---

### Task 6: mlp/random-forest/gradient-boosting → Professional

**Files:**
- Create: `plugins/surroforge_professional/src/surroforge_professional/trainers/__init__.py`
- Rename (git mv): `src/surroforge/trainers/{mlp,random_forest,gradient_boosting}.py` → `plugins/surroforge_professional/src/surroforge_professional/trainers/`
- Modify: die drei verschobenen Module (`required_tier`-Attribut, mlp-Docstring), `src/surroforge/plugins/registry.py` (`_BUILTIN`), `plugins/surroforge_professional/pyproject.toml` (Entry-Points, scikit-learn-Dep)
- Test: `plugins/surroforge_professional/tests/test_gating.py` (neu), `tests/test_plugins.py` (Builtin-Assertion)

**Interfaces:**
- Consumes: Seam/Gate/Registry (Tasks 1–3), Authorizer (Task 5).
- Produces: `surroforge_professional.trainers.mlp:MLPTrainer`, `...random_forest:RandomForestTrainer`, `...gradient_boosting:GradientBoostingTrainer` — jeweils mit `required_tier = "professional"`; Core-`_BUILTIN[TRAINER_GROUP]` enthält nur noch polynomial/kriging/rbf/reduced-field. Task 7 (Test-Migration) und Task 10 (Beispiele) hängen daran.

- [ ] **Step 1: Failing Gating-Tests schreiben** — `plugins/surroforge_professional/tests/test_gating.py` (neu):

```python
"""Resolution-level gating of the Professional trainers (Wave 1)."""

from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.plugins import available_trainers, get_trainer


@pytest.fixture
def unlicensed(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.mark.parametrize("method", ["mlp", "random-forest", "gradient-boosting"])
def test_pro_method_denied_without_license(unlicensed, method):
    with pytest.raises(PluginError) as err:
        get_trainer(method)
    assert "'professional' license" in str(err.value)
    assert "SURROFORGE_PRO_LICENSE" in str(err.value)  # hint from the authorizer


@pytest.mark.parametrize("method", ["mlp", "random-forest", "gradient-boosting"])
def test_pro_method_resolves_with_license(licensed, method):
    cls = get_trainer(method)
    assert cls.name == method
    assert cls.required_tier == "professional"


def test_pro_methods_listed_via_entry_points(licensed):
    assert {"mlp", "random-forest", "gradient-boosting"} <= set(available_trainers())
```

(`licensed` kommt aus der bestehenden `conftest.py`.)

- [ ] **Step 2: Fehlschlag verifizieren**

Run: `python -m pytest plugins/surroforge_professional/tests/test_gating.py -q`
Expected: FAILs — die Methoden lösen noch als freie Core-Builtins auf (kein Raise, kein `required_tier`).

- [ ] **Step 3: Module verschieben + Paket anlegen**

```bash
mkdir -p plugins/surroforge_professional/src/surroforge_professional/trainers
git mv src/surroforge/trainers/mlp.py \
       plugins/surroforge_professional/src/surroforge_professional/trainers/mlp.py
git mv src/surroforge/trainers/random_forest.py \
       plugins/surroforge_professional/src/surroforge_professional/trainers/random_forest.py
git mv src/surroforge/trainers/gradient_boosting.py \
       plugins/surroforge_professional/src/surroforge_professional/trainers/gradient_boosting.py
```

`plugins/surroforge_professional/src/surroforge_professional/trainers/__init__.py` (neu):

```python
"""Professional surrogate methods (licensed): mlp, random-forest, gradient-boosting, torch-mlp."""
```

(Die verschobenen Module importieren weiter `from surroforge.trainers.base import BaseTrainer` und `from surroforge.errors import TrainerError` — beide bleiben Core, keine Import-Änderung nötig.)

- [ ] **Step 4: `required_tier` deklarieren** — in jedem der drei Module direkt unter dem `name`-Klassenattribut:

`trainers/mlp.py` (Zeile ~65):
```python
    name = "mlp"
    required_tier = "professional"
```
`trainers/random_forest.py` (Zeile ~64):
```python
    name = "random-forest"
    required_tier = "professional"
```
`trainers/gradient_boosting.py` (Zeile ~53):
```python
    name = "gradient-boosting"
    required_tier = "professional"
```

Im mlp-Docstring die veraltete Core-Prämisse anpassen — alt (Zeilen 4–6):
```
``sklearn.neural_network.MLPRegressor`` rather than a deep-learning framework, so
it adds **no new dependency** (scikit-learn is already Core, BSD-licensed → NFR8),
plugs into the project's sklearn-style integer ``random_state`` reproducibility
```
neu:
```
``sklearn.neural_network.MLPRegressor`` rather than a deep-learning framework, so
it needs no deep-learning stack (scikit-learn only; a Professional method since
the Wave-1 editions split), plugs into the project's sklearn-style integer
``random_state`` reproducibility
```

- [ ] **Step 5: Core-`_BUILTIN` kürzen** — in `src/surroforge/plugins/registry.py`:

```python
_BUILTIN: dict[str, dict[str, str]] = {
    TRAINER_GROUP: {
        "polynomial": "surroforge.trainers.polynomial:PolynomialTrainer",
        "kriging": "surroforge.trainers.kriging:KrigingTrainer",
        "rbf": "surroforge.trainers.rbf:RBFTrainer",
        "reduced-field": "surroforge.trainers.reduced_field:ReducedFieldTrainer",
    },
```

(Exporter/Reporter-Teile unverändert.)

- [ ] **Step 6: Entry-Points + Dependency** — in `plugins/surroforge_professional/pyproject.toml`:

`dependencies`-Liste um `"scikit-learn>=1.3",` ergänzen. Neuer Block:

```toml
[project.entry-points."surroforge.trainers"]
mlp = "surroforge_professional.trainers.mlp:MLPTrainer"
random-forest = "surroforge_professional.trainers.random_forest:RandomForestTrainer"
gradient-boosting = "surroforge_professional.trainers.gradient_boosting:GradientBoostingTrainer"
```

Reinstall: `./.venv/bin/pip install -e plugins/surroforge_professional`

- [ ] **Step 7: Tests grün + Core-Suite prüfen**

Run: `python -m pytest plugins/surroforge_professional/tests/test_gating.py -q`
Expected: PASS.

Run: `python -m pytest tests/ -q`
Expected: FAILs NUR in `tests/test_conformance.py` (ImportError der drei verschobenen Module), `tests/test_mlp.py`, `tests/test_random_forest.py`, `tests/test_gradient_boosting.py` — genau die Dateien, die Task 7 migriert. Bricht anderes, STOPP und analysieren (nicht raten).

Grün-Check des Zwischenzustands (alles außer den vier bekannten Dateien MUSS grün sein):

```bash
python -m pytest tests/ -q \
  --ignore=tests/test_conformance.py --ignore=tests/test_mlp.py \
  --ignore=tests/test_random_forest.py --ignore=tests/test_gradient_boosting.py
```
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/surroforge/trainers src/surroforge/plugins/registry.py \
        plugins/surroforge_professional/src/surroforge_professional/trainers \
        plugins/surroforge_professional/pyproject.toml \
        plugins/surroforge_professional/tests/test_gating.py
git commit -m "feat(editions)!: move mlp/random-forest/gradient-boosting to surroforge_professional"
```

---

### Task 7: Test-Migration (drei Trainer-Suiten + Konformanz)

**Files:**
- Rename (git mv): `tests/{test_mlp,test_random_forest,test_gradient_boosting}.py` → `plugins/surroforge_professional/tests/`
- Modify: die drei migrierten Dateien (Imports, `pytestmark`), `tests/test_conformance.py` (Imports + Params kürzen)
- Create: `plugins/surroforge_professional/tests/test_conformance_professional.py`

**Interfaces:**
- Consumes: verschobene Trainer-Module (Task 6), `licensed`-Fixture (Bestand), `check_trainer`-API (`report.passed`, `report.summary()`, `report.raise_for_status()`).
- Produces: grüne Gesamt-Suite: `pytest tests/` und `pytest plugins/surroforge_professional/tests` beide PASS.

- [ ] **Step 1: Dateien verschieben**

```bash
git mv tests/test_mlp.py plugins/surroforge_professional/tests/test_mlp.py
git mv tests/test_random_forest.py plugins/surroforge_professional/tests/test_random_forest.py
git mv tests/test_gradient_boosting.py plugins/surroforge_professional/tests/test_gradient_boosting.py
```

- [ ] **Step 2: Imports + Lizenz-Markierung anpassen** — in jeder der drei Dateien:

(a) Import des Trainer-Moduls umstellen:
```python
# alt:                                             # neu:
from surroforge.trainers.mlp import MLPTrainer     from surroforge_professional.trainers.mlp import MLPTrainer
from surroforge.trainers.random_forest import ...  from surroforge_professional.trainers.random_forest import ...
from surroforge.trainers.gradient_boosting import ...  from surroforge_professional.trainers.gradient_boosting import ...
```
(b) Direkt unter den Imports einfügen (Registry-/Engine-Pfade in diesen Suiten brauchen die Lizenz; Direkt-Konstruktion stört sie nicht):
```python
pytestmark = pytest.mark.usefixtures("licensed")
```
(c) Falls eine Datei `from tests.conftest import ...` importiert (prüfen mit `rg -n "tests.conftest" <datei>`): das Importierte in die Datei inline kopieren oder auf `plugins/surroforge_professional/tests/conftest.py` übertragen — NICHT quer auf `tests/` referenzieren.

- [ ] **Step 3: Core-Konformanz kürzen** — in `tests/test_conformance.py`:

Imports entfernen:
```python
from surroforge.trainers.gradient_boosting import GradientBoostingTrainer
from surroforge.trainers.mlp import MLPTrainer
from surroforge.trainers.random_forest import RandomForestTrainer
```
`BUILTINS`-Liste ersetzen durch:
```python
BUILTINS = [
    pytest.param(PolynomialTrainer, id="polynomial"),
    pytest.param(KrigingTrainer, id="kriging", marks=_needs_smt),
    pytest.param(RBFTrainer, id="rbf", marks=_needs_smt),
    # Default inner is polynomial, so reduced-field is conformant without SMT.
    pytest.param(ReducedFieldTrainer, id="reduced-field"),
]
```

- [ ] **Step 4: Professional-Konformanz anlegen** — `plugins/surroforge_professional/tests/test_conformance_professional.py` (neu):

```python
"""The shipped conformance suite, run against the Professional sklearn trainers.

torch-mlp's conformance run lives in test_torch_mlp.py (needs the [torch] extra).
"""

from __future__ import annotations

import pytest

from surroforge.conformance import check_trainer
from surroforge_professional.trainers.gradient_boosting import GradientBoostingTrainer
from surroforge_professional.trainers.mlp import MLPTrainer
from surroforge_professional.trainers.random_forest import RandomForestTrainer

PRO_TRAINERS = [
    pytest.param(MLPTrainer, id="mlp"),
    pytest.param(GradientBoostingTrainer, id="gradient-boosting"),
    pytest.param(RandomForestTrainer, id="random-forest"),
]


@pytest.mark.parametrize("trainer_cls", PRO_TRAINERS)
def test_professional_trainer_is_conformant(trainer_cls):
    report = check_trainer(trainer_cls)
    assert report.passed, report.summary()
    report.raise_for_status()  # must not raise
```

- [ ] **Step 5: Beide Suiten grün**

Run: `python -m pytest tests/ -q && python -m pytest plugins/surroforge_professional/tests -q`
Expected: beide PASS, keine Skips durch fehlende Module.

- [ ] **Step 6: Commit**

```bash
git add tests/test_conformance.py plugins/surroforge_professional/tests \
        tests/test_mlp.py tests/test_random_forest.py tests/test_gradient_boosting.py
git commit -m "test(editions): migrate mlp/rf/gb suites to professional; split conformance runs"
```

(Die drei `tests/test_*.py`-Pfade sind die Löschseite des git mv.)

---

### Task 8: torch-mlp + pt-Exporter → Professional; surroforge-torch auflösen

**Files:**
- Rename (git mv): `plugins/surroforge-torch/src/surroforge_torch/trainer.py` → `plugins/surroforge_professional/src/surroforge_professional/trainers/_torch_impl.py`; `.../pt_exporter.py` → `plugins/surroforge_professional/src/surroforge_professional/pt_exporter.py`; `plugins/surroforge-torch/tests/test_torch_mlp.py` → `plugins/surroforge_professional/tests/test_torch_mlp.py`
- Create: `plugins/surroforge_professional/src/surroforge_professional/trainers/torch_mlp.py` (Guard)
- Modify: `_torch_impl.py` (required_tier + Docstring), `pyproject.toml` ([torch]-Extra + 2 Entry-Points), migrierte Testdatei
- Delete: `plugins/surroforge-torch/` (Rest: pyproject, README)

**Interfaces:**
- Consumes: Seam/Gate (Tasks 1–3), Authorizer (Task 5).
- Produces: `surroforge_professional.trainers.torch_mlp:TorchMLPTrainer` (Guard-Modul; ohne torch eine Stub-Klasse, deren `__init__` `PluginError` wirft), `surroforge_professional.pt_exporter:PtExporter`; Extra `surroforge_professional[torch]`.

- [ ] **Step 1: Verschieben**

```bash
git mv plugins/surroforge-torch/src/surroforge_torch/trainer.py \
       plugins/surroforge_professional/src/surroforge_professional/trainers/_torch_impl.py
git mv plugins/surroforge-torch/src/surroforge_torch/pt_exporter.py \
       plugins/surroforge_professional/src/surroforge_professional/pt_exporter.py
git mv plugins/surroforge-torch/tests/test_torch_mlp.py \
       plugins/surroforge_professional/tests/test_torch_mlp.py
```

- [ ] **Step 2: Impl anpassen** — in `_torch_impl.py`:

(a) Unter `name = "torch-mlp"` (Zeile ~120):
```python
    name = "torch-mlp"
    required_tier = "professional"
```
(b) Docstring-Anfang aktualisieren — alt (Zeilen 3–9):
```
This is the first heavyweight-framework method and the first real exercise of
SurroForge's external plugin seam: it ships as a *separate distribution*
(``surroforge-torch``) that registers ``torch-mlp`` through the ``surroforge.trainers``
entry point only — never Core's ``_BUILTIN`` table. Core stays torch-free; installing
this package makes ``model.method: torch-mlp`` resolvable. Switching backends is a
one-line config edit: ``torch-mlp`` mirrors the built-in ``mlp``'s exact hyperparameter
surface, so the same YAML trains either scikit-learn or PyTorch.
```
neu:
```
This is the heavyweight-framework counterpart of ``mlp``. Since the Wave-1
editions split it ships inside ``surroforge_professional`` behind the optional
``[torch]`` extra and registers ``torch-mlp`` through the ``surroforge.trainers``
entry point (via the guard module ``trainers/torch_mlp.py``, resolvable without
torch installed) — never Core's ``_BUILTIN`` table. Core stays torch-free.
Switching backends is a one-line config edit: ``torch-mlp`` mirrors ``mlp``'s
exact hyperparameter surface, so the same YAML trains either scikit-learn or
PyTorch.
```

- [ ] **Step 3: Guard-Modul** — `plugins/surroforge_professional/src/surroforge_professional/trainers/torch_mlp.py` (neu):

```python
"""torch-mlp guard module — the entry-point target, resolvable without torch.

The real implementation (``_torch_impl``) subclasses ``nn.Module`` at module
level, so importing it requires torch. This guard lets ``ep.load()`` (and the
licence gate, which runs at resolution) succeed without the ``[torch]`` extra;
constructing the stub then explains exactly what to install.
"""

from __future__ import annotations

from typing import Any

try:
    from surroforge_professional.trainers._torch_impl import TorchMLPTrainer
except ModuleNotFoundError as exc:  # torch (or a torch submodule) missing
    if exc.name is None or not exc.name.split(".")[0] == "torch":
        raise  # a real bug in the impl module must not be masked

    from surroforge.errors import PluginError

    class TorchMLPTrainer:  # type: ignore[no-redef]
        """Stub: keeps resolution/gating working; construction explains the extra."""

        name = "torch-mlp"
        required_tier = "professional"
        export_formats: frozenset[str] = frozenset()

        def __init__(self, **_: Any) -> None:
            raise PluginError(
                "torch-mlp requires the torch extra: "
                "pip install 'surroforge_professional[torch]'"
            )


__all__ = ["TorchMLPTrainer"]
```

- [ ] **Step 4: pyproject** — in `plugins/surroforge_professional/pyproject.toml`:

`[project.optional-dependencies]` um `torch = ["torch>=2.0"]` ergänzen (neben `dev`/`doe`); den Trainer-Entry-Point-Block erweitern und den Exporter-Block neu anlegen:

```toml
[project.entry-points."surroforge.trainers"]
mlp = "surroforge_professional.trainers.mlp:MLPTrainer"
random-forest = "surroforge_professional.trainers.random_forest:RandomForestTrainer"
gradient-boosting = "surroforge_professional.trainers.gradient_boosting:GradientBoostingTrainer"
torch-mlp = "surroforge_professional.trainers.torch_mlp:TorchMLPTrainer"

[project.entry-points."surroforge.exporters"]
pt = "surroforge_professional.pt_exporter:PtExporter"
```

- [ ] **Step 5: Migrierte Tests anpassen** — in `plugins/surroforge_professional/tests/test_torch_mlp.py`:

(a) `from surroforge_torch.trainer import TorchMLPTrainer` → `from surroforge_professional.trainers.torch_mlp import TorchMLPTrainer`. Weitere `surroforge_torch`-Referenzen finden: `rg -n "surroforge_torch" plugins/surroforge_professional/tests/test_torch_mlp.py` — jede auf den neuen Modulpfad umstellen (z. B. `monkeypatch.setattr`-Ziele auf `surroforge_professional.trainers._torch_impl`).
(b) Unter den Imports: `pytestmark = pytest.mark.usefixtures("licensed")`.
(c) Neuer Test ans Ende (Guard liefert bei installiertem torch die echte Klasse):

```python
def test_guard_module_exports_real_trainer_when_torch_installed():
    from surroforge_professional.trainers._torch_impl import TorchMLPTrainer as Impl
    from surroforge_professional.trainers.torch_mlp import TorchMLPTrainer as Guarded

    assert Guarded is Impl
    assert Guarded.required_tier == "professional"
```

(Der torch-lose Stub-Zweig ist im Dev-Venv nicht ausführbar — torch ist installiert; er wird durch Code-Review + den obigen `exc.name`-Filter abgesichert.)

- [ ] **Step 6: Altes Paket entfernen + Installs umstellen**

```bash
git rm -r plugins/surroforge-torch
./.venv/bin/pip uninstall -y surroforge-torch
./.venv/bin/pip install -e 'plugins/surroforge_professional[torch]'
```

- [ ] **Step 7: Tests grün**

Run: `python -m pytest plugins/surroforge_professional/tests -q && python -m pytest tests/ -q`
Expected: beide PASS; `torch-mlp` löst mit `licensed` auf, ohne Lizenz `PluginError`.

Zusatz-Verifikation (Shell):
```bash
SURROFORGE_PRO_LICENSE=dev ./.venv/bin/python -c \
  "from surroforge.plugins import get_trainer; print(get_trainer('torch-mlp').name)"
```
Expected: `torch-mlp`

- [ ] **Step 8: Commit**

```bash
git add -A plugins/surroforge-torch plugins/surroforge_professional
git commit -m "feat(editions)!: absorb torch-mlp + pt exporter into surroforge_professional ([torch] extra)"
```

---

### Task 9: `surroforge_enterprise` anlegen; autoencoder-field → Enterprise

**Files:**
- Create: `plugins/surroforge_enterprise/pyproject.toml`, `.../src/surroforge_enterprise/__init__.py`, `.../trainers/__init__.py`, `.../trainers/autoencoder_field.py` (Guard), `plugins/surroforge_enterprise/tests/conftest.py`
- Rename (git mv): `plugins/surroforge-autoencoder/src/surroforge_autoencoder/trainer.py` → `plugins/surroforge_enterprise/src/surroforge_enterprise/trainers/_autoencoder_impl.py`; `plugins/surroforge-autoencoder/tests/test_autoencoder_field.py` → `plugins/surroforge_enterprise/tests/test_autoencoder_field.py`
- Modify: `_autoencoder_impl.py` (required_tier), migrierte Testdatei
- Delete: `plugins/surroforge-autoencoder/`
- Test: `plugins/surroforge_enterprise/tests/test_enterprise_gating.py` (neu)

**Interfaces:**
- Consumes: Seam/Gate (Tasks 1–3), Authorizer aus professional (Task 5 — enterprise registriert KEINEN eigenen).
- Produces: Distribution `surroforge_enterprise` (Proprietary, dependency auf `surroforge_professional`), Entry-Point `autoencoder-field = "surroforge_enterprise.trainers.autoencoder_field:AutoencoderFieldTrainer"` mit `required_tier = "enterprise"`.

- [ ] **Step 1: Failing Tier-Tests schreiben** — `plugins/surroforge_enterprise/tests/test_enterprise_gating.py` (neu; Datei zuerst, Paket kommt in Step 2):

```python
"""autoencoder-field is an *enterprise*-tier method: a professional licence is not enough."""

from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.plugins import get_trainer


def test_professional_license_does_not_unlock_autoencoder(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "local-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(PluginError, match="'enterprise' license"):
        get_trainer("autoencoder-field")


def test_enterprise_license_unlocks_autoencoder(licensed_enterprise):
    cls = get_trainer("autoencoder-field")
    assert cls.name == "autoencoder-field"
    assert cls.required_tier == "enterprise"


def test_no_license_denies_autoencoder(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(PluginError, match="'enterprise' license"):
        get_trainer("autoencoder-field")
```

`plugins/surroforge_enterprise/tests/conftest.py` (neu):

```python
from __future__ import annotations

import pytest


@pytest.fixture
def licensed_enterprise(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "enterprise-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))
```

- [ ] **Step 2: Paket-Skelett**

`plugins/surroforge_enterprise/pyproject.toml` (neu):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "surroforge_enterprise"
version = "0.1.0"
description = "Enterprise edition for SurroForge: autoencoder field surrogate (more to come in Phase 3)."
readme = { text = "Enterprise edition for SurroForge.", content-type = "text/markdown" }
requires-python = ">=3.10"
license = { text = "Proprietary" }
authors = [{ name = "SurroForge" }]
keywords = ["surrogate", "surroforge", "autoencoder", "field-rom", "enterprise"]
dependencies = [
    "surroforge",
    "surroforge_professional",
    "numpy>=1.24",
]

[project.optional-dependencies]
torch = ["torch>=2.0"]
dev = ["pytest>=8.0", "ruff>=0.4", "mypy>=1.9"]

[project.entry-points."surroforge.trainers"]
autoencoder-field = "surroforge_enterprise.trainers.autoencoder_field:AutoencoderFieldTrainer"

[tool.hatch.build.targets.wheel]
packages = ["src/surroforge_enterprise"]

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
```

`src/surroforge_enterprise/__init__.py`:
```python
"""SurroForge Enterprise plugin."""

from __future__ import annotations

__version__ = "0.1.0"
```

`src/surroforge_enterprise/trainers/__init__.py`:
```python
"""Enterprise surrogate methods (licensed, enterprise tier): autoencoder-field."""
```

- [ ] **Step 3: Impl verschieben + Guard**

```bash
mkdir -p plugins/surroforge_enterprise/src/surroforge_enterprise/trainers
git mv plugins/surroforge-autoencoder/src/surroforge_autoencoder/trainer.py \
       plugins/surroforge_enterprise/src/surroforge_enterprise/trainers/_autoencoder_impl.py
git mv plugins/surroforge-autoencoder/tests/test_autoencoder_field.py \
       plugins/surroforge_enterprise/tests/test_autoencoder_field.py
```

In `_autoencoder_impl.py` unter `name = "autoencoder-field"` (Zeile ~102):
```python
    name = "autoencoder-field"
    required_tier = "enterprise"
```

`trainers/autoencoder_field.py` (Guard, neu — Muster identisch zu Task 8 Step 3):

```python
"""autoencoder-field guard module — the entry-point target, resolvable without torch."""

from __future__ import annotations

from typing import Any

try:
    from surroforge_enterprise.trainers._autoencoder_impl import AutoencoderFieldTrainer
except ModuleNotFoundError as exc:  # torch (or a torch submodule) missing
    if exc.name is None or not exc.name.split(".")[0] == "torch":
        raise  # a real bug in the impl module must not be masked

    from surroforge.errors import PluginError

    class AutoencoderFieldTrainer:  # type: ignore[no-redef]
        """Stub: keeps resolution/gating working; construction explains the extra."""

        name = "autoencoder-field"
        required_tier = "enterprise"
        export_formats: frozenset[str] = frozenset()

        def __init__(self, **_: Any) -> None:
            raise PluginError(
                "autoencoder-field requires the torch extra: "
                "pip install 'surroforge_enterprise[torch]'"
            )


__all__ = ["AutoencoderFieldTrainer"]
```

- [ ] **Step 4: Migrierte Tests anpassen** — in `plugins/surroforge_enterprise/tests/test_autoencoder_field.py`:

(a) `rg -n "surroforge_autoencoder" plugins/surroforge_enterprise/tests/test_autoencoder_field.py` — jede Referenz auf `surroforge_enterprise.trainers.autoencoder_field` (Klassenimport) bzw. `..._autoencoder_impl` (monkeypatch-Ziele) umstellen.
(b) Der pyproject-Test (`test_pyproject_declares_autoencoder_field_entry_point`, liest `Path("plugins/surroforge-autoencoder/pyproject.toml")` cwd-relativ): Pfad auf `plugins/surroforge_enterprise/pyproject.toml` ändern; erwarteten Entry-Point-Wert auf `"surroforge_enterprise.trainers.autoencoder_field:AutoencoderFieldTrainer"` aktualisieren; falls er das `license`-Feld asserted, neu `Proprietary` erwarten.
(c) Falls die Datei `get_trainer`/`available_trainers` nutzt (`rg -n "get_trainer|available_trainers" <datei>`): unter den Imports `pytestmark = pytest.mark.usefixtures("licensed_enterprise")` einfügen.

- [ ] **Step 5: Altes Paket entfernen + Installs**

```bash
git rm -r plugins/surroforge-autoencoder
./.venv/bin/pip uninstall -y surroforge-autoencoder
./.venv/bin/pip install -e 'plugins/surroforge_enterprise[torch]'
```

- [ ] **Step 6: Tests grün**

Run: `python -m pytest plugins/surroforge_enterprise/tests -q && python -m pytest tests/ plugins/surroforge_professional/tests -q`
Expected: alle PASS — insbesondere: Pro-Lizenz schaltet autoencoder-field NICHT frei, Enterprise-Lizenz schon.

- [ ] **Step 7: Commit**

```bash
git add -A plugins/surroforge-autoencoder plugins/surroforge_enterprise
git commit -m "feat(editions)!: new surroforge_enterprise package; autoencoder-field is enterprise-tier"
```

---

### Task 10: Beispiele 06–09 + Beispiel-Doku

**Files:**
- Modify: `examples/06_mlp_surrogate.py`, `examples/07_pod_nn_field.py`, `examples/08_torch_mlp.py`, `examples/09_naca0012_openfoam/ingest_and_train.py`, `examples/09_naca0012_openfoam/README.md`, `docs/examples.md`, `examples/README.md`

**Interfaces:**
- Consumes: neue Paket-/Extra-Namen (Tasks 6–9). Reine Text-/Docstring-Änderungen — keine Logik.

- [ ] **Step 1: Beispiel 06** — Docstring-Prämisse ersetzen. Alt:

```
The `mlp` method is a multi-layer perceptron (scikit-learn's `MLPRegressor`, with
an input `StandardScaler` and the small-data-friendly `lbfgs` solver). It is a
genuine neural network but rides Core scikit-learn, so it needs no deep-learning
framework and no extra backend.
```
Neu:
```
The `mlp` method is a multi-layer perceptron (scikit-learn's `MLPRegressor`, with
an input `StandardScaler` and the small-data-friendly `lbfgs` solver). It is a
genuine neural network backed by scikit-learn — no deep-learning framework needed.
Since the Wave-1 editions split, `mlp` is a Professional method: install the
`surroforge_professional` plugin and set a license, e.g. `SURROFORGE_PRO_LICENSE=dev`.
```

- [ ] **Step 2: Beispiel 07** — ans ENDE des Modul-Docstrings (vor dem schließenden `"""`) anhängen:

```
Note (editions): the `mlp` inner is a Professional method — install
`surroforge_professional` and set `SURROFORGE_PRO_LICENSE`; the polynomial
inner runs on Core alone.
```

- [ ] **Step 3: Beispiel 08** — Docstring-Absatz ersetzen. Alt:

```
SurroForge ships `mlp` (scikit-learn's `MLPRegressor`) in Core. The optional
`surroforge-torch` plugin adds `torch-mlp`, a PyTorch MLP that registers on the
same `surroforge.trainers` entry point.
```
Neu:
```
Both halves are Professional methods since the Wave-1 editions split: `mlp`
(scikit-learn) and `torch-mlp` (PyTorch) ship in the `surroforge_professional`
plugin — the PyTorch half additionally needs the `[torch]` extra
(`pip install 'surroforge_professional[torch]'`) — and require a license
(e.g. `SURROFORGE_PRO_LICENSE=dev`). `torch-mlp` registers on the same
`surroforge.trainers` entry point.
```
Danach: `rg -n "surroforge-torch" examples/08_torch_mlp.py` — verbleibende Nennungen auf `surroforge_professional[torch]` umformulieren.

- [ ] **Step 4: Beispiel 09 (NACA)** — `rg -n "surroforge-torch|surroforge_torch" examples/09_naca0012_openfoam/` und Treffer umstellen. In `ingest_and_train.py` ans Ende des Modul-Docstrings anhängen:

```
Note (editions): the neural-network candidates (`mlp`, `torch-mlp`) are
Professional methods — `pip install 'surroforge_professional[torch]'` and set
`SURROFORGE_PRO_LICENSE`. The polynomial candidates run on Core alone.
```
In `examples/09_naca0012_openfoam/README.md` einen gleichlautenden Hinweis-Absatz in den Setup-/Requirements-Abschnitt einfügen (Abschnitt per `rg -n "install|Setup|Voraussetzung" examples/09_naca0012_openfoam/README.md` lokalisieren).

- [ ] **Step 5: Tabellenzeilen** — in `docs/examples.md` UND `examples/README.md` die drei Zeilen ersetzen (Spalte 3, „braucht"):

| Beispiel | neue Spalte 3 |
|---|---|
| 06 | `Professional license (`mlp`); ONNX export adds `[onnx]`` |
| 07 | `Core (polynomial inner); the `mlp` inner needs a Professional license` |
| 08 | `Professional license; the PyTorch half needs `surroforge_professional[torch]`` |

(Zeilen exakt lokalisieren: `rg -n "06_mlp_surrogate|07_pod_nn_field|08_torch_mlp" docs/examples.md examples/README.md`; nur Spalte 3 ändern, Beschreibungs-Spalte von 08 zusätzlich: `the \`surroforge-torch\` plugin` → `the \`[torch]\` extra`.)

- [ ] **Step 6: Verifizieren + Commit**

```bash
rg -n "surroforge-torch|surroforge_torch|surroforge-autoencoder|surroforge_autoencoder" \
   examples/ docs/examples.md examples/README.md ; echo "expect: no hits"
SURROFORGE_PRO_LICENSE=dev ./.venv/bin/python examples/06_mlp_surrogate.py > /dev/null && echo OK
git add examples/06_mlp_surrogate.py examples/07_pod_nn_field.py examples/08_torch_mlp.py \
        examples/09_naca0012_openfoam/ingest_and_train.py examples/09_naca0012_openfoam/README.md \
        docs/examples.md examples/README.md
git commit -m "docs(examples): editions notes for 06-09; torch plugin is now professional[torch]"
```

Expected: der rg findet keine Treffer mehr (außer ggf. historischen superpowers-Docs — die sind tabu und werden NICHT angefasst); Beispiel 06 läuft mit Lizenz durch.

---

### Task 11: PRD §3.2/§4.8/§6.2/§9.2/§9.3 + CHANGELOG

**Files:**
- Modify: `PRD.md`, `CHANGELOG.md`

**Interfaces:**
- Consumes: finale Editions-Zuordnung (autoencoder = Enterprise) aus Tasks 6–9.

- [ ] **Step 1: PRD-Edits** (exakte alt→neu-Ersetzungen):

(a) §3.2-Matrix-Zeile — alt:
```
| Neuronale Netze (sklearn-MLP, PyTorch, Autoencoder) | — | ✓ | ✓ |
```
neu:
```
| Neuronale Netze (sklearn-MLP, PyTorch) | — | ✓ | ✓ |
```
(b) §3.2: direkt VOR der Zeile `| Active Learning, Multi-Fidelity | — | — | ✓ |` neue Zeile einfügen:
```
| Autoencoder-Feld-Surrogate (nichtlineare Reduktion) | — | — | ✓ |
```
(c) Trennprinzip-Absatz — alt (Satzende):
```
löst den Lizenz-Check aus.
```
neu:
```
löst den Lizenz-Check aus — stufen-bewusst: das Autoencoder-Feld-Surrogat verlangt die Enterprise-Stufe.
```
(d) §4.8 (Methoden Phase 3 — Enterprise): Bullet ergänzen:
```
- **Autoencoder-Feld-Surrogate** (nichtlineare Feld-Reduktion, über PyTorch)
```
(e) §9.2-Inhalte — alt: `neuronale Netze (sklearn-MLP, PyTorch, Autoencoder)` → neu: `neuronale Netze (sklearn-MLP, PyTorch)`.
(f) §9.3-Inhalte: Bullet `- Autoencoder-Feld-Surrogat` ergänzen (vor `- Active Learning mit Acquisition-Funktionen`).
(g) §6.2 kommerzielles Paket-Listing — alt:
```
surroforge-professional/  # Erw. Methoden (RF/Boosting/NN), Export, Repro/Lineage, HTML-Reports, DoE
surroforge-enterprise/    # Active Learning, Multi-Fidelity, Audit-PDF
```
neu:
```
surroforge_professional/  # Erw. Methoden (RF/Boosting/NN), Export, Repro/Lineage, HTML-Reports, DoE
surroforge_enterprise/    # Autoencoder-Feld-Surrogat, Active Learning, Multi-Fidelity, Audit-PDF
```

- [ ] **Step 2: CHANGELOG** — unter `## [Unreleased]` (vor dem ersten `### Added`-Block) einfügen:

```markdown
### Changed — Editionen (Breaking)

- **`mlp`, `random-forest`, `gradient-boosting` sind Professional-Methoden** und leben jetzt
  im `surroforge_professional`-Plugin (Entry-Points statt Core-Builtins). Auflösung ohne
  Lizenz schlägt mit einem klaren `PluginError` fehl; `SURROFORGE_PRO_LICENSE` (oder
  `~/.surroforge/professional.lic`) schaltet frei. Core behält Kriging, RBF, Polynom und
  `reduced-field` (mit Nicht-NN-Inner).
- **`torch-mlp` + `pt`-Exporter** sind in `surroforge_professional` aufgegangen (optionales
  Extra: `pip install 'surroforge_professional[torch]'`); das separate `surroforge-torch`-Paket
  entfällt.
- **`autoencoder-field` ist eine Enterprise-Methode** im neuen `surroforge_enterprise`-Paket;
  eine Professional-Lizenz schaltet sie nicht frei (Stufen: professional < enterprise). Das
  separate `surroforge-autoencoder`-Paket entfällt.
- **Plugin-Seam (additiv, API v1 unverändert):** Plugins können `required_tier` deklarieren;
  Editionen registrieren Authorizer über die neue Entry-Point-Gruppe `surroforge.authorizers`.
- **Hinweis:** bestehende lokale Run-Stores mit `mlp`/`torch-mlp`-Modellen (Pickle-Modulpfade)
  müssen neu trainiert werden; die Modulpfade haben sich durch den Umzug geändert.
```

- [ ] **Step 3: Commit**

```bash
git add PRD.md CHANGELOG.md
git commit -m "docs(prd): autoencoder-field is enterprise-tier; record Wave-1 editions split"
```

---

### Task 12: Voll-Verifikation (ci.sh) + Abschluss

**Files:** keine neuen (nur Fixes, falls Gates rot sind)

- [ ] **Step 1: Installations-Zustand prüfen**

```bash
./.venv/bin/pip list 2>/dev/null | rg -i "surroforge"
```
Expected: `surroforge` (editable), `surroforge_professional 0.2.0` (editable), `surroforge_enterprise 0.1.0` (editable) — KEIN `surroforge-torch`, KEIN `surroforge-autoencoder`.

- [ ] **Step 2: Alle Quality-Gates**

Run: `scripts/ci.sh`
Expected: Summary am Ende — jede Zeile `PASS` (ruff, mypy, pyright, import-linter, pytest Core, pytest je Plugin `plugins/*/tests`, …). Rote Gates: Ursache beheben (fix-forward), Gate erneut laufen lassen; bei Unklarheit STOPP statt raten.

- [ ] **Step 3: Manuelle End-to-End-Stichprobe**

```bash
./.venv/bin/python -c "from surroforge.plugins import get_trainer, available_trainers; \
print(sorted(available_trainers()))"
SURROFORGE_PRO_LICENSE=dev ./.venv/bin/python -c \
  "from surroforge.plugins import get_trainer; print(get_trainer('mlp').required_tier)"
./.venv/bin/python -c "from surroforge.plugins import get_trainer; get_trainer('mlp')" \
  2>&1 | rg -q "professional' license" && echo "GATED OK"
SURROFORGE_PRO_LICENSE=enterprise ./.venv/bin/python -c \
  "from surroforge.plugins import get_trainer; print(get_trainer('autoencoder-field').name)"
```
Expected: Trainer-Liste enthält alle Methoden; `professional`; `GATED OK`; `autoencoder-field`.

- [ ] **Step 4: Abschluss-Commit (nur falls Step 2/3 Fixes erzeugt haben)**

```bash
git status --short   # zeigen, dann gezielt adden + committen
```

---

## Außerhalb dieses Plans (bewusst)

- **Editions-Artefakt** (claude.ai) auf „Autoencoder = Enterprise" aktualisieren — macht der Hauptagent nach Abschluss, kein Repo-Task.
- Export-/Befehls-Gating (W2), reproduce/Lineage (W3), echte Lizenz-Validierung (M3), Nutzer-Doku-Sweep — siehe `docs/superpowers/plans/2026-07-08-editions-code-migration.md`.
- Historische superpowers-Specs/Pläne (z. B. `2026-06-30-random-forest*.md`) beschreiben den alten Core-Zustand — NICHT anfassen.
