# Editions Welle 3: reproduce/Lineage-Feature-Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `reproduce` und die Herkunfts-Anzeigen (`info`, Markdown-Report, `compare`) verlangen eine Professional-Lizenz; die Lineage-Erfassung (lineage.json) bleibt unveränderte Core-Infrastruktur.

**Architecture:** Ein neues Core-Feature-Gate `gate.require_license(feature, tier)` neben dem W1-Seam (gleiche Authorizer, gleicher Hint, gemeinsamer Meldungs-Helfer). `reproduce` gated hart im Service; `info`/Report sanft (Feld-Schnitt: Identität frei, Herkunft Pro); `compare` auf Service-Ebene (Keys entfallen unlizenziert). Kein neuer Fehlertyp, keine Extraktion.

**Tech Stack:** Python 3.13 (`.venv/`), pytest, Jinja2 (Report-Template), W1-Authorizer-Gruppe `surroforge.authorizers`.

**Spec:** `docs/superpowers/specs/2026-07-10-editions-wave3-lineage-design.md`

## Global Constraints

- **Venv:** alle Befehle über `.venv/bin/…` (System-Python ist 3.9).
- **Git:** Identität repo-lokal `Georg Babij <g.babij@gmail.com>`; **KEIN `Co-Authored-By:`-Trailer**; `git add` nur explizite Pfade; direkt auf `main`; **nicht pushen**.
- **W1-Meldung byte-identisch:** `check()` muss nach dem Refactor exakt dieselbe Meldung erzeugen — `'{name}' requires a '{tier}' license, and no installed SurroForge edition grants it. {hints}` (rstrip) — bestehende Tests in `tests/test_gate.py`/`tests/test_plugins.py`/`plugins/surroforge_professional/tests/test_gating.py` asserten sie.
- **Frozen String (unangetastet):** `LICENSE_ERROR` in `plugins/surroforge_professional/src/surroforge_professional/license.py`.
- **Feature-Namen:** `"reproduce"` (harter Befehl) und `"lineage"` (die drei Anzeige-Oberflächen) — keine weiteren.
- **Editions-Neutralität:** Core-Code nennt keine Aktivierungs-Mechanik; der Hint kommt aus dem Authorizer. „Professional" als Editionsname in Hilfetexten/Doku ist erlaubt (W2-Präzedenz).
- **Fixture-Disziplin:** `granted_license` (Fake-Authorizer) NIE in Tests, die unlizenziertes Verhalten asserten — der Fake ignoriert Env/HOME. Unlizenziert-Tests neutralisieren Env + HOME (W2-Muster: `delenv SURROFORGE_PRO_LICENSE` + `setenv HOME → tmp_path / "no-licence-home"`).
- **Zeilenlänge 100** (ruff).

---

### Task 1: Feature-Gate `require_license` + gemeinsamer Meldungs-Helfer

**Files:**
- Modify: `src/surroforge/plugins/gate.py:36-52` (`check()` refaktorieren, `_denied_message` + `require_license` ergänzen)
- Test: `tests/test_gate.py` (ans Dateiende anhängen)

**Interfaces:**
- Consumes: `_load_authorizers()` (W1, gecacht), `PluginError`.
- Produces: `gate.require_license(feature: str, *, tier: str = "professional") -> None` (wirft `PluginError` mit `'{feature}' requires a '{tier}' license, …`); `gate._denied_message(name: str, tier: str) -> str`. Tasks 2-5 rufen `require_license`; alle Tests monkeypatchen weiterhin `gate._load_authorizers`.

- [ ] **Step 1: Failing Tests schreiben**

Ans Ende von `tests/test_gate.py` anhängen:

```python
# --- require_license: the Wave-3 feature gate ---------------------------------


def test_feature_denied_without_any_authorizer(monkeypatch):
    monkeypatch.setattr(gate, "_load_authorizers", lambda: ())
    with pytest.raises(PluginError, match=r"'reproduce' requires a 'professional' license"):
        gate.require_license("reproduce")


def test_feature_permitted_by_granting_authorizer(monkeypatch):
    monkeypatch.setattr(
        gate, "_load_authorizers", lambda: (lambda tier: tier == "professional",)
    )
    gate.require_license("reproduce")  # must not raise


def test_feature_gate_appends_authorizer_hint(monkeypatch):
    def deny(tier: str) -> bool:
        return False

    deny.hint = "Set THE_LICENSE_VAR."
    monkeypatch.setattr(gate, "_load_authorizers", lambda: (deny,))
    with pytest.raises(PluginError, match=r"'lineage' requires .* Set THE_LICENSE_VAR"):
        gate.require_license("lineage")


def test_feature_gate_respects_tier_argument(monkeypatch):
    monkeypatch.setattr(
        gate, "_load_authorizers", lambda: (lambda tier: tier == "professional",)
    )
    with pytest.raises(PluginError, match=r"'audit' requires a 'enterprise' license"):
        gate.require_license("audit", tier="enterprise")
```

- [ ] **Step 2: Tests laufen rot**

Run: `.venv/bin/python -m pytest tests/test_gate.py -v -k feature`
Expected: 4× FAIL — `AttributeError: module 'surroforge.plugins.gate' has no attribute 'require_license'`

- [ ] **Step 3: Implementieren**

In `src/surroforge/plugins/gate.py` den Block `def check(…)` (Zeilen 36-52) ersetzen durch:

```python
def check(metadata: PluginMetadata) -> None:
    """Authorise a resolved plugin: free plugins pass, tier-declaring ones are gated.

    The authorizer scan only runs for tier-declaring plugins, so the free
    (Core) resolution path pays no overhead.
    """
    if metadata.required_tier is None:
        return
    if any(authorize(metadata.required_tier) for authorize in _load_authorizers()):
        return
    raise PluginError(_denied_message(metadata.name, metadata.required_tier))


def require_license(feature: str, *, tier: str = "professional") -> None:
    """Gate a Core feature surface behind an edition tier (Wave 3).

    The resolution seam (:func:`check`) covers everything reached through the
    plugin registry; feature surfaces like ``reproduce`` or the lineage panels
    never resolve a plugin, so they call this directly — same authorizer group,
    same hint mechanics, same editions-neutral message, naming the FEATURE
    instead of a plugin.
    """
    if any(authorize(tier) for authorize in _load_authorizers()):
        return
    raise PluginError(_denied_message(feature, tier))


def _denied_message(name: str, tier: str) -> str:
    """The editions-neutral denial, shared by the seam gate and the feature gate.

    Core never names an activation mechanism; an installed edition's authorizer
    may carry a ``hint`` attribute that is appended verbatim. Wave-1 tests assert
    this wording — keep it byte-identical.
    """
    authorizers = _load_authorizers()
    hints = " ".join(h for a in authorizers if (h := getattr(a, "hint", "")))
    message = (
        f"'{name}' requires a '{tier}' license, "
        "and no installed SurroForge edition grants it."
    )
    return f"{message} {hints}".rstrip()
```

- [ ] **Step 4: Tests laufen grün (inkl. W1-Regression)**

Run: `.venv/bin/python -m pytest tests/test_gate.py tests/test_plugins.py -q`
Expected: alle PASS (die bestehenden W1-Meldungs-Tests beweisen die Byte-Identität).

- [ ] **Step 5: Commit**

```bash
git add src/surroforge/plugins/gate.py tests/test_gate.py
git commit -m "feat(gate): require_license feature gate beside the W1 seam, shared denial message"
```

---

### Task 2: `reproduce` hart gaten + Test-Fixture-Welle

`services.reproduce` bekommt das Gate; damit brechen ALLE Core-Tests, die reproduce aufrufen — Fixture und Injektionen landen deshalb im selben Commit.

**Files:**
- Modify: `src/surroforge/services.py:31` (Import) und `src/surroforge/services.py:163-171` (Gate)
- Modify: `tests/conftest.py` (Fixture `granted_license` nach `licensed_export`)
- Modify (autouse-Fixture nach den Imports): `tests/test_reproduce.py`, `tests/test_reproduce_field.py`, `tests/test_reproduce_scalar_identity.py`, `tests/test_probe_coverage.py`, `tests/test_reduction_config.py`, `tests/test_field_ingest.py`
- Modify (nur Signaturen): `tests/test_cli.py:121` (`test_reproduce_cli`), `tests/test_scenarios.py:71` (`test_scenario_reproduction`), `tests/test_reduced_field.py:299` (`test_kriging_inner_field_run_reproduces`), `tests/test_openfoam.py:348` (`test_sweep_trains_through_reduced_field_pipeline`)
- Create: `tests/test_lineage_gating.py`; Modify: `tests/test_cli.py` (CLI-Deny-Test)

**Interfaces:**
- Consumes: `gate.require_license("reproduce")` (Task 1); Fixtures `project`/`poly_config` (Core-conftest), Helfer `_init_project_with_data`/`_write_config`/`runner`/`app` (test_cli.py).
- Produces: Fixture `granted_license` (tests/conftest.py) — Tasks 3-5 injizieren es; Datei `tests/test_lineage_gating.py` — Tasks 3-5 hängen Tests an.

- [ ] **Step 1: Fixture in `tests/conftest.py`** (direkt nach `licensed_export` einfügen)

```python
@pytest.fixture
def granted_license(monkeypatch):
    """Bypass the Wave-3 feature gate with a fake always-yes authorizer.

    For tests of Core functionality that sits BEHIND the gate (reproduce math,
    field reproduce, lineage panels) — works on a Core-only checkout, unlike
    ``licensed_export`` (which needs the real professional authorizer). Never
    use it in tests that assert UNLICENSED behavior: the fake ignores env/HOME.
    """
    from surroforge.plugins import gate

    monkeypatch.setattr(gate, "_load_authorizers", lambda: ((lambda tier: True),))
```

- [ ] **Step 2: Deny-Tests schreiben (rot, weil das Gate noch fehlt — der Deny-Test schlägt fehl, weil noch NICHTS wirft)**

Neue Datei `tests/test_lineage_gating.py`:

```python
"""Wave-3 feature gating: reproduce/lineage surfaces are Professional.

Core-side behavior tests. Unlicensed tests neutralize env + HOME (W2 pattern)
so a real dev-machine licence cannot defeat them; the licensed side uses the
``granted_license`` fake authorizer, so this file runs on a Core-only checkout.
"""

from __future__ import annotations

import pytest

from surroforge.core.config import parse_config
from surroforge.engine import run_training
from surroforge.errors import PluginError
from tests.conftest import poly_config


@pytest.fixture
def unlicensed(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-licence-home"))


def test_reproduce_service_denied_without_license(unlicensed, project):
    from surroforge.services import reproduce

    run = run_training(project, parse_config(poly_config("polynomial")))
    with pytest.raises(PluginError, match=r"'reproduce' requires a 'professional' license"):
        reproduce(project, run.run_id)


def test_reproduce_service_works_with_granted_license(granted_license, project):
    from surroforge.services import assert_reproducible, reproduce

    run = run_training(project, parse_config(poly_config("polynomial")))
    result = reproduce(project, run.run_id)
    assert_reproducible(result)  # must not raise
```

In `tests/test_cli.py` direkt nach `test_reproduce_cli` einfügen:

```python
def test_reproduce_cli_denied_without_license(tmp_path, monkeypatch):
    # Wave 3: reproduce is a Professional feature — clean gate error, exit 1.
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-licence-home"))
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])

    result = runner.invoke(app, ["--config-dir", str(root), "reproduce", "run-001"])

    assert result.exit_code == 1
    assert "'reproduce' requires a 'professional' license" in result.output
    assert "Traceback" not in result.output
```

- [ ] **Step 3: Deny-Tests laufen rot**

Run: `.venv/bin/python -m pytest tests/test_lineage_gating.py::test_reproduce_service_denied_without_license tests/test_cli.py::test_reproduce_cli_denied_without_license -v`
Expected: beide FAIL (`DID NOT RAISE PluginError` bzw. exit_code 0 statt 1) — reproduce ist noch ungegatet.

- [ ] **Step 4: Gate implementieren**

`src/surroforge/services.py:31` — alt:

```python
from surroforge.plugins import get_comparison_reporter, get_exporter, get_reporter
```

neu:

```python
from surroforge.plugins import gate, get_comparison_reporter, get_exporter, get_reporter
```

In `reproduce()` (Zeile ~163) direkt nach dem Docstring (nach Zeile 171) einfügen:

```python
    gate.require_license("reproduce")
```

- [ ] **Step 5: Fixture-Welle über die bestehenden reproduce-Tests**

In JEDEM der sechs Rein-reproduce-Module — `tests/test_reproduce.py`, `tests/test_reproduce_field.py`, `tests/test_reproduce_scalar_identity.py`, `tests/test_probe_coverage.py`, `tests/test_reduction_config.py`, `tests/test_field_ingest.py` — direkt nach dem Import-Block einfügen (identischer Block; diese Module enthalten keine Unlizenziert-Tests, autouse ist dort sicher):

```python
@pytest.fixture(autouse=True)
def _grant_license(granted_license):
    """reproduce is Professional since Wave 3; this module tests the Core math behind it."""
```

(Falls ein Modul `pytest` noch nicht importiert: `import pytest` ergänzen.)

Vier Misch-Modul-Signaturen erweitern (NUR den Parameter, Körper unverändert):
- `tests/test_cli.py:121`: `def test_reproduce_cli(tmp_path):` → `def test_reproduce_cli(tmp_path, granted_license):`
- `tests/test_scenarios.py:71`: `def test_scenario_reproduction(tmp_path):` → `def test_scenario_reproduction(tmp_path, granted_license):`
- `tests/test_reduced_field.py:299`: `def test_kriging_inner_field_run_reproduces(tmp_path):` → `def test_kriging_inner_field_run_reproduces(tmp_path, granted_license):`
- `tests/test_openfoam.py:348`: `def test_sweep_trains_through_reduced_field_pipeline(tmp_path):` → `def test_sweep_trains_through_reduced_field_pipeline(tmp_path, granted_license):`

- [ ] **Step 6: Alles grün**

Run: `.venv/bin/python -m pytest tests -q`
Expected: alle PASS (Kernprobe: `tests/test_reproduce*.py`, `tests/test_lineage_gating.py`, `tests/test_cli.py` grün).

- [ ] **Step 7: Commit**

```bash
git add src/surroforge/services.py tests/conftest.py tests/test_lineage_gating.py \
  tests/test_reproduce.py tests/test_reproduce_field.py tests/test_reproduce_scalar_identity.py \
  tests/test_probe_coverage.py tests/test_reduction_config.py tests/test_field_ingest.py \
  tests/test_cli.py tests/test_scenarios.py tests/test_reduced_field.py tests/test_openfoam.py
git commit -m "feat(editions)!: reproduce requires a professional license (Wave 3)"
```

---

### Task 3: `info` sanft — Identität frei, Herkunft Professional

**Files:**
- Modify: `src/surroforge/cli/main.py:26` (Import) und `src/surroforge/cli/main.py:599-615` (`info`-Body)
- Modify: `tests/test_cli.py:232` (`test_info_cli` Signatur) + neuer Deny-Test

**Interfaces:**
- Consumes: `gate.require_license("lineage")` (Task 1), `PluginError` (in cli/main.py seit W2 importiert), `granted_license` (Task 2).
- Produces: nichts Neues für spätere Tasks (gleicher Feld-Schnitt wird in Task 4 wiederverwendet, aber dort eigenständig implementiert).

- [ ] **Step 1: Deny-Test schreiben** (in `tests/test_cli.py` nach `test_info_cli`)

```python
def test_info_cli_hides_provenance_without_license(tmp_path, monkeypatch):
    # Wave 3: identity + metrics stay free; hashes/seed/versions are Professional.
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-licence-home"))
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])

    result = runner.invoke(app, ["--config-dir", str(root), "info", "run-001"])

    assert result.exit_code == 0
    assert "experiment" in result.output
    assert "polynomial" in result.output
    assert "provenance hidden:" in result.output
    assert "config_hash" not in result.output
    assert "data_hash" not in result.output
    assert "rmse" in result.output  # metrics stay free
```

Und Signatur `tests/test_cli.py:232`: `def test_info_cli(tmp_path):` → `def test_info_cli(tmp_path, granted_license):` (asserted `config_hash` — braucht jetzt die Lizenz).

- [ ] **Step 2: Deny-Test läuft rot**

Run: `.venv/bin/python -m pytest tests/test_cli.py::test_info_cli_hides_provenance_without_license -v`
Expected: FAIL — `"provenance hidden:" not in result.output` (info zeigt noch alles).

- [ ] **Step 3: Implementieren**

`src/surroforge/cli/main.py:26` — alt:

```python
from surroforge.plugins import available_commands, get_exporter, get_trainer
```

neu:

```python
from surroforge.plugins import available_commands, gate, get_exporter, get_trainer
```

`info`-Body (Zeilen 599-615) ersetzen durch:

```python
@app.command()
@_guard
def info(run_id: str = typer.Argument(..., help="Run id to inspect.")) -> None:
    """Show detailed information about a run (PRD 4.2)."""
    run = _project().run(run_id)
    lineage = run.lineage
    _info(f"Run {run.run_id}")
    for key in ("experiment", "method", "created_at", "data_source"):
        _info(f"  {key:<18}: {lineage.get(key, '-')}")
    _info(f"  {'onnx_artifact':<18}: {'yes' if run.has_onnx() else 'no'}")
    # Provenance is a Professional surface (Wave 3); identity and metrics stay
    # free. The hidden-line carries the gate's own editions-neutral message.
    try:
        gate.require_license("lineage")
    except PluginError as exc:
        _info(f"  (provenance hidden: {exc})")
    else:
        for key in (
            "config_hash", "data_hash", "seed", "prediction_hash",
            "python_version", "surroforge_version", "parent_run",
        ):
            _info(f"  {key:<18}: {lineage.get(key, '-')}")
        _info(f"  {'dependencies':<18}: {', '.join(lineage.get('dependencies', []))}")
    _info("")
    _metrics_summary(run)
```

- [ ] **Step 4: Tests laufen grün**

Run: `.venv/bin/python -m pytest tests/test_cli.py -q`
Expected: alle PASS (inkl. `test_info_cli` mit Lizenz und dem neuen Deny-Test).

- [ ] **Step 5: Commit**

```bash
git add src/surroforge/cli/main.py tests/test_cli.py
git commit -m "feat(editions)!: info shows provenance only with a professional license (Wave 3)"
```

---

### Task 4: Markdown-Report sanft — gleicher Feld-Schnitt

**Files:**
- Modify: `src/surroforge/reporters/markdown.py` (Imports, Template-Hunks, `render()`-Gate)
- Modify: `tests/test_reporters.py:22` (Signatur) + neuer Deny-Test

**Interfaces:**
- Consumes: `gate.require_license("lineage")`, `PluginError`, `granted_license`, Fixture `trained_run` (Core-conftest).
- Produces: Template-Kontext-Keys `provenance: bool`, `provenance_notice: str | None` (nur intern).

- [ ] **Step 1: Deny-Test schreiben** (in `tests/test_reporters.py` nach `test_markdown_reports_metrics_and_lineage`)

```python
def test_markdown_report_omits_provenance_without_license(trained_run, monkeypatch, tmp_path):
    # Wave 3: the report keeps every free section; provenance values give way to
    # a single italic notice carrying the gate's message. (The config YAML block
    # stays — it is the user's own input, incl. split.seed; gated is the DERIVED
    # provenance: hashes, fingerprint, env versions, dependencies.)
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-licence-home"))
    text = MarkdownReporter().render(trained_run)
    assert "## Reproducibility" in text  # the section survives, its values do not
    assert "*Provenance omitted: " in text
    assert "'lineage' requires a 'professional' license" in text
    assert trained_run.data_hash not in text
    assert trained_run.config_hash not in text
    assert "| rmse |" in text  # metrics stay free
    assert "## Configuration" in text
```

Und Signatur `tests/test_reporters.py:22`: `def test_markdown_reports_metrics_and_lineage(trained_run):` → `def test_markdown_reports_metrics_and_lineage(trained_run, granted_license):`.

- [ ] **Step 2: Deny-Test läuft rot**

Run: `.venv/bin/python -m pytest tests/test_reporters.py -v -k provenance`
Expected: FAIL — `"*Provenance omitted: " not in text`.

- [ ] **Step 3: Implementieren**

`src/surroforge/reporters/markdown.py` — Imports ergänzen (nach `from surroforge.core.run import Run`):

```python
from surroforge.errors import PluginError
from surroforge.plugins import gate
```

Vier Template-Hunks in `_TEMPLATE` (exakte Ersetzungen):

1. Zeile 25-26 — alt:

```
{% if parent_run %}**Reproduction of:** {{ parent_run }}
{% endif %}
```

neu:

```
{% if provenance and parent_run %}**Reproduction of:** {{ parent_run }}
{% endif %}
```

2. Zeile 33 — alt:

```
- **Data hash:** `{{ data_hash }}`
```

neu:

```
{% if provenance %}- **Data hash:** `{{ data_hash }}`
{% endif %}
```

(Die Folgezeile ist eine Leerzeile — sie bleibt außerhalb des if.)

3. Zeilen 68-69 — alt:

```
- **Config hash:** `{{ config_hash }}`
- **Random seed:** {{ seed }}
```

neu:

```
{% if provenance %}- **Config hash:** `{{ config_hash }}`
- **Random seed:** {{ seed }}
{% endif %}
```

(Achtung: die alte Zeile 70 `- **Test fraction:** {{ test_fraction }}` bleibt danach unverändert stehen.)

4. Reproducibility-Abschnitt (Zeilen 75-83) — alt:

```
## Reproducibility

- **Python:** {{ python_version }}
- **SurroForge:** {{ surroforge_version }}
- **Prediction fingerprint:** `{{ prediction_hash }}`
- **Dependencies:**
{% for dep in dependencies %}    - {{ dep }}
{% endfor %}
*Reproduce this run with* `surroforge reproduce {{ run_id }}`.
```

neu:

```
## Reproducibility
{% if provenance %}
- **Python:** {{ python_version }}
- **SurroForge:** {{ surroforge_version }}
- **Prediction fingerprint:** `{{ prediction_hash }}`
- **Dependencies:**
{% for dep in dependencies %}    - {{ dep }}
{% endfor %}
*Reproduce this run with* `surroforge reproduce {{ run_id }}`.
{% else %}
*Provenance omitted: {{ provenance_notice }}*
{% endif %}
```

In `render()` direkt vor `context = {` einfügen:

```python
        # Provenance is a Professional surface (Wave 3): without a licence the
        # report keeps every free section and replaces the provenance values
        # with one notice carrying the gate's editions-neutral message.
        provenance_notice: str | None = None
        try:
            gate.require_license("lineage")
        except PluginError as exc:
            provenance_notice = str(exc)
```

und im `context`-Dict (nach `"dependencies": …,`) ergänzen:

```python
            "provenance": provenance_notice is None,
            "provenance_notice": provenance_notice,
```

- [ ] **Step 4: Tests laufen grün**

Run: `.venv/bin/python -m pytest tests/test_reporters.py tests/test_two_source_report.py -q`
Expected: alle PASS (der Überschriften-Test läuft unlizenziert weiter — `## Reproducibility` bleibt; test_two_source_report asserted nur freie Teile).

- [ ] **Step 5: Commit**

```bash
git add src/surroforge/reporters/markdown.py tests/test_reporters.py
git commit -m "feat(editions)!: markdown report gates provenance behind the professional license (Wave 3)"
```

---

### Task 5: `compare` — Lineage-Keys auf Service-Ebene, CLI konditional

**Files:**
- Modify: `src/surroforge/services.py:389-397` (`compare_runs`)
- Modify: `src/surroforge/cli/main.py:411-420` (`_render_comparison`) + neuer Helfer nach `_render_lineage` (Zeile ~499)
- Modify: `tests/test_cli.py:217` (`test_compare_cli` Signatur) + neuer Deny-Test; `tests/test_lineage_gating.py` (zwei API-Tests anhängen)

**Interfaces:**
- Consumes: `gate.require_license("lineage")`, `granted_license`, `unlicensed`-Fixture (test_lineage_gating.py, Task 2); der `gate`-Import in `cli/main.py` existiert seit Task 3, der in `services.py` seit Task 2.
- Produces: `compare_runs`-Ergebnis ohne `lineage`/`lineage_summary`-Keys, wenn unlizenziert (dokumentierter API-Kontrakt).

- [ ] **Step 1: Deny-Tests schreiben**

In `tests/test_lineage_gating.py` anhängen:

```python
def test_compare_runs_strips_lineage_without_license(unlicensed, project):
    from surroforge.services import compare_runs

    r1 = run_training(project, parse_config(poly_config("polynomial")))
    r2 = run_training(project, parse_config(poly_config("polynomial")))
    result = compare_runs(project, [r1.run_id, r2.run_id])
    assert "lineage_summary" not in result
    assert all("lineage" not in entry for entry in result["runs"])
    assert result["best"] is not None  # ranking/metrics stay free


def test_compare_runs_keeps_lineage_with_granted_license(granted_license, project):
    from surroforge.services import compare_runs

    r1 = run_training(project, parse_config(poly_config("polynomial")))
    r2 = run_training(project, parse_config(poly_config("polynomial")))
    result = compare_runs(project, [r1.run_id, r2.run_id])
    assert result["lineage_summary"]["same_data"] is True
    assert all("lineage" in entry for entry in result["runs"])
```

In `tests/test_cli.py` nach `test_compare_cli` einfügen und Signatur `tests/test_cli.py:217` erweitern (`def test_compare_cli(tmp_path):` → `def test_compare_cli(tmp_path, granted_license):`; der Test asserted `"Lineage" in result.output`):

```python
def test_compare_cli_hides_lineage_without_license(tmp_path, monkeypatch):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-licence-home"))
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])

    result = runner.invoke(
        app, ["--config-dir", str(root), "compare", "--runs", "run-001,run-002"]
    )

    assert result.exit_code == 0, result.output
    assert "Ranking by test rmse" in result.output
    assert "Lineage: hidden (" in result.output
    assert "data_hash" not in result.output
```

- [ ] **Step 2: Deny-Tests laufen rot**

Run: `.venv/bin/python -m pytest tests/test_lineage_gating.py -v -k compare` und `.venv/bin/python -m pytest tests/test_cli.py::test_compare_cli_hides_lineage_without_license -v`
Expected: FAIL (`lineage_summary` noch im Ergebnis; CLI rendert noch das Panel).

- [ ] **Step 3: Service implementieren**

`src/surroforge/services.py`, `compare_runs` — alt (Zeilen ~394-397):

```python
    if not run_ids:
        raise DataError("Provide at least one run id to compare.")
    runs = [project.run(rid) for rid in run_ids]
    return build_comparison(runs, metric=metric)
```

neu:

```python
    if not run_ids:
        raise DataError("Provide at least one run id to compare.")
    runs = [project.run(rid) for rid in run_ids]
    result = build_comparison(runs, metric=metric)
    # Lineage is a Professional surface (Wave 3): the aggregation is built either
    # way (capture is Core infra), but an unlicensed result carries no lineage
    # view — one boundary for the CLI and the Python API alike. The comparability
    # warnings stay: they protect conclusions, they don't display provenance.
    try:
        gate.require_license("lineage")
    except PluginError:
        for entry in result["runs"]:
            entry.pop("lineage", None)
        result.pop("lineage_summary", None)
    return result
```

`PluginError`-Import in services.py prüfen: `from surroforge.errors import …` (Zeile ~20-30) muss `PluginError` enthalten — falls nicht, ergänzen.

- [ ] **Step 4: CLI implementieren**

`src/surroforge/cli/main.py`, `_render_comparison` (Zeile ~411) — alt:

```python
    _render_config_diff(result)
    _render_lineage(result)
```

neu:

```python
    _render_config_diff(result)
    if "lineage_summary" in result:
        _render_lineage(result)
    else:
        _render_lineage_notice()
```

Nach dem Ende von `_render_lineage` (nach Zeile ~498) einfügen:

```python
def _render_lineage_notice() -> None:
    # compare_runs stripped the lineage view (Wave 3): explain with the gate's
    # own editions-neutral message instead of a second hand-written one.
    try:
        gate.require_license("lineage")
    except PluginError as exc:
        _info(f"Lineage: hidden ({exc})")
```

- [ ] **Step 5: Tests laufen grün**

Run: `.venv/bin/python -m pytest tests/test_lineage_gating.py tests/test_cli.py tests/test_comparison.py tests/test_scenarios.py -q`
Expected: alle PASS (`test_comparison.py` testet `build_comparison` direkt — bleibt ungegatet und unberührt; `test_scenario_method_comparison` asserted keine Lineage-Keys).

- [ ] **Step 6: Commit**

```bash
git add src/surroforge/services.py src/surroforge/cli/main.py tests/test_cli.py tests/test_lineage_gating.py
git commit -m "feat(editions)!: compare strips the lineage view without a professional license (Wave 3)"
```

---

### Task 6: Cross-Edition-Integration + W2-Follow-up-Regressionstest

**Files:**
- Modify: `tests/test_training.py` (nach `test_train_onnx_without_license_skips_gracefully`)
- Modify: `plugins/surroforge_professional/tests/test_gating.py` (ans Dateiende)

**Interfaces:**
- Consumes: Core-Fixtures `project`/`poly_config`/`make_samples`; professional-Fixtures `licensed`/`project`/`trained_run` (conftest) und `unlicensed` (test_gating.py, modul-lokal); `run_training`/`parse_config` (in test_training.py importiert).
- Produces: nichts — Regressionsschlösser.

- [ ] **Step 1: W2-Follow-up-Test** (in `tests/test_training.py` nach `test_train_onnx_without_license_skips_gracefully` einfügen; erwartet SOFORT grün — Regressionsschloss auf bestehendem Verhalten, kein TDD-rot; wäre er rot, ist das ein echter Defekt → melden, nicht anpassen)

```python
def test_train_then_predict_without_license(project, monkeypatch, tmp_path):
    # Second lock on the Wave-2/3 invariant: the run store is licence-free —
    # an unlicensed user trains AND predicts from the stored model, end to end.
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-licence-home"))
    from surroforge.services import predict_with_run
    from tests.conftest import make_samples

    run = run_training(project, parse_config(poly_config("polynomial")))
    new_points = make_samples(n=5, seed=99).drop(columns=["max_stress"])
    preds = predict_with_run(run, new_points)
    assert "max_stress" in preds.columns
    assert len(preds) == 5
```

- [ ] **Step 2: professional-Integrationstests (echter Authorizer)** — in `plugins/surroforge_professional/tests/test_gating.py` anhängen:

```python
# --- Feature gate (Wave 3) -----------------------------------------------------


def test_reproduce_works_with_real_license(licensed, project, trained_run):
    from surroforge.services import assert_reproducible, reproduce

    result = reproduce(project, trained_run.run_id)
    assert_reproducible(result)  # must not raise


def test_reproduce_denied_by_real_authorizer_names_hint(unlicensed, project, trained_run):
    from surroforge.services import reproduce

    with pytest.raises(PluginError) as err:
        reproduce(project, trained_run.run_id)
    assert "'reproduce' requires a 'professional' license" in str(err.value)
    assert "SURROFORGE_PRO_LICENSE" in str(err.value)  # hint from the real authorizer
```

- [ ] **Step 3: Tests laufen grün**

Run: `.venv/bin/python -m pytest tests/test_training.py plugins/surroforge_professional/tests/test_gating.py -q`
Expected: alle PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_training.py plugins/surroforge_professional/tests/test_gating.py
git commit -m "test(editions): real-authorizer reproduce integration + unlicensed train->predict lock"
```

---

### Task 7: PRD, Doku, Beispiel, CHANGELOG, MEMORY — Abschluss-Sweep

**Files:**
- Modify: `PRD.md:362,364-366,261`, `examples/03_reproduction.py` (Docstring), `examples/README.md:16`, `docs/examples.md:17`, `docs/scenarios/reproduction.md` (nach dem Goal-Absatz), `docs/quickstart.md:107,133-135`, `docs/index.md:26-28`, `README.md:101-106`, `docs/reference/cli.md:47-50,69-72`, `docs/reference/python-api.md:76,86,88`, `CHANGELOG.md` (Editionen-Block), `MEMORY.md` — (quickstart.md:100 „a complete lineage record" bleibt bewusst unverändert: die Datei wird weiterhin geschrieben)

**Interfaces:**
- Consumes: Endzustand Tasks 1-6.
- Produces: Doku-Schlussstand Welle 3.

- [ ] **Step 1: PRD**

Zeile 362 — alt:

```
Im **freien Kern** enthält ein Run `config.snapshot.yaml`, `model.pkl` (interner Store für `predict`), `metrics.json` und `log.txt`. `model.onnx` (Export) und `lineage.json` (Herkunft) werden erst in **Professional** geschrieben.
```

neu:

```
Jeder Run enthält `config.snapshot.yaml`, `model.pkl` (interner Store für `predict`), `metrics.json`, `log.txt` und `lineage.json` — die Herkunfts-**Erfassung** ist Core-interne Infrastruktur (sie trägt `predict` und den Upgrade-Pfad: früher trainierte Runs bleiben nach Lizenzkauf reproduzierbar). **Nutzbar** wird die Herkunft erst in Professional (`reproduce`, Anzeige in `info`/Report/`compare`); `model.onnx` (Export) wird erst in Professional geschrieben.
```

§7.2 (Zeilen 364-366) — alt:

```
### 7.2 Run-Metadaten (lineage.json, Professional)

`lineage.json` wird von Professional geschrieben. Pflichtfelder pro Run:
```

neu:

```
### 7.2 Run-Metadaten (lineage.json; Erfassung Core-intern, Nutzung Professional)

`lineage.json` wird von Core für jeden Run geschrieben (interne Infrastruktur); die Professional-Oberflächen (`reproduce`, `info`, Report, `compare`) machen sie nutzbar. Pflichtfelder pro Run:
```

§5.2 letzter Aufzählungspunkt (Zeile 261) — alt:

```
- Der freie Kern speichert Konfiguration und Modell für Vorhersagen, jedoch ohne Herkunfts-Hashes und ohne `reproduce`
```

neu:

```
- Der freie Kern speichert Konfiguration und Modell für Vorhersagen; Herkunfts-Daten werden intern erfasst, sind ohne Lizenz aber weder einsehbar noch per `reproduce` nutzbar
```

- [ ] **Step 2: Beispiel 03 + Beispiel-Indizes**

`examples/03_reproduction.py` — im Docstring nach `This mirrors `surroforge reproduce <run-id>`.` einfügen:

```
Since the editions split, `reproduce` is a Professional feature: set
`SURROFORGE_PRO_LICENSE` (any value activates the stub licence) before running.
The recording itself happens on every run, licensed or not.
```

`examples/README.md:16` und `docs/examples.md:17`: in der `03_reproduction.py`-Zeile die letzte Spalte `Core only` → `Professional license (`reproduce`)`.

- [ ] **Step 3: Doku-Site**

`docs/scenarios/reproduction.md` — nach dem Goal-Absatz (Zeile 4) einfügen:

```markdown

!!! info "Professional feature"
    `reproduce` and the provenance views (`info`, report, `compare` lineage) need a
    Professional license. The recording itself happens on every run — Core writes
    `runs/<id>/lineage.json` regardless — so runs trained before a license was
    installed remain reproducible afterwards.
```

`docs/quickstart.md:107` — alt: `surroforge info run-001         # full lineage: hashes, seed, versions` → neu: `surroforge info run-001         # identity + metrics; provenance needs Professional`.
`docs/quickstart.md` §10 (Reproduce-Intro, Zeilen 133-135): nach „…you can verify it reproduces:" den Satz einfügen: `Reproduction is a **Professional** feature (the recording happens on every run).`

`docs/index.md:26-28` — im Reproducibility-Bullet `…re-runs from the config and returns a verdict.` → `…re-runs from the config and returns a verdict (Professional; the recording itself happens on every run).`

`README.md:101-106` („Reproducibility"-Abschnitt): am Absatzende anfügen: `Recording happens on every run; verifying and viewing it (\`reproduce\`, the provenance in \`info\`/reports/\`compare\`) is a Professional feature.`

`docs/reference/cli.md:47-50` (`info`): am Absatzende anfügen: `The provenance block (hashes, seed, versions, dependencies) needs a Professional license; identity fields and metrics stay free.`
`docs/reference/cli.md:69-72` (`reproduce`): nach dem ersten Satz einfügen: `A Professional feature (the recording happens on every run).`

`docs/reference/python-api.md`:
- Zeile 76 (`lineage`-Member): `| \`lineage\` | The full lineage mapping. |` → `| \`lineage\` | The full lineage mapping (raw data stays readable; the gated surfaces are \`reproduce\` and the aggregated views). |`
- Zeile 86 (`reproduce`-Zeile): am Beschreibungsende anfügen: `Professional (licensed).`
- Zeile 88 (`compare_runs`-Zeile): am Beschreibungsende anfügen: `Without a Professional license the result omits \`lineage\`/\`lineage_summary\`.`

- [ ] **Step 4: CHANGELOG + MEMORY**

`CHANGELOG.md` — im Abschnitt `### Changed — Editionen (Breaking)` nach dem Welle-2-Bullet einfügen:

```markdown
- **Reproduzierbarkeit & Lineage sind Professional (Welle 3):** `reproduce` (CLI und
  `services.reproduce`) verlangt eine Lizenz; `info`, der Markdown-Report und `compare`
  zeigen Herkunft (Hashes, Seed, Versionen, Dependencies) nur noch lizenziert und blenden
  sonst einen Hinweis ein; `compare_runs` lässt die Lineage-Keys unlizenziert weg. Die
  **Erfassung** bleibt Core: jeder Run schreibt `lineage.json` weiter (`predict` und der
  Upgrade-Pfad hängen daran). Neu: `gate.require_license(feature)` als Core-Feature-Gate
  neben dem W1-Seam.
```

`MEMORY.md`:
- Im Bullet „**Reproduzierbarkeit:**" (Zeile ~73) nach dem ersten Satz einfügen: `Seit Welle 3 sind die Oberflächen (reproduce, Herkunfts-Anzeige in info/Report/compare) Professional; die Erfassung bleibt Core.`
- In „Offene Punkte" (Zeile ~108): `reproduce/Lineage-Gating (Welle 3),` aus der Aufzählung streichen (übrig bleiben u. a. skalierbare GP, echte Lizenz-Aktivierung (M3), Support-Infrastruktur).

- [ ] **Step 5: Gesamte Quality-Gates**

Run: `scripts/ci.sh`
Expected: alle Gates grün (ruff, mypy, pyright, import-linter, pytest Core + `plugins/*/tests`, Beispiele mit injizierter Lizenz; `docs`-Gate darf wie in W2 wegen fehlendem mkdocs SKIP sein).

- [ ] **Step 6: Commit**

```bash
git add PRD.md examples/03_reproduction.py examples/README.md docs/examples.md \
  docs/scenarios/reproduction.md docs/quickstart.md docs/index.md README.md \
  docs/reference/cli.md docs/reference/python-api.md CHANGELOG.md MEMORY.md
git commit -m "docs(editions): Wave-3 lineage gating sweep (PRD reconciliation, example 03, docs, changelog, memory)"
```
