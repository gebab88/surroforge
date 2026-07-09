# Editions Welle 2: Export-Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Die Exporter `pickle`/`onnx` ziehen lizenzpflichtig nach `surroforge_professional` (der `pt`-Exporter wird mitgegatet); Train/Predict bleiben über einen neuen Core-Model-Store frei.

**Architecture:** Erst wird der Run-Store (`model.pkl` schreiben/lesen) von den Exportern entkoppelt (`persistence/model_io.py`), dann `train --onnx` auf den W1-Seam umgestellt (reines Refactoring), dann der atomare Umzug der Exporter mit `required_tier="professional"` + Entry-Point-Umbau. Alle Nutzer-Exportpfade laufen durch `get_exporter` → der W1-Seam gated vollständig, keine Self-Gates.

**Tech Stack:** Python 3.13 (`.venv/`), pytest, hatchling-Entry-Points, W1-Seam (`required_tier` + `surroforge.authorizers`).

**Spec:** `docs/superpowers/specs/2026-07-09-editions-wave2-export-design.md`

## Global Constraints

- **Venv:** alle Befehle über `.venv/bin/…` (System-Python ist 3.9).
- **Git:** Identität ist repo-lokal `Georg Babij <g.babij@gmail.com>`; **KEIN `Co-Authored-By:`-Trailer, niemals**; `git add` nur mit expliziten Pfaden; direkt auf `main` committen; **nicht pushen**.
- **Frozen String (W1, nicht anfassen):** `LICENSE_ERROR = "Professional license required. Set SURROFORGE_PRO_LICENSE or install ~/.surroforge/professional.lic."` in `plugins/surroforge_professional/src/surroforge_professional/license.py` (ein Test asserted `==`).
- **Editions-Neutralität Core:** Core-Fehlermeldungen nennen keine Env-Variablen/Lizenzpfade; der Aktivierungs-Hint kommt vom Authorizer (W1-Mechanik). Neue Core-Texte dürfen "Professional" als Editionsnamen nennen (Präzedenz: README/Doku), aber keine Aktivierungs-Mechanik.
- **Zeilenlänge 100** (ruff), Kommentare/Docstrings erklären das *Warum* und referenzieren PRD-Abschnitte.
- Nach Änderungen an einer der beiden `pyproject.toml` müssen die editierbaren Installs aufgefrischt werden, sonst sieht die Entry-Point-Metadata den Umbau nicht (Task 3, Step 8).

---

### Task 1: Core-Model-Store `persistence/model_io.py`

Der Run-Store schreibt/liest `model.pkl` heute über die Exporter (`engine.py:233`, `run.py:262`). Das wird Core-Persistenz, damit der Exporter-Umzug (Task 3) Train/Predict nicht anfasst.

**Files:**
- Create: `src/surroforge/persistence/model_io.py`
- Create: `tests/test_model_io.py`
- Modify: `src/surroforge/persistence/__init__.py`
- Modify: `src/surroforge/engine.py:33-35` (Import) und `src/surroforge/engine.py:233` (Pickle-Write)
- Modify: `src/surroforge/core/run.py:261-266` (`Run.load_model`)

**Interfaces:**
- Consumes: `surroforge.trainers.base.BaseTrainer` (nur als Typ), `pickle` (stdlib).
- Produces: `dump_model(trainer: BaseTrainer, path: Path) -> Path` und `load_model(path: str | Path) -> BaseTrainer`, beide re-exportiert aus `surroforge.persistence`. Task 3 (Professional-`PickleExporter`) delegiert an `dump_model`.

- [ ] **Step 1: Failing Test schreiben**

Neue Datei `tests/test_model_io.py`:

```python
"""Run-store model serialisation (Wave 2: decoupled from the Professional exporters)."""

from __future__ import annotations

import numpy as np

from surroforge.persistence import dump_model, load_model
from surroforge.trainers.polynomial import PolynomialTrainer


def test_model_roundtrip_preserves_predictions(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (50, 3))
    y = x @ np.array([[1.0], [2.0], [3.0]]) + 0.5
    model = PolynomialTrainer(degree=2)
    model.fit(x, y)

    path = dump_model(model, tmp_path / "model.pkl")
    assert path.is_file()
    restored = load_model(path)
    assert np.allclose(restored.predict(x), model.predict(x))
```

- [ ] **Step 2: Test läuft rot**

Run: `.venv/bin/python -m pytest tests/test_model_io.py -v`
Expected: FAIL — `ImportError: cannot import name 'dump_model' from 'surroforge.persistence'`

- [ ] **Step 3: Modul implementieren**

Neue Datei `src/surroforge/persistence/model_io.py`:

```python
"""Run-store model serialisation (the ``model.pkl`` artifact).

Deliberately Core persistence, not an exporter: train writes and predict reads
the stored model regardless of which editions are installed (PRD 3.2: export is
Professional, the run store is not). The Professional ``pickle`` *exporter*
delegates to :func:`dump_model`, so a user-facing pickle export stays
byte-compatible with the run store.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from surroforge.trainers.base import BaseTrainer


def dump_model(trainer: BaseTrainer, path: Path) -> Path:
    """Write the fitted ``trainer`` to ``path`` (highest pickle protocol)."""
    path = Path(path)
    with open(path, "wb") as handle:
        pickle.dump(trainer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model(path: str | Path) -> BaseTrainer:
    """Load a trainer previously written by :func:`dump_model`."""
    with open(path, "rb") as handle:
        return pickle.load(handle)
```

In `src/surroforge/persistence/__init__.py` ergänzen (Import-Block und `__all__`, alphabetisch einsortiert):

```python
from surroforge.persistence.model_io import dump_model, load_model
```

und in `__all__`: `"dump_model",` (nach `"capture_environment"`) sowie `"load_model",` (nach `"is_locally_produced"`).

- [ ] **Step 4: Test läuft grün**

Run: `.venv/bin/python -m pytest tests/test_model_io.py -v`
Expected: PASS

- [ ] **Step 5: Engine auf `dump_model` umstellen**

In `src/surroforge/engine.py` den Import `from surroforge.exporters.pickle_exporter import PickleExporter` (Zeile 34) **entfernen** und die Persistence-Import-Zeile (35) erweitern (geklammert — einzeilig wären es 101 Zeichen, ruff-Limit ist 100):

```python
from surroforge.persistence import (
    capture_environment,
    dump_model,
    write_json,
    write_npz,
    write_yaml,
)
```

(Zeile 33 `from surroforge.exporters.onnx_exporter import ONNXExporter` bleibt — Task 2 entfernt sie.)

Zeile 233 ersetzen — alt:

```python
    PickleExporter().export(final_model, run_dir / MODEL_PICKLE, n_features=n_features)
```

neu:

```python
    dump_model(final_model, run_dir / MODEL_PICKLE)
```

- [ ] **Step 6: `Run.load_model` auf `load_model` umstellen**

In `src/surroforge/core/run.py:261-266` — alt:

```python
    def load_model(self) -> Any:
        from surroforge.exporters.pickle_exporter import load_pickle

        self._require()
        _warn_if_foreign_artifact(self.model_path, self.lineage)
        return load_pickle(self.model_path)
```

neu (lazy Import wie bisher; das lokale `load_model` ist die Modul-Funktion, kein Rekursions-Selbstbezug):

```python
    def load_model(self) -> Any:
        from surroforge.persistence.model_io import load_model

        self._require()
        _warn_if_foreign_artifact(self.model_path, self.lineage)
        return load_model(self.model_path)
```

- [ ] **Step 7: Core-Suite läuft grün**

Run: `.venv/bin/python -m pytest tests -q`
Expected: alle PASS (Verhalten unverändert; `tests/test_exporters.py` läuft noch gegen die Core-Exporter, die bis Task 3 bleiben).

- [ ] **Step 8: Commit**

```bash
git add src/surroforge/persistence/model_io.py src/surroforge/persistence/__init__.py src/surroforge/engine.py src/surroforge/core/run.py tests/test_model_io.py
git commit -m "feat(persistence): model_io run-store serialisation, decoupled from exporters"
```

---

### Task 2: `train --onnx` durch den Seam (reines Refactoring)

Der Train-Zeit-ONNX-Export umgeht den Seam per Direktimport (`engine.py:33` + `285`). Umstellung auf `get_exporter("onnx")` **innerhalb** des bestehenden best-effort try/except. In diesem Task sind die Exporter noch Core-Builtins ohne `required_tier` → Verhalten identisch, bestehende Tests bleiben das Netz. Das Gating-Verhalten entsteht in Task 3 und wird in Task 4 getestet.

**Files:**
- Modify: `src/surroforge/engine.py:33` (Import) und `src/surroforge/engine.py:282-288` (ONNX-Block)
- Modify: `src/surroforge/cli/main.py:25-26` (Imports) und `src/surroforge/cli/main.py:343-346` (Post-Train-Hinweis)

**Interfaces:**
- Consumes: `surroforge.plugins.get_exporter(name) -> type` (wirft `PluginError` bei unbekanntem Format oder — ab Task 3 — fehlender Lizenz).
- Produces: `run_training(..., export_onnx=True)` löst `onnx` über den Seam auf; jede Auflösungs-Exception wird zur Log-Zeile `ONNX export skipped: {exc}` (Run läuft durch, `lineage["onnx_exported"] is False`). Der CLI-Hinweis `ONNX export available — …` erscheint nur, wenn `get_exporter("onnx")` auflösbar ist.

- [ ] **Step 1: Engine umstellen**

In `src/surroforge/engine.py` den Import `from surroforge.exporters.onnx_exporter import ONNXExporter` (Zeile 33) **entfernen** und Zeile 36 erweitern:

```python
from surroforge.plugins import get_exporter, get_trainer
```

Block Zeile 282-288 — alt:

```python
    onnx_exported = False
    if export_onnx and final_model.supports_export("onnx"):
        try:
            ONNXExporter().export(final_model, run_dir / MODEL_ONNX, n_features=n_features)
            onnx_exported = True
        except Exception as exc:  # ONNX is best-effort; never fail a run over it
            log(f"ONNX export skipped: {exc}")
```

neu:

```python
    onnx_exported = False
    if export_onnx and final_model.supports_export("onnx"):
        try:
            # Resolved through the plugin seam: since Wave 2 the ONNX exporter is
            # Professional, so resolution itself may raise (unlicensed or not
            # installed) — the same best-effort skip covers that (PRD 3.2).
            exporter = get_exporter("onnx")()
            exporter.export(final_model, run_dir / MODEL_ONNX, n_features=n_features)
            onnx_exported = True
        except Exception as exc:  # ONNX is best-effort; never fail a run over it
            log(f"ONNX export skipped: {exc}")
```

- [ ] **Step 2: CLI-Post-Train-Hinweis absichern**

In `src/surroforge/cli/main.py` die Imports erweitern — Zeile 25/26 alt:

```python
from surroforge.errors import DataError, SurroForgeError
from surroforge.plugins import available_commands, get_trainer
```

neu:

```python
from surroforge.errors import DataError, PluginError, SurroForgeError
from surroforge.plugins import available_commands, get_exporter, get_trainer
```

Hinweis-Block Zeile 343-346 — alt:

```python
    # Discoverability: ONNX is now opt-in, so point users at the on-demand path
    # when the trained method actually supports it (e.g. polynomial).
    if not onnx and "onnx" in getattr(get_trainer(cfg.model.method), "export_formats", frozenset()):
        _info(f"  ONNX export available — run 'surroforge export {run.run_id} --format onnx'.")
```

neu:

```python
    # Discoverability: ONNX is now opt-in, so point users at the on-demand path
    # when the trained method supports it AND the exporter actually resolves —
    # since Wave 2 it is Professional; don't advertise a command that would fail.
    if not onnx and "onnx" in getattr(get_trainer(cfg.model.method), "export_formats", frozenset()):
        try:
            get_exporter("onnx")
        except PluginError:
            pass
        else:
            _info(f"  ONNX export available — run 'surroforge export {run.run_id} --format onnx'.")
```

- [ ] **Step 3: Suiten laufen grün (Refactoring-Nachweis)**

Run: `.venv/bin/python -m pytest tests -q && .venv/bin/ruff check src tests && .venv/bin/mypy src`
Expected: alle PASS / keine Findings (Exporter sind noch ungegatete Builtins — identisches Verhalten).

- [ ] **Step 4: Commit**

```bash
git add src/surroforge/engine.py src/surroforge/cli/main.py
git commit -m "refactor(engine,cli): resolve the train-time ONNX exporter through the plugin seam"
```

---

### Task 3: Exporter-Umzug nach surroforge_professional + Gating (atomarer Migrations-Commit)

Der eigentliche Umzug. Zwischenzustände sind zwangsläufig rot (Core-Tests importieren die Exporter-Module), deshalb **ein** Commit am Ende. Muster ist der W1-Methoden-Umzug.

**Files:**
- git mv: `src/surroforge/exporters/pickle_exporter.py` → `plugins/surroforge_professional/src/surroforge_professional/exporters/pickle_exporter.py`
- git mv: `src/surroforge/exporters/onnx_exporter.py` → `plugins/surroforge_professional/src/surroforge_professional/exporters/onnx_exporter.py`
- git mv: `plugins/surroforge_professional/src/surroforge_professional/pt_exporter.py` → `plugins/surroforge_professional/src/surroforge_professional/exporters/pt_exporter.py`
- git mv: `tests/test_exporters.py` → `plugins/surroforge_professional/tests/test_exporters.py`
- Create: `plugins/surroforge_professional/src/surroforge_professional/exporters/__init__.py`
- Modify: die drei verschobenen Exporter (`required_tier`, Delegation, Docstrings)
- Modify: `src/surroforge/exporters/__init__.py` (Docstring), `src/surroforge/plugins/registry.py:39-42` (`_BUILTIN`)
- Modify: `pyproject.toml:80-82` (Exporter-EPs raus), `plugins/surroforge_professional/pyproject.toml` (EPs + `[onnx]`-Extra)
- Modify: `tests/conftest.py` (Fixture `licensed_export`), `tests/test_plugins.py:8-16,42-46`, `tests/test_cli.py:81`, `tests/test_scenarios.py:23-24`, `tests/test_training.py:95`
- Modify: `plugins/surroforge_professional/tests/test_gating.py` (Exporter-Gating-Tests)

**Interfaces:**
- Consumes: `surroforge.persistence.model_io.dump_model` (Task 1); W1-Seam (`required_tier`-Klassenattribut, Authorizer-Gate); `surroforge.exporters.base.BaseExporter` (bleibt Core).
- Produces: `surroforge_professional.exporters.pickle_exporter.PickleExporter`, `…onnx_exporter.ONNXExporter`, `…pt_exporter.PtExporter` — alle mit `required_tier = "professional"`, registriert als Entry-Points `pickle`/`onnx`/`pt`; Core-Fixture `licensed_export` (skippt ohne installiertes professional, setzt sonst `SURROFORGE_PRO_LICENSE`); `_BUILTIN[EXPORTER_GROUP] == {}`.

- [ ] **Step 1: Dateien verschieben**

```bash
mkdir -p plugins/surroforge_professional/src/surroforge_professional/exporters
git mv src/surroforge/exporters/pickle_exporter.py plugins/surroforge_professional/src/surroforge_professional/exporters/pickle_exporter.py
git mv src/surroforge/exporters/onnx_exporter.py plugins/surroforge_professional/src/surroforge_professional/exporters/onnx_exporter.py
git mv plugins/surroforge_professional/src/surroforge_professional/pt_exporter.py plugins/surroforge_professional/src/surroforge_professional/exporters/pt_exporter.py
git mv tests/test_exporters.py plugins/surroforge_professional/tests/test_exporters.py
```

- [ ] **Step 2: Exporter-Unterpaket anlegen und die drei Module anpassen**

Neue Datei `plugins/surroforge_professional/src/surroforge_professional/exporters/__init__.py`:

```python
"""Licensed model exporters (pickle, ONNX, PyTorch .pt) — Professional tier (Wave 2)."""
```

`exporters/pickle_exporter.py` vollständig ersetzen durch:

```python
"""Pickle exporter — Professional since Wave 2 (PRD 3.2, 4.4).

Delegates to Core's run-store serialisation (``persistence.model_io``), so a
user-facing pickle export stays byte-compatible with the ``model.pkl`` the run
store writes at train time. The gate sits in registry resolution (W1 seam);
the class itself carries no licence logic.
"""

from __future__ import annotations

from pathlib import Path

from surroforge.exporters.base import BaseExporter
from surroforge.persistence.model_io import dump_model
from surroforge.trainers.base import BaseTrainer


class PickleExporter(BaseExporter):
    name = "pickle"
    extension = ".pkl"
    required_tier = "professional"

    def export(self, trainer: BaseTrainer, destination: Path, *, n_features: int) -> Path:
        return dump_model(trainer, Path(destination))
```

(Die Modul-Funktion `load_pickle` **entfällt** — Ersatz ist `surroforge.persistence.model_io.load_model`; einzige Nutzer waren `run.py` [Task 1] und die mitgezogenen Tests [Step 6].)

`exporters/onnx_exporter.py`: Docstring-Kopfzeile ersetzen (Zeile 1) durch
`"""ONNX exporter — Professional since Wave 2 (PRD 3.2, 4.4).` und nach `extension = ".onnx"` einfügen:

```python
    required_tier = "professional"
```

`exporters/pt_exporter.py`: nach `extension = ".pt"` einfügen:

```python
    required_tier = "professional"
```

und den Modul-Docstring ersetzen durch:

```python
"""Native PyTorch ``.pt`` exporter (needs the [torch] extra) — licensed since Wave 2.

Registered through the ``surroforge.exporters`` entry point (never Core's
``_BUILTIN`` table). It mirrors the ONNX exporter: it delegates to the trainer's
:meth:`to_pt`, so only a method that declares ``pt`` in ``export_formats`` (today,
``torch-mlp``) can be exported this way. The module import stays torch-free, so
the licence seam runs before any torch import.
"""
```

- [ ] **Step 3: Core-Seite ausdünnen**

`src/surroforge/exporters/__init__.py` — Docstring-Zeile 1 ersetzen:

```python
"""Exporter interface. The concrete exporters (pickle/onnx/pt) are Professional (Wave 2)."""
```

(Rest der Datei — `BaseExporter`-Re-Export — bleibt.)

`src/surroforge/plugins/registry.py:39-42` — alt:

```python
    EXPORTER_GROUP: {
        "pickle": "surroforge.exporters.pickle_exporter:PickleExporter",
        "onnx": "surroforge.exporters.onnx_exporter:ONNXExporter",
    },
```

neu:

```python
    # Core ships no exporters since Wave 2 (PRD 3.2): pickle/onnx/pt are licensed
    # and registered by surroforge_professional. The run store writes/reads
    # model.pkl through persistence.model_io, independent of any exporter.
    EXPORTER_GROUP: {},
```

- [ ] **Step 4: Entry-Points umbauen**

`pyproject.toml` (Root): den Block Zeilen 80-82 **ersatzlos streichen**:

```toml
[project.entry-points."surroforge.exporters"]
pickle = "surroforge.exporters.pickle_exporter:PickleExporter"
onnx = "surroforge.exporters.onnx_exporter:ONNXExporter"
```

`plugins/surroforge_professional/pyproject.toml`: den Exporter-EP-Block (Zeilen 33-34) ersetzen — alt:

```toml
[project.entry-points."surroforge.exporters"]
pt = "surroforge_professional.pt_exporter:PtExporter"
```

neu:

```toml
[project.entry-points."surroforge.exporters"]
pickle = "surroforge_professional.exporters.pickle_exporter:PickleExporter"
onnx = "surroforge_professional.exporters.onnx_exporter:ONNXExporter"
pt = "surroforge_professional.exporters.pt_exporter:PtExporter"
```

und das `[onnx]`-Extra (Zeile 25) erweitern — alt:

```toml
onnx = ["onnx>=1.15", "onnxruntime>=1.17"]
```

neu (mlp/random-forest `to_onnx` brauchen `skl2onnx`; bisher kam es implizit über Cores `[onnx]`-Extra):

```toml
onnx = ["skl2onnx>=1.16", "onnx>=1.15", "onnxruntime>=1.17"]
```

- [ ] **Step 5: Core-Fixture `licensed_export` + Anpassung der drei Export-Tests**

In `tests/conftest.py` nach dem `trained_run`-Fixture einfügen:

```python
@pytest.fixture
def licensed_export(monkeypatch):
    """Grant the stub Professional licence for tests that export models (Wave 2).

    Skips cleanly on a Core-only checkout: without surroforge_professional there
    is no exporter to license — those paths are covered by the plugin's own suite.
    """
    pytest.importorskip("surroforge_professional")
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "test-license")
```

Drei Signaturen erweitern (nur den Parameter ergänzen, Testkörper unverändert):

- `tests/test_cli.py:81`: `def test_export_and_report(tmp_path):` → `def test_export_and_report(tmp_path, licensed_export):`
- `tests/test_scenarios.py:23`: `def test_scenario_greenfield(tmp_path):` → `def test_scenario_greenfield(tmp_path, licensed_export):` — und unter dem Docstring (Zeile 24) die Zeile ergänzen: `# Export is Professional since Wave 2; the rest of the scenario is Core.`
- `tests/test_training.py:95`: `def test_onnx_exported_when_requested(project):` → `def test_onnx_exported_when_requested(project, licensed_export):`

(`test_onnx_export_is_opt_in_by_default` exportiert nicht → bleibt lizenz-frei.)

- [ ] **Step 6: Verschobene Exporter-Tests umschreiben**

`plugins/surroforge_professional/tests/test_exporters.py` vollständig ersetzen durch (nutzt die dortigen Fixtures `licensed`/`project`/`trained_run` aus `plugins/surroforge_professional/tests/conftest.py`; Direkt-Instanzierung umgeht den Seam bewusst — das Gate sitzt in der Registry-Auflösung):

```python
"""Model export: pickle round-trip and ONNX (incl. runtime parity) (PRD 4.4).

Core tests until Wave 2; the exporters are Professional now, so the suite lives
here. Direct class instantiation bypasses the licence seam by design (the gate
sits in registry resolution); everything going through ``export_model`` /
``Run.export`` runs with the ``licensed`` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from surroforge.core.config import parse_config
from surroforge.engine import run_training
from surroforge.errors import ExportError
from surroforge.persistence import load_model
from surroforge.services import export_model
from surroforge.trainers.kriging import KrigingTrainer
from surroforge.trainers.polynomial import PolynomialTrainer
from surroforge_professional.exporters.onnx_exporter import ONNXExporter
from surroforge_professional.exporters.pickle_exporter import PickleExporter


def _fitted_polynomial(n_features=3):
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (50, n_features))
    y = x @ np.arange(1, n_features + 1).reshape(-1, 1) + 0.5
    model = PolynomialTrainer(degree=2)
    model.fit(x, y)
    return model, x


def _kriging_run(project):
    pytest.importorskip("smt")
    config = parse_config(
        {
            "experiment": "beam-kriging",
            "data": {
                "source": "beam",
                "inputs": ["length", "width", "load"],
                "outputs": ["max_stress"],
                "split": {"test_fraction": 0.2, "seed": 42},
            },
            "model": {"method": "kriging"},
            "validation": {"cross_validation": 5, "metrics": ["rmse", "r2", "mae"]},
        }
    )
    return run_training(project, config)


def test_pickle_roundtrip_preserves_predictions(tmp_path):
    model, x = _fitted_polynomial()
    path = PickleExporter().export(model, tmp_path / "m.pkl", n_features=3)
    assert path.is_file()
    restored = load_model(path)
    assert np.allclose(restored.predict(x), model.predict(x))


def test_onnx_export_and_runtime_parity(tmp_path):
    pytest.importorskip("onnxruntime")
    model, x = _fitted_polynomial()
    path = ONNXExporter().export(model, tmp_path / "m.onnx", n_features=3)
    assert path.is_file()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path))
    onnx_out = sess.run(None, {"input": x.astype(np.float32)})[0]
    # ONNX runs in float32, so compare with a tolerance.
    assert np.allclose(onnx_out.reshape(-1), model.predict(x).reshape(-1), atol=1e-3)


def test_onnx_export_rejects_unsupported_method(tmp_path):
    pytest.importorskip("smt")
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (30, 1))
    model = KrigingTrainer()
    model.fit(x, 2 * x)
    with pytest.raises(ExportError):
        ONNXExporter().export(model, tmp_path / "m.onnx", n_features=1)


def test_export_model_service_default_path(licensed, trained_run):
    # No explicit output -> artifact lands inside the run directory.
    path = export_model(trained_run, "pickle")
    assert path == trained_run.run_dir / "model.pkl"
    assert path.is_file()


def test_run_export_method_binds_to_service(licensed, trained_run):
    # Run.export is the object-level binding of export_model: same default path/result.
    path = trained_run.export("pickle")
    assert path == trained_run.run_dir / "model.pkl"
    assert path.is_file()
    restored = load_model(path)
    assert restored is not None


def test_run_export_propagates_gate_error(licensed, project):
    # The binding carries the declaration-driven gate: kriging cannot export to ONNX.
    run = _kriging_run(project)
    with pytest.raises(ExportError):
        run.export("onnx")


def test_export_model_service_onnx_for_kriging_errors(licensed, project):
    run = _kriging_run(project)
    with pytest.raises(ExportError):
        export_model(run, "onnx")
```

- [ ] **Step 7: Registry-Tests in Core umbauen + Exporter-Gating-Tests in professional**

`tests/test_plugins.py`: im Import-Block (Zeilen 8-16) `available_exporters,` und `get_exporter,` **streichen** (verbleiben: `available_reporters`, `available_trainers`, `get_comparison_reporter`, `get_reporter`, `get_trainer`). Die Funktion `test_exporters_and_reporters_registered` (Zeilen 42-46) ersetzen durch:

```python
def test_reporters_registered():
    assert "markdown" in available_reporters()
    assert get_reporter("markdown").name == "markdown"


def test_exporters_not_in_core_builtin_table():
    # pickle/onnx moved to surroforge_professional (Wave 2); Core's fallback table
    # must not resurrect them for a source checkout.
    from surroforge.plugins.registry import _BUILTIN, EXPORTER_GROUP

    assert _BUILTIN[EXPORTER_GROUP] == {}


def test_core_only_install_has_no_exporters(monkeypatch):
    # Without any edition installed there is no export format at all — and the
    # error is the editions-neutral "Unknown export format", not a licence pitch.
    from surroforge.plugins import registry

    monkeypatch.setattr(registry, "_entry_points_grouped", lambda group: {})
    assert registry.available_exporters() == []
    with pytest.raises(PluginError, match="Unknown export format"):
        registry.get_exporter("pickle")
```

`plugins/surroforge_professional/tests/test_gating.py`: Import-Zeile 8 erweitern zu

```python
from surroforge.plugins import available_exporters, available_trainers, get_exporter, get_trainer
```

und am Dateiende anhängen:

```python
# --- Exporters (Wave 2) --------------------------------------------------------


@pytest.mark.parametrize("fmt", ["pickle", "onnx", "pt"])
def test_exporter_denied_without_license(unlicensed, fmt):
    with pytest.raises(PluginError) as err:
        get_exporter(fmt)
    assert "'professional' license" in str(err.value)
    assert "SURROFORGE_PRO_LICENSE" in str(err.value)  # hint from the authorizer


@pytest.mark.parametrize("fmt", ["pickle", "onnx", "pt"])
def test_exporter_resolves_with_license(licensed, fmt):
    cls = get_exporter(fmt)
    assert cls.name == fmt
    assert cls.required_tier == "professional"


def test_exporters_listed_via_entry_points(licensed):
    assert {"pickle", "onnx", "pt"} <= set(available_exporters())
```

- [ ] **Step 8: Editierbare Installs auffrischen (Entry-Point-Metadata)**

```bash
.venv/bin/pip install -e . --no-deps
.venv/bin/pip install -e plugins/surroforge_professional --no-deps
```

Expected: beide enden mit `Successfully installed …`.

- [ ] **Step 9: Alle Suiten laufen grün**

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest plugins/surroforge_professional/tests -q
.venv/bin/python -m pytest plugins/surroforge_enterprise/tests -q
```

Expected: PASS überall (enterprise unberührt, mitgeprüft wegen der EP-Umbauten).

- [ ] **Step 10: Ein atomarer Commit**

```bash
git add src/surroforge/exporters/__init__.py src/surroforge/plugins/registry.py pyproject.toml \
  src/surroforge/exporters/pickle_exporter.py src/surroforge/exporters/onnx_exporter.py \
  plugins/surroforge_professional/src/surroforge_professional/exporters \
  plugins/surroforge_professional/src/surroforge_professional/pt_exporter.py \
  plugins/surroforge_professional/pyproject.toml plugins/surroforge_professional/tests/test_exporters.py \
  plugins/surroforge_professional/tests/test_gating.py \
  tests/conftest.py tests/test_plugins.py tests/test_cli.py tests/test_scenarios.py \
  tests/test_training.py tests/test_exporters.py
git commit -m "feat(editions)!: pickle/onnx exporters move to surroforge_professional, licence-gated (Wave 2)"
```

(Die `git mv`-Quellpfade sind durch den mv bereits im Index; die obige Liste deckt Quell- und Zielseite der Renames ab. `git status` vor dem Commit prüfen: es darf nichts Ungelistetes geben.)

---

### Task 4: Verhaltens-Tests — sanfter Skip + unterdrückter Hinweis

Zwei neue Tests, die den W2-Kontrakt festschreiben. Sie müssen **sofort grün** sein (das Verhalten existiert seit Task 2+3) — sie sind das Regressionsschloss, nicht TDD-rot.

**Files:**
- Modify: `tests/test_training.py` (nach `test_onnx_exported_when_requested`)
- Modify: `tests/test_cli.py` (nach `test_export_and_report`)

**Interfaces:**
- Consumes: Task-2-Verhalten (`ONNX export skipped: {exc}`-Log, Hint-Guard), Task-3-Gating; Fixtures `project`/`poly_config` (Core-conftest), Helfer `_init_project_with_data`/`_write_config` (in `tests/test_cli.py` vorhanden), CLI-`runner`/`app` (ebd.).
- Produces: nichts für spätere Tasks — reine Regressionstests.

- [ ] **Step 1: Engine-Skip-Test schreiben**

In `tests/test_training.py` direkt nach `test_onnx_exported_when_requested` (Zeile 99) einfügen:

```python
def test_train_onnx_without_license_skips_gracefully(project, monkeypatch, tmp_path):
    # Wave 2: the ONNX exporter is Professional. An unlicensed train --onnx must
    # not fail the run — the best-effort block logs a skip instead. The assertion
    # sticks to the skip prefix so it holds both with surroforge_professional
    # installed (licence message) and without (unknown-format message).
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-licence-home"))
    run = run_training(project, parse_config(poly_config("polynomial")), export_onnx=True)
    assert not run.has_onnx()
    assert run.lineage["onnx_exported"] is False
    assert "ONNX export skipped:" in (run.run_dir / "log.txt").read_text()
```

(`HOME` wird wie in den W1-Gating-Tests umgebogen, damit eine echte `~/.surroforge/professional.lic` auf der Entwickler-Maschine den Test nicht unterläuft.)

- [ ] **Step 2: CLI-Hint-Tests schreiben**

In `tests/test_cli.py` direkt nach `test_export_and_report` einfügen:

```python
def test_train_hint_suppressed_when_exporter_unlicensed(tmp_path, monkeypatch):
    # Wave 2: don't advertise 'surroforge export … --format onnx' when resolving
    # the exporter would fail (unlicensed or professional not installed).
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-licence-home"))
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)

    result = runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])

    assert result.exit_code == 0
    assert "ONNX export available" not in result.output


def test_train_hint_appears_when_exporter_licensed(tmp_path, licensed_export):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)

    result = runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])

    assert result.exit_code == 0
    assert "ONNX export available" in result.output
```

- [ ] **Step 3: Tests laufen grün**

Run: `.venv/bin/python -m pytest tests/test_training.py tests/test_cli.py -q`
Expected: PASS (inkl. der drei neuen Tests; sollten sie rot sein, ist das ein echter Task-2/3-Defekt — nicht die Tests anpassen, sondern den Defekt melden).

- [ ] **Step 4: Commit**

```bash
git add tests/test_training.py tests/test_cli.py
git commit -m "test: lock in Wave-2 export gating (graceful train skip, guarded CLI hint)"
```

---

### Task 5: Texte, Doku, Beispiele, MEMORY, CHANGELOG

Alle Nutzer-sichtbaren Texte auf den W2-Stand. Keine Verhaltensänderung außer zwei Strings (CLI-Hilfetext, Beispiel 01).

**Files:**
- Modify: `src/surroforge/cli/main.py:525` (`--format`-Hilfetext), `src/surroforge/core/run.py:268-277` (`Run.export`-Docstring)
- Modify: `examples/01_quickstart.py:24-30,85-88`, `examples/06_mlp_surrogate.py:18-20,99-107`, `examples/09_naca0012_openfoam/README.md:64`
- Modify: `docs/quickstart.md:117-125`, `docs/reference/cli.md:33-39,59-62`, `docs/reference/python-api.md:60,78,91`, `docs/install.md:8,31`, `docs/examples.md:9,15,20`, `README.md:23,53`
- Modify: `MEMORY.md:41,43,108-109`, `CHANGELOG.md` (Unreleased → Editionen-Block)

**Interfaces:**
- Consumes: Task-3-Endzustand (Exporter Professional, `model_io` Core).
- Produces: nichts für spätere Tasks — Doku-Schlussstand von Welle 2.

- [ ] **Step 1: CLI-Hilfetext und `Run.export`-Docstring**

`src/surroforge/cli/main.py:525` — alt:

```python
    format: str = typer.Option("onnx", "--format", "-f", help="Export format: onnx or pickle."),
```

neu:

```python
    format: str = typer.Option(
        "onnx",
        "--format",
        "-f",
        help="Export format — pickle/onnx need a Professional license; pt also the [torch] extra.",
    ),
```

`src/surroforge/core/run.py` — im `Run.export`-Docstring alles ab „``fmt`` is any registered export format" bis zum Docstring-Ende (Zeilen 273-277) ersetzen durch:

```python
        ``fmt`` is any registered export format the run's method declares. Since the
        Wave-2 editions split every export format is provided by a plugin: ``pickle``
        and ``onnx`` come with ``surroforge_professional`` (licensed), ``pt``
        additionally needs its ``[torch]`` extra. An unsupported pair raises
        :class:`~surroforge.errors.ExportError`. ``output`` defaults to ``model<ext>``
        inside the run directory.
        """
```

- [ ] **Step 2: Beispiele 01, 06 und 09-README**

`examples/01_quickstart.py`: im Import-Block (Zeilen 25-30) die Zeile `export_model,` streichen. Schritt 7 (Zeilen 85-88) — alt:

```python
    # 7. Export the trained model. Pickle is always available; ONNX is declared
    #    per method (polynomial supports it, with the `onnx` extra installed).
    model_path = export_model(run, "pickle")
    print(f"\nExported model -> {model_path.name}")
```

neu:

```python
    # 7. The fitted model is already persisted in the run store — predict works
    #    from it directly. Standalone export (pickle/ONNX via `export_model` or
    #    `surroforge export`) is a Professional feature since the editions split.
    print(f"\nStored model artifact -> {run.model_path.name}")
```

`examples/06_mlp_surrogate.py`: Docstring-Zeilen 18-20 — alt:

```
Self-contained: synthesises its own data in a temp dir and cleans up. ONNX
export uses the optional `[onnx]` extra and degrades gracefully to pickle
when it is absent.
```

neu:

```
Self-contained: synthesises its own data in a temp dir and cleans up. Export is
Professional (like `mlp` itself); ONNX conversion uses the plugin's `[onnx]`
extra (`pip install 'surroforge_professional[onnx]'`) and degrades gracefully
to a pickle export when the conversion stack is absent.
```

und Zeile 107 — alt: `        print("Install the extra for ONNX: pip install 'surroforge[onnx]'")` — neu:

```python
        print("Install the extra for ONNX: pip install 'surroforge_professional[onnx]'")
```

`examples/09_naca0012_openfoam/README.md:64`: im Satz „`ingest_and_train.py` also **exports** the trained surrogates: …" direkt nach `the trained surrogates` einfügen: ` (model export needs the Professional license already required for the NN candidates)`.

- [ ] **Step 3: Doku-Site**

`docs/quickstart.md` §8 (Zeilen 117-125) ersetzen durch (äußere Fence hier 4 Backticks, weil der Ersatztext selbst einen bash-Block enthält):

````markdown
## 8. Export the model

Standalone model export is a **Professional** feature — the trained model itself
is already stored in the run directory, so predict/report/reproduce work without
any export:

```bash
surroforge export run-001 --format pickle           # Professional license required
surroforge export run-001 --format onnx             # polynomial only (needs the onnx stack)
```

A method that does not support a format fails with a clear message — export support
is declared per method, not guessed.
````

`docs/reference/cli.md:33-39` — den `train`-Absatz ersetzen durch:

```markdown
### `surroforge train -c <config.yaml> [--onnx]`
Train a surrogate from a YAML [configuration](configuration.md). Prints the new run
id and a train/test/CV metrics summary. `--config/-c` is required. The model is
always saved in the run store; ONNX export is **opt-in** with `--onnx` (it pulls in
and runs `skl2onnx`, the dominant cost of a cheap method) — otherwise produce it
later on demand with `surroforge export <run> --format onnx`. Model export is a
Professional feature: without a license, `--onnx` logs `ONNX export skipped: …`
and the run still completes.
```

`docs/reference/cli.md:59-62` — den `export`-Absatz ersetzen durch:

```markdown
### `surroforge export <run_id> [-f FORMAT] [-o OUTPUT]`
Export a run's model — a **Professional** feature (licensed). `--format/-f` is
`onnx` (default), `pickle`, or a plugin format such as `pt`; ONNX only for methods
that declare it (e.g. polynomial, mlp). An unsupported method/format pair fails
with a clear message. The run store's own `model.pkl` is written at train time
regardless — predict never needs an export.
```

`docs/reference/python-api.md`:
- Zeile 60 (`fit`-Zeile): `ONNX export is opt-in (`export_onnx=True`); otherwise produce it later from the saved model via `export_model`.` ersetzen durch `ONNX export is opt-in (`export_onnx=True`) and needs a Professional license (unlicensed, the export is skipped and the run completes); otherwise produce it later via `export_model`.`
- Zeile 78 (`has_onnx()`-Zeile): unverändert lassen.
- Zeile 91 (`export_model`-Zeile) ersetzen durch: `| `export_model(run, fmt, output=None) -> Path` | Export to `pickle`/`onnx`/a plugin format — Professional (licensed); clear `ExportError` for unsupported pairs. |`

`docs/install.md`: Zeile 8 — alt `pip install 'surroforge[onnx]'    # + ONNX export` → neu `pip install 'surroforge[onnx]'    # + ONNX conversion backend (export is Professional)`; Zeile 31 — Beschreibung der `onnx`-Extra-Zeile von `ONNX model export` → `ONNX conversion backend (model export itself is Professional)`.

`docs/examples.md`: Zeile 9 — `[onnx] for ONNX export` → `[onnx] for the ONNX conversion backend`; Zeile 15 (Beispiel 01) — Beschreibung `… train → predict → export → report → reproduce` → `… train → predict → report → reproduce` (Export ist aus 01 raus); Zeile 20 (Beispiel 06) — `ONNX export adds `[onnx]`` → `ONNX export adds `surroforge_professional[onnx]``.

`README.md`: Zeile 23 — Kommentar `# + ONNX export` → `# + ONNX conversion backend (export is Professional)`; Zeile 53 — `surroforge export run-001 --format onnx` → `surroforge export run-001 --format onnx                    # Professional: model export`.

- [ ] **Step 4: MEMORY.md und CHANGELOG**

`MEMORY.md`:
- Zeile 41 — alt `├── exporters/       # base, pickle (immer), onnx; pt kommt über surroforge_professional[torch]` → neu `├── exporters/       # nur base (Schnittstelle); pickle/onnx/pt sind Professional (Welle 2)`
- Zeile 43 — alt `├── persistence/     # store: atomare I/O, Run-IDs, Environment-Capture` → neu `├── persistence/     # store: atomare I/O, Run-IDs, Environment-Capture; model_io: Run-Store-Pickle`
- Zeilen 108-109: in der Aufzählung offener Punkte `Export-Gating (Welle 2),` streichen (es bleiben skalierbare GP, reproduce/Lineage-Gating (Welle 3), echte Lizenz-Aktivierung (M3), Support-Infrastruktur).

`CHANGELOG.md`: im Abschnitt `### Changed — Editionen (Breaking)` nach dem `autoencoder-field`-Bullet einfügen:

```markdown
- **Modell-Export ist Professional (Welle 2):** die Exporter `pickle` und `onnx` leben jetzt
  (wie `pt`) lizenzpflichtig in `surroforge_professional`; `surroforge export`, `Run.export`
  und `export_model` verlangen eine Lizenz. `train --onnx` ohne Lizenz überspringt den
  Export sanft (`ONNX export skipped: …`) und lässt den Run durchlaufen. Der Run-Store
  selbst (`model.pkl` schreiben/lesen für predict) bleibt frei in Core
  (`surroforge.persistence.model_io`); das interne `load_pickle` ist durch
  `persistence.model_io.load_model` ersetzt.
```

- [ ] **Step 5: Gesamte Quality-Gates**

Run: `scripts/ci.sh`
Expected: alle Gates grün (ruff, mypy, pyright, import-linter, pytest Core + `plugins/*/tests`, inkl. `tests/test_examples.py`, das die Beispiele mit injizierter Lizenz fährt — Beispiel 01 läuft jetzt auch ohne).

- [ ] **Step 6: Commit**

```bash
git add src/surroforge/cli/main.py src/surroforge/core/run.py \
  examples/01_quickstart.py examples/06_mlp_surrogate.py examples/09_naca0012_openfoam/README.md \
  docs/quickstart.md docs/reference/cli.md docs/reference/python-api.md docs/install.md \
  docs/examples.md README.md MEMORY.md CHANGELOG.md
git commit -m "docs(editions): Wave-2 export gating sweep (help texts, docs, examples, memory, changelog)"
```
