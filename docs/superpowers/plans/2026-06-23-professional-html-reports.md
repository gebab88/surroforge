# Professional HTML Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an installable `surroforge-professional` plugin that registers a licensed static HTML reporter.

**Architecture:** Keep SurroForge Core unchanged. Add a self-contained plugin under `plugins/surroforge-professional/` that registers `html` through `surroforge.reporters`, checks a local Professional license gate, builds a plugin-private report context from `Run`, and renders a single-file escaped HTML document.

**Tech Stack:** Python 3.10+, hatchling, pytest, Jinja2, SurroForge plugin entry points, existing `Run` and `generate_report` APIs.

---

## File Structure

Create these files:

- `plugins/surroforge-professional/pyproject.toml`: package metadata, editable-install support, and `surroforge.reporters` entry point.
- `plugins/surroforge-professional/README.md`: install, license, CLI/API usage, and v1 scope.
- `plugins/surroforge-professional/src/surroforge_professional/__init__.py`: package version export.
- `plugins/surroforge-professional/src/surroforge_professional/license.py`: local license gate.
- `plugins/surroforge-professional/src/surroforge_professional/reports/__init__.py`: reporter package marker.
- `plugins/surroforge-professional/src/surroforge_professional/reports/context.py`: plugin-private `Run` to template-context conversion.
- `plugins/surroforge-professional/src/surroforge_professional/reports/html.py`: `HtmlReporter` and static Jinja HTML template.
- `plugins/surroforge-professional/tests/conftest.py`: plugin-local fixtures for scalar training runs and license setup.
- `plugins/surroforge-professional/tests/test_registration.py`: reporter entry-point registration.
- `plugins/surroforge-professional/tests/test_license.py`: local license gate tests.
- `plugins/surroforge-professional/tests/test_html_report.py`: scalar HTML content, escaping, missing-data, and service integration tests.
- `plugins/surroforge-professional/tests/test_field_degradation.py`: Field-ROM optional section and degradation tests.

Do not modify Core source files for v1. Do not stage `.superpowers/` files.

## Task 1: Scaffold Plugin Package And Entry-Point Registration

**Files:**
- Create: `plugins/surroforge-professional/pyproject.toml`
- Create: `plugins/surroforge-professional/src/surroforge_professional/__init__.py`
- Create: `plugins/surroforge-professional/src/surroforge_professional/reports/__init__.py`
- Create: `plugins/surroforge-professional/src/surroforge_professional/reports/html.py`
- Create: `plugins/surroforge-professional/tests/test_registration.py`

- [ ] **Step 1: Write the failing registration test**

Create `plugins/surroforge-professional/tests/test_registration.py`:

```python
from __future__ import annotations

from surroforge.plugins import available_reporters, get_reporter
from surroforge_professional.reports.html import HtmlReporter


def test_html_reporter_registered():
    assert "html" in available_reporters()
    assert get_reporter("html") is HtmlReporter
```

- [ ] **Step 2: Run the test to verify it fails before scaffold**

Run:

```bash
.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_registration.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'surroforge_professional'`.

- [ ] **Step 3: Create package metadata and minimal reporter**

Create `plugins/surroforge-professional/pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "surroforge-professional"
version = "0.1.0"
description = "Professional report formats for SurroForge."
readme = "README.md"
requires-python = ">=3.10"
license = "Proprietary"
authors = [{ name = "SurroForge" }]
keywords = ["surrogate", "surroforge", "reports", "html", "professional"]
dependencies = [
    "surroforge",
    "jinja2>=3.1",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4", "mypy>=1.9"]

[project.entry-points."surroforge.reporters"]
html = "surroforge_professional.reports.html:HtmlReporter"

[tool.hatch.build.targets.wheel]
packages = ["src/surroforge_professional"]

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
```

Create `plugins/surroforge-professional/src/surroforge_professional/__init__.py`:

```python
"""SurroForge Professional plugin."""

from __future__ import annotations

__version__ = "0.1.0"
```

Create `plugins/surroforge-professional/src/surroforge_professional/reports/__init__.py`:

```python
"""Professional report generators."""
```

Create `plugins/surroforge-professional/src/surroforge_professional/reports/html.py`:

```python
"""Static HTML report for SurroForge Professional."""

from __future__ import annotations

from surroforge.core.run import Run
from surroforge.reporters.base import BaseReporter


class HtmlReporter(BaseReporter):
    name = "html"
    extension = ".html"

    def render(self, run: Run) -> str:
        return "<!doctype html>\n<title>SurroForge Report</title>\n"
```

- [ ] **Step 4: Install the plugin in editable mode**

Run:

```bash
.venv/bin/python -m pip install -e plugins/surroforge-professional
```

Expected: package installs as `surroforge-professional-0.1.0`.

- [ ] **Step 5: Run the registration test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_registration.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit scaffold**

Run:

```bash
git add plugins/surroforge-professional/pyproject.toml plugins/surroforge-professional/src/surroforge_professional/__init__.py plugins/surroforge-professional/src/surroforge_professional/reports/__init__.py plugins/surroforge-professional/src/surroforge_professional/reports/html.py plugins/surroforge-professional/tests/test_registration.py
git commit -m "feat(professional): scaffold html reporter plugin"
```

## Task 2: Add Local Professional License Gate

**Files:**
- Create: `plugins/surroforge-professional/src/surroforge_professional/license.py`
- Create: `plugins/surroforge-professional/tests/test_license.py`
- Modify: `plugins/surroforge-professional/src/surroforge_professional/reports/html.py`

- [ ] **Step 1: Write failing license tests**

Create `plugins/surroforge-professional/tests/test_license.py`:

```python
from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge_professional.license import (
    LICENSE_ERROR,
    has_license,
    license_file,
    require_license,
)
from surroforge_professional.reports.html import HtmlReporter


def test_license_missing_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert has_license() is False
    with pytest.raises(PluginError, match="Professional license required"):
        require_license()


def test_env_license_permits_render(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "local-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert has_license() is True
    require_license()


def test_file_license_permits_render(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    path = license_file()
    path.parent.mkdir(parents=True)
    path.write_text("local-dev-license\n")

    assert has_license() is True
    require_license()


def test_reporter_checks_license_before_reading_run(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(PluginError, match="Professional license required"):
        HtmlReporter().render(object())


def test_missing_license_message_is_actionable(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(PluginError) as exc:
        require_license()

    assert str(exc.value) == LICENSE_ERROR
    assert "SURROFORGE_PRO_LICENSE" in str(exc.value)
    assert "~/.surroforge/professional.lic" in str(exc.value)
```

- [ ] **Step 2: Run license tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_license.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'surroforge_professional.license'`.

- [ ] **Step 3: Implement the license gate**

Create `plugins/surroforge-professional/src/surroforge_professional/license.py`:

```python
"""Local Professional license gate."""

from __future__ import annotations

import os
from pathlib import Path

from surroforge.errors import PluginError

LICENSE_ENV = "SURROFORGE_PRO_LICENSE"
LICENSE_FILE = Path("~/.surroforge/professional.lic")
LICENSE_ERROR = (
    "Professional license required for HTML reports. Set SURROFORGE_PRO_LICENSE "
    "or install ~/.surroforge/professional.lic."
)


def license_file() -> Path:
    return LICENSE_FILE.expanduser()


def has_license() -> bool:
    env_value = os.environ.get(LICENSE_ENV, "").strip()
    return bool(env_value) or license_file().is_file()


def require_license() -> None:
    if not has_license():
        raise PluginError(LICENSE_ERROR)
```

Replace `plugins/surroforge-professional/src/surroforge_professional/reports/html.py` with:

```python
"""Static HTML report for SurroForge Professional."""

from __future__ import annotations

from surroforge.core.run import Run
from surroforge.reporters.base import BaseReporter
from surroforge_professional.license import require_license


class HtmlReporter(BaseReporter):
    name = "html"
    extension = ".html"

    def render(self, run: Run) -> str:
        require_license()
        return "<!doctype html>\n<title>SurroForge Report</title>\n"
```

- [ ] **Step 4: Run license and registration tests**

Run:

```bash
.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_registration.py plugins/surroforge-professional/tests/test_license.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit license gate**

Run:

```bash
git add plugins/surroforge-professional/src/surroforge_professional/license.py plugins/surroforge-professional/src/surroforge_professional/reports/html.py plugins/surroforge-professional/tests/test_license.py
git commit -m "feat(professional): require local license for html reports"
```

## Task 3: Render Scalar HTML Report Content

**Files:**
- Create: `plugins/surroforge-professional/tests/conftest.py`
- Create: `plugins/surroforge-professional/tests/test_html_report.py`
- Create: `plugins/surroforge-professional/src/surroforge_professional/reports/context.py`
- Modify: `plugins/surroforge-professional/src/surroforge_professional/reports/html.py`

- [ ] **Step 1: Add plugin-local fixtures**

Create `plugins/surroforge-professional/tests/conftest.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import run_training


@pytest.fixture
def licensed(monkeypatch, tmp_path):
    monkeypatch.setenv("SURROFORGE_PRO_LICENSE", "local-dev-license")
    monkeypatch.setenv("HOME", str(tmp_path))


def make_samples(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    length = rng.uniform(1.0, 3.0, n)
    width = rng.uniform(0.5, 1.5, n)
    load = rng.uniform(10.0, 50.0, n)
    max_stress = 2.5 * length**2 + 1.3 * width * load + 0.7 * load + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {"length": length, "width": width, "load": load, "max_stress": max_stress}
    )


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "samples.csv"
    make_samples().to_csv(path, index=False)
    return path


@pytest.fixture
def project(tmp_path, sample_csv):
    proj = Project.init(tmp_path / "proj", name="professional-test")
    proj.add_dataset(sample_csv, name="beam")
    return proj


@pytest.fixture
def trained_run(project):
    config = parse_config(
        {
            "experiment": "beam-professional-html",
            "data": {
                "source": "beam",
                "inputs": ["length", "width", "load"],
                "outputs": ["max_stress"],
                "split": {"test_fraction": 0.2, "seed": 42},
            },
            "model": {"method": "polynomial", "degree": 2},
            "validation": {"cross_validation": 5, "metrics": ["rmse", "r2", "mae"]},
        }
    )
    return run_training(project, config)
```

- [ ] **Step 2: Write failing scalar report tests**

Create `plugins/surroforge-professional/tests/test_html_report.py`:

```python
from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.services import generate_report
from surroforge_professional.reports.html import HtmlReporter


def test_html_contains_required_sections(licensed, trained_run):
    text = HtmlReporter().render(trained_run)

    assert text.startswith("<!doctype html>")
    assert "<h1>SurroForge Report</h1>" in text
    for heading in (
        "Executive Summary",
        "Quality Snapshot",
        "Data And Model",
        "Validation Details",
        "Reproducibility Appendix",
    ):
        assert heading in text
    assert "<script" not in text.lower()
    assert "https://" not in text
    assert "http://" not in text


def test_html_reports_metrics_and_lineage(licensed, trained_run):
    text = HtmlReporter().render(trained_run)

    assert trained_run.run_id in text
    assert trained_run.config.experiment in text
    assert trained_run.config.model.method in text
    assert trained_run.data_hash in text
    assert trained_run.config_hash in text
    assert trained_run.lineage["prediction_hash"] in text
    assert f"surroforge reproduce {trained_run.run_id}" in text
    for metric in trained_run.config.validation.metrics:
        assert f">{metric}<" in text
    for column in trained_run.config.data.inputs + trained_run.config.data.outputs:
        assert column in text


def test_generate_report_writes_html_under_reports_dir(licensed, project, trained_run):
    path, text = generate_report(project, trained_run, fmt="html")

    assert path == project.reports_dir / f"{trained_run.run_id}.html"
    assert path.read_text() == text
    assert "SurroForge Report" in text


def test_missing_required_train_metrics_fail_clearly(licensed, trained_run, monkeypatch):
    monkeypatch.setattr(
        trained_run,
        "metrics",
        {"test": {}, "cross_validation": {}},
        raising=False,
    )

    with pytest.raises(PluginError, match=r"metrics\['train'\]"):
        HtmlReporter().render(trained_run)
```

- [ ] **Step 3: Run scalar report tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_html_report.py -q
```

Expected: FAIL because the current reporter returns only the minimal scaffold HTML and has no context builder.

- [ ] **Step 4: Implement scalar report context**

Create `plugins/surroforge-professional/src/surroforge_professional/reports/context.py`:

```python
"""Build template context for Professional HTML reports."""

from __future__ import annotations

from typing import Any

import numpy as np

from surroforge.core.run import Run
from surroforge.errors import PluginError


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.5g}"
    except (TypeError, ValueError):
        return str(value)


def _require_key(mapping: dict[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise PluginError(f"Cannot render HTML report: missing {label}.")
    return mapping[key]


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PluginError(f"Cannot render HTML report: {label} must be a mapping.")
    return value


def _metric_rows(metric_names: list[str], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    train = _require_dict(_require_key(metrics, "train", "metrics['train']"), "metrics['train']")
    test = _require_dict(_require_key(metrics, "test", "metrics['test']"), "metrics['test']")
    cv = _require_dict(
        _require_key(metrics, "cross_validation", "metrics['cross_validation']"),
        "metrics['cross_validation']",
    )

    rows: list[dict[str, Any]] = []
    raw_values: list[float] = []
    for metric in metric_names:
        train_value = _require_key(train, metric, f"train metric '{metric}'")
        test_value = _require_key(test, metric, f"test metric '{metric}'")
        cv_record = _require_dict(
            _require_key(cv, metric, f"cross-validation metric '{metric}'"),
            f"cross-validation metric '{metric}'",
        )
        cv_mean = _require_key(cv_record, "mean", f"cross-validation mean for '{metric}'")
        cv_std = _require_key(cv_record, "std", f"cross-validation std for '{metric}'")
        numeric = [float(train_value), float(test_value), float(cv_mean), float(cv_std)]
        raw_values.extend(abs(value) for value in numeric)
        rows.append(
            {
                "name": metric,
                "train": fmt(train_value),
                "test": fmt(test_value),
                "cv_mean": fmt(cv_mean),
                "cv_std": fmt(cv_std),
                "raw_test": float(test_value),
            }
        )

    max_value = max(raw_values) if raw_values else 0.0
    for row in rows:
        width = 0 if max_value <= 0.0 else int(round(abs(row["raw_test"]) / max_value * 100))
        row["test_bar"] = max(2, min(100, width)) if max_value > 0.0 else 0
    return rows


def build_context(run: Run) -> dict[str, Any]:
    config = run.config
    metrics = run.metrics
    lineage = run.lineage
    metric_names = list(config.validation.metrics)
    rows = _metric_rows(metric_names, metrics)
    cv = _require_dict(
        _require_key(metrics, "cross_validation", "metrics['cross_validation']"),
        "metrics['cross_validation']",
    )

    return {
        "run_id": run.run_id,
        "experiment": config.experiment,
        "method": config.model.method,
        "created_at": lineage.get("created_at", ""),
        "parent_run": lineage.get("parent_run"),
        "data_source": lineage.get("data_source", config.data.source),
        "data_hash": run.data_hash,
        "config_hash": run.config_hash,
        "prediction_hash": lineage.get("prediction_hash", ""),
        "seed": run.seed,
        "n_samples": metrics.get("n_samples", "?"),
        "n_inputs": metrics.get("n_inputs", len(config.data.inputs)),
        "n_outputs": metrics.get("n_outputs", len(config.data.outputs)),
        "inputs": list(config.data.inputs),
        "outputs": list(config.data.outputs),
        "hyperparameters": dict(config.model.hyperparameters),
        "metric_rows": rows,
        "cv_folds": cv.get("folds", config.validation.cross_validation.folds),
        "test_fraction": config.data.split.test_fraction,
        "config_yaml": config.to_yaml(),
        "python_version": lineage.get("python_version", ""),
        "surroforge_version": lineage.get("surroforge_version", ""),
        "dependencies": [str(dep) for dep in lineage.get("dependencies", [])],
        "two_source": None,
        "fmt": fmt,
        "np_version": np.__version__,
    }
```

- [ ] **Step 5: Implement static scalar HTML template**

Replace `plugins/surroforge-professional/src/surroforge_professional/reports/html.py` with:

```python
"""Static HTML report for SurroForge Professional."""

from __future__ import annotations

from jinja2 import Environment

from surroforge.core.run import Run
from surroforge.reporters.base import BaseReporter
from surroforge_professional.license import require_license
from surroforge_professional.reports.context import build_context

_ENV = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True)

_TEMPLATE = _ENV.from_string(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SurroForge Report {{ run_id }}</title>
  <style>
    :root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f7f8fa; color: #1f2933; }
    main { max-width: 1120px; margin: 0 auto; padding: 32px 24px 48px; }
    h1, h2, h3 { margin: 0; line-height: 1.2; }
    h1 { font-size: 32px; }
    h2 { font-size: 22px; margin-bottom: 16px; }
    h3 { font-size: 16px; margin-bottom: 8px; }
    section { background: #ffffff; border: 1px solid #d9dee7; border-radius: 8px; padding: 20px; margin-top: 18px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { border-bottom: 1px solid #e5e9f0; padding: 9px 8px; text-align: left; vertical-align: top; }
    th { background: #f1f4f8; font-weight: 650; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    pre { white-space: pre-wrap; background: #111827; color: #f9fafb; padding: 14px; border-radius: 6px; overflow-x: auto; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 16px; }
    .fact { border: 1px solid #e1e6ef; border-radius: 8px; padding: 12px; background: #fbfcfe; }
    .label { display: block; color: #52606d; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 5px; }
    .value { word-break: break-word; font-weight: 600; }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
    .metric-card { border: 1px solid #dfe5ee; border-radius: 8px; padding: 14px; background: #fbfcfe; }
    .metric-name { font-weight: 700; margin-bottom: 10px; }
    .metric-line { display: flex; justify-content: space-between; gap: 12px; font-size: 14px; margin-top: 5px; }
    .bar { height: 7px; background: #e7ecf3; border-radius: 999px; overflow: hidden; margin-top: 10px; }
    .bar-fill { height: 100%; background: #44546a; border-radius: 999px; }
    .columns { display: flex; flex-wrap: wrap; gap: 8px; padding: 0; margin: 8px 0 0; list-style: none; }
    .columns li { border: 1px solid #d8dee8; background: #f6f8fb; border-radius: 999px; padding: 4px 9px; font-size: 13px; }
    .appendix-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; }
    .muted { color: #66788a; }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>SurroForge Report</h1>
      <p class="muted">Professional HTML report for run <code>{{ run_id }}</code></p>
    </header>

    <section>
      <h2>Executive Summary</h2>
      <div class="summary-grid">
        <div class="fact"><span class="label">Run ID</span><span class="value">{{ run_id }}</span></div>
        <div class="fact"><span class="label">Experiment</span><span class="value">{{ experiment }}</span></div>
        <div class="fact"><span class="label">Method</span><span class="value">{{ method }}</span></div>
        <div class="fact"><span class="label">Created</span><span class="value">{{ created_at }}</span></div>
        <div class="fact"><span class="label">Data Hash</span><code class="value">{{ data_hash }}</code></div>
        <div class="fact"><span class="label">Config Hash</span><code class="value">{{ config_hash }}</code></div>
        <div class="fact"><span class="label">Prediction Fingerprint</span><code class="value">{{ prediction_hash }}</code></div>
      </div>
    </section>

    <section>
      <h2>Quality Snapshot</h2>
      <div class="metric-grid">
        {% for row in metric_rows %}
        <article class="metric-card">
          <div class="metric-name">{{ row.name }}</div>
          <div class="metric-line"><span>Train</span><strong>{{ row.train }}</strong></div>
          <div class="metric-line"><span>Test</span><strong>{{ row.test }}</strong></div>
          <div class="metric-line"><span>CV mean</span><strong>{{ row.cv_mean }}</strong></div>
          <div class="metric-line"><span>CV std</span><strong>{{ row.cv_std }}</strong></div>
          <div class="bar" aria-label="Relative test metric magnitude">
            <div class="bar-fill" style="width: {{ row.test_bar }}%"></div>
          </div>
        </article>
        {% endfor %}
      </div>
    </section>

    <section>
      <h2>Data And Model</h2>
      <div class="summary-grid">
        <div class="fact"><span class="label">Source</span><span class="value">{{ data_source }}</span></div>
        <div class="fact"><span class="label">Samples</span><span class="value">{{ n_samples }}</span></div>
        <div class="fact"><span class="label">Inputs</span><span class="value">{{ n_inputs }}</span></div>
        <div class="fact"><span class="label">Outputs</span><span class="value">{{ n_outputs }}</span></div>
      </div>
      <h3>Input Columns</h3>
      <ul class="columns">{% for column in inputs %}<li>{{ column }}</li>{% endfor %}</ul>
      <h3>Output Columns</h3>
      <ul class="columns">{% for column in outputs %}<li>{{ column }}</li>{% endfor %}</ul>
      <h3>Hyperparameters</h3>
      {% if hyperparameters %}
      <table>
        <thead><tr><th>Name</th><th>Value</th></tr></thead>
        <tbody>{% for key, value in hyperparameters.items() %}<tr><td>{{ key }}</td><td>{{ value }}</td></tr>{% endfor %}</tbody>
      </table>
      {% else %}
      <p class="muted">Defaults</p>
      {% endif %}
    </section>

    <section>
      <h2>Validation Details</h2>
      <table>
        <thead><tr><th>Metric</th><th>Train</th><th>Test</th><th>CV mean</th><th>CV std</th></tr></thead>
        <tbody>
        {% for row in metric_rows %}
          <tr><td>{{ row.name }}</td><td>{{ row.train }}</td><td>{{ row.test }}</td><td>{{ row.cv_mean }}</td><td>{{ row.cv_std }}</td></tr>
        {% endfor %}
        </tbody>
      </table>
      <p class="muted">Cross-validation: {{ cv_folds }}-fold.</p>
    </section>

    {% if two_source %}
    <section>
      <h2>Field-ROM Details</h2>
    </section>
    {% endif %}

    <section>
      <h2>Reproducibility Appendix</h2>
      <div class="appendix-grid">
        <div class="fact"><span class="label">Python</span><span class="value">{{ python_version }}</span></div>
        <div class="fact"><span class="label">SurroForge</span><span class="value">{{ surroforge_version }}</span></div>
        <div class="fact"><span class="label">Random Seed</span><span class="value">{{ seed }}</span></div>
        <div class="fact"><span class="label">Test Fraction</span><span class="value">{{ test_fraction }}</span></div>
      </div>
      <h3>Dependencies</h3>
      <ul>{% for dep in dependencies %}<li>{{ dep }}</li>{% endfor %}</ul>
      <h3>Configuration</h3>
      <pre>{{ config_yaml }}</pre>
      <p>Reproduce this run with <code>surroforge reproduce {{ run_id }}</code>.</p>
    </section>
  </main>
</body>
</html>
"""
)


class HtmlReporter(BaseReporter):
    name = "html"
    extension = ".html"

    def render(self, run: Run) -> str:
        require_license()
        return _TEMPLATE.render(**build_context(run))
```

- [ ] **Step 6: Run scalar report tests**

Run:

```bash
.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_html_report.py -q
```

Expected: PASS.

- [ ] **Step 7: Run plugin tests so far**

Run:

```bash
.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_registration.py plugins/surroforge-professional/tests/test_license.py plugins/surroforge-professional/tests/test_html_report.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit scalar report rendering**

Run:

```bash
git add plugins/surroforge-professional/src/surroforge_professional/reports/context.py plugins/surroforge-professional/src/surroforge_professional/reports/html.py plugins/surroforge-professional/tests/conftest.py plugins/surroforge-professional/tests/test_html_report.py
git commit -m "feat(professional): render scalar html reports"
```

## Task 4: Pin Escaping And Required-Data Failures

**Files:**
- Modify: `plugins/surroforge-professional/tests/test_html_report.py`
- Modify: `plugins/surroforge-professional/src/surroforge_professional/reports/context.py`

- [ ] **Step 1: Add escaping and missing-CV tests**

Append to `plugins/surroforge-professional/tests/test_html_report.py`:

```python
from types import SimpleNamespace


class _EvilConfig:
    experiment = "<script>alert(1)</script>"
    data = SimpleNamespace(
        source="beam<script>",
        inputs=["length<script>"],
        outputs=["stress<script>"],
        split=SimpleNamespace(test_fraction=0.2),
    )
    model = SimpleNamespace(method="poly<script>", hyperparameters={"degree": "<b>2</b>"})
    validation = SimpleNamespace(
        metrics=["rmse"],
        cross_validation=SimpleNamespace(folds=5),
    )

    def to_yaml(self) -> str:
        return "experiment: <script>alert(1)</script>"


class _EvilRun:
    run_id = "run-evil<script>"
    config = _EvilConfig()
    metrics = {
        "n_samples": 10,
        "n_inputs": 1,
        "n_outputs": 1,
        "train": {"rmse": 1.0},
        "test": {"rmse": 2.0},
        "cross_validation": {"folds": 5, "rmse": {"mean": 1.5, "std": 0.1}},
    }
    lineage = {
        "created_at": "<script>now()</script>",
        "data_hash": "data<script>",
        "config_hash": "config<script>",
        "prediction_hash": "pred<script>",
        "python_version": "3<script>",
        "surroforge_version": "0<script>",
        "dependencies": ["numpy<script>"],
    }
    data_hash = "data<script>"
    config_hash = "config<script>"
    seed = 7


def test_dynamic_content_is_html_escaped(licensed):
    text = HtmlReporter().render(_EvilRun())

    assert "<script>" not in text
    assert "<b>2</b>" not in text
    assert "&lt;script&gt;" in text
    assert "&lt;b&gt;2&lt;/b&gt;" in text


def test_missing_cross_validation_metric_fails_clearly(licensed, trained_run, monkeypatch):
    metrics = dict(trained_run.metrics)
    metrics["cross_validation"] = {"folds": 5}
    monkeypatch.setattr(trained_run, "metrics", metrics, raising=False)

    with pytest.raises(PluginError, match="cross-validation metric 'rmse'"):
        HtmlReporter().render(trained_run)
```

- [ ] **Step 2: Run the new tests**

Run:

```bash
.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_html_report.py::test_dynamic_content_is_html_escaped plugins/surroforge-professional/tests/test_html_report.py::test_missing_cross_validation_metric_fails_clearly -q
```

Expected: PASS. If the escaping test fails, confirm `Environment(autoescape=True)` is still used and no dynamic values are passed through a Jinja `safe` filter.

- [ ] **Step 3: Run the full scalar report test file**

Run:

```bash
.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_html_report.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit safety tests**

Run:

```bash
git add plugins/surroforge-professional/tests/test_html_report.py plugins/surroforge-professional/src/surroforge_professional/reports/context.py
git commit -m "test(professional): pin html escaping and report failures"
```

## Task 5: Add Field-ROM Optional Section And Degradation

**Files:**
- Create: `plugins/surroforge-professional/tests/test_field_degradation.py`
- Modify: `plugins/surroforge-professional/src/surroforge_professional/reports/context.py`
- Modify: `plugins/surroforge-professional/src/surroforge_professional/reports/html.py`

- [ ] **Step 1: Write failing Field-ROM tests**

Create `plugins/surroforge-professional/tests/test_field_degradation.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import run_training
from surroforge_professional.reports.html import HtmlReporter


def _field_run(tmp_path):
    rng = np.random.default_rng(42)
    n_dof, n = 28, 64
    modes = np.linalg.qr(rng.standard_normal((n_dof, 3)))[0]
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    amps = np.column_stack([5.0 * x[:, 0], 2.0 * x[:, 1] ** 2, 2.0 * x[:, 0] * x[:, 1]])
    y = amps @ modes.T
    cols: dict[str, np.ndarray] = {"p0": x[:, 0], "p1": x[:, 1]}
    for j in range(n_dof):
        cols[f"f{j}"] = y[:, j]
    csv = tmp_path / "field.csv"
    pd.DataFrame(cols).to_csv(csv, index=False)
    proj = Project.init(tmp_path / "proj", name="rom")
    proj.add_dataset(csv, name="snaps")
    cfg = parse_config(
        {
            "experiment": "rom-html",
            "data": {
                "source": "snaps",
                "inputs": ["p0", "p1"],
                "outputs": ["f*"],
                "split": {"test_fraction": 0.25, "seed": 1},
            },
            "model": {
                "method": "reduced-field",
                "reduction": {"basis": "pod", "rank": 2},
                "inner": {"method": "polynomial", "degree": 2},
            },
            "validation": {"cross_validation": 3, "metrics": ["energy_l2", "truncation_error", "rmse", "r2"]},
        }
    )
    return run_training(proj, cfg)


def test_complete_field_run_renders_two_source_section(licensed, tmp_path):
    run = _field_run(tmp_path)
    text = HtmlReporter().render(run)

    assert "Field-ROM Details" in text
    assert "total field error" in text
    assert "truncation floor" in text
    assert "surrogate error" in text
    assert "Reduction rank" in text


def test_incomplete_field_records_omit_field_section(licensed, tmp_path, monkeypatch):
    run = _field_run(tmp_path)
    metrics = dict(run.metrics)
    metrics["cross_validation"] = dict(metrics["cross_validation"])
    metrics["cross_validation"]["truncation_error"] = {"mean": 0.1, "std": 0.01}
    monkeypatch.setattr(run, "metrics", metrics, raising=False)

    text = HtmlReporter().render(run)

    assert "Field-ROM Details" not in text
    assert "Validation Details" in text
```

- [ ] **Step 2: Run Field-ROM tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_field_degradation.py -q
```

Expected: FAIL because `two_source` is always `None` and the template does not render details.

- [ ] **Step 3: Add two-source context builder**

Append this function to `plugins/surroforge-professional/src/surroforge_professional/reports/context.py` above `build_context`:

```python
def _two_source_context(config: Any, metrics: dict[str, Any], lineage: dict[str, Any]) -> dict[str, Any] | None:
    metric_names = set(config.validation.metrics)
    field_names = ("energy_l2", "truncation_error")
    if not set(field_names) <= metric_names:
        return None

    train = metrics.get("train", {})
    test = metrics.get("test", {})
    cv = metrics.get("cross_validation", {})
    if not all(name in train and name in test for name in field_names):
        return None
    if not all(name in cv and cv[name].get("per_fold") for name in field_names):
        return None
    if len(cv["energy_l2"]["per_fold"]) != len(cv["truncation_error"]["per_fold"]):
        return None

    from surroforge.validators.metrics import two_source_split

    fold_splits = [
        two_source_split(energy, truncation)
        for energy, truncation in zip(
            cv["energy_l2"]["per_fold"],
            cv["truncation_error"]["per_fold"],
            strict=True,
        )
    ]
    components = ("total", "truncation", "surrogate")
    train_split = two_source_split(train["energy_l2"], train["truncation_error"])
    test_split = two_source_split(test["energy_l2"], test["truncation_error"])
    cv_mean = {name: float(np.mean([split[name] for split in fold_splits])) for name in components}
    cv_std = {name: float(np.std([split[name] for split in fold_splits])) for name in components}
    labels = {
        "total": "total field error",
        "truncation": "truncation floor",
        "surrogate": "surrogate error",
    }
    reduction = lineage.get("reduction") if isinstance(lineage.get("reduction"), dict) else {}
    return {
        "rows": [
            {
                "label": labels[name],
                "train": fmt(train_split[name]),
                "test": fmt(test_split[name]),
                "cv_mean": fmt(cv_mean[name]),
                "cv_std": fmt(cv_std[name]),
            }
            for name in components
        ],
        "reduction_rank": reduction.get("rank"),
        "energy_captured": fmt(reduction["energy_captured"]) if "energy_captured" in reduction else "",
        "truncation_per_fold": ", ".join(fmt(value) for value in cv["truncation_error"]["per_fold"]),
    }
```

Then change the `build_context` return value from:

```python
        "two_source": None,
```

to:

```python
        "two_source": _two_source_context(config, metrics, lineage),
```

- [ ] **Step 4: Render Field-ROM details in the template**

Replace the current `{% if two_source %}` block in `plugins/surroforge-professional/src/surroforge_professional/reports/html.py` with:

```html
    {% if two_source %}
    <section>
      <h2>Field-ROM Details</h2>
      <table>
        <thead><tr><th>Error source</th><th>Train</th><th>Test</th><th>CV mean</th><th>CV std</th></tr></thead>
        <tbody>
        {% for row in two_source.rows %}
          <tr><td>{{ row.label }}</td><td>{{ row.train }}</td><td>{{ row.test }}</td><td>{{ row.cv_mean }}</td><td>{{ row.cv_std }}</td></tr>
        {% endfor %}
        </tbody>
      </table>
      {% if two_source.reduction_rank %}
      <p>Reduction rank: <strong>{{ two_source.reduction_rank }}</strong></p>
      {% endif %}
      {% if two_source.energy_captured %}
      <p>Energy captured: <strong>{{ two_source.energy_captured }}</strong></p>
      {% endif %}
      <p class="muted">Per-fold truncation floors: {{ two_source.truncation_per_fold }}</p>
    </section>
    {% endif %}
```

- [ ] **Step 5: Run Field-ROM tests**

Run:

```bash
.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_field_degradation.py -q
```

Expected: PASS.

- [ ] **Step 6: Run all plugin tests**

Run:

```bash
.venv/bin/python -m pytest plugins/surroforge-professional/tests -q
```

Expected: PASS.

- [ ] **Step 7: Commit Field-ROM support**

Run:

```bash
git add plugins/surroforge-professional/src/surroforge_professional/reports/context.py plugins/surroforge-professional/src/surroforge_professional/reports/html.py plugins/surroforge-professional/tests/test_field_degradation.py
git commit -m "feat(professional): render field html diagnostics"
```

## Task 6: Add Plugin README And Final Verification

**Files:**
- Create: `plugins/surroforge-professional/README.md`

- [ ] **Step 1: Write plugin README**

Create `plugins/surroforge-professional/README.md`:

````markdown
# surroforge-professional

Professional report formats for SurroForge.

## Install For Development

From the repository root:

```bash
.venv/bin/python -m pip install -e plugins/surroforge-professional
```

## License Gate

HTML reports require a local Professional license marker. For development, set:

```bash
export SURROFORGE_PRO_LICENSE=local-dev-license
```

Alternatively create:

```bash
mkdir -p ~/.surroforge
printf '%s\n' local-dev-license > ~/.surroforge/professional.lic
```

There is no online activation in v1.

## CLI Usage

```bash
surroforge report run-001 --format html
```

The report is written to:

```text
reports/run-001.html
```

## Python API

```python
from surroforge import Project
from surroforge.services import generate_report

project = Project.open("beam-study")
run = project.run("run-001")
path, html = generate_report(project, run, fmt="html")
print(path)
```

## v1 Scope

- Static single-file HTML.
- Embedded CSS.
- No required JavaScript.
- No external network assets.
- Markdown report parity plus metric cards and simple visual structure.
- Optional Field-ROM details when the run carries complete metrics and lineage.

PDF audit reports and interactive charts are outside the v1 package.
````

- [ ] **Step 2: Run formatting and plugin tests**

Run:

```bash
.venv/bin/python -m ruff check plugins/surroforge-professional
.venv/bin/python -m pytest plugins/surroforge-professional/tests -q
```

Expected: both commands PASS.

- [ ] **Step 3: Run Core reporter smoke tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_reporters.py tests/test_plugins.py -q
```

Expected: PASS. This checks the existing Markdown reporter and plugin registry still behave.

- [ ] **Step 4: Run full test suite if time permits**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: PASS with the repository's existing skip count. If this is too slow during execution, record that only plugin tests and reporter smoke tests were run.

- [ ] **Step 5: Check worktree**

Run:

```bash
git status --short
```

Expected: only Professional plugin files are modified or staged. `.superpowers/` may remain untracked from brainstorming and must not be staged for this feature.

- [ ] **Step 6: Commit README and verification fixes**

Run:

```bash
git add plugins/surroforge-professional/README.md plugins/surroforge-professional
git commit -m "docs(professional): document html reports plugin"
```

If no files changed after Step 1, skip this commit and record that README was already included.

## Final Verification Checklist

- `html` appears in `available_reporters()` after editable plugin install.
- Missing license raises a `PluginError` with the approved message.
- Env-var license permits rendering.
- File license permits rendering.
- `generate_report(project, run, fmt="html")` writes `reports/<run-id>.html`.
- Rendered HTML starts with `<!doctype html>`.
- Rendered HTML contains no `<script>` tag and no external `http://` or `https://` assets.
- Scalar reports include Executive Summary, Quality Snapshot, Data And Model, Validation Details, and Reproducibility Appendix.
- Prediction fingerprint appears in the reproducibility/run identity content.
- Field-ROM details appear only when complete `energy_l2` and `truncation_error` data is present.
- Existing Core Markdown report tests still pass.
