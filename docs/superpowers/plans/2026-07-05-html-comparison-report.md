# Professional HTML Comparison Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `surroforge compare --format html` renders a self-contained, interactive HTML comparison report via a new comparison-reporter plugin contract, implemented in the Professional plugin.

**Architecture:** Core gains a parallel plugin contract (`BaseComparisonReporter` ABC, entry-point group `surroforge.comparison_reporters`, `generate_comparison_report` service, `--format/--output` on `compare`). The Professional plugin implements it in a new `reports/comparison.py` (context builder + Jinja template string, same pattern as the single-run `html.py`), with cross-run chart data built in `charts.py` and the uPlot assets + base CSS shared with the single-run report.

**Tech Stack:** Python 3.13 (`.venv`), Typer CLI, Jinja2 (autoescape), uPlot (vendored, MIT), pytest.

**Spec:** `docs/superpowers/specs/2026-07-05-html-comparison-report-design.md`

## Global Constraints

- **Never push.** Commits stay local until the user has seen commits + changed files (CLAUDE.md rule).
- **No `Co-Authored-By:` trailer.** Repo-local identity `Georg Babij <g.babij@gmail.com>` is configured; plain `git commit -m` is correct. Commit directly to `main`.
- Use `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/mypy`, `.venv/bin/pyright`.
- Ruff: line-length 100, rules `E,F,I,UP,B,W`. mypy gate is `mypy src`. **pyright includes `plugins/` — the current baseline is 0 errors and must stay 0.**
- The plugin test suite is NOT collected by root pytest; run it explicitly: `.venv/bin/pytest plugins/surroforge-professional/tests`.
- Entry-point group name exactly: `surroforge.comparison_reporters`. Reporter name exactly: `html`. Default output file exactly: `reports/comparison.html`.
- Default `--format` is `terminal` and the terminal path must stay byte-identical to today.
- The license gate is `require_license()` (raises `PluginError` with "Professional license required…"); tests use the existing `licensed` fixture (`SURROFORGE_PRO_LICENSE` env).
- The report must be self-contained: no external `src=`/`href=`/`@import`/`url(http`; all strings via Jinja autoescape; chart JSON via `_charts_json` (escapes every `<` as the JSON sequence backslash-u003c).

---

### Task 1: Core contract + registry group

**Files:**
- Modify: `src/surroforge/reporters/base.py` (19 lines — add second ABC)
- Modify: `src/surroforge/plugins/registry.py` (group constant, empty `_BUILTIN` entry, two functions)
- Modify: `src/surroforge/plugins/__init__.py` (re-exports)
- Test: `tests/test_plugins.py` (append)

**Interfaces:**
- Consumes: existing `_resolve(group, name, kind)` / `_names(group)` in registry.py.
- Produces: `BaseComparisonReporter` (attrs `name: str`, `extension: str`; abstract `render(self, comparison: dict[str, Any]) -> str`), `get_comparison_reporter(name: str) -> type`, `available_comparison_reporters() -> list[str]`, constant `COMPARISON_REPORTER_GROUP = "surroforge.comparison_reporters"`. Tasks 2 and 4 rely on these exact names.

**Critical detail:** `_resolve` (registry.py:99) and `_names` (registry.py:120) index `_BUILTIN[group]` directly — a group missing from `_BUILTIN` raises `KeyError`. The new group therefore gets an **empty** `_BUILTIN` entry.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_plugins.py`:

```python
def test_unknown_comparison_reporter_raises():
    with pytest.raises(PluginError) as err:
        get_comparison_reporter("nope")
    assert "comparison report format" in str(err.value)
    assert "nope" in str(err.value)


def test_comparison_reporter_resolves_from_entry_point(monkeypatch):
    from surroforge.plugins import registry

    class FakeComparisonReporter:
        name = "fake-html"
        extension = ".html"

    class FakeEP:
        name = "fake-html"
        value = "fake.module:FakeComparisonReporter"

        def load(self):
            return FakeComparisonReporter

    def fake_grouped(group):
        if group == registry.COMPARISON_REPORTER_GROUP:
            return {"fake-html": [FakeEP()]}
        return {}

    monkeypatch.setattr(registry, "_entry_points_grouped", fake_grouped)
    assert registry.get_comparison_reporter("fake-html") is FakeComparisonReporter
    assert "fake-html" in registry.available_comparison_reporters()
```

Match the file's existing imports (it already imports `pytest` and `PluginError`; add `get_comparison_reporter` to its `from surroforge.plugins import ...` line). The registry docstring documents that tests inject fakes by replacing `_entry_points_grouped` — this is the sanctioned pattern. If `check_api_version` rejects the fake class, mirror how `tests/test_plugin_seam.py` fakes declare their API version.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_plugins.py -v`
Expected: the two new tests FAIL with `ImportError`/`AttributeError` (`get_comparison_reporter` does not exist); all pre-existing tests PASS.

- [ ] **Step 3: Implement**

`src/surroforge/reporters/base.py` — change the `typing` situation and append the ABC (final file content of the additions):

```python
from typing import Any     # new import at top, after the abc import


class BaseComparisonReporter(ABC):
    """Renders the JSON-serializable dict that ``build_comparison`` produces."""

    #: Registry name / ``--format`` value.
    name: str = "base"
    #: File extension (including the dot) for the rendered report.
    extension: str = ""

    @abstractmethod
    def render(self, comparison: dict[str, Any]) -> str:
        """Return the comparison report as text."""
```

`src/surroforge/plugins/registry.py`:
- After `COMMAND_GROUP = "surroforge.commands"` add:
  `COMPARISON_REPORTER_GROUP = "surroforge.comparison_reporters"`
- In the `_BUILTIN` literal, after the `REPORTER_GROUP` entry add:

```python
    # Core ships no comparison reporter; the terminal rendering lives in the CLI.
    COMPARISON_REPORTER_GROUP: {},
```

- After `get_reporter` add:

```python
def get_comparison_reporter(name: str) -> type:
    return _resolve(COMPARISON_REPORTER_GROUP, name, "comparison report format")
```

- After `available_reporters` add:

```python
def available_comparison_reporters() -> list[str]:
    return sorted(_names(COMPARISON_REPORTER_GROUP))
```

`src/surroforge/plugins/__init__.py`: add `available_comparison_reporters` and `get_comparison_reporter` to the `from surroforge.plugins.registry import (...)` block and to `__all__` (both lists are alphabetically sorted — keep that).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_plugins.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Static gates**

Run: `.venv/bin/ruff check src tests && .venv/bin/mypy src | tail -1 && .venv/bin/pyright 2>/dev/null | grep -E '^[0-9]+ error' `
Expected: ruff `All checks passed!`; mypy `Success: no issues found in 46 source files`; pyright `0 errors, ...`.

- [ ] **Step 6: Commit**

```bash
git add src/surroforge/reporters/base.py src/surroforge/plugins/registry.py src/surroforge/plugins/__init__.py tests/test_plugins.py
git commit -m "feat(core): add the comparison-reporter plugin contract"
```

---

### Task 2: Service function + `compare --format/--output`

**Files:**
- Modify: `src/surroforge/services.py` (import + one function after `generate_report`, ~line 511)
- Modify: `src/surroforge/cli/main.py` (the `compare` command, lines 358–367)
- Test: `tests/test_reporters.py`, `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `get_comparison_reporter` (Task 1), existing `compare_runs` (services.py:388).
- Produces: `generate_comparison_report(project: Project, run_ids: list[str], *, metric: str = "rmse", format: str = "html", output: str | Path | None = None) -> Path`. Task 4's end-to-end tests and Task 5's acceptance call the CLI exactly as specified here.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reporters.py`:

```python
class _StubComparisonReporter:
    name = "stub"
    extension = ".stub"

    def render(self, comparison):
        return f"comparison of {len(comparison['runs'])} runs by {comparison['metric']}"


def test_generate_comparison_report_writes_under_reports_dir(project, trained_run, monkeypatch):
    import surroforge.services as services

    monkeypatch.setattr(services, "get_comparison_reporter", lambda name: _StubComparisonReporter)
    path = services.generate_comparison_report(
        project, [trained_run.run_id], metric="rmse", format="stub"
    )
    assert path == project.reports_dir / "comparison.stub"
    assert path.read_text() == "comparison of 1 runs by rmse"


def test_generate_comparison_report_custom_output(project, trained_run, tmp_path, monkeypatch):
    import surroforge.services as services

    monkeypatch.setattr(services, "get_comparison_reporter", lambda name: _StubComparisonReporter)
    dest = tmp_path / "nested" / "cmp.stub"
    path = services.generate_comparison_report(project, [trained_run.run_id], output=dest)
    assert path == dest
    assert dest.is_file()
```

Append to `tests/test_cli.py`:

```python
def test_compare_format_html_calls_service(tmp_path, monkeypatch):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])

    calls = {}

    def fake_generate(project, run_ids, *, metric="rmse", format="html", output=None):
        calls.update(run_ids=run_ids, metric=metric, format=format, output=output)
        out = root / "reports" / "comparison.html"
        out.write_text("<!doctype html>")
        return out

    monkeypatch.setattr("surroforge.services.generate_comparison_report", fake_generate)
    result = runner.invoke(
        app, ["--config-dir", str(root), "compare", "--runs", "run-001", "-f", "html"]
    )
    assert result.exit_code == 0, result.output
    assert "comparison.html" in result.output
    assert calls == {"run_ids": ["run-001"], "metric": "rmse", "format": "html", "output": None}


def test_compare_terminal_with_output_errors(tmp_path):
    root = _init_project_with_data(tmp_path)
    cfg = _write_config(root)
    runner.invoke(app, ["--config-dir", str(root), "train", "--config", str(cfg)])
    result = runner.invoke(
        app,
        ["--config-dir", str(root), "compare", "--runs", "run-001", "--output", "x.html"],
    )
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "--format" in result.output
    assert "Traceback" not in result.output
```

The existing `test_compare_cli` (tests/test_cli.py:193) stays untouched — it is the regression guard that the default terminal path is unchanged.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_reporters.py tests/test_cli.py -v -k "comparison or compare"`
Expected: the three new tests FAIL (`AttributeError: ... has no attribute 'generate_comparison_report'` / unknown `--format` option); `test_compare_cli` PASSES.

- [ ] **Step 3: Implement**

`src/surroforge/services.py` — extend the plugins import (line 31) to:

```python
from surroforge.plugins import get_comparison_reporter, get_exporter, get_reporter
```

and append after `generate_report` (line 511):

```python
def generate_comparison_report(
    project: Project,
    run_ids: list[str],
    *,
    metric: str = "rmse",
    format: str = "html",
    output: str | Path | None = None,
) -> Path:
    """Render a multi-run comparison report and write it under ``reports/`` (PRD 4.1).

    Builds the comparison via :func:`compare_runs` and renders it with the
    comparison reporter registered under ``format`` (e.g. the Professional HTML
    comparison report).
    """
    comparison = compare_runs(project, run_ids, metric=metric)
    reporter = get_comparison_reporter(format)()
    text = reporter.render(comparison)
    if output is None:
        output = project.reports_dir / f"comparison{reporter.extension}"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return output
```

`src/surroforge/cli/main.py` — replace the `compare` command (lines 358–367) with:

```python
@app.command()
@_guard
def compare(
    runs: str = typer.Option(..., "--runs", "-r", help="Comma-separated run ids."),
    metric: str = typer.Option("rmse", "--metric", "-m", help="Metric used to rank runs."),
    format: str = typer.Option(
        "terminal",
        "--format",
        "-f",
        help="Output format: 'terminal' prints tables; anything else is resolved as a "
        "comparison-reporter plugin (e.g. 'html' from surroforge-professional).",
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Destination file (needs a file-producing --format)."
    ),
) -> None:
    """Compare several training runs side by side (PRD 4.1)."""
    project = _project()
    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    if format == "terminal":
        if output is not None:
            raise DataError("--output needs a file-producing --format (e.g. --format html).")
        _render_comparison(services.compare_runs(project, run_ids, metric=metric))
        return
    path = services.generate_comparison_report(
        project, run_ids, metric=metric, format=format, output=output
    )
    _success(f"Wrote {format} comparison report to {path}")
```

(`DataError` is already imported in main.py:25; `_guard` renders it as a clean `Error:` line.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_reporters.py tests/test_cli.py -v`
Expected: ALL PASS (including all pre-existing tests).

- [ ] **Step 5: Static gates**

Run: `.venv/bin/ruff check src tests && .venv/bin/mypy src | tail -1 && .venv/bin/pyright 2>/dev/null | grep -E '^[0-9]+ error'`
Expected: clean / `Success` / `0 errors`.

- [ ] **Step 6: Commit**

```bash
git add src/surroforge/services.py src/surroforge/cli/main.py tests/test_reporters.py tests/test_cli.py
git commit -m "feat(core): render comparison reports via compare --format"
```

---

### Task 3: Cross-run chart data (`build_comparison_charts`)

**Files:**
- Modify: `plugins/surroforge-professional/src/surroforge_professional/reports/charts.py` (append one function)
- Test: `plugins/surroforge-professional/tests/test_charts.py` (append)

**Interfaces:**
- Consumes: the `build_comparison` result dict (shape: `comparison.py:191-200` — `runs` entries carry `run_id` and `metrics{name: {train, test, cv_mean, cv_std}}`; top level carries `metric_names`, `best`, `best_per_metric`).
- Produces: `build_comparison_charts(comparison: dict[str, Any]) -> dict[str, Any] | None` returning `{"runs": [str], "metrics": [str], "best": str | None, "best_per_metric": dict, "per_metric": {name: {"train": [float|None], "test": [...], "cv_mean": [...], "cv_std": [...]}}}` with all lists parallel to `runs`. Task 4's template/JS consumes exactly these keys.

- [ ] **Step 1: Write the failing tests** — append to `plugins/surroforge-professional/tests/test_charts.py`:

```python
def _comparison_entry(run_id, **metric_values):
    return {
        "run_id": run_id,
        "metrics": {
            name: {"train": v, "test": v, "cv_mean": v, "cv_std": 0.01}
            for name, v in metric_values.items()
        },
    }


def _comparison(entries, metric_names, best="r1", best_per_metric=None):
    return {
        "metric": "rmse",
        "metric_names": metric_names,
        "runs": entries,
        "best": best,
        "best_per_metric": best_per_metric or {},
    }


def test_comparison_charts_shape():
    from surroforge_professional.reports.charts import build_comparison_charts

    charts = build_comparison_charts(
        _comparison(
            [_comparison_entry("r1", rmse=0.2, r2=0.9), _comparison_entry("r2", rmse=0.3, r2=0.8)],
            ["rmse", "r2"],
            best_per_metric={"rmse": "r1", "r2": "r1"},
        )
    )
    assert charts is not None
    assert charts["runs"] == ["r1", "r2"]
    assert charts["metrics"] == ["rmse", "r2"]
    assert charts["best"] == "r1"
    assert charts["best_per_metric"] == {"rmse": "r1", "r2": "r1"}
    assert charts["per_metric"]["rmse"]["test"] == [0.2, 0.3]
    assert charts["per_metric"]["rmse"]["cv_std"] == [0.01, 0.01]


def test_comparison_charts_missing_metric_becomes_none_hole():
    from surroforge_professional.reports.charts import build_comparison_charts

    charts = build_comparison_charts(
        _comparison(
            [_comparison_entry("r1", rmse=0.2), _comparison_entry("r2")], ["rmse"]
        )
    )
    assert charts is not None
    assert charts["per_metric"]["rmse"]["test"] == [0.2, None]


def test_comparison_charts_none_when_nothing_plottable():
    from surroforge_professional.reports.charts import build_comparison_charts

    entry = {"run_id": "r1", "metrics": {"rmse": {"train": None, "test": None,
                                                  "cv_mean": None, "cv_std": None}}}
    assert build_comparison_charts(_comparison([entry], ["rmse"])) is None


def test_comparison_charts_never_raises_on_garbage():
    from surroforge_professional.reports.charts import build_comparison_charts

    assert build_comparison_charts({}) is None
    assert build_comparison_charts({"runs": "junk", "metric_names": ["rmse"]}) is None
    assert build_comparison_charts(
        {"runs": [{"run_id": "r1", "metrics": "junk"}], "metric_names": ["rmse"]}
    ) is None
    assert build_comparison_charts(
        _comparison([_comparison_entry("r1", rmse=float("nan"))], ["rmse"])
    ) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest plugins/surroforge-professional/tests/test_charts.py -v -k comparison`
Expected: FAIL with `ImportError: cannot import name 'build_comparison_charts'`.

- [ ] **Step 3: Implement** — append to `charts.py` (its module docstring already promises pure/never-raise semantics; this function keeps them):

```python
def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def build_comparison_charts(comparison: dict[str, Any]) -> dict[str, Any] | None:
    """Cross-run chart arrays from a ``build_comparison`` result, or ``None``.

    Lists are parallel to ``runs``; a value a run lacks becomes ``None`` (uPlot
    renders a gap). Metrics with no finite value in any run are dropped; if no
    metric survives, the whole chart payload is ``None``.
    """
    entries = comparison.get("runs")
    metric_names = comparison.get("metric_names")
    if not isinstance(entries, list) or not isinstance(metric_names, list):
        return None
    runs: list[str] = []
    per_run: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or "run_id" not in entry:
            return None
        runs.append(str(entry["run_id"]))
        metrics = entry.get("metrics")
        per_run.append(metrics if isinstance(metrics, dict) else {})

    per_metric: dict[str, dict[str, list[float | None]]] = {}
    kept: list[str] = []
    for name in metric_names:
        series: dict[str, list[float | None]] = {
            "train": [], "test": [], "cv_mean": [], "cv_std": []
        }
        for metrics in per_run:
            record = metrics.get(name)
            record = record if isinstance(record, dict) else {}
            for key, values in series.items():
                values.append(_finite(record.get(key)))
        if any(v is not None for v in series["train"] + series["test"] + series["cv_mean"]):
            kept.append(str(name))
            per_metric[str(name)] = series
    if not runs or not kept:
        return None

    best = comparison.get("best")
    best_per_metric = comparison.get("best_per_metric")
    return {
        "runs": runs,
        "metrics": kept,
        "best": best if isinstance(best, str) else None,
        "best_per_metric": best_per_metric if isinstance(best_per_metric, dict) else {},
        "per_metric": per_metric,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest plugins/surroforge-professional/tests/test_charts.py -v`
Expected: ALL PASS (pre-existing chart tests included).

- [ ] **Step 5: Static gates**

Run: `.venv/bin/ruff check plugins/surroforge-professional && .venv/bin/pyright 2>/dev/null | grep -E '^[0-9]+ error'`
Expected: clean / `0 errors`.

- [ ] **Step 6: Commit**

```bash
git add plugins/surroforge-professional/src/surroforge_professional/reports/charts.py plugins/surroforge-professional/tests/test_charts.py
git commit -m "feat(professional): add cross-run comparison chart data"
```

---

### Task 4: HTML comparison reporter (template, entry point, shared CSS)

**Files:**
- Modify: `plugins/surroforge-professional/src/surroforge_professional/reports/html.py` (extract `_BASE_CSS`)
- Create: `plugins/surroforge-professional/src/surroforge_professional/reports/comparison.py`
- Modify: `plugins/surroforge-professional/pyproject.toml` (entry point)
- Test: `plugins/surroforge-professional/tests/test_comparison_report.py` (new), plus a `second_run` fixture in `plugins/surroforge-professional/tests/conftest.py`

**Interfaces:**
- Consumes: `BaseComparisonReporter` (Task 1), `generate_comparison_report` (Task 2), `build_comparison_charts` (Task 3), and from the existing modules: `require_license` (license.py), `fmt` + `_charts_json` (context.py), `_UPLOT_JS`/`_UPLOT_CSS`/`_UPLOT_LICENSE_COMMENT` (html.py module globals).
- Produces: `HtmlComparisonReporter` registered as entry point `html` in group `surroforge.comparison_reporters`; `_BASE_CSS` in `html.py` (used by both templates).

**Design note (documented deviation):** the spec asks for side-by-side grouped bars. uPlot's stock `paths.bars` draws every series centred on the x value; true side-by-side needs the `disp`-based offset math from uPlot's demo code, which nothing in this pipeline can test. The template below instead renders **layered bars** (train widest 0.72, test 0.5, CV mean narrowest 0.28 — widest drawn first, so all three stay readable), matching the overlay aesthetic the single-run report's metric chart already ships. The ±std whiskers and the legend readout carry the exact values either way. Per-metric best marking: a "best" chip in the chart heading plus a `★` suffix on that run's x-axis label (the spec's accent-colour bar would need per-datapoint fills — same untestable `disp` machinery).

- [ ] **Step 1: Extract `_BASE_CSS` in `html.py`**

Cut the CSS body out of the template literal: everything between the template's opening `  <style>` and closing `  </style>` (html.py lines 41–188 — from `    :root {` down to and including the `@media (max-width: 720px) { ... }` block's closing `    }`) moves verbatim into a new module-level constant defined after `_UPLOT_LICENSE_COMMENT`:

```python
# Shared page styling for the Professional reports. The comparison report embeds the
# same block, so the two documents render with one visual language and cannot drift.
_BASE_CSS = """
    :root {
      ... (the moved 148 lines, byte-identical) ...
    }
"""
```

In the template, the emptied block becomes exactly:

```
  <style>{{ base_css|safe }}</style>
```

and `HtmlReporter.render`'s `_TEMPLATE.render(...)` call gains the argument `base_css=_BASE_CSS,`.

- [ ] **Step 2: Verify the single-run report is unchanged**

Run: `.venv/bin/pytest plugins/surroforge-professional/tests/test_html_report.py -v`
Expected: ALL PASS (the rendered page is byte-identical up to the moved whitespace inside `<style>`).

- [ ] **Step 3: Add the `second_run` fixture** — append to `plugins/surroforge-professional/tests/conftest.py`:

```python
@pytest.fixture
def second_run(project):
    config = parse_config(
        {
            "experiment": "beam-professional-html-2",
            "data": {
                "source": "beam",
                "inputs": ["length", "width", "load"],
                "outputs": ["max_stress"],
                "split": {"test_fraction": 0.2, "seed": 42},
            },
            "model": {"method": "polynomial", "degree": 1},
            "validation": {"cross_validation": 5, "metrics": ["rmse", "r2", "mae"]},
        }
    )
    return run_training(project, config)
```

(`parse_config` and `run_training` are already imported in that conftest. Degree 1 vs. the existing `trained_run`'s degree 2 makes `config_diff` non-empty — the diff-section test depends on it.)

- [ ] **Step 4: Write the failing tests** — create `plugins/surroforge-professional/tests/test_comparison_report.py`:

```python
"""Professional HTML comparison report (spec 2026-07-05)."""

from __future__ import annotations

import json
import re

import pytest

from surroforge.comparison import build_comparison
from surroforge.errors import PluginError
from surroforge_professional.reports.comparison import HtmlComparisonReporter


def _minimal_comparison(**overrides):
    base = {
        "metric": "rmse",
        "metric_names": ["rmse"],
        "runs": [
            {
                "run_id": "run-001",
                "method": "polynomial",
                "experiment": "exp",
                "created_at": "2026-07-05T12:00:00",
                "rank": 1,
                "delta": 0.0,
                "metrics": {
                    "rmse": {"train": 0.1, "test": 0.2, "cv_mean": 0.15, "cv_std": 0.01}
                },
                "lineage": {
                    "seed": 0, "data_hash": "abc123", "config_hash": "def456",
                    "parent_run": None,
                },
            }
        ],
        "best": "run-001",
        "best_per_metric": {"rmse": "run-001"},
        "config_diff": {},
        "lineage_summary": {"same_data": True, "data_hash_groups": {"abc123": ["run-001"]}},
        "warnings": [],
    }
    base.update(overrides)
    return base


@pytest.fixture
def comparison(trained_run, second_run):
    return build_comparison([trained_run, second_run], metric="rmse")


@pytest.fixture
def html(licensed, comparison):
    return HtmlComparisonReporter().render(comparison)


def test_requires_license(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(PluginError, match="license"):
        HtmlComparisonReporter().render(_minimal_comparison())


def test_smoke_and_sections(html):
    assert html.startswith("<!doctype html>")
    for heading in (
        "SurroForge Run Comparison",
        "Best Run",
        "Ranking",
        "Metric Charts",
        "All Metrics",
        "Configuration Differences",
        "Lineage",
    ):
        assert heading in html


def test_self_contained(html):
    for marker in ('src="http', 'href="http', "@import", "url(http"):
        assert marker not in html


def test_ranking_shows_every_run_and_marks_best(html, comparison):
    for entry in comparison["runs"]:
        assert entry["run_id"] in html
    assert comparison["best"] in html
    assert ">best</span>" in html


def test_config_diff_lists_differing_field(html):
    assert "model.degree" in html


def test_chart_json_embeds_all_runs(html, comparison):
    match = re.search(
        r'<script type="application/json" id="sf-comparison-charts">(.*?)</script>',
        html,
        re.S,
    )
    assert match
    data = json.loads(match.group(1))
    assert data["runs"] == [entry["run_id"] for entry in comparison["runs"]]
    assert set(data["metrics"]) <= set(comparison["metric_names"])
    first = data["per_metric"][data["metrics"][0]]
    assert len(first["test"]) == len(data["runs"])


def test_uplot_attribution_present(html):
    assert "uPlot v1.6.32 (MIT License)" in html


def test_warning_banner(licensed):
    text = HtmlComparisonReporter().render(
        _minimal_comparison(warnings=["Runs use different datasets (data_hash differs)."])
    )
    assert "Warnings" in text
    assert "different datasets" in text


def test_hostile_strings_are_escaped(licensed):
    comparison = _minimal_comparison()
    comparison["runs"][0]["experiment"] = '<script>alert("xss")</script>'
    comparison["config_diff"] = {
        "model.note": {"run-001": "<img src=x onerror=alert(1)>"}
    }
    text = HtmlComparisonReporter().render(comparison)
    assert '<script>alert("xss")</script>' not in text
    assert "<img src=x" not in text
    assert "&lt;script&gt;" in text


def test_single_run_comparison_renders(licensed):
    text = HtmlComparisonReporter().render(_minimal_comparison())
    assert text.startswith("<!doctype html>")


def test_charts_section_omitted_when_nothing_plottable(licensed):
    comparison = _minimal_comparison()
    comparison["runs"][0]["metrics"] = {
        "rmse": {"train": None, "test": None, "cv_mean": None, "cv_std": None}
    }
    text = HtmlComparisonReporter().render(comparison)
    assert "sf-comparison-charts" not in text
    assert "Metric Charts" not in text


def test_generate_comparison_report_end_to_end(licensed, project, trained_run, second_run):
    from surroforge.services import generate_comparison_report

    path = generate_comparison_report(
        project, [trained_run.run_id, second_run.run_id], metric="rmse", format="html"
    )
    assert path == project.reports_dir / "comparison.html"
    assert path.read_text().startswith("<!doctype html>")


def test_cli_compare_format_html_full_stack(licensed, project, trained_run, second_run, tmp_path):
    from typer.testing import CliRunner

    from surroforge.cli.main import app

    out = tmp_path / "cmp.html"
    result = CliRunner().invoke(
        app,
        [
            "--config-dir", str(project.root), "compare",
            "--runs", f"{trained_run.run_id},{second_run.run_id}",
            "--format", "html", "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.read_text().startswith("<!doctype html>")
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `.venv/bin/pytest plugins/surroforge-professional/tests/test_comparison_report.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'surroforge_professional.reports.comparison'`.

- [ ] **Step 6: Create `comparison.py`** — full content:

```python
"""Static HTML comparison report for SurroForge Professional.

Consumes the JSON-serializable dict that :func:`surroforge.comparison.build_comparison`
produces and renders one self-contained page: best-run callout, ranking, per-metric
charts, the full metric table, config diff, and lineage. Shares the visual language
and the vendored uPlot bundle with the single-run report in
:mod:`surroforge_professional.reports.html`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jinja2 import Environment

from surroforge.reporters.base import BaseComparisonReporter
from surroforge_professional.license import require_license
from surroforge_professional.reports.charts import build_comparison_charts
from surroforge_professional.reports.context import _charts_json, fmt
from surroforge_professional.reports.html import (
    _BASE_CSS,
    _UPLOT_CSS,
    _UPLOT_JS,
    _UPLOT_LICENSE_COMMENT,
)


def _num_fmt(value: Any) -> str:
    return fmt(value) if isinstance(value, (int, float)) else "-"


def _fmt_cv(mean: Any, std: Any) -> str:
    if not isinstance(mean, (int, float)):
        return "-"
    if isinstance(std, (int, float)):
        return f"{fmt(mean)}±{fmt(std)}"
    return fmt(mean)


def _abbrev(value: Any, n: int = 10) -> str:
    text = str(value) if value not in (None, "") else "-"
    return f"{text[:n]}…" if len(text) > n else text


def _best_summary(comparison: dict[str, Any]) -> dict[str, Any] | None:
    best = comparison["best"]
    if best is None:
        return None
    entry = next((e for e in comparison["runs"] if e["run_id"] == best), None)
    if entry is None:
        return None
    record = entry["metrics"].get(comparison["metric"]) or {}
    return {
        "run_id": best,
        "method": entry["method"],
        "experiment": entry["experiment"],
        "test_fmt": _num_fmt(record.get("test")),
    }


def _best_chips(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    chips = [{"metric": m, "run_id": r} for m, r in comparison["best_per_metric"].items()]
    if len({chip["run_id"] for chip in chips}) > 1:
        return chips
    return []


def _ranking_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    metric = comparison["metric"]
    entries = comparison["runs"]
    ranked = sorted((e for e in entries if e["rank"] is not None), key=lambda e: e["rank"])
    unranked = [e for e in entries if e["rank"] is None]
    rows = []
    for entry in ranked + unranked:
        record = entry["metrics"].get(metric) or {}
        delta = entry["delta"]
        rows.append(
            {
                "rank": str(entry["rank"]) if entry["rank"] is not None else "–",
                "run_id": entry["run_id"],
                "method": entry["method"],
                "experiment": entry["experiment"],
                "test_fmt": _num_fmt(record.get("test")),
                "delta_fmt": "—" if delta in (None, 0.0) else f"+{fmt(delta)}",
                "is_best": entry["run_id"] == comparison["best"],
            }
        )
    return rows


def _metric_table(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    names = comparison["metric_names"]
    best = comparison["best_per_metric"]
    rows = []
    for entry in comparison["runs"]:
        cells = []
        for name in names:
            record = entry["metrics"].get(name) or {}
            cells.append(
                {
                    "train_fmt": _num_fmt(record.get("train")),
                    "test_fmt": _num_fmt(record.get("test")),
                    "cv_fmt": _fmt_cv(record.get("cv_mean"), record.get("cv_std")),
                    "is_best": best.get(name) == entry["run_id"],
                }
            )
        rows.append({"run_id": entry["run_id"], "cells": cells})
    return rows


def _config_diff_rows(comparison: dict[str, Any], run_ids: list[str]) -> list[dict[str, Any]]:
    diff = comparison["config_diff"]
    rows = []
    for path in sorted(diff):
        values = diff[path]
        cells = []
        for rid in run_ids:
            value = values.get(rid, "<absent>")
            absent = value == "<absent>"
            cells.append({"absent": absent, "text": "" if absent else str(value)})
        rows.append({"path": path, "cells": cells})
    return rows


def _lineage(comparison: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "run_id": entry["run_id"],
            "seed": entry["lineage"]["seed"],
            "data_hash": _abbrev(entry["lineage"]["data_hash"]),
            "config_hash": _abbrev(entry["lineage"]["config_hash"]),
            "parent_run": entry["lineage"]["parent_run"] or "—",
            "created_at": str(entry["created_at"])[:19],
        }
        for entry in comparison["runs"]
    ]
    summary = comparison["lineage_summary"]
    groups = [
        {"data_hash": _abbrev(data_hash), "run_ids": run_ids}
        for data_hash, run_ids in summary["data_hash_groups"].items()
    ]
    return {"rows": rows, "same_data": summary["same_data"], "groups": groups}


def _build_context(comparison: dict[str, Any]) -> dict[str, Any]:
    run_ids = [entry["run_id"] for entry in comparison["runs"]]
    charts = build_comparison_charts(comparison)
    return {
        "metric": comparison["metric"],
        "n_runs": len(run_ids),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "best": _best_summary(comparison),
        "best_chips": _best_chips(comparison),
        "warnings": comparison["warnings"],
        "ranking_rows": _ranking_rows(comparison),
        "metric_names": comparison["metric_names"],
        "metric_table": _metric_table(comparison),
        "run_ids": run_ids,
        "config_diff_rows": _config_diff_rows(comparison, run_ids),
        "config_identical": not comparison["config_diff"],
        "lineage": _lineage(comparison),
        "charts": charts,
        "charts_json": _charts_json(charts),
    }


_ENV = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True)
_TEMPLATE = _ENV.from_string(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SurroForge Comparison</title>
  <style>{{ base_css|safe }}</style>
  <style>
    .warnings {
      background: #fdf6e3;
      border: 1px solid #e0c56e;
      border-radius: 8px;
      margin: 0 0 20px;
      padding: 12px 16px;
    }
    .warnings ul { margin: 4px 0 0; }
    .chip {
      background: var(--accent-soft);
      border-radius: 999px;
      color: var(--accent);
      display: inline-block;
      font-size: 0.85rem;
      margin: 0 6px 6px 0;
      padding: 3px 10px;
    }
    .absent { color: var(--muted); font-style: italic; }
    tr.best td { background: var(--accent-soft); }
  </style>
  {% if charts %}<style>{{ uplot_css|safe }}</style>{% endif %}
</head>
<body>
  <main>
    <header>
      <h1>SurroForge Run Comparison</h1>
      <p class="lede">{{ n_runs }} runs | ranked by test {{ metric }} | generated {{ generated_at }}</p>
    </header>

    {% if warnings %}
    <div class="warnings" role="alert">
      <strong>Warnings</strong>
      <ul>{% for warning in warnings %}<li>{{ warning }}</li>{% endfor %}</ul>
    </div>
    {% endif %}

    {% if best %}
    <section aria-labelledby="best-run">
      <h2 id="best-run">Best Run</h2>
      <div class="summary-grid">
        <div class="fact"><span class="label">Run</span><span class="value mono">{{ best.run_id }}</span></div>
        <div class="fact"><span class="label">Method</span><span class="value">{{ best.method }}</span></div>
        <div class="fact"><span class="label">Experiment</span><span class="value">{{ best.experiment }}</span></div>
        <div class="fact"><span class="label">Test {{ metric }}</span><span class="value mono">{{ best.test_fmt }}</span></div>
      </div>
      {% if best_chips %}
      <p style="margin-top:10px">Best per metric:
        {% for chip in best_chips %}<span class="chip">{{ chip.metric }}: {{ chip.run_id }}</span>{% endfor %}
      </p>
      {% endif %}
    </section>
    {% endif %}

    <section aria-labelledby="ranking">
      <h2 id="ranking">Ranking</h2>
      <table>
        <thead>
          <tr><th>Rank</th><th>Run</th><th>Method</th><th>Experiment</th><th>Test {{ metric }}</th><th>Δ to best</th></tr>
        </thead>
        <tbody>
          {% for row in ranking_rows %}
          <tr{% if row.is_best %} class="best"{% endif %}>
            <td>{{ row.rank }}</td>
            <td class="mono">{{ row.run_id }}{% if row.is_best %} <span class="badge">best</span>{% endif %}</td>
            <td>{{ row.method }}</td>
            <td>{{ row.experiment }}</td>
            <td class="mono">{{ row.test_fmt }}</td>
            <td class="mono">{{ row.delta_fmt }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    {% if charts %}
    <section aria-labelledby="metric-charts">
      <h2 id="metric-charts">Metric Charts</h2>
      {% for name in charts.metrics %}
      <h3>{{ name }}{% if charts.best_per_metric.get(name) %} — best: <span class="chip">{{ charts.best_per_metric.get(name) }}</span>{% endif %}</h3>
      <div id="sf-metric-{{ loop.index0 }}" class="sf-chart"></div>
      {% endfor %}
    </section>
    {% endif %}

    <section aria-labelledby="all-metrics">
      <h2 id="all-metrics">All Metrics</h2>
      <table>
        <thead>
          <tr><th>Run</th>{% for name in metric_names %}<th>{{ name }} (train / test / cv)</th>{% endfor %}</tr>
        </thead>
        <tbody>
          {% for row in metric_table %}
          <tr>
            <td class="mono">{{ row.run_id }}</td>
            {% for cell in row.cells %}
            <td class="mono">{{ cell.train_fmt }} / {{ cell.test_fmt }}{% if cell.is_best %} <span class="badge">best</span>{% endif %} / {{ cell.cv_fmt }}</td>
            {% endfor %}
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    <section aria-labelledby="config-diff">
      <h2 id="config-diff">Configuration Differences</h2>
      {% if config_identical %}
      <p>Configurations are identical.</p>
      {% else %}
      <table>
        <thead><tr><th>Field</th>{% for rid in run_ids %}<th class="mono">{{ rid }}</th>{% endfor %}</tr></thead>
        <tbody>
          {% for row in config_diff_rows %}
          <tr>
            <td class="mono">{{ row.path }}</td>
            {% for cell in row.cells %}
            <td class="mono">{% if cell.absent %}<span class="absent">absent</span>{% else %}{{ cell.text }}{% endif %}</td>
            {% endfor %}
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% endif %}
    </section>

    <section aria-labelledby="lineage">
      <h2 id="lineage">Lineage</h2>
      <table>
        <thead>
          <tr><th>Run</th><th>Seed</th><th>Data Hash</th><th>Config Hash</th><th>Parent</th><th>Created</th></tr>
        </thead>
        <tbody>
          {% for row in lineage.rows %}
          <tr>
            <td class="mono">{{ row.run_id }}</td>
            <td class="mono">{{ row.seed }}</td>
            <td class="mono">{{ row.data_hash }}</td>
            <td class="mono">{{ row.config_hash }}</td>
            <td class="mono">{{ row.parent_run }}</td>
            <td class="mono">{{ row.created_at }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% if lineage.same_data %}
      <p>All runs share the same data (data_hash {{ lineage.groups[0].data_hash }}).</p>
      {% else %}
      <p>Runs use different datasets:</p>
      <ul>
        {% for group in lineage.groups %}
        <li><span class="mono">{{ group.data_hash }}</span>: {{ group.run_ids|join(', ') }}</li>
        {% endfor %}
      </ul>
      {% endif %}
    </section>

    {% if charts %}
    <section aria-labelledby="viz-lib">
      <h2 id="viz-lib">Visualization Library</h2>
      <p>Interactive charts rendered with uPlot (MIT License), bundled inline.</p>
    </section>
    <script type="application/json" id="sf-comparison-charts">{{ charts_json|safe }}</script>
    {{ uplot_license_comment|safe }}
    <script>{{ uplot_js|safe }}</script>
    <script>
    (function () {
      var holder = document.getElementById('sf-comparison-charts');
      if (!holder || typeof uPlot === 'undefined') return;
      var data;
      try { data = JSON.parse(holder.textContent); } catch (err) { return; }

      var palette = ['#2056a3', '#1f8a5b', '#b58a1f'];

      function dims(node) {
        var w = node.clientWidth || 640;
        return { width: w, height: Math.max(220, Math.round(w * 0.45)) };
      }
      function mount(node, opts, series) {
        if (!node) return null;
        var d = dims(node);
        opts.width = d.width; opts.height = d.height;
        var u = new uPlot(opts, series, node);
        window.addEventListener('resize', function () {
          var n = dims(node); u.setSize({ width: n.width, height: n.height });
        });
        return u;
      }

      var xs = data.runs.map(function (_, i) { return i; });
      data.metrics.forEach(function (name, mi) {
        var pm = data.per_metric[name];
        var node = document.getElementById('sf-metric-' + mi);
        if (!node || !pm) return;
        var bestRun = data.best_per_metric[name];

        // y-range covering bars AND whiskers (cv_mean ± std), padded 5%.
        var lo = 0, hi = -Infinity;
        ['train', 'test', 'cv_mean'].forEach(function (key) {
          pm[key].forEach(function (v) {
            if (v == null) return;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          });
        });
        data.runs.forEach(function (_, i) {
          var mean = pm.cv_mean[i], std = pm.cv_std[i];
          if (mean == null || std == null) return;
          if (mean + std > hi) hi = mean + std;
          if (mean - std < lo) lo = mean - std;
        });
        if (hi === -Infinity) return;
        var pad = 0.05 * ((hi - lo) || Math.abs(hi) || 1);

        // Layered bars: widest drawn first, narrowest on top, so all three stay
        // readable at every x; exact values live in the legend readout.
        var layers = [
          { label: 'train',   values: pm.train,   size: 0.72 },
          { label: 'test',    values: pm.test,    size: 0.5 },
          { label: 'CV mean', values: pm.cv_mean, size: 0.28 }
        ];
        var series = [{}];
        var chartData = [xs];
        layers.forEach(function (layer, i) {
          chartData.push(layer.values);
          series.push({
            label: layer.label, stroke: palette[i], fill: palette[i],
            paths: uPlot.paths.bars({ size: [layer.size, 100] })
          });
        });
        mount(node, {
          scales: {
            x: { time: false },
            y: { range: [lo === 0 ? 0 : lo - pad, hi + pad] }
          },
          axes: [
            { label: 'run', values: function (u, vals) {
                return vals.map(function (v) {
                  var rid = data.runs[v];
                  if (rid == null) return '';
                  return rid + (rid === bestRun ? ' \\u2605' : '');
                });
            } },
            { label: name }
          ],
          series: series,
          hooks: {
            draw: [function (u) {
              // ±std whiskers on the CV-mean bar.
              var ctx = u.ctx;
              ctx.save();
              ctx.strokeStyle = '#172033';
              ctx.lineWidth = 1;
              data.runs.forEach(function (_, i) {
                var mean = pm.cv_mean[i], std = pm.cv_std[i];
                if (mean == null || std == null) return;
                var cx = u.valToPos(i, 'x', true);
                var yLo = u.valToPos(mean - std, 'y', true);
                var yHi = u.valToPos(mean + std, 'y', true);
                var cap = 5;
                ctx.beginPath();
                ctx.moveTo(cx, yLo); ctx.lineTo(cx, yHi);
                ctx.moveTo(cx - cap, yLo); ctx.lineTo(cx + cap, yLo);
                ctx.moveTo(cx - cap, yHi); ctx.lineTo(cx + cap, yHi);
                ctx.stroke();
              });
              ctx.restore();
            }]
          }
        }, chartData);
      });
    })();
    </script>
    {% endif %}
  </main>
</body>
</html>
"""
)


class HtmlComparisonReporter(BaseComparisonReporter):
    name = "html"
    extension = ".html"

    def render(self, comparison: dict[str, Any]) -> str:
        require_license()
        return _TEMPLATE.render(
            **_build_context(comparison),
            base_css=_BASE_CSS,
            uplot_js=_UPLOT_JS,
            uplot_css=_UPLOT_CSS,
            uplot_license_comment=_UPLOT_LICENSE_COMMENT,
        )
```

**Jinja gotcha, deliberately avoided:** config-diff rows use the key `cells`, NOT `values` — `row.values` in Jinja resolves to the dict *method* `values()`, not the key. Also note the axis-label star is `\\u2605` inside the Python string so the rendered JS contains `★`.

- [ ] **Step 7: Register the entry point** — in `plugins/surroforge-professional/pyproject.toml`, after the `[project.entry-points."surroforge.commands"]` table add:

```toml
[project.entry-points."surroforge.comparison_reporters"]
html = "surroforge_professional.reports.comparison:HtmlComparisonReporter"
```

Then refresh the editable install's metadata (entry points live in dist-info, not the source tree):

Run: `.venv/bin/pip install -e plugins/surroforge-professional --no-deps -q && .venv/bin/python -c "from surroforge.plugins import available_comparison_reporters; print(available_comparison_reporters())"`
Expected: `['html']`

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/pytest plugins/surroforge-professional/tests -v`
Expected: ALL PASS (new file's 13 tests + all pre-existing plugin tests).

- [ ] **Step 9: Static gates**

Run: `.venv/bin/ruff check plugins/surroforge-professional && .venv/bin/pyright 2>/dev/null | grep -E '^[0-9]+ error'`
Expected: clean / `0 errors`.

- [ ] **Step 10: Commit**

```bash
git add plugins/surroforge-professional/src/surroforge_professional/reports/html.py \
        plugins/surroforge-professional/src/surroforge_professional/reports/comparison.py \
        plugins/surroforge-professional/pyproject.toml \
        plugins/surroforge-professional/tests/conftest.py \
        plugins/surroforge-professional/tests/test_comparison_report.py
git commit -m "feat(professional): HTML comparison report behind compare --format html"
```

---

### Task 5: Docs, real-data acceptance, full regression

**Files:**
- Modify: `README.md` (one line in the quickstart, ~line 47)
- Modify: `docs/scenarios/method-comparison.md` (one example + one sentence, ~line 46)

**Interfaces:** consumes the finished feature; produces the final green light. No code changes.

- [ ] **Step 1: Docs** — in `README.md` directly under the existing line
`surroforge compare --runs run-001,run-002` add:

```
surroforge compare --runs run-001,run-002 --format html   # Professional: interactive HTML report
```

In `docs/scenarios/method-comparison.md`, directly under the existing
`surroforge compare --runs run-001,run-002 --metric rmse` code line add (inside the same code block):

```
surroforge compare --runs run-001,run-002 --format html -o comparison.html   # Professional plugin
```

and after that code block append the sentence:

```
With the Professional plugin installed, `--format html` writes a self-contained
interactive comparison report (`reports/comparison.html` by default) instead of the
terminal tables.
```

- [ ] **Step 2: Real-data acceptance against the saved NACA runs (read-only on that project)**

```bash
SURROFORGE_PRO_LICENSE=local-dev-license .venv/bin/surroforge \
  --config-dir examples/naca0012_openfoam/ex09_real/naca compare \
  --runs run-001,run-002,run-003,run-004,run-005,run-006 -f html \
  -o /private/tmp/claude-501/-Users-georg-Software-surroforge/35056efb-276a-40ab-95eb-ab773c0f6980/scratchpad/naca-vergleich.html
```

Expected: exit 0, "Wrote html comparison report to …". Then verify content:

```bash
head -c 60 /private/tmp/claude-501/-Users-georg-Software-surroforge/35056efb-276a-40ab-95eb-ab773c0f6980/scratchpad/naca-vergleich.html
grep -c 'run-006' /private/tmp/claude-501/-Users-georg-Software-surroforge/35056efb-276a-40ab-95eb-ab773c0f6980/scratchpad/naca-vergleich.html
```

Expected: starts `<!doctype html>`; `run-006` appears (>0). These six runs mix field and coefficient experiments, so the **warnings banner must appear** (different experiments/outputs): `grep -c 'Warnings' …` > 0. Report the output path so the user can open it in a browser.

- [ ] **Step 3: Full regression**

Run: `scripts/ci.sh` (600000 ms timeout) and `.venv/bin/pytest plugins/surroforge-professional/tests -q`
Expected: all required ci.sh gates green (pytest 0 failed; pyright 0 errors); plugin suite all green.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/scenarios/method-comparison.md
git commit -m "docs: document compare --format html"
```

- [ ] **Step 5: Final report to the user** — commits + changed files + acceptance evidence + the path to `naca-vergleich.html`. **Do not push** — the user reviews first.
