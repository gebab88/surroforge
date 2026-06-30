# Interactive HTML Visualizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four interactive charts (parity, residual, per-fold CV, metric comparison) to the SurroForge Professional HTML report, rendered with vendored uPlot, fully self-contained.

**Architecture:** A small additive Core change persists holdout *actual* values next to the existing prediction fingerprint. A new, isolated `build_charts(run)` function in the Professional plugin turns a `Run` into numeric chart arrays; the HTML template inlines vendored uPlot (MIT) plus a small init script that draws the charts and wires a multi-output dropdown.

**Tech Stack:** Python 3.10+, scikit-learn-trained runs, Jinja2 (autoescape), numpy, uPlot 1.6.32 (MIT, vendored as static asset — *not* a pip dependency).

## Global Constraints

- **No new pip dependency.** uPlot is vendored as a static asset under the plugin package; existing deps (`jinja2>=3.1`, `numpy>=1.24`) are unchanged.
- **Self-contained HTML, no network:** the rendered report must contain no external resource load — assert absence of `src="http`, `href="http`, `@import`, and `url(http`. (uPlot's banner comment `https://github.com/leeoniya/uPlot` is a comment, not a resource load, and is allowed.)
- **XSS-safe:** every dynamic value flows through Jinja autoescape; the embedded chart JSON has every `<` replaced by its `<` escape and is emitted with `|safe`.
- **License gate unchanged:** `require_license()` runs first in `HtmlReporter.render()`.
- **Additive Core change must not alter any hash or `reproduce()` behavior** — `holdout_actual` feeds no hash and is never read by the reproduce verdict.
- **uPlot pinned to v1.6.32 (MIT)**, attribution carried inline in the report.
- **Lint:** plugin uses ruff line-length 100; run `ruff check` on touched trees before each commit.
- **Graceful degradation:** `build_charts` never raises on missing/old/malformed run data — it omits a chart or returns `None`; the report always renders.

---

### Task 1: Core — persist holdout actuals + `Run.holdout_actual`

**Files:**
- Modify: `src/surroforge/engine.py` (around lines 161-164 and 296)
- Modify: `src/surroforge/core/run.py` (after the `holdout_fingerprint` property, ~line 142)
- Test: `tests/test_training.py`

**Interfaces:**
- Consumes: existing engine locals `y_test` (line 124) and `_FINGERPRINT_MAX_ROWS = 512`; the `lineage` dict assembled at ~line 286.
- Produces: lineage key `"holdout_actual"` (list, row-aligned with `holdout_fingerprint`); `Run.holdout_actual -> list[Any]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_training.py` (the file already imports `np`, `parse_config`, `run_training`, and `poly_config`; add `import pandas as pd` and `from surroforge.core.project import Project` at the top if not present):

```python
def test_holdout_actual_persisted_and_aligned_with_fingerprint(project):
    run = run_training(project, parse_config(poly_config("polynomial")))
    actual = np.asarray(run.holdout_actual, dtype=float)
    pred = np.asarray(run.holdout_fingerprint, dtype=float)
    assert "holdout_actual" in run.lineage
    assert actual.shape == pred.shape          # same rows AND same columns
    assert actual.ndim == 2 and actual.shape[1] == 1
    assert np.isfinite(actual).all()


def test_holdout_actual_absent_returns_empty_list(project):
    run = run_training(project, parse_config(poly_config("polynomial")))
    run.lineage.pop("holdout_actual")          # simulate a pre-feature run
    assert run.holdout_actual == []


def test_holdout_actual_multi_output_shape(tmp_path):
    rng = np.random.default_rng(1)
    n = 60
    a = rng.uniform(0.0, 1.0, n)
    b = rng.uniform(0.0, 1.0, n)
    df = pd.DataFrame({"a": a, "b": b, "y1": 2 * a + b, "y2": a - 3 * b + 0.5})
    csv = tmp_path / "multi.csv"
    df.to_csv(csv, index=False)
    proj = Project.init(tmp_path / "p", name="multi")
    proj.add_dataset(csv, name="multi")
    cfg = parse_config(
        {
            "experiment": "multi",
            "data": {
                "source": "multi",
                "inputs": ["a", "b"],
                "outputs": ["y1", "y2"],
                "split": {"test_fraction": 0.25, "seed": 1},
            },
            "model": {"method": "polynomial", "degree": 1},
            "validation": {"cross_validation": 4, "metrics": ["rmse", "r2"]},
        }
    )
    run = run_training(proj, cfg)
    actual = np.asarray(run.holdout_actual, dtype=float)
    pred = np.asarray(run.holdout_fingerprint, dtype=float)
    assert actual.shape == pred.shape
    assert actual.shape[1] == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_training.py -k holdout_actual -v`
Expected: FAIL — `Run` has no attribute `holdout_actual` / `"holdout_actual"` not in lineage.

- [ ] **Step 3: Persist the actuals in the engine**

In `src/surroforge/engine.py`, immediately after the `prediction_hash = _prediction_fingerprint(holdout_fingerprint)` line (currently line 164), insert:

```python
    # Holdout *actuals* row-aligned with the fingerprint above (same split, same
    # [:_FINGERPRINT_MAX_ROWS] slice over the same x_test row order), so the
    # Professional report can plot predicted-vs-actual. Inert to every hash and to
    # reproduce(), which keys on holdout_fingerprint — this is report-only data.
    holdout_actual = np.ascontiguousarray(y_test, dtype=np.float64)[:_FINGERPRINT_MAX_ROWS]
```

Then in the `lineage = {` dict, directly after the existing line
`"holdout_fingerprint": holdout_fingerprint.tolist(),` (line 296), add:

```python
        "holdout_actual": holdout_actual.tolist(),
```

- [ ] **Step 4: Add the `Run` accessor**

In `src/surroforge/core/run.py`, directly after the `holdout_fingerprint` property (which ends at line 142), add:

```python
    @property
    def holdout_actual(self) -> list[Any]:
        """Out-of-sample holdout actual targets, row-aligned with :attr:`holdout_fingerprint`.

        Persisted next to the prediction fingerprint so the Professional HTML report can
        plot predicted-vs-actual. Inert to the reproduce verdict (which keys on the
        fingerprint) and to every hash. Empty list for runs trained before this was
        recorded.
        """
        return self.lineage.get("holdout_actual", [])
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_training.py -k holdout_actual -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Verify nothing else regressed + lint**

Run: `python -m pytest tests/test_training.py tests/test_reproduce_scalar_identity.py tests/test_reproduce.py -q && ruff check src tests`
Expected: PASS, "All checks passed!".

- [ ] **Step 7: Commit**

```bash
git add src/surroforge/engine.py src/surroforge/core/run.py tests/test_training.py
git commit -m "feat(core): persist holdout actuals for report parity plots"
```

---

### Task 2: Plugin — `build_charts(run)` data builder

**Files:**
- Create: `plugins/surroforge-professional/src/surroforge_professional/reports/charts.py`
- Modify: `plugins/surroforge-professional/tests/conftest.py` (add a `multi_run` fixture)
- Test: `plugins/surroforge-professional/tests/test_charts.py`

**Interfaces:**
- Consumes: `Run.holdout_actual`, `Run.holdout_fingerprint`, `Run.reduction`, `Run.config` (`.data.outputs`, `.validation.metrics`), `Run.metrics` (`train`/`test`/`cross_validation` with `per_fold`/`mean`).
- Produces: `build_charts(run: Run) -> dict | None`. When non-`None`, keys are a subset of `{"parity", "outputs", "cv", "metric_comparison"}`:
  - `parity`: `{"actual": list[list[float]], "predicted": list[list[float]]}`, each shaped **outputs × points** (one inner list per output column).
  - `outputs`: `list[str]` output labels (present only with `parity`).
  - `cv`: `{"metrics": list[str], "folds": list[int], "per_fold": {metric: list[float]}, "mean": {metric: float}}`.
  - `metric_comparison`: `{"metrics": list[str], "train": list[float], "test": list[float], "cv_mean": list[float]}`.

- [ ] **Step 1: Add the `multi_run` fixture**

In `plugins/surroforge-professional/tests/conftest.py` (already imports `np`, `pd`, `parse_config`, `Project`, `run_training`), append:

```python
@pytest.fixture
def multi_sample_csv(tmp_path):
    rng = np.random.default_rng(1)
    n = 80
    a = rng.uniform(0.0, 1.0, n)
    b = rng.uniform(0.0, 1.0, n)
    df = pd.DataFrame({"a": a, "b": b, "y1": 2 * a + b, "y2": a - 3 * b + 0.5})
    path = tmp_path / "multi.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def multi_run(tmp_path, multi_sample_csv):
    proj = Project.init(tmp_path / "mproj", name="multi-professional")
    proj.add_dataset(multi_sample_csv, name="multi")
    config = parse_config(
        {
            "experiment": "multi-output",
            "data": {
                "source": "multi",
                "inputs": ["a", "b"],
                "outputs": ["y1", "y2"],
                "split": {"test_fraction": 0.25, "seed": 7},
            },
            "model": {"method": "polynomial", "degree": 1},
            "validation": {"cross_validation": 4, "metrics": ["rmse", "r2"]},
        }
    )
    return run_training(proj, config)
```

- [ ] **Step 2: Write the failing tests**

Create `plugins/surroforge-professional/tests/test_charts.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

from surroforge.core.config import parse_config
from surroforge_professional.reports.charts import build_charts


def _cfg(outputs, metrics=("rmse", "r2")):
    return parse_config(
        {
            "experiment": "x",
            "data": {
                "source": "s",
                "inputs": ["a", "b"],
                "outputs": list(outputs),
                "split": {"test_fraction": 0.2, "seed": 1},
            },
            "model": {"method": "polynomial", "degree": 1},
            "validation": {"cross_validation": 3, "metrics": list(metrics)},
        }
    )


def _metrics():
    return {
        "train": {"rmse": 0.10, "r2": 0.99},
        "test": {"rmse": 0.20, "r2": 0.97},
        "cross_validation": {
            "folds": 3,
            "rmse": {"mean": 0.18, "std": 0.02, "per_fold": [0.16, 0.18, 0.20]},
            "r2": {"mean": 0.96, "std": 0.01, "per_fold": [0.95, 0.96, 0.97]},
        },
    }


def _fake(*, outputs=("y1",), actual=None, predicted=None, reduction=None, metrics=None):
    return SimpleNamespace(
        config=_cfg(outputs),
        reduction=reduction or {},
        holdout_actual=[] if actual is None else actual,
        holdout_fingerprint=[] if predicted is None else predicted,
        metrics=_metrics() if metrics is None else metrics,
    )


def test_scalar_single_output_full_charts(trained_run):
    charts = build_charts(trained_run)
    assert charts is not None
    assert set(charts) >= {"parity", "cv", "metric_comparison", "outputs"}
    assert len(charts["parity"]["actual"]) == 1        # one output series
    assert len(charts["parity"]["actual"][0]) == len(charts["parity"]["predicted"][0])
    assert charts["outputs"] == list(trained_run.config.data.outputs)


def test_multi_output_parity_two_series(multi_run):
    charts = build_charts(multi_run)
    assert charts["outputs"] == ["y1", "y2"]
    assert len(charts["parity"]["actual"]) == 2
    assert len(charts["parity"]["predicted"]) == 2


def test_parity_transposes_to_output_major():
    run = _fake(
        outputs=("y1", "y2"),
        actual=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        predicted=[[1.1, 2.1], [2.9, 3.8], [5.2, 5.9]],
    )
    charts = build_charts(run)
    assert charts["parity"]["actual"][0] == [1.0, 3.0, 5.0]   # column 0
    assert charts["parity"]["actual"][1] == [2.0, 4.0, 6.0]   # column 1


def test_field_run_omits_parity():
    run = _fake(actual=[[1.0]], predicted=[[1.0]], reduction={"basis": "pod", "rank": 3})
    charts = build_charts(run)
    assert "parity" not in charts
    assert "cv" in charts and "metric_comparison" in charts


def test_old_run_without_actuals_omits_parity_keeps_cv():
    run = _fake(actual=[], predicted=[[1.0], [2.0]])
    charts = build_charts(run)
    assert "parity" not in charts
    assert "cv" in charts and "metric_comparison" in charts


def test_shape_mismatch_omits_parity_without_raising():
    run = _fake(actual=[[1.0], [2.0]], predicted=[[1.0]])
    charts = build_charts(run)            # must not raise
    assert "parity" not in charts


def test_non_numeric_actuals_omit_parity_without_raising():
    run = _fake(actual=[["oops"], ["bad"]], predicted=[[1.0], [2.0]])
    charts = build_charts(run)            # must not raise
    assert "parity" not in charts


def test_cv_present_with_per_fold():
    charts = build_charts(_fake(actual=[[1.0]], predicted=[[1.0]]))
    assert charts["cv"]["metrics"] == ["rmse", "r2"]
    assert charts["cv"]["folds"] == [1, 2, 3]
    assert charts["cv"]["per_fold"]["rmse"] == [0.16, 0.18, 0.20]


def test_returns_none_when_nothing_plottable():
    run = _fake(
        actual=[], predicted=[],
        metrics={"train": {}, "test": {}, "cross_validation": {}},
    )
    assert build_charts(run) is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest plugins/surroforge-professional/tests/test_charts.py -v`
Expected: FAIL — `ModuleNotFoundError: surroforge_professional.reports.charts`.

- [ ] **Step 4: Implement `charts.py`**

Create `plugins/surroforge-professional/src/surroforge_professional/reports/charts.py`:

```python
"""Chart-data assembly for the Professional interactive HTML report.

Pure and render-free: ``build_charts(run)`` turns a persisted :class:`Run` into the
numeric arrays the HTML template feeds to uPlot. It never raises on missing, old, or
malformed run data — a chart whose inputs are absent is simply omitted, and if nothing
is plottable the whole section is dropped (``build_charts`` returns ``None``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from surroforge.core.run import Run


def _as_2d(values: Any) -> np.ndarray | None:
    """Coerce a stored list (1-D or 2-D) into a finite 2-D float array, else ``None``."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def _output_labels(run: Run, n_cols: int) -> list[str]:
    outputs = [str(o) for o in run.config.data.outputs]
    if len(outputs) == n_cols:
        return outputs
    return [f"output {i + 1}" for i in range(n_cols)]


def _parity(run: Run) -> dict[str, Any] | None:
    # Field/ROM runs store a reconstructed field as the fingerprint and keep no field
    # actuals, so predicted-vs-actual is meaningless there — skip it.
    if run.reduction:
        return None
    actual = _as_2d(run.holdout_actual)
    predicted = _as_2d(run.holdout_fingerprint)
    if actual is None or predicted is None or actual.shape != predicted.shape:
        return None
    # Output-major (one inner list per output column) so the selector picks a column.
    return {"actual": actual.T.tolist(), "predicted": predicted.T.tolist()}


def _cv(run: Run) -> dict[str, Any] | None:
    metrics = run.metrics
    if not isinstance(metrics, dict):
        return None
    cv = metrics.get("cross_validation")
    if not isinstance(cv, dict):
        return None
    per_fold: dict[str, list[float]] = {}
    mean: dict[str, float] = {}
    n_folds = 0
    kept: list[str] = []
    for name in run.config.validation.metrics:
        rec = cv.get(name)
        if not isinstance(rec, dict):
            continue
        arr = _as_2d(rec.get("per_fold"))
        if arr is None or arr.shape[1] != 1:
            continue
        flat = arr.ravel().tolist()
        per_fold[name] = flat
        kept.append(name)
        n_folds = max(n_folds, len(flat))
        try:
            mean[name] = float(rec["mean"])
        except (TypeError, ValueError, KeyError):
            mean[name] = float(np.mean(flat))
    if not kept:
        return None
    return {
        "metrics": kept,
        "folds": list(range(1, n_folds + 1)),
        "per_fold": per_fold,
        "mean": mean,
    }


def _metric_comparison(run: Run) -> dict[str, Any] | None:
    metrics = run.metrics
    if not isinstance(metrics, dict):
        return None
    train = metrics.get("train")
    test = metrics.get("test")
    cv = metrics.get("cross_validation")
    if not all(isinstance(section, dict) for section in (train, test, cv)):
        return None
    kept: list[str] = []
    tr: list[float] = []
    te: list[float] = []
    cm: list[float] = []
    for name in run.config.validation.metrics:
        cvrec = cv.get(name)
        if name not in train or name not in test or not isinstance(cvrec, dict):
            continue
        try:
            tval = float(train[name])
            eval_ = float(test[name])
            mval = float(cvrec["mean"])
        except (TypeError, ValueError, KeyError):
            continue
        kept.append(name)
        tr.append(tval)
        te.append(eval_)
        cm.append(mval)
    if not kept:
        return None
    return {"metrics": kept, "train": tr, "test": te, "cv_mean": cm}


def build_charts(run: Run) -> dict[str, Any] | None:
    parity = _parity(run)
    cv = _cv(run)
    comparison = _metric_comparison(run)
    if parity is None and cv is None and comparison is None:
        return None
    charts: dict[str, Any] = {}
    if parity is not None:
        charts["parity"] = parity
        charts["outputs"] = _output_labels(run, len(parity["actual"]))
    if cv is not None:
        charts["cv"] = cv
    if comparison is not None:
        charts["metric_comparison"] = comparison
    return charts
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest plugins/surroforge-professional/tests/test_charts.py -v`
Expected: PASS (10 tests).

- [ ] **Step 6: Lint**

Run: `ruff check plugins/surroforge-professional`
Expected: "All checks passed!".

- [ ] **Step 7: Commit**

```bash
git add plugins/surroforge-professional/src/surroforge_professional/reports/charts.py \
        plugins/surroforge-professional/tests/test_charts.py \
        plugins/surroforge-professional/tests/conftest.py
git commit -m "feat(professional): add build_charts data builder for HTML report"
```

---

### Task 3: Plugin — vendor uPlot, wire context + template + init script

**Files:**
- Create: `plugins/surroforge-professional/src/surroforge_professional/reports/assets/uplot.min.js`
- Create: `.../reports/assets/uplot.min.css`
- Create: `.../reports/assets/uplot.LICENSE`
- Modify: `plugins/surroforge-professional/pyproject.toml`
- Modify: `.../reports/context.py`
- Modify: `.../reports/html.py`
- Test: `plugins/surroforge-professional/tests/test_html_report.py`

**Interfaces:**
- Consumes: `build_charts` (Task 2); the existing `build_context(run)` return dict; `HtmlReporter.render`.
- Produces: context keys `charts` (dict|None) and `charts_json` (str); a rendered `Visualizations` section and inline uPlot.

- [ ] **Step 1: Vendor the uPlot assets (pinned v1.6.32, MIT)**

Run, from the repo root `/Users/georg/Software/surroforge`:

```bash
ASSETS="plugins/surroforge-professional/src/surroforge_professional/reports/assets"
mkdir -p "$ASSETS"
cd /tmp
curl -sSL --max-time 60 -o uplot-1.6.32.tgz "https://registry.npmjs.org/uplot/-/uplot-1.6.32.tgz"
tar xzf uplot-1.6.32.tgz
cd /Users/georg/Software/surroforge
cp /tmp/package/dist/uPlot.iife.min.js "$ASSETS/uplot.min.js"
cp /tmp/package/dist/uPlot.min.css "$ASSETS/uplot.min.css"
cp /tmp/package/LICENSE "$ASSETS/uplot.LICENSE"
ls -l "$ASSETS"
```

Expected: three files; `uplot.min.js` ~51 KB (starts with `/*! https://github.com/leeoniya/uPlot (v1.6.32) */ var uPlot=...`), `uplot.min.css` ~1.8 KB, `uplot.LICENSE` the MIT text ("Copyright (c) 2022 Leon Sorokin").

- [ ] **Step 2: Ensure the assets are packaged**

In `plugins/surroforge-professional/pyproject.toml`, after the existing
`[tool.hatch.build.targets.wheel]` block (the `packages = [...]` line), add:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/surroforge_professional/reports/assets" = "surroforge_professional/reports/assets"
```

- [ ] **Step 3: Write the failing tests**

Edit `plugins/surroforge-professional/tests/test_html_report.py`.

First, **replace** the body of the existing `test_html_contains_required_sections` (it currently asserts `"<script" not in text` and bare `https://`/`http://`, both now invalid) with:

```python
def test_html_contains_required_sections(licensed, trained_run):
    text = HtmlReporter().render(trained_run)

    assert text.startswith("<!doctype html>")
    assert "<h1>SurroForge Report</h1>" in text
    for heading in (
        "Executive Summary",
        "Quality Snapshot",
        "Visualizations",
        "Data And Model",
        "Validation Details",
        "Reproducibility Appendix",
    ):
        assert heading in text
    # uPlot is inlined; the report must still load NO external resource.
    assert 'src="http' not in text
    assert 'href="http' not in text
    assert "@import" not in text
    assert "url(http" not in text.lower()
```

Next, **replace** the existing `test_dynamic_report_values_are_escaped` (its `assert "<script" not in text.lower()` is now wrong — uPlot's own `<script>` is expected) with this version (note the added `reduction`/`holdout_actual`/`holdout_fingerprint` keys on the fake, and payload-specific escaping assertions):

```python
def test_dynamic_report_values_are_escaped(licensed):
    config = parse_config(
        {
            "experiment": "<script>alert(1)</script>",
            "data": {
                "source": "<b>beam</b>",
                "inputs": ["length<script>"],
                "outputs": ["max_stress<b>"],
                "split": {"test_fraction": 0.2, "seed": 42},
            },
            "model": {"method": "polynomial", "degree": 2},
            "validation": {"cross_validation": 5, "metrics": ["rmse"]},
        }
    )
    run = SimpleNamespace(
        run_id="run-hostile",
        config=config,
        reduction={},
        holdout_actual=[],
        holdout_fingerprint=[],
        metrics={
            "train": {"rmse": 1.0},
            "test": {"rmse": 2.0},
            "cross_validation": {"rmse": {"mean": 1.5, "std": 0.1}, "folds": 5},
            "n_samples": 12,
            "n_inputs": 1,
            "n_outputs": 1,
        },
        lineage={
            "created_at": "2026-06-24T00:00:00Z",
            "data_source": "<b>beam</b>",
            "data_hash": "data-hash",
            "config_hash": "config-hash",
            "prediction_hash": "prediction-hash",
            "dependencies": ["pkg<script>==1", "pkg<b>==2"],
            "python_version": "3.13",
            "surroforge_version": "0.1",
            "seed": 42,
        },
        seed=42,
    )

    text = HtmlReporter().render(run)

    # The injected markup must be neutralized everywhere it could land.
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "<b>beam</b>" not in text
    assert "&lt;b&gt;beam&lt;/b&gt;" in text
    # uPlot's own <script> is expected, but no external resource may load.
    assert 'src="http' not in text
    assert 'href="http' not in text
```

Then **append** these new tests to the same file (add `import json` and `import re` at the top of the file):

```python
def test_html_embeds_chart_data_and_uplot(licensed, trained_run):
    text = HtmlReporter().render(trained_run)

    assert 'id="sf-charts"' in text
    assert "uPlot" in text                       # the library is inlined
    assert 'id="sf-parity"' in text
    assert 'id="sf-residual"' in text
    assert 'id="sf-cv"' in text
    assert 'id="sf-cmp"' in text

    match = re.search(
        r'<script type="application/json" id="sf-charts">(.*?)</script>', text, re.S
    )
    assert match is not None
    payload = json.loads(match.group(1))
    assert "parity" in payload
    assert "cv" in payload
    assert "metric_comparison" in payload


def test_html_single_output_has_no_selector(licensed, trained_run):
    text = HtmlReporter().render(trained_run)
    assert 'id="sf-output"' not in text


def test_html_multi_output_has_output_selector(licensed, multi_run):
    text = HtmlReporter().render(multi_run)
    assert 'id="sf-output"' in text
    assert ">y1<" in text
    assert ">y2<" in text


def test_uplot_mit_attribution_present(licensed, trained_run):
    text = HtmlReporter().render(trained_run)
    assert "uPlot" in text
    assert "MIT" in text
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `python -m pytest plugins/surroforge-professional/tests/test_html_report.py -v`
Expected: FAIL — no `Visualizations`/`sf-charts`/`uPlot` yet; `charts`/`charts_json` not in context.

- [ ] **Step 5: Wire the chart data into the context**

In `.../reports/context.py`, add to the imports at the top:

```python
import json

from surroforge_professional.reports.charts import build_charts
```

Add this helper just above `def build_context(run: Run)`:

```python
def _charts_json(charts: dict[str, Any] | None) -> str:
    """Serialize chart data for the in-page JSON block.

    Every ``<`` becomes its ``\\u003c`` escape so the payload can never break out of
    the surrounding ``<script>`` element (XSS-safe; the data is numeric regardless).
    """
    if not charts:
        return "{}"
    return json.dumps(charts, allow_nan=False).replace("<", "\\u003c")
```

In `build_context`, compute the charts near the end and add two keys to the returned
dict. Just before the `return {` statement add:

```python
    charts = build_charts(run)
```

and inside the returned dict (e.g. right after the `"reproduce_command": ...` line) add:

```python
        "charts": charts,
        "charts_json": _charts_json(charts),
```

- [ ] **Step 6: Load uPlot + render the charts in `html.py`**

In `.../reports/html.py`, add the asset loader near the top (after the existing imports):

```python
from importlib.resources import files


def _asset(name: str) -> str:
    return (files("surroforge_professional.reports") / "assets" / name).read_text(
        encoding="utf-8"
    )


_UPLOT_JS = _asset("uplot.min.js")
_UPLOT_CSS = _asset("uplot.min.css")
# Carry the MIT licence inline with the bundled library (its terms require the notice to
# travel with the Software). Neutralise any "--" so it can't terminate the HTML comment.
_UPLOT_LICENSE_COMMENT = (
    "<!-- uPlot v1.6.32 (MIT License)\n"
    + _asset("uplot.LICENSE").replace("--", "- -")
    + "\n-->"
)
```

Add the chart-container CSS to the existing `<style>` block. Insert these rules just
before the `@media (max-width: 720px)` line:

```css
    .sf-chart {
      margin: 8px 0 22px;
      width: 100%;
    }
    .sf-select {
      display: inline-block;
      font-size: 0.9rem;
      margin: 6px 0 10px;
    }
    .sf-select select { margin-left: 6px; }
```

Inline the uPlot stylesheet: in the template `<head>`, immediately after the existing
`</style>` and before `</head>`, add:

```html
  {% if charts %}<style>{{ uplot_css|safe }}</style>{% endif %}
```

Add the Visualizations section. Insert it between the Quality Snapshot section's closing
`</section>` and the `Data And Model` section's opening `<section ...>`:

```html
    {% if charts %}
    <section aria-labelledby="visualizations">
      <h2 id="visualizations">Visualizations</h2>
      {% if charts.parity %}
      <h3>Predicted vs. actual</h3>
      {% if charts.outputs|length > 1 %}
      <label class="sf-select">Output
        <select id="sf-output">
          {% for name in charts.outputs %}<option value="{{ loop.index0 }}">{{ name }}</option>{% endfor %}
        </select>
      </label>
      {% endif %}
      <div id="sf-parity" class="sf-chart"></div>
      <h3>Residuals</h3>
      <div id="sf-residual" class="sf-chart"></div>
      {% endif %}
      {% if charts.cv %}
      <h3>Cross-validation by fold</h3>
      <div id="sf-cv" class="sf-chart"></div>
      {% endif %}
      {% if charts.metric_comparison %}
      <h3>Train / test / CV-mean</h3>
      <div id="sf-cmp" class="sf-chart"></div>
      {% endif %}
    </section>
    {% endif %}
```

Add the attribution line. In the Reproducibility Appendix section, just before its
closing `</section>`, add:

```html
      {% if charts %}
      <h3>Visualization Library</h3>
      <p>Interactive charts rendered with uPlot (MIT License), bundled inline.</p>
      {% endif %}
```

Add the data block, the inlined library, and the init script. Insert immediately before
the closing `</main>` tag:

```html
    {% if charts %}
    <script type="application/json" id="sf-charts">{{ charts_json|safe }}</script>
    {{ uplot_license_comment|safe }}
    <script>{{ uplot_js|safe }}</script>
    <script>
    (function () {
      var holder = document.getElementById('sf-charts');
      if (!holder || typeof uPlot === 'undefined') return;
      var data;
      try { data = JSON.parse(holder.textContent); } catch (err) { return; }

      var palette = ['#2056a3', '#1f8a5b', '#b1432f', '#7a4fb5', '#b58a1f'];

      function dims(node) {
        var w = node.clientWidth || 640;
        return { width: w, height: Math.max(220, Math.round(w * 0.58)) };
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
      function scatter(label, color) {
        return {
          label: label, stroke: color, fill: color,
          paths: function () { return null; },
          points: { show: true, size: 6 }
        };
      }
      function refline(label) {
        return { label: label, stroke: '#9aa7bd', width: 1 };
      }

      if (data.parity) {
        var sel = document.getElementById('sf-output');
        var uP = null, uR = null;
        function ordered(col, byPredicted) {
          var a = data.parity.actual[col], p = data.parity.predicted[col];
          var order = a.map(function (_, i) { return i; });
          order.sort(function (i, j) {
            return byPredicted ? (p[i] - p[j]) : (a[i] - a[j]);
          });
          return {
            a: order.map(function (i) { return a[i]; }),
            p: order.map(function (i) { return p[i]; })
          };
        }
        function draw(col) {
          var par = ordered(col, false);
          if (uP) uP.destroy();
          uP = mount(document.getElementById('sf-parity'), {
            scales: { x: { time: false } },
            axes: [{ label: 'actual' }, { label: 'predicted' }],
            series: [{}, scatter('predicted', palette[0]), refline('y = x')]
          }, [par.a, par.p, par.a]);

          var res = ordered(col, true);
          var resid = res.p.map(function (v, i) { return v - res.a[i]; });
          var zero = res.p.map(function () { return 0; });
          if (uR) uR.destroy();
          uR = mount(document.getElementById('sf-residual'), {
            scales: { x: { time: false } },
            axes: [{ label: 'predicted' }, { label: 'residual' }],
            series: [{}, scatter('residual', palette[2]), refline('zero')]
          }, [res.p, resid, zero]);
        }
        draw(0);
        if (sel) sel.addEventListener('change', function () {
          draw(parseInt(sel.value, 10) || 0);
        });
      }

      if (data.cv) {
        var cvSeries = [data.cv.folds];
        var cvOpts = [{}];
        data.cv.metrics.forEach(function (m, i) {
          cvSeries.push(data.cv.per_fold[m]);
          cvOpts.push({
            label: m, stroke: palette[i % palette.length], width: 2,
            points: { show: true, size: 5 }
          });
        });
        mount(document.getElementById('sf-cv'), {
          scales: { x: { time: false } },
          axes: [
            { label: 'fold', values: function (u, vals) {
                return vals.map(function (v) { return 'fold ' + v; });
            } },
            {}
          ],
          series: cvOpts
        }, cvSeries);
      }

      if (data.metric_comparison) {
        var mc = data.metric_comparison;
        var xs = mc.metrics.map(function (_, i) { return i; });
        var bars = uPlot.paths.bars({ size: [0.6, 60] });
        mount(document.getElementById('sf-cmp'), {
          scales: { x: { time: false } },
          axes: [
            { label: 'metric', values: function (u, vals) {
                return vals.map(function (v) { return mc.metrics[v] != null ? mc.metrics[v] : ''; });
            } },
            {}
          ],
          series: [
            {},
            { label: 'train', stroke: palette[0], fill: palette[0], paths: bars },
            { label: 'test', stroke: palette[1], fill: palette[1], paths: bars },
            { label: 'CV mean', stroke: palette[4], fill: palette[4], paths: bars }
          ]
        }, [xs, mc.train, mc.test, mc.cv_mean]);
      }
    })();
    </script>
    {% endif %}
```

Finally, change `HtmlReporter.render` to pass the asset constants to the template:

```python
    def render(self, run: Run) -> str:
        require_license()
        return _TEMPLATE.render(
            **build_context(run),
            uplot_js=_UPLOT_JS,
            uplot_css=_UPLOT_CSS,
            uplot_license_comment=_UPLOT_LICENSE_COMMENT,
        )
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest plugins/surroforge-professional/tests/test_html_report.py -v`
Expected: PASS (all tests, including the rewritten and new ones).

- [ ] **Step 8: Run the full plugin suite + lint**

Run: `python -m pytest plugins/surroforge-professional/tests -q && ruff check plugins/surroforge-professional`
Expected: PASS, "All checks passed!".

- [ ] **Step 9: Smoke-check a real rendered report (manual, optional but recommended)**

Render `trained_run` to a file and confirm it opens with charts. The CI tests verify
structure, inlining, JSON validity, no external assets, and XSS — they do **not** render
pixels (no headless browser). A quick eyeball:

```bash
python -c "
from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import run_training
import os, tempfile, pandas as pd, numpy as np
os.environ['SURROFORGE_PRO_LICENSE'] = 'local-dev'
d = tempfile.mkdtemp(); rng = np.random.default_rng(0); n=80
df = pd.DataFrame({'length':rng.uniform(1,3,n),'width':rng.uniform(.5,1.5,n),'load':rng.uniform(10,50,n)})
df['max_stress'] = 2.5*df.length**2 + 1.3*df.width*df.load + 0.7*df.load
csv = d+'/s.csv'; df.to_csv(csv, index=False)
p = Project.init(d+'/proj', name='smoke'); p.add_dataset(csv, name='beam')
cfg = parse_config({'experiment':'smoke','data':{'source':'beam','inputs':['length','width','load'],'outputs':['max_stress'],'split':{'test_fraction':0.2,'seed':42}},'model':{'method':'polynomial','degree':2},'validation':{'cross_validation':5,'metrics':['rmse','r2','mae']}})
run = run_training(p, cfg)
from surroforge_professional.reports.html import HtmlReporter
open('/tmp/sf-report.html','w').write(HtmlReporter().render(run))
print('wrote /tmp/sf-report.html')
"
open /tmp/sf-report.html
```

Expected: the report opens with parity, residual, per-fold CV, and metric-comparison
charts that respond to hover/drag-zoom.

- [ ] **Step 10: Commit**

```bash
git add plugins/surroforge-professional/src/surroforge_professional/reports/assets \
        plugins/surroforge-professional/pyproject.toml \
        plugins/surroforge-professional/src/surroforge_professional/reports/context.py \
        plugins/surroforge-professional/src/surroforge_professional/reports/html.py \
        plugins/surroforge-professional/tests/test_html_report.py
git commit -m "feat(professional): interactive uPlot charts in the HTML report"
```

---

### Task 4: Docs + CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/index.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add the CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]`, add a new subsection immediately after the
`### Added — Methods` block (before `### Added — Plugins`):

```markdown
### Added — Reports

- **Interactive HTML visualizations** — the `surroforge-professional` `html` reporter now
  embeds four interactive charts: a **parity** plot (predicted vs. actual), a **residual**
  plot, **per-fold cross-validation** values, and a **train/test/CV-mean** metric
  comparison. Rendered with **uPlot** (MIT, ~51 KB) vendored and inlined — the report stays
  a single self-contained file with **no network/external assets** and remains XSS-safe
  (chart data is numeric and `<`-escaped in its JSON block). Multi-output models get a
  dropdown to switch the parity/residual plots between output columns; hovering shows exact
  values, drag-zoom and double-click-reset are available. Charts degrade gracefully: runs
  trained before this release (no holdout actuals) and field/ROM runs simply omit the
  parity/residual plots while keeping the CV and comparison charts. To support the parity
  plot, Core now persists out-of-sample **holdout actuals** alongside the prediction
  fingerprint, exposed as **`Run.holdout_actual`** — additive lineage that feeds no hash
  and does not affect `reproduce()`.
```

- [ ] **Step 2: Add a docs mention**

Locate the HTML-report mention in the docs:

Run: `grep -n -i "html" docs/index.md`

After the line that describes the Professional HTML report, add a sentence (matching the
surrounding prose style):

```markdown
The Professional HTML report includes interactive charts — parity, residual, per-fold
cross-validation, and a train/test/CV-mean comparison — rendered with inlined uPlot, so
the report stays a single self-contained file.
```

If `docs/index.md` has no HTML-report mention, add the sentence to the section that lists
report formats (run `grep -rn -i "report" docs/index.md` to find it).

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md docs/index.md
git commit -m "docs: document interactive HTML visualizations"
```

---

## Final Verification

After all tasks, run the whole suite and lint from the repo root:

```bash
python -m pytest tests plugins/surroforge-professional/tests -q
ruff check src tests plugins/surroforge-professional
```

Expected: all tests pass, "All checks passed!". Then proceed to the whole-branch review
and `superpowers:finishing-a-development-branch`.
