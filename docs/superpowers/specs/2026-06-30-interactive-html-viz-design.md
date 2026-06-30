# Interactive HTML Visualizations Design

## Goal

Add **interactive charts** to the SurroForge Professional HTML report (PRD 9.2,
"HTML-Reports mit interaktiven Visualisierungen"). The existing `html` reporter is
deliberately static (no JavaScript). This adds four charts — a **parity** plot
(predicted vs. actual), a **residual** plot, **per-fold cross-validation** bars, and
a **train/test/CV-mean metric comparison** — rendered with a small vendored charting
library, fully self-contained (no network, no external assets), and gated by the same
local Professional license as the rest of the reporter.

A small **additive Core change** persists the holdout *actual* values next to the
existing prediction fingerprint so the parity/residual plots have ground truth to plot
against.

## User-Facing Behaviour

Unchanged invocation:

```
surroforge report <run> --format html
# or: generate_report(project, run, fmt="html")  ->  reports/<run-id>.html
```

The produced HTML gains a **Visualizations** section (after *Quality Snapshot*). Charts
are interactive: hover shows exact values via a crosshair tooltip, drag-select zooms,
double-click resets. For multi-output models a **dropdown selector** switches the parity
and residual plots between output columns. With JavaScript disabled the charts simply do
not render; every number remains available in the existing tables, so the report stays
informative.

## Architecture

Chosen structure (brainstorm "Approach A"): a dedicated, isolated-testable data builder
plus a vendored asset, following the existing `reports/` layout (`html.py` + `context.py`).

- `src/surroforge/engine.py` (**Core**, Apache 2.0) — persist holdout actuals.
- `src/surroforge/core/run.py` (**Core**) — `Run.holdout_actual` accessor.
- `plugins/surroforge-professional/src/surroforge_professional/reports/charts.py` (**new**) —
  `build_charts(run) -> dict | None`, the only non-trivial logic.
- `.../reports/context.py` — call `build_charts`, attach `charts` + `charts_json`.
- `.../reports/html.py` — Visualizations template section, inline uPlot, init script.
- `.../reports/assets/uplot.min.js`, `.../assets/uplot.min.css` (**new**, package data) —
  vendored [uPlot](https://github.com/leeoniya/uPlot) (MIT, ~40 KB), inlined at render.
- `.../reports/assets/uplot.LICENSE` (**new**) — uPlot's MIT license text (NFR8 attribution).

## Core Change — persist holdout actuals

Today `engine.py` records only `holdout_fingerprint` (the out-of-sample holdout
*predictions* `holdout_test_pred`, sliced to `_FINGERPRINT_MAX_ROWS = 512`). The parity
plot also needs the matching *actuals*.

In `engine.py`, beside the existing fingerprint:

```python
holdout_actual = np.ascontiguousarray(y_test, dtype=np.float64)[:_FINGERPRINT_MAX_ROWS]
...
lineage["holdout_actual"] = holdout_actual.tolist()
```

`holdout_fingerprint` (from `holdout_test_pred`) and `holdout_actual` (from `y_test`)
come from the **same** `train_test_split` and are sliced to the **same** first 512 rows,
so `actual[i]` pairs with `predicted[i]` row-for-row. Both serialize as nested lists for
multi-output (shape `rows × outputs`) and as flat lists for single output, mirroring the
existing fingerprint serialization.

`Run` gains a mirror accessor:

```python
@property
def holdout_actual(self) -> list[Any]:
    """Out-of-sample holdout actual targets, row-aligned with holdout_fingerprint."""
    return self.lineage.get("holdout_actual", [])
```

**Why this is safe / additive:**

- `holdout_actual` feeds **no hash** — `config_hash`, `data_hash`, `spec_hash`, and
  `prediction_hash` are each computed from their own inputs, not from whole-lineage bytes.
- It is **inert to `reproduce()`**, which compares `holdout_fingerprint` only.
- It lives in `lineage.json` next to `holdout_fingerprint` for guaranteed row-alignment
  and symmetry (the prediction fingerprint is already there).
- Old runs lack the key → `Run.holdout_actual` returns `[]` → charts degrade gracefully.

## Charts Builder (`reports/charts.py`)

A pure function, isolated from rendering:

```python
def build_charts(run: Run) -> dict | None
```

Returns `None` when nothing is plottable (the template omits the whole Visualizations
section). Otherwise returns a dict whose sub-keys are each present only when that chart
can be built:

```python
{
    "outputs": ["cl", "cd"],          # output column names, for the selector
    "parity": {                       # scalar runs only
        "actual": [[...], [...]],     # shape: rows x outputs (always 2-D)
        "predicted": [[...], [...]],  # same shape, row-aligned
    },                                # residual derives from this client-side (pred - actual)
    "cv": {
        "metrics": ["rmse", "r2"],
        "folds": [1, 2, 3, 4, 5],
        "per_fold": {"rmse": [...], "r2": [...]},
        "mean": {"rmse": 0.1, "r2": 0.98},
    },
    "metric_comparison": {
        "metrics": ["rmse", "r2"],
        "train": [...], "test": [...], "cv_mean": [...],
    },
}
```

Build conditions and rules:

| Sub-chart | Built when | Notes |
| --- | --- | --- |
| `parity` | `holdout_actual` and `holdout_fingerprint` both non-empty, coerce to 2-D, **same `rows × outputs` shape**, and run is **not** a field run | residual is derived client-side from these arrays (DRY — no duplicate payload) |
| `cv` | `metrics['cross_validation'][m]` has a non-empty `per_fold` list for the config metrics | one series per metric; folds are `1..k` |
| `metric_comparison` | `metrics` has `train`/`test`/`cross_validation` for the config metrics | grouped bars: train, test, CV-mean per metric |

- **Multi-output:** `outputs = config.data.outputs`. Parity/residual arrays are always
  shaped `rows × outputs`; the selector chooses a column. If the prediction column count
  differs from `len(outputs)` (e.g. a pattern), fall back to generic labels
  `output 1..k`.
- **Field / ROM runs:** detected via `bool(run.reduction)`. For these,
  `holdout_fingerprint` is a reconstructed *field* (not scalar predictions) and no field
  actuals are stored, so `parity`/`residual` are omitted; `cv` and `metric_comparison`
  still build. (The existing Field-ROM Details table is unchanged.)
- **Graceful degradation / robustness:** the builder never raises on missing, old, or
  malformed data — a sub-chart whose inputs are absent, non-numeric, or shape-mismatched
  is simply omitted (defensive style matching `context.py`). If every sub-chart is
  omitted, `build_charts` returns `None`.

`build_context` calls `build_charts(run)` and adds two context keys: `charts` (the dict
or `None`) and `charts_json` (XSS-safe serialized payload, see below).

## Rendering, uPlot Vendoring, Interactivity, XSS (`reports/html.py`)

- **Visualizations section** rendered (after *Quality Snapshot*) only when `charts` is not
  `None`; each chart's markup is guarded on its sub-key.
- **uPlot vendored inline:** `assets/uplot.min.js` and `assets/uplot.min.css` are read once
  at module import via `importlib.resources` and inlined into `<style>`/`<script>`. uPlot's
  MIT license text (`assets/uplot.LICENSE`) is carried as an HTML comment plus a short
  attribution line in the Reproducibility Appendix (NFR8).
- **XSS-safe data embedding:** chart data is emitted in
  `<script type="application/json" id="sf-charts">…</script>`. `charts_json` is built in
  Python with `json.dumps(...)` then every `<` character replaced by its JSON Unicode
  escape (the six-character sequence for code point U+003C) — so neither `</script>` nor
  `<!--` can break out of the block — and rendered with Jinja `|safe`. The init script reads it via
  `JSON.parse(document.getElementById('sf-charts').textContent)`. Data is numeric only;
  output names additionally pass through Jinja autoescape in the dropdown markup.
- **Init script (small, inline):** constructs the uPlot charts from the parsed JSON; wires
  the output selector so switching redraws parity + residual; computes residual as
  `predicted - actual` per point; configures the cursor (crosshair + tooltip with exact
  values) and drag-zoom / double-click reset (uPlot built-ins).
- **Fully self-contained:** no CDN, no fonts, no external/network assets. Consistent with
  the existing reporter's no-network guarantee.

## Error Handling & Backward Compatibility

- `require_license()` runs first in `render()`; without a license a `PluginError` is raised
  before any run data is read (unchanged).
- `build_charts` never raises on data problems — it omits charts and can return `None`; the
  report always renders.
- Shape mismatch / non-numeric / corrupt lineage → the affected sub-chart is omitted, no
  crash.
- Pre-change runs (no `holdout_actual`) → parity/residual omitted, CV/metric charts remain.

## Scope

**In scope (v1):** parity plot (scalar runs), residual plot (client-derived), per-fold CV
bars, train/test/CV-mean metric comparison, multi-output dropdown selector, uPlot vendored
inline with MIT attribution, Core `holdout_actual` persistence + `Run.holdout_actual`
accessor, graceful degradation for old/field/malformed runs, license gating (inherited),
CHANGELOG + docs entries.

**Out of scope (YAGNI / other Phase-2 items):** parity/residual for field runs (no field
actuals persisted), charts spanning *multiple* runs (that is the separate "Erweiterte
Run-Vergleiche" item), PDF output, exporting charts as images, configurable chart
selection, dark-mode/theming, extra downsampling beyond the existing 512-row cap.

## Validation / Testing

**Core (`tests/`):**

- The engine writes `holdout_actual` row-aligned with `holdout_fingerprint` (same length,
  same order) for single- and multi-output runs.
- `Run.holdout_actual` returns the persisted list; absence returns `[]` (back-compat).

**Plugin (`plugins/surroforge-professional/tests/`):**

- `build_charts`: expected structure for scalar single- and multi-output runs; `parity`
  omitted for a field run; `cv` present when `per_fold` exists; `metric_comparison`
  present; `parity`/`residual` omitted when `holdout_actual` is empty (old run); no crash
  on a shape-mismatch / malformed lineage.
- Render: output contains the Visualizations `<section>`; the embedded JSON parses; uPlot
  is inlined; **no external asset references** (assert no CDN `http(s)://` `src`/`href` in
  the output); a malicious output column name is escaped in the dropdown (XSS).
- The license gate is still enforced (no license → `PluginError`, no render).
- Determinism: the same run yields the same chart data.
