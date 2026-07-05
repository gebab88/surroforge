# Professional HTML Comparison Report Design

## Goal

Ship Cycle 2 of the extended-run-comparisons work
(`docs/superpowers/specs/2026-06-30-extended-run-comparisons-design.md`): a
**self-contained, interactive HTML report comparing multiple runs**, rendered by the
Professional plugin from the JSON-serializable dict that Core's `build_comparison`
already produces. One CLI entry point: `surroforge compare --format html`.

Out of scope: extending the `build_comparison` data model (no cross-run per-fold or
parity data), a markdown comparison format, any change to the existing terminal
rendering.

## Architecture Decision (Approach A)

The existing reporter contract (`BaseReporter.render(run: Run)`) and the `report`
command are strictly single-run; a comparison consumes a different input shape. Core
therefore gains a **separate, parallel comparison-reporter contract** — a small ABC, a
new entry-point group, and a service function — mirroring the existing reporter pattern
1:1. Rejected alternatives: generalizing `BaseReporter` with an optional
`render_comparison` (capability checks via `hasattr`, muddied typing), and importing the
plugin directly from the CLI (breaks the entry-point architecture).

## CLI Behaviour

```bash
surroforge compare --runs run-001,run-002,run-003 --metric rmse            # unchanged: terminal
surroforge compare --runs run-001,run-002 --format html                   # → reports/comparison.html
surroforge compare --runs run-001,run-002 -f html -o vergleich.html       # explicit path
```

- `--format/-f` defaults to `terminal` → exactly today's code path, zero behaviour
  change for existing users.
- `--format html` → Core resolves the comparison reporter via the new registry group,
  renders, writes the file, and prints the written path (same UX as `report`).
- `--output/-o` is only valid for file-producing formats; the default for `html` is
  **`reports/comparison.html`** inside the project (deliberately simple and
  overwriting — run-id chains in filenames get unwieldy for N runs; archive via `-o`).
  `--format terminal --output …` is a clear CLI error.
- Error cases follow the `report` precedent: plugin not installed → `PluginError`
  ("unknown comparison reporter 'html'", listing available formats); plugin present but
  unlicensed → `LicenseError` from `require_license()` inside the reporter.
- Docs in scope: one sentence + example each in `README.md` (compare section) and
  `docs/scenarios/method-comparison.md`.

## Core Changes (4 files, ~60 lines)

1. **Contract — `src/surroforge/reporters/base.py`**: a second, minimal ABC next to
   `BaseReporter`:

   ```python
   class BaseComparisonReporter(ABC):
       """Renders the JSON-serializable dict that build_comparison produces."""
       name: str
       extension: str

       @abstractmethod
       def render(self, comparison: dict[str, Any]) -> str: ...
   ```

   The input is exactly the `build_comparison` dict — no new data type; the dict is the
   documented contract (`comparison.py` module docstring).

2. **Registry — `src/surroforge/plugins/registry.py`**:
   `COMPARISON_REPORTER_GROUP = "surroforge.comparison_reporters"`, plus
   `get_comparison_reporter(name)` and `available_comparison_reporters()` delegating to
   the existing `_resolve`/`_entry_points_grouped` helpers, identical to `get_reporter`.
   **No `_BUILTIN` entry**: Core ships no comparison reporter; terminal rendering stays
   CLI code, not a plugin.

3. **Service — `src/surroforge/services.py`**: `generate_comparison_report(project,
   run_ids, *, metric="rmse", format="html", output=None) -> Path` — calls the existing
   `compare_runs`, resolves the reporter, renders, writes to `output` or
   `reports/comparison<extension>` (same write/dir pattern as `generate_report`),
   returns the path.

4. **CLI — `src/surroforge/cli/main.py`**, `compare` command: new options `--format/-f`
   (default `"terminal"`) and `--output/-o` (default `None`). `terminal` → the
   unchanged existing branch (erroring if `--output` is set); anything else →
   `services.generate_comparison_report(...)` and "Comparison report written to
   `<path>`".

Architecture invariants: import-linter contracts untouched (services already imports
the registry; core still imports no adapters/plugins); everything mypy- and
pyright-clean.

## Professional Plugin

One new module `plugins/surroforge-professional/src/surroforge_professional/reports/comparison.py`,
following `html.py`'s structure (one module = reporter + Jinja template string):

```python
class HtmlComparisonReporter(BaseComparisonReporter):
    name = "html"
    extension = ".html"

    def render(self, comparison: dict[str, Any]) -> str:
        require_license()                                  # gate first, like html.py
        context = _build_context(comparison)
        return _TEMPLATE.render(**context, uplot_js=..., uplot_css=..., uplot_license=...)
```

- **`_build_context(comparison)`** (private, same module): pure formatting into
  template rows — ranking rows (rank, run, method, experiment, test value, Δ to best),
  the all-metrics table (train / test / cv mean±std per run×metric, `-` when missing),
  config-diff rows, lineage groups, warnings — plus `charts`/`charts_json`. No
  computation: the semantics (ranking, best, diff) arrive precomputed from
  `build_comparison`.
- **Cross-run chart data** as `build_comparison_charts(comparison) -> dict | None` in
  **`charts.py`** (chart-data builders live there; same guarantees as `build_charts`:
  pure, never raises, `None` when nothing is plottable). Shape:

  ```python
  {"runs": [...run_ids], "metrics": [...], "best": run_id,
   "per_metric": {name: {"train": [...], "test": [...], "cv_mean": [...], "cv_std": [...]}}}
  # lists parallel to runs; missing values as None (uPlot null)
  ```

- **Asset sharing instead of duplication** — targeted small changes to `html.py`:
  `comparison.py` imports the module globals `_UPLOT_JS`/`_UPLOT_CSS`/
  `_UPLOT_LICENSE_COMMENT` from `.html`; the ~150-line base-CSS block moves out of the
  template literal into a module constant `_BASE_CSS` in `html.py`, and **both**
  templates embed it (`{{ base_css|safe }}`) so the two reports look identical and
  cannot drift. `_charts_json` (XSS escape: every `<` becomes the JSON escape sequence backslash-u003c) is reused from `context.py`.
- **Registration** in `plugins/surroforge-professional/pyproject.toml`:

  ```toml
  [project.entry-points."surroforge.comparison_reporters"]
  html = "surroforge_professional.reports.comparison:HtmlComparisonReporter"
  ```

## Page Layout & Charts

Same visual language as the single-run report; one self-contained HTML, no network
access.

1. **Header**: "Run comparison", ranking metric, number of runs, generation timestamp.
2. **Best-run callout**: the winner (`comparison["best"]`) with method + test value;
   `best_per_metric` as chips when metrics disagree on the winner.
3. **Warnings banner** (yellow) when `warnings` is non-empty — *before* the numbers, so
   nobody compares apples to oranges unknowingly.
4. **Ranking table**: rank, run id, method, experiment, test `<metric>`, Δ to best;
   unrankable runs below with "–".
5. **Charts — one block per metric** (see below).
6. **All-metrics table**: per run×metric `train / test / cv mean±std`.
7. **Config diff**: only differing paths (precomputed in the dict), one column per run,
   `<absent>` in italics.
8. **Lineage**: data-hash groups ("all runs on the same data" vs. group list), seed /
   config hash / parent per run.

**Chart block per metric** (uPlot, vendored assets): x-axis = the runs; **three bar
series** — train, test, CV mean — as grouped bars, rendered by adapting the custom
bar-draw path the single-run init script already uses (adapted, not reinvented).
**±std as whiskers** on the CV-mean bar, drawn in the same draw pass; exact values
(incl. ±std) additionally in the tooltip/legend readout. Documented fallback if whisker
drawing turns out disproportionately complex during implementation: ±std in the tooltip
only — the information is never lost. **Best marking**: the per-metric best run's bar
group (`best_per_metric`) gets an accent colour + "best" badge under the run label.
Missing values → `null` (uPlot leaves a gap), no fake zero bars.

**Degenerate cases**: 1 run → page renders normally (single-row ranking);
`build_comparison_charts` → `None` → chart section omitted, rest stays; a metric
missing for one run → `null` gaps, no crash.

**Security**: all strings (run ids, experiments, config values) via Jinja autoescape;
chart payload via the existing `_charts_json` escaping; no external resources (the
existing self-containment test is mirrored for the comparison report).

## Tests & Verification

**Core tests** (`tests/`, run by root pytest + `ci.sh`):
- Registry: `get_comparison_reporter("nope")` → `PluginError` with the available list;
  successful resolution via mocked entry points — following the existing registry test
  patterns.
- Service: `generate_comparison_report` with a stubbed reporter — writes
  `reports/comparison.html`, honours the `--output` override, returns the path, passes
  `metric` through.
- CLI: Typer-runner tests — `--format html` calls the service; `--format terminal
  --output x` → error; without the new flags, byte-identical terminal behaviour
  (regression guard).

**Plugin tests** (`plugins/surroforge-professional/tests/`, own suite):
- `test_charts.py` additions: `build_comparison_charts` — shape (parallel lists, `None`
  holes), `None` when nothing plottable, never raises (even on broken metric dicts) —
  the same guarantees as `build_charts`, tested the same way.
- New `test_comparison_report.py`, mirroring `test_html_report.py`: doctype smoke;
  **self-containment** (no external `src=`/`href=`/`@import`/`url(http`); all 8 section
  headings; ranking values + best badge; warnings banner appears for differing
  datasets; **XSS test** with a hostile experiment name and config value; chart JSON
  extracted via regex + `json.loads` + shape check; **license gate**: without the
  `licensed` fixture → `LicenseError`; end-to-end via `generate_comparison_report` with
  two genuinely trained runs (conftest gains a second scalar run next to
  `trained_run`).

**Gates & process:**
- The plugin suite is **not** in root CI — it is checked explicitly per task
  (`pytest plugins/surroforge-professional/tests`), and **pyright includes `plugins/`**
  — new code must keep the freshly won 0-error baseline.
- Per task: ruff + focused tests; at the end: full `scripts/ci.sh` + plugin suite.
- **Real-data acceptance**: `surroforge --config-dir examples/naca0012_openfoam/ex09_real/naca
  compare --runs run-001,…,run-006 -f html -o <scratch>/naca-vergleich.html` against the
  six saved NACA runs (read-only) — proves the report on real CFD comparison data; the
  file can then be opened in a browser.

## Commit Plan

Small, single-purpose commits directly to `main` (solo project): Core contract +
registry + service + CLI; plugin chart builder; plugin reporter + template; tests may
land with their feature commits; docs touch-up last.
