# Professional HTML Reports Design

Date: 2026-06-23
Status: approved for implementation planning

## Context

SurroForge Core already supports report generation through the `surroforge.reporters`
entry-point group. The built-in `MarkdownReporter` renders deterministic run reports
from a `Run` object, including data overview, model specification, validation metrics,
optional Field-ROM diagnostics, configuration, and reproducibility metadata.

The product roadmap places HTML reports in the Professional edition. This design keeps
Core unchanged and adds the first Professional report format as a separate plugin package
inside the current repository.

## Goals

- Add an HTML report format as a Professional plugin, not as a Core feature.
- Keep the v1 report static and offline-readable as a single `.html` file.
- Preserve Markdown report coverage while improving scanability with simple static
  visual structure.
- Enforce a local Professional license gate before rendering.
- Avoid JavaScript and Plotly in v1 while leaving room for later interactive reports.

## Non-Goals

- No Core command changes.
- No online license activation.
- No PDF audit report generation.
- No interactive charts, Plotly bundle, or browser-side filtering in v1.
- No shared Core report-context abstraction unless later duplication proves painful.

## Decisions

### Packaging

Create a plugin package at `plugins/surroforge-professional/`, following the pattern
already used by `plugins/surroforge-torch/`.

The package registers an HTML reporter through the existing entry-point seam:

```toml
[project.entry-points."surroforge.reporters"]
html = "surroforge_professional.reports.html:HtmlReporter"
```

The reporter implements the existing `BaseReporter` contract:

```python
class HtmlReporter(BaseReporter):
    name = "html"
    extension = ".html"

    def render(self, run: Run) -> str:
        ...
```

With the plugin installed, existing Core surfaces work without changes:

```bash
surroforge report run-001 --format html
```

and:

```python
generate_report(project, run, fmt="html")
```

### Architecture

The selected approach is a self-contained Professional reporter in the plugin. It reads
only the existing `Run` object and produces HTML text. The plugin owns:

- HTML template.
- Embedded CSS.
- Static visual components such as metric cards, badges, bars, and tables.
- Local Professional license check.
- Plugin tests and README.

Core remains responsible only for resolving the reporter entry point and writing the
returned text to `reports/<run-id>.html`.

This avoids pulling Professional-specific layout, licensing, or report-context choices
into the open-source Core package.

## Data Flow

`HtmlReporter.render(run)` builds a plugin-internal view context from:

- `run.run_id`
- `run.config`
- `run.metrics`
- `run.lineage`
- `run.data_hash`
- `run.config_hash`
- `run.seed`
- `run.holdout_fingerprint` when needed for Field-ROM diagnostics

The context is intentionally private to the plugin. Markdown and HTML can duplicate a
small amount of data extraction in v1; a shared Core report-context builder is deferred
until both reporters need enough common logic to justify the abstraction.

## Report Structure

The v1 layout is "Executive Summary + Details".

1. Executive Summary
   - Run ID.
   - Experiment.
   - Method.
   - Created time.
   - Data hash.
   - Config hash.
   - Prediction fingerprint.

2. Quality Snapshot
   - Metric cards for train, test, and cross-validation values.
   - Simple static bars or badges to make comparisons scannable.
   - No JavaScript dependency.

3. Data And Model
   - Data source when available.
   - Sample count.
   - Input and output column names.
   - Method and hyperparameters.

4. Validation Details
   - Full metric table matching Markdown report content.
   - Train, test, CV mean, and CV standard deviation.

5. Field-ROM Details
   - Rendered only when required metrics and lineage data are present.
   - Two-source error split.
   - Reduction rank and captured energy when available.
   - Per-fold truncation floors when available.

6. Reproducibility Appendix
   - YAML configuration snapshot.
   - Python version.
   - SurroForge version.
   - Dependencies.
   - Reproduce command.

The prediction fingerprint belongs in run identity and reproducibility context, not in
the quality-metric summary. It is the SHA-256 hash over out-of-sample holdout predictions
recorded for reproduction checks.

## License Gate

The v1 Professional gate is local-only. Rendering is allowed when either of these is
present:

- Non-empty `SURROFORGE_PRO_LICENSE` environment variable.
- Existing license file at `~/.surroforge/professional.lic`.

There is no network check and no activation flow in v1.

When the gate fails, the plugin raises a SurroForge-compatible domain exception with a
clear message:

```text
Professional license required for HTML reports. Set SURROFORGE_PRO_LICENSE or install ~/.surroforge/professional.lic.
```

## Error Handling And Safety

Required data must fail clearly:

- Missing train metrics.
- Missing test metrics.
- Missing cross-validation metrics.
- Missing config or lineage fields needed for the base report identity.

Optional Field-ROM data must degrade gracefully:

- If all required Field-ROM metrics and lineage fields are present, render the Field-ROM
  details section.
- If those records are absent or incomplete, omit the section and still render the
  scalar-compatible report.

All dynamic text inserted into HTML must be escaped, including:

- Experiment names.
- Column names.
- Data source paths.
- Hash strings.
- YAML config text.
- Dependency strings.
- Error or warning text included in future report variants.

The report must be a static HTML document with embedded CSS and no required external
assets. This keeps reports readable offline and avoids leaking project paths or metadata
through network asset loads.

## Tests

Plugin tests live under `plugins/surroforge-professional/tests/`.

Required coverage:

- Entry-point registration: `html` resolves as a reporter when the plugin is installed.
- License gate: missing license raises a clear domain exception.
- License gate: `SURROFORGE_PRO_LICENSE` permits rendering.
- License gate: `~/.surroforge/professional.lic` permits rendering.
- HTML content: run ID, method, data hash, config hash, prediction fingerprint, metrics,
  and reproduce command appear.
- Escaping: hostile strings from config or lineage are escaped, not rendered as markup.
- Field-ROM degradation: complete Field-ROM data renders the optional section; incomplete
  Field-ROM records omit it without crashing.
- Service integration: `generate_report(project, run, fmt="html")` writes
  `reports/<run-id>.html`.

## Documentation

The plugin README documents:

- Editable development install.
- `surroforge report run-001 --format html`.
- Python API usage through `generate_report`.
- Local license gate.
- v1 scope and non-goals.

Core documentation does not need a broad rewrite for v1. A later Professional docs page
can link from Core docs after the plugin package is stable.

## Acceptance Criteria

- `surroforge-professional` can be installed in editable mode from
  `plugins/surroforge-professional/`.
- With a valid local license gate, `surroforge report <run-id> --format html` writes
  `reports/<run-id>.html`.
- Without the local license gate, the command fails with a clear Professional-license
  message.
- Generated HTML is single-file, static, escaped, and readable offline.
- Scalar reports render the expected sections.
- Field-ROM reports render optional diagnostics only when the required source data is
  complete.
- Existing Core reporter behavior remains unchanged.

## Implementation Boundary

This design is ready for implementation planning. The next step is a concrete plan using
the writing-plans skill. No implementation code should be written before that plan is
approved.
