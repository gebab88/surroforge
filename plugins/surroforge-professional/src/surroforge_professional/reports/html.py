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
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SurroForge Report</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172033;
      --muted: #5f6c80;
      --line: #d8dee8;
      --panel: #f6f8fb;
      --accent: #2056a3;
      --accent-soft: #e7effc;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
      line-height: 1.5;
    }
    main {
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }
    header {
      border-bottom: 2px solid var(--ink);
      margin-bottom: 28px;
      padding-bottom: 18px;
    }
    h1, h2, h3 { line-height: 1.2; }
    h1 {
      font-size: 2rem;
      margin: 0 0 10px;
    }
    h2 {
      border-bottom: 1px solid var(--line);
      font-size: 1.35rem;
      margin: 34px 0 16px;
      padding-bottom: 8px;
    }
    h3 {
      font-size: 1rem;
      margin: 18px 0 8px;
    }
    p { margin: 0 0 12px; }
    code, pre {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92em;
    }
    pre {
      background: #f3f5f8;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow-x: auto;
      padding: 14px;
      white-space: pre;
    }
    table {
      border-collapse: collapse;
      width: 100%;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: var(--panel);
      color: #253149;
      font-weight: 650;
    }
    ul {
      margin: 8px 0 0;
      padding-left: 20px;
    }
    .lede {
      color: var(--muted);
      margin: 0;
    }
    .summary-grid, .model-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    .fact, .metric-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .label {
      color: var(--muted);
      display: block;
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
    }
    .value {
      display: block;
      margin-top: 3px;
      overflow-wrap: anywhere;
    }
    .metric-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    }
    .metric-card h3 {
      margin-top: 0;
    }
    .metric-values {
      display: grid;
      gap: 6px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin: 0;
    }
    .metric-values div {
      background: white;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
    }
    .badge {
      background: var(--accent-soft);
      border-radius: 999px;
      color: var(--accent);
      display: inline-block;
      font-weight: 700;
      padding: 3px 9px;
    }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
    }
    @media (max-width: 720px) {
      main { padding: 22px 16px 36px; }
      table { display: block; overflow-x: auto; }
      th, td { white-space: nowrap; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>SurroForge Report</h1>
      <p class="lede">{{ experiment }} | {{ method }} | {{ run_id }}</p>
    </header>

    <section aria-labelledby="executive-summary">
      <h2 id="executive-summary">Executive Summary</h2>
      <div class="summary-grid">
        <div class="fact"><span class="label">Run ID</span><span class="value mono">{{ run_id }}</span></div>
        <div class="fact"><span class="label">Experiment</span><span class="value">{{ experiment }}</span></div>
        <div class="fact"><span class="label">Method</span><span class="value">{{ method }}</span></div>
        <div class="fact"><span class="label">Created</span><span class="value">{{ created_at }}</span></div>
        <div class="fact"><span class="label">Data Hash</span><span class="value mono">{{ data_hash }}</span></div>
        <div class="fact"><span class="label">Config Hash</span><span class="value mono">{{ config_hash }}</span></div>
        <div class="fact"><span class="label">Prediction Hash</span><span class="value mono">{{ prediction_hash }}</span></div>
      </div>
    </section>

    <section aria-labelledby="quality-snapshot">
      <h2 id="quality-snapshot">Quality Snapshot</h2>
      <p>Cross-validation: <span class="badge">{{ cv_folds }} folds</span></p>
      <div class="metric-grid">
        {% for row in metric_rows %}
        <article class="metric-card">
          <h3>{{ row.name }}</h3>
          <div class="metric-values">
            <div><span class="label">Train</span><span class="value mono">{{ row.train_fmt }}</span></div>
            <div><span class="label">Test</span><span class="value mono">{{ row.test_fmt }}</span></div>
            <div><span class="label">CV Mean</span><span class="value mono">{{ row.cv_mean_fmt }}</span></div>
            <div><span class="label">CV Std</span><span class="value mono">{{ row.cv_std_fmt }}</span></div>
          </div>
        </article>
        {% endfor %}
      </div>
    </section>

    <section aria-labelledby="data-and-model">
      <h2 id="data-and-model">Data And Model</h2>
      <div class="model-grid">
        <div>
          <h3>Data</h3>
          <p><span class="label">Source</span><span class="value">{{ data_source }}</span></p>
          <p><span class="label">Samples</span><span class="value">{{ n_samples }}</span></p>
          <p><span class="label">Inputs ({{ n_inputs }})</span></p>
          <ul>{% for column in inputs %}<li>{{ column }}</li>{% endfor %}</ul>
          <p><span class="label">Outputs ({{ n_outputs }})</span></p>
          <ul>{% for column in outputs %}<li>{{ column }}</li>{% endfor %}</ul>
        </div>
        <div>
          <h3>Model</h3>
          <p><span class="label">Method</span><span class="value">{{ method }}</span></p>
          <p><span class="label">Hyperparameters</span></p>
          {% if hyperparameters %}
          <table>
            <thead><tr><th>Name</th><th>Value</th></tr></thead>
            <tbody>
              {% for param in hyperparameters %}
              <tr><td>{{ param.key }}</td><td>{{ param.value }}</td></tr>
              {% endfor %}
            </tbody>
          </table>
          {% else %}
          <p>Defaults</p>
          {% endif %}
        </div>
      </div>
    </section>

    <section aria-labelledby="validation-details">
      <h2 id="validation-details">Validation Details</h2>
      <table>
        <thead>
          <tr><th>Metric</th><th>Train</th><th>Test</th><th>CV Mean</th><th>CV Std</th></tr>
        </thead>
        <tbody>
          {% for row in metric_rows %}
          <tr>
            <td>{{ row.name }}</td>
            <td class="mono">{{ row.train_fmt }}</td>
            <td class="mono">{{ row.test_fmt }}</td>
            <td class="mono">{{ row.cv_mean_fmt }}</td>
            <td class="mono">{{ row.cv_std_fmt }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    {% if two_source %}
    <section aria-labelledby="field-rom-details">
      <h2 id="field-rom-details">Field-ROM Details</h2>
    </section>
    {% endif %}

    <section aria-labelledby="reproducibility-appendix">
      <h2 id="reproducibility-appendix">Reproducibility Appendix</h2>
      <div class="summary-grid">
        <div class="fact"><span class="label">Random Seed</span><span class="value mono">{{ seed }}</span></div>
        <div class="fact"><span class="label">Test Fraction</span><span class="value mono">{{ fmt(test_fraction) }}</span></div>
        <div class="fact"><span class="label">Python</span><span class="value mono">{{ python_version }}</span></div>
        <div class="fact"><span class="label">SurroForge</span><span class="value mono">{{ surroforge_version }}</span></div>
      </div>
      <h3>Dependencies</h3>
      {% if dependencies %}
      <ul>{% for dep in dependencies %}<li class="mono">{{ dep }}</li>{% endfor %}</ul>
      {% else %}
      <p>None recorded.</p>
      {% endif %}
      <h3>Config YAML</h3>
      <pre>{{ config_yaml }}</pre>
      <h3>Reproduce</h3>
      <p><code>{{ reproduce_command }}</code></p>
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
