"""Static HTML report for SurroForge Professional."""

from __future__ import annotations

from importlib.resources import files

from jinja2 import Environment

from surroforge.core.run import Run
from surroforge.reporters.base import BaseReporter
from surroforge_professional.license import require_license
from surroforge_professional.reports.context import build_context


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
    @media (max-width: 720px) {
      main { padding: 22px 16px 36px; }
      table { display: block; overflow-x: auto; }
      th, td { white-space: nowrap; }
    }
  </style>
  {% if charts %}<style>{{ uplot_css|safe }}</style>{% endif %}
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
      <table>
        <thead>
          <tr><th>Error source</th><th>Train</th><th>Test</th><th>CV mean</th><th>CV std</th></tr>
        </thead>
        <tbody>
          {% for row in two_source.rows %}
          <tr>
            <td>{{ row.label }}</td>
            <td class="mono">{{ row.train_fmt }}</td>
            <td class="mono">{{ row.test_fmt }}</td>
            <td class="mono">{{ row.cv_mean_fmt }}</td>
            <td class="mono">{{ row.cv_std_fmt }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% if two_source.reduction_rank is not none %}
      <p><span class="label">Reduction rank</span><span class="value mono">{{ two_source.reduction_rank }}</span></p>
      {% endif %}
      {% if two_source.energy_captured_fmt is not none %}
      <p><span class="label">Energy captured</span><span class="value mono">{{ two_source.energy_captured_fmt }}</span></p>
      {% endif %}
      <p><span class="label">Per-fold truncation floors</span><span class="value mono">{{ two_source.truncation_per_fold }}</span></p>
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
      {% if charts %}
      <h3>Visualization Library</h3>
      <p>Interactive charts rendered with uPlot (MIT License), bundled inline.</p>
      {% endif %}
    </section>
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
        return _TEMPLATE.render(
            **build_context(run),
            uplot_js=_UPLOT_JS,
            uplot_css=_UPLOT_CSS,
            uplot_license_comment=_UPLOT_LICENSE_COMMENT,
        )
