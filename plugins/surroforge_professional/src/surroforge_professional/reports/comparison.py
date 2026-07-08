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
