from __future__ import annotations

import json
import math
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
    assert charts is not None
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
    assert charts is not None
    assert charts["parity"]["actual"][0] == [1.0, 3.0, 5.0]   # column 0
    assert charts["parity"]["actual"][1] == [2.0, 4.0, 6.0]   # column 1


def test_field_run_omits_parity():
    run = _fake(actual=[[1.0]], predicted=[[1.0]], reduction={"basis": "pod", "rank": 3})
    charts = build_charts(run)
    assert charts is not None
    assert "parity" not in charts
    assert "cv" in charts and "metric_comparison" in charts


def test_old_run_without_actuals_omits_parity_keeps_cv():
    run = _fake(actual=[], predicted=[[1.0], [2.0]])
    charts = build_charts(run)
    assert charts is not None
    assert "parity" not in charts
    assert "cv" in charts and "metric_comparison" in charts


def test_shape_mismatch_omits_parity_without_raising():
    run = _fake(actual=[[1.0], [2.0]], predicted=[[1.0]])
    charts = build_charts(run)            # must not raise
    assert charts is not None
    assert "parity" not in charts


def test_non_numeric_actuals_omit_parity_without_raising():
    run = _fake(actual=[["oops"], ["bad"]], predicted=[[1.0], [2.0]])
    charts = build_charts(run)            # must not raise
    assert charts is not None
    assert "parity" not in charts


def test_cv_present_with_per_fold():
    charts = build_charts(_fake(actual=[[1.0]], predicted=[[1.0]]))
    assert charts is not None
    assert charts["cv"]["metrics"] == ["rmse", "r2"]
    assert charts["cv"]["folds"] == [1, 2, 3]
    assert charts["cv"]["per_fold"]["rmse"] == [0.16, 0.18, 0.20]


def test_returns_none_when_nothing_plottable():
    run = _fake(
        actual=[], predicted=[],
        metrics={"train": {}, "test": {}, "cross_validation": {}},
    )
    assert build_charts(run) is None


def test_non_finite_metric_mean_is_dropped_from_comparison():
    metrics = _metrics()
    metrics["cross_validation"]["r2"]["mean"] = float("nan")
    run = _fake(actual=[[1.0]], predicted=[[1.0]], metrics=metrics)
    charts = build_charts(run)
    assert charts is not None
    assert "r2" not in charts["metric_comparison"]["metrics"]
    assert "rmse" in charts["metric_comparison"]["metrics"]
    assert all(math.isfinite(v) for v in charts["metric_comparison"]["cv_mean"])


def test_charts_json_safe_after_nan_metric():
    from surroforge_professional.reports.context import _charts_json
    metrics = _metrics()
    metrics["cross_validation"]["r2"]["mean"] = float("nan")
    run = _fake(actual=[[1.0]], predicted=[[1.0]], metrics=metrics)
    out = _charts_json(build_charts(run))   # must not raise
    json.loads(out)                          # valid JSON


def test_charts_json_escapes_angle_brackets():
    from surroforge_professional.reports.context import _charts_json
    out = _charts_json({"outputs": ["<script>x</script>"]})
    assert "<" not in out
    assert "\\u003c" in out


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
