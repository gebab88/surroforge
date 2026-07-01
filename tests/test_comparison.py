from __future__ import annotations

from types import SimpleNamespace

import pytest

from surroforge.comparison import build_comparison
from surroforge.core.config import parse_config
from surroforge.errors import DataError


def _cfg(*, method="polynomial", degree=2, n_estimators=None, metrics=("rmse", "r2", "mae"),
         experiment="exp", inputs=("a", "b"), outputs=("y",), seed=42):
    model: dict = {"method": method}
    if method == "polynomial":
        model["degree"] = degree
    if n_estimators is not None:
        model["n_estimators"] = n_estimators
    return parse_config(
        {
            "experiment": experiment,
            "data": {
                "source": "s", "inputs": list(inputs), "outputs": list(outputs),
                "split": {"test_fraction": 0.2, "seed": seed},
            },
            "model": model,
            "validation": {"cross_validation": 5, "metrics": list(metrics)},
        }
    )


def _metrics(*, names=("rmse", "r2", "mae"), test=None):
    test = test or {"rmse": 0.02, "r2": 0.98, "mae": 0.01}
    train = {"rmse": 0.01, "r2": 0.99, "mae": 0.008}
    cv = {"rmse": (0.02, 0.002), "r2": (0.98, 0.001), "mae": (0.01, 0.001)}
    return {
        "train": {n: train[n] for n in names if n in train},
        "test": {n: test[n] for n in names if n in test},
        "cross_validation": {
            "folds": 5,
            **{n: {"mean": cv[n][0], "std": cv[n][1]} for n in names if n in cv},
        },
    }


def _run(run_id, *, config=None, metrics=None, data_hash="d0", config_hash="c0",
         created_at="2026-06-30T00:00:00Z", seed=42, parent_run=None):
    config = config if config is not None else _cfg()
    return SimpleNamespace(
        run_id=run_id, method=config.model.method, config=config,
        metrics=metrics if metrics is not None else _metrics(),
        created_at=created_at, seed=seed, data_hash=data_hash,
        config_hash=config_hash, parent_run=parent_run,
    )


def test_result_shape():
    result = build_comparison([_run("run-001"), _run("run-002")])
    assert result["metric"] == "rmse"
    assert result["metric_names"] == ["rmse", "r2", "mae"]
    assert len(result["runs"]) == 2
    entry = result["runs"][0]
    assert set(entry) >= {"run_id", "method", "experiment", "created_at", "metrics",
                          "rank", "delta", "lineage"}
    assert set(entry["metrics"]["rmse"]) == {"train", "test", "cv_mean", "cv_std"}
    assert set(entry["lineage"]) == {"seed", "data_hash", "config_hash", "parent_run"}


def test_ranking_and_delta_lower_is_better():
    a = _run("A", metrics=_metrics(test={"rmse": 0.02, "r2": 0.98, "mae": 0.01}))
    b = _run("B", metrics=_metrics(test={"rmse": 0.01, "r2": 0.95, "mae": 0.02}))
    result = build_comparison([a, b], metric="rmse")
    by_id = {e["run_id"]: e for e in result["runs"]}
    assert result["best"] == "B"
    assert by_id["B"]["rank"] == 1 and by_id["B"]["delta"] == 0.0
    assert by_id["A"]["rank"] == 2
    assert by_id["A"]["delta"] == pytest.approx(0.01)


def test_ranking_higher_is_better_metric():
    a = _run("A", metrics=_metrics(test={"rmse": 0.02, "r2": 0.98, "mae": 0.01}))
    b = _run("B", metrics=_metrics(test={"rmse": 0.01, "r2": 0.95, "mae": 0.02}))
    result = build_comparison([a, b], metric="r2")
    assert result["best"] == "A"  # higher r2 is better


def test_best_per_metric():
    a = _run("A", metrics=_metrics(test={"rmse": 0.02, "r2": 0.99, "mae": 0.05}))
    b = _run("B", metrics=_metrics(test={"rmse": 0.01, "r2": 0.90, "mae": 0.01}))
    result = build_comparison([a, b])
    assert result["best_per_metric"] == {"rmse": "B", "r2": "A", "mae": "B"}


def test_config_diff_only_differing_paths():
    a = _run("A", config=_cfg(degree=2))
    b = _run("B", config=_cfg(degree=3))
    diff = build_comparison([a, b])["config_diff"]
    assert diff["model.degree"] == {"A": 2, "B": 3}
    assert "model.method" not in diff  # both polynomial


def test_config_diff_identical_is_empty():
    a = _run("A", config=_cfg(degree=2))
    b = _run("B", config=_cfg(degree=2))
    assert build_comparison([a, b])["config_diff"] == {}


def test_config_diff_absent_key_sentinel():
    a = _run("A", config=_cfg(method="random-forest", n_estimators=300))
    b = _run("B", config=_cfg(method="random-forest", n_estimators=None))
    diff = build_comparison([a, b])["config_diff"]
    assert diff["model.n_estimators"] == {"A": 300, "B": "<absent>"}


def test_lineage_summary_same_and_different_data():
    same = build_comparison([_run("A", data_hash="h1"), _run("B", data_hash="h1")])
    assert same["lineage_summary"]["same_data"] is True
    diff = build_comparison([_run("A", data_hash="h1"), _run("B", data_hash="h2")])
    assert diff["lineage_summary"]["same_data"] is False
    assert diff["lineage_summary"]["data_hash_groups"] == {"h1": ["A"], "h2": ["B"]}


def test_warning_on_different_metric_sets_and_missing_ranking_metric():
    a = _run("A")  # rmse, r2, mae
    b = _run("B", config=_cfg(metrics=("rmse",)), metrics=_metrics(names=("rmse",)))
    result = build_comparison([a, b], metric="mae")
    text = " ".join(result["warnings"])
    assert "different metric sets" in text
    assert "mae" in text and "B" in text  # B unranked on mae
    by_id = {e["run_id"]: e for e in result["runs"]}
    assert by_id["B"]["rank"] is None
    assert by_id["B"]["metrics"]["mae"]["test"] is None


def test_warning_on_different_data_and_experiment():
    a = _run("A", config=_cfg(experiment="exp-1"), data_hash="h1")
    b = _run("B", config=_cfg(experiment="exp-2"), data_hash="h2")
    text = " ".join(build_comparison([a, b])["warnings"])
    assert "different datasets" in text
    assert "different experiments" in text


def test_single_run():
    result = build_comparison([_run("A")])
    assert result["best"] == "A"
    assert result["runs"][0]["rank"] == 1
    assert result["runs"][0]["delta"] == 0.0
    assert result["config_diff"] == {}


def test_empty_runs_raises():
    with pytest.raises(DataError):
        build_comparison([])
