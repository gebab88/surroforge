"""Two-source error split in the Markdown report (Story R3.2, PRD 4.6).

The report derives the split at render time from the r3-1 metrics: Pythagoras
gives ``surrogate = sqrt(max(energy_l2² − truncation², 0))`` per split and per
fold, and Parseval (orthonormal POD basis) makes that surrogate component the
coefficient-space error — one table is the #42 coefficient-vs-field view. The
deception fixture below is the story's point: r2 and energy_captured both look
great while the truncation floor carries essentially all of the error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import run_training
from surroforge.services import generate_report
from surroforge.validators.metrics import two_source_split

SECTION = "## Two-Source Error Split"
ROW_LABELS = ("total field error", "truncation floor", "surrogate error")
SPLICE = "-fold.\n\n## Configuration"  # the exact scalar text at the insertion site


# --- two_source_split unit tests (Task 1) ---------------------------------------------

def test_split_is_pythagorean_algebra():
    # 3-4-5 triangle, recomputed not golden: total 5, floor 3 -> surrogate 4.
    split = two_source_split(5.0, 3.0)
    assert split == {"total": 5.0, "truncation": 3.0, "surrogate": 4.0}


def test_split_clamps_rounding_noise_instead_of_nan():
    # Rounding can push the floor infinitesimally above the total (r3-1 review:
    # the derived gap amplifies 1-ulp noise to ~1e-8) — must clamp, never nan.
    split = two_source_split(0.1, 0.1 + 1e-15)
    assert split["surrogate"] == 0.0


def test_split_zero_case():
    assert two_source_split(0.0, 0.0) == {"total": 0.0, "truncation": 0.0, "surrogate": 0.0}


# --- report fixtures --------------------------------------------------------------------

def _field_run(
    tmp_path,
    a3,
    degree=2,
    metrics=("energy_l2", "truncation_error", "rmse", "r2"),
    model=None,
):
    """Pinned rank-3 field (story Dev Notes), trained rank-2 through the engine."""
    rng = np.random.default_rng(42)
    n_dof, n = 40, 80
    modes = np.linalg.qr(rng.standard_normal((n_dof, 3)))[0]
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    amps = np.column_stack([5.0 * x[:, 0], 2.0 * x[:, 1] ** 2, a3 * x[:, 0] * x[:, 1]])
    y = amps @ modes.T
    cols: dict[str, np.ndarray] = {"p0": x[:, 0], "p1": x[:, 1]}
    for j in range(n_dof):
        cols[f"f{j}"] = y[:, j]
    csv = tmp_path / "field.csv"
    pd.DataFrame(cols).to_csv(csv, index=False)
    proj = Project.init(tmp_path / "proj", name="rom")
    proj.add_dataset(csv, name="snaps")
    cfg = parse_config(
        {
            "experiment": "rom-two-source",
            "data": {
                "source": "snaps",
                "inputs": ["p0", "p1"],
                "outputs": ["f*"],
                "split": {"test_fraction": 0.25, "seed": 1},
            },
            "model": model
            or {
                "method": "reduced-field",
                "reduction": {"basis": "pod", "rank": 2},
                "inner": {"method": "polynomial", "degree": degree},
            },
            "validation": {"cross_validation": 3, "metrics": list(metrics)},
        }
    )
    return proj, run_training(proj, cfg)


def _table_cells(text: str, row_label: str) -> list[float]:
    """Parse [train, test, cv_mean, cv_std] floats from the section's table row."""
    for line in text.splitlines():
        if line.startswith(f"| {row_label}"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            return [float(c) for c in cells[1:]]
    raise AssertionError(f"row {row_label!r} not found in report")


# --- table correctness (AC 1, 2) ---------------------------------------------------------

def test_two_source_table_renders_consistent_values(tmp_path):
    proj, run = _field_run(tmp_path, a3=2.0)
    _path, text = generate_report(proj, run)

    assert SECTION in text
    for label in ROW_LABELS:
        assert f"| {label}" in text

    # Train and test columns match the split of the respective persisted metrics
    # (recomputed, not golden); Pythagoras holds for the rendered numbers.
    for col, split_name in ((0, "train"), (1, "test")):
        expected = two_source_split(
            run.metrics[split_name]["energy_l2"], run.metrics[split_name]["truncation_error"]
        )
        total = _table_cells(text, "total field error")[col]
        floor = _table_cells(text, "truncation floor")[col]
        surrogate = _table_cells(text, "surrogate error")[col]
        assert total == pytest.approx(expected["total"], rel=1e-4)
        assert floor == pytest.approx(expected["truncation"], rel=1e-4)
        assert surrogate == pytest.approx(expected["surrogate"], abs=1e-6)
        assert total**2 == pytest.approx(floor**2 + surrogate**2, abs=1e-6)

    # Reduction context line from lineage (Parseval/guidance text around the table).
    reduction = run.lineage["reduction"]
    assert f"rank {reduction['rank']}" in text
    assert f"{reduction['energy_captured']:.5g}" in text
    assert "coefficient-space error" in text  # the Parseval line (#42)
    assert "add modes" in text and "add data" in text  # the #25 guidance


# --- the deception case (AC 3) ------------------------------------------------------------

def test_great_r2_from_tiny_basis_is_exposed(tmp_path):
    proj, run = _field_run(tmp_path, a3=0.8)
    test = run.metrics["test"]
    split = two_source_split(test["energy_l2"], test["truncation_error"])

    # Every classic signal looks great...
    assert test["r2"] > 0.85
    assert run.lineage["reduction"]["energy_captured"] > 0.98
    # ...while the split exposes a floor carrying essentially all the error.
    assert split["truncation"] > 0.05
    assert split["surrogate"] < 1e-6  # 1e-6, not 1e-12: BLAS margin discipline

    _path, text = generate_report(proj, run)
    assert SECTION in text
    assert f"{split['truncation']:.5g}" in text  # the floor is visibly rendered


# --- CV aggregation has teeth (AC 1: per-fold first, Jensen) ------------------------------

def test_cv_mean_aggregates_per_fold_splits_not_means(tmp_path):
    # Degree-1 inner underfits -> real, differing per-fold in-span error. Only this
    # fixture discriminates: mean-of-per-fold-splits != sqrt-of-aggregated-means
    # at the report's 5g rendering (~0.19907 vs ~0.19925 on the pin).
    proj, run = _field_run(tmp_path, a3=0.8, degree=1)
    cv = run.metrics["cross_validation"]
    folds = [
        two_source_split(e, t)
        for e, t in zip(
            cv["energy_l2"]["per_fold"], cv["truncation_error"]["per_fold"], strict=True
        )
    ]
    expected_mean = float(np.mean([f["surrogate"] for f in folds]))
    jensen_wrong = two_source_split(cv["energy_l2"]["mean"], cv["truncation_error"]["mean"])[
        "surrogate"
    ]
    assert f"{expected_mean:.5g}" != f"{jensen_wrong:.5g}"  # the fixture discriminates

    _path, text = generate_report(proj, run)
    rendered = _table_cells(text, "surrogate error")[2]  # CV mean column
    assert rendered == pytest.approx(expected_mean, rel=1e-4)
    assert f"{rendered:.5g}" != f"{jensen_wrong:.5g}"

    # The std cells are also per-fold-derived (review patch: a std-only bug — e.g.
    # rendering 0 or another component's std — must not survive the suite).
    for label, comp in (
        ("total field error", "total"),
        ("truncation floor", "truncation"),
        ("surrogate error", "surrogate"),
    ):
        expected_std = float(np.std([f[comp] for f in folds]))
        assert _table_cells(text, label)[3] == pytest.approx(
            expected_std, rel=1e-3, abs=1e-6
        )


# --- absence regression (AC 4) -------------------------------------------------------------

def test_scalar_report_is_unchanged(tmp_path):
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, size=(60, 2))
    frame = pd.DataFrame(
        {"a": x[:, 0], "b": x[:, 1], "y": 2.0 * x[:, 0] + x[:, 1] ** 2}
    )
    csv = tmp_path / "scalar.csv"
    frame.to_csv(csv, index=False)
    proj = Project.init(tmp_path / "proj", name="scalar")
    proj.add_dataset(csv, name="data")
    run = run_training(
        proj,
        parse_config(
            {
                "experiment": "scalar",
                "data": {
                    "source": "data",
                    "inputs": ["a", "b"],
                    "outputs": ["y"],
                    "split": {"test_fraction": 0.25, "seed": 1},
                },
                "model": {"method": "polynomial", "degree": 2},
                "validation": {"cross_validation": 3, "metrics": ["rmse", "r2"]},
            }
        ),
    )
    _path, text = generate_report(proj, run)
    assert SECTION not in text
    # Splice-point pin: the text between the CV line and the next section is
    # exactly what it was before this story — no jinja whitespace drift.
    assert SPLICE in text


def test_field_run_without_the_metrics_gets_no_section(tmp_path):
    # Legacy path: a field run configured with rmse only has no numbers to derive
    # from — the report must render exactly the old shape.
    proj, run = _field_run(tmp_path, a3=2.0, metrics=("rmse",))
    _path, text = generate_report(proj, run)
    assert SECTION not in text
    assert SPLICE in text


def test_incomplete_cv_records_degrade_to_legacy_report(tmp_path):
    # Foreign or hand-edited CV records (review patch): the legacy report renders
    # fine from mean/std alone (train/test scalars are hard-required by the main
    # table pre-R3.2), so the section must degrade silently — not crash on
    # zip(strict=True) or render nan cells from np.mean([]).
    proj, run = _field_run(tmp_path, a3=2.0)
    cv = run.metrics["cross_validation"]

    cv["truncation_error"]["per_fold"].pop()  # unequal lengths
    _path, text = generate_report(proj, run)
    assert SECTION not in text
    assert SPLICE in text

    cv["truncation_error"]["per_fold"] = []  # empty list
    _path, text = generate_report(proj, run)
    assert SECTION not in text
    assert SPLICE in text

    del cv["energy_l2"]["per_fold"]  # key absent entirely
    _path, text = generate_report(proj, run)
    assert SECTION not in text
    assert SPLICE in text


# --- reduction line is optional (method-agnostic metrics since r3-1) -----------------------

def test_unreduced_method_renders_split_without_reduction_line(tmp_path):
    # Field metrics are method-agnostic (r3-1): a plain polynomial run has no
    # lineage["reduction"], so the section must render WITHOUT the context line
    # rather than failing on the missing key.
    proj, run = _field_run(tmp_path, a3=2.0, model={"method": "polynomial", "degree": 2})
    assert "reduction" not in run.lineage  # the precondition this test exists for

    _path, text = generate_report(proj, run)
    assert SECTION in text
    for label in ROW_LABELS:
        assert f"| {label}" in text
    assert "Reduction: rank" not in text
