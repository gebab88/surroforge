"""Reproducibility guarantee (PRD 4.5, 5.2)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from surroforge.core.config import parse_config
from surroforge.data.loader import select_columns
from surroforge.engine import run_training
from surroforge.errors import ReproducibilityError
from surroforge.services import (
    REPRODUCTION_TOLERANCE,
    ReproduceResult,
    assert_reproducible,
    reproduce,
)
from tests.conftest import make_samples, poly_config


def test_reproduce_is_bit_identical(project):
    cfg = parse_config(poly_config("polynomial"))
    original = run_training(project, cfg)

    result = reproduce(project, original.run_id)
    assert result.bit_identical
    assert result.max_abs_diff == 0.0
    assert result.data_hash_match
    assert result.reproduction.parent_run == original.run_id
    assert_reproducible(result)  # does not raise


def test_two_independent_runs_share_prediction_hash(project):
    cfg = parse_config(poly_config("polynomial"))
    a = run_training(project, cfg)
    b = run_training(project, cfg)
    assert a.lineage["prediction_hash"] == b.lineage["prediction_hash"]
    assert a.config_hash == b.config_hash


def test_kriging_reproducible(project):
    cfg = parse_config(poly_config("kriging"))
    original = run_training(project, cfg)
    result = reproduce(project, original.run_id)
    assert result.max_abs_diff <= 1e-9


def test_reproduce_detects_data_change(project):
    # A file-path source has its hash recomputed from disk each run, so editing
    # the file makes reproduction detect the change (PRD 5.2).
    src = project.root / "data" / "live.csv"
    make_samples(seed=1).to_csv(src, index=False)
    raw = poly_config("polynomial")
    raw["data"]["source"] = "data/live.csv"
    original = run_training(project, parse_config(raw))

    make_samples(seed=2).to_csv(src, index=False)  # underlying data changes
    result = reproduce(project, original.run_id)

    assert not result.data_hash_match
    assert result.verdict == "data_changed"
    with pytest.raises(ReproducibilityError):
        assert_reproducible(result)


def test_reproduce_emits_verdict(project):
    cfg = parse_config(poly_config("polynomial"))
    original = run_training(project, cfg)
    result = reproduce(project, original.run_id)
    assert result.verdict == "reproduced"


def test_fingerprint_is_out_of_sample(project):
    # Teeth: the persisted fingerprint must be the HOLDOUT model's predictions on
    # the held-out test split (out-of-sample). The final saved model trained on ALL
    # data, so its predictions on those same points are IN-SAMPLE — and for
    # interpolating Kriging near-exact, which is precisely why fingerprinting them
    # was the original defect. This test fails if the fingerprint is ever reverted
    # to the final model on in-sample points.
    cfg = parse_config(poly_config("kriging"))
    run = run_training(project, cfg)
    stored = np.asarray(run.holdout_fingerprint, dtype=float)
    assert stored.size > 0

    frame, _, _ = project.resolve_source(cfg.data.source)
    x, y = select_columns(frame, cfg.data.inputs, cfg.data.outputs)
    _, x_test, _, _ = train_test_split(
        x, y,
        test_size=cfg.data.split.test_fraction,
        random_state=cfg.data.split.seed,
        shuffle=True,
    )
    final_in_sample = np.ascontiguousarray(
        run.load_model().predict(x_test), dtype=float
    )[: len(stored)]

    assert stored.shape == final_in_sample.shape
    # The out-of-sample holdout predictions differ from the final model's in-sample
    # predictions by far more than the reproduction tolerance.
    assert float(np.max(np.abs(stored - final_in_sample))) > REPRODUCTION_TOLERANCE


def test_assert_reproducible_raises_on_numeric_divergence():
    # AC3: a genuine over-tolerance divergence raises, independent of data change.
    result = ReproduceResult(
        parent=None,
        reproduction=None,
        bit_identical=False,
        max_abs_diff=1.0,
        data_hash_match=True,
        verdict="diverged",
    )
    with pytest.raises(ReproducibilityError):
        assert_reproducible(result)
