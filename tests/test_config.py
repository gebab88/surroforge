"""Configuration validation (PRD 4.3)."""

from __future__ import annotations

import pytest

from surroforge.core.config import parse_config
from surroforge.errors import ConfigError
from tests.conftest import poly_config


def test_valid_config_parses():
    cfg = parse_config(poly_config())
    assert cfg.experiment == "beam-polynomial"
    assert cfg.model.method == "polynomial"
    assert cfg.model.hyperparameters == {"degree": 2}
    assert cfg.validation.cross_validation.folds == 5
    assert cfg.data.split.test_fraction == 0.2


def test_cross_validation_int_shorthand():
    cfg = parse_config(poly_config())
    assert cfg.validation.cross_validation.folds == 5
    assert cfg.validation.cross_validation.shuffle is True


def test_defaults_applied_when_optional_fields_missing():
    raw = poly_config()
    del raw["data"]["split"]
    del raw["validation"]["metrics"]
    cfg = parse_config(raw)
    assert cfg.data.split.test_fraction == 0.2
    assert cfg.data.split.seed == 42
    assert cfg.validation.metrics == ["rmse", "r2", "mae"]


@pytest.mark.parametrize("missing", ["experiment", "data", "model", "validation"])
def test_missing_required_top_level_field(missing):
    raw = poly_config()
    del raw[missing]
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_missing_required_data_field():
    raw = poly_config()
    del raw["data"]["inputs"]
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_unknown_top_level_field_rejected():
    raw = poly_config()
    raw["unexpected"] = True
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_duplicate_columns_rejected():
    raw = poly_config()
    raw["data"]["inputs"] = ["length", "length", "load"]
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_bad_test_fraction_rejected():
    raw = poly_config()
    raw["data"]["split"]["test_fraction"] = 1.5
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_canonical_json_is_order_independent():
    a = parse_config(poly_config())
    raw = poly_config()
    # Reorder keys; canonical form must be identical.
    raw["validation"] = {"metrics": ["rmse", "r2", "mae"], "cross_validation": 5}
    b = parse_config(raw)
    assert a.canonical_json() == b.canonical_json()
