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


# --- model.reduction / model.inner blocks (Story R1.2) --------------------------------

def _reduced_config(reduction: dict, inner: dict | None = None) -> dict:
    raw = poly_config()
    raw["model"] = {"method": "reduced-field", "reduction": reduction}
    if inner is not None:
        raw["model"]["inner"] = inner
    return raw


@pytest.mark.parametrize(
    "reduction",
    [
        {"basis": "pod", "energy": 0.999},
        {"basis": "pod", "rank": 12},
        {"rank": "adaptive"},  # basis defaults to pod
    ],
)
def test_valid_reduction_blocks_parse(reduction):
    cfg = parse_config(_reduced_config(reduction))
    assert cfg.model.reduction is not None
    assert cfg.model.reduction.basis == "pod"


@pytest.mark.parametrize(
    ("reduction", "field"),
    [
        ({"energy": 0.9, "rank": 3}, "reduction"),  # both set
        ({"basis": "pod"}, "reduction"),  # neither set
        ({"basis": "svd", "energy": 0.9}, "basis"),
        ({"energy": 1.5}, "energy"),
        ({"energy": 0}, "energy"),
        ({"rank": 0}, "rank"),
        ({"rank": -1}, "rank"),
        ({"rank": True}, "rank"),  # bool must not coerce to rank=1
        ({"energy": True}, "energy"),  # bool must not coerce to energy=1.0
    ],
)
def test_invalid_reduction_raises_config_error_naming_field(reduction, field):
    with pytest.raises(ConfigError, match=field):
        parse_config(_reduced_config(reduction))


def test_inner_block_parses_method_plus_hyperparameters():
    cfg = parse_config(
        _reduced_config({"energy": 0.999}, inner={"method": "kriging", "corr": "squar_exp"})
    )
    assert cfg.model.inner is not None
    assert cfg.model.inner.method == "kriging"
    assert cfg.model.inner.hyperparameters == {"corr": "squar_exp"}


def test_hyperparameters_synthesizes_reduction_and_inner_dicts():
    # The engine's trainer_cls(**hyperparameters) factory must receive the same dict
    # kwargs as the pre-R1.2 extra-dict form (AC5: the flow is preserved).
    cfg = parse_config(
        _reduced_config(
            {"basis": "pod", "energy": 0.999},
            inner={"method": "polynomial", "degree": 2},
        )
    )
    hyper = cfg.model.hyperparameters
    assert hyper["reduction"] == {"basis": "pod", "energy": 0.999}  # unset rank dropped
    assert hyper["inner"] == {"method": "polynomial", "degree": 2}


def test_scalar_config_serialization_unchanged_by_reduction_fields():
    # AC4's only net for an exclude_none regression: the R2.4 guard pins fingerprint
    # bytes but NOT the config_hash value, so `reduction: null` leaking into the dump
    # would change every scalar config_hash and still pass R2.4. Pin the model section
    # by construction (exactly the user's keys — nothing added, nothing dropped),
    # never a golden hash literal.
    cfg = parse_config(poly_config())
    dumped = cfg.canonical_dict()
    assert dumped["model"] == {"method": "polynomial", "degree": 2}
    assert cfg.model.hyperparameters == {"degree": 2}  # no synthesized keys
    assert "reduction" not in cfg.to_yaml()
    assert "inner" not in cfg.to_yaml()


def test_null_hyperparameter_extra_rejected():
    # A null extra would be silently dropped by exclude_none serialization while the
    # trainer was constructed WITH it (snapshot lies about the run). Reject loudly.
    raw = poly_config()
    raw["model"]["degree"] = None
    with pytest.raises(ConfigError, match="degree"):
        parse_config(raw)


def test_null_inner_hyperparameter_extra_rejected():
    raw = _reduced_config({"energy": 0.999}, inner={"method": "kriging", "corr": None})
    with pytest.raises(ConfigError, match="corr"):
        parse_config(raw)


def test_declared_none_reduction_inner_still_allowed():
    # Explicit `reduction: null` / `inner: null` on the DECLARED fields means "absent" —
    # only None-valued hyperparameter extras are rejected.
    raw = poly_config()
    raw["model"]["reduction"] = None
    raw["model"]["inner"] = None
    cfg = parse_config(raw)
    assert cfg.model.reduction is None
    assert cfg.model.hyperparameters == {"degree": 2}
    assert cfg.canonical_dict()["model"] == {"method": "polynomial", "degree": 2}
