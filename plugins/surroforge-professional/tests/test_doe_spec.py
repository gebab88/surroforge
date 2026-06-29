from __future__ import annotations

import pytest

from surroforge_professional.doe.spec import (
    CategoricalFactor,
    ContinuousFactor,
    DiscreteFactor,
    DoeError,
    load_spec,
)


def _write(tmp_path, text):
    path = tmp_path / "doe.yaml"
    path.write_text(text)
    return path


def test_load_valid_lhs_spec(tmp_path):
    spec = load_spec(
        _write(
            tmp_path,
            """
design: lhs
samples: 10
seed: 1
factors:
  - {name: aoa, type: continuous, bounds: [-5, 16]}
  - {name: speed, type: discrete, levels: [20, 40, 60]}
  - {name: turb, type: categorical, categories: [SA, kOmega]}
""",
        )
    )
    assert spec.design == "lhs"
    assert spec.samples == 10
    assert spec.seed == 1
    assert spec.case_prefix == "case"
    assert isinstance(spec.factors[0], ContinuousFactor)
    assert isinstance(spec.factors[1], DiscreteFactor)
    assert isinstance(spec.factors[2], CategoricalFactor)


def test_missing_file_raises_doe_error(tmp_path):
    with pytest.raises(DoeError, match="not found"):
        load_spec(tmp_path / "nope.yaml")


def test_unknown_design_raises_doe_error(tmp_path):
    with pytest.raises(DoeError):
        load_spec(_write(tmp_path, "design: grid\nsamples: 4\nfactors: [{name: a, type: discrete, levels: [1,2]}]\n"))


def test_samples_forbidden_for_full_factorial(tmp_path):
    with pytest.raises(DoeError, match="full-factorial does not take 'samples'"):
        load_spec(
            _write(
                tmp_path,
                "design: full-factorial\nsamples: 4\nfactors: [{name: a, type: discrete, levels: [1,2]}]\n",
            )
        )


def test_full_factorial_continuous_needs_num_levels(tmp_path):
    with pytest.raises(DoeError, match="num_levels"):
        load_spec(
            _write(
                tmp_path,
                "design: full-factorial\nfactors: [{name: a, type: continuous, bounds: [0, 1]}]\n",
            )
        )


def test_sampling_design_requires_samples(tmp_path):
    with pytest.raises(DoeError, match="requires 'samples'"):
        load_spec(_write(tmp_path, "design: lhs\nfactors: [{name: a, type: discrete, levels: [1,2]}]\n"))


def test_duplicate_factor_names_rejected(tmp_path):
    with pytest.raises(DoeError, match="duplicate factor name"):
        load_spec(
            _write(
                tmp_path,
                "design: lhs\nsamples: 4\nfactors: [{name: a, type: discrete, levels: [1]}, "
                "{name: a, type: discrete, levels: [2]}]\n",
            )
        )


def test_bounds_require_lo_below_hi(tmp_path):
    with pytest.raises(DoeError, match="lo < hi"):
        load_spec(
            _write(
                tmp_path,
                "design: lhs\nsamples: 4\nfactors: [{name: a, type: continuous, bounds: [3, 1]}]\n",
            )
        )


def test_empty_levels_rejected(tmp_path):
    with pytest.raises(DoeError, match="non-empty"):
        load_spec(_write(tmp_path, "design: lhs\nsamples: 4\nfactors: [{name: a, type: discrete, levels: []}]\n"))
