from __future__ import annotations

import pytest

from surroforge_professional.doe.sampling import generate
from surroforge_professional.doe.spec import DoeSpec


def _spec(**kw):
    return DoeSpec(**kw)


def test_full_factorial_is_cartesian_product():
    spec = _spec(
        design="full-factorial",
        factors=[
            {"name": "a", "type": "discrete", "levels": [1, 2, 3]},
            {"name": "b", "type": "categorical", "categories": ["x", "y"]},
        ],
    )
    frame = generate(spec)
    assert list(frame.columns) == ["a", "b"]
    assert len(frame) == 6  # 3 x 2
    assert set(frame["b"]) == {"x", "y"}
    assert set(frame["a"]) == {1.0, 2.0, 3.0}


def test_full_factorial_continuous_uses_linspace():
    spec = _spec(
        design="full-factorial",
        factors=[{"name": "a", "type": "continuous", "bounds": [0, 1], "num_levels": 3}],
    )
    frame = generate(spec)
    assert list(frame["a"]) == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("design", ["lhs", "sobol", "random"])
def test_sampling_designs_respect_count_and_bounds(design):
    spec = _spec(
        design=design,
        samples=8,
        seed=1,
        factors=[
            {"name": "aoa", "type": "continuous", "bounds": [-5, 16]},
            {"name": "speed", "type": "discrete", "levels": [20, 40, 60]},
            {"name": "turb", "type": "categorical", "categories": ["SA", "kOmega"]},
        ],
    )
    frame = generate(spec)
    assert len(frame) == 8
    assert frame["aoa"].between(-5, 16).all()
    assert set(frame["speed"]).issubset({20.0, 40.0, 60.0})
    assert set(frame["turb"]).issubset({"SA", "kOmega"})


@pytest.mark.parametrize("design", ["lhs", "sobol", "random"])
def test_sampling_is_deterministic_per_seed(design):
    kw = dict(
        design=design,
        samples=8,
        factors=[{"name": "a", "type": "continuous", "bounds": [0, 1]}],
    )
    same = generate(_spec(seed=1, **kw)).equals(generate(_spec(seed=1, **kw)))
    differ = generate(_spec(seed=1, **kw)).equals(generate(_spec(seed=2, **kw)))
    assert same is True
    assert differ is False


def test_sobol_non_power_of_two_warns():
    spec = _spec(
        design="sobol",
        samples=6,
        factors=[{"name": "a", "type": "continuous", "bounds": [0, 1]}],
    )
    with pytest.warns(UserWarning, match="power of two"):
        generate(spec)
