"""Sampling strategies that turn a DoeSpec into a table of points."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from surroforge_professional.doe.spec import (
    CategoricalFactor,
    ContinuousFactor,
    DiscreteFactor,
    DoeError,
    DoeSpec,
)


def _require_qmc():
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise DoeError(
            "LHS and Sobol designs require scipy. Install it with "
            "`pip install 'surroforge-professional[doe]'`."
        ) from exc
    return qmc


def _factorial_levels(factor) -> list:
    if isinstance(factor, ContinuousFactor):
        lo, hi = factor.bounds
        return [float(v) for v in np.linspace(lo, hi, factor.num_levels)]
    if isinstance(factor, DiscreteFactor):
        return [float(v) for v in factor.levels]
    return list(factor.categories)


def _discrete_choices(factor) -> list | None:
    if isinstance(factor, DiscreteFactor):
        return [float(v) for v in factor.levels]
    if isinstance(factor, CategoricalFactor):
        return list(factor.categories)
    return None


def _map_unit(factor, column: np.ndarray) -> list:
    if isinstance(factor, ContinuousFactor):
        lo, hi = factor.bounds
        return [float(v) for v in (lo + column * (hi - lo))]
    choices = _discrete_choices(factor)
    k = len(choices)
    index = np.minimum((column * k).astype(int), k - 1)
    return [choices[i] for i in index]


def _unit_hypercube(spec: DoeSpec, dimensions: int) -> np.ndarray:
    if spec.design == "lhs":
        qmc = _require_qmc()
        return qmc.LatinHypercube(d=dimensions, seed=spec.seed).random(spec.samples)
    if spec.design == "sobol":
        qmc = _require_qmc()
        n = spec.samples
        if n & (n - 1):  # not a power of two
            warnings.warn(
                f"Sobol sampling is most uniform at a power of two; got samples={n}.",
                UserWarning,
                stacklevel=2,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence scipy's own balance warning
            return qmc.Sobol(d=dimensions, seed=spec.seed).random(n)
    rng = np.random.default_rng(spec.seed)
    return rng.random((spec.samples, dimensions))


def generate(spec: DoeSpec) -> pd.DataFrame:
    """Return a DataFrame of sample points, one column per factor."""
    if spec.design == "full-factorial":
        columns = [_factorial_levels(f) for f in spec.factors]
        rows = list(itertools.product(*columns))
        data = {f.name: [row[i] for row in rows] for i, f in enumerate(spec.factors)}
        return pd.DataFrame(data)

    unit = _unit_hypercube(spec, len(spec.factors))
    data = {f.name: _map_unit(f, unit[:, i]) for i, f in enumerate(spec.factors)}
    return pd.DataFrame(data)
