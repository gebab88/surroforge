"""DoE specification schema, loader and semantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from surroforge.errors import SurroForgeError


class DoeError(SurroForgeError):
    """Raised for an invalid DoE specification or generation request."""


class _FactorBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class ContinuousFactor(_FactorBase):
    type: Literal["continuous"]
    bounds: tuple[float, float]
    num_levels: int | None = None


class DiscreteFactor(_FactorBase):
    type: Literal["discrete"]
    levels: list[float]


class CategoricalFactor(_FactorBase):
    type: Literal["categorical"]
    categories: list[str]


Factor = Annotated[
    Union[ContinuousFactor, DiscreteFactor, CategoricalFactor],
    Field(discriminator="type"),
]


class DoeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    design: Literal["full-factorial", "lhs", "sobol", "random"]
    samples: int | None = None
    seed: int = 0
    case_prefix: str = "case"
    factors: list[Factor]


def validate_spec(spec: DoeSpec) -> None:
    """Semantic validation beyond field shapes; raises :class:`DoeError`."""
    if not spec.factors:
        raise DoeError("a DoE spec must declare at least one factor.")

    names = [f.name for f in spec.factors]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise DoeError(f"duplicate factor name(s): {', '.join(dupes)}.")

    for factor in spec.factors:
        if isinstance(factor, ContinuousFactor):
            lo, hi = factor.bounds
            if not lo < hi:
                raise DoeError(f"factor '{factor.name}': bounds require lo < hi, got [{lo}, {hi}].")
            if factor.num_levels is not None and factor.num_levels < 2:
                raise DoeError(
                    f"factor '{factor.name}': num_levels must be >= 2, got {factor.num_levels}."
                )
        elif isinstance(factor, DiscreteFactor):
            if not factor.levels:
                raise DoeError(f"factor '{factor.name}': levels must be non-empty.")
        else:
            if not factor.categories:
                raise DoeError(f"factor '{factor.name}': categories must be non-empty.")

    if spec.design == "full-factorial":
        if spec.samples is not None:
            raise DoeError(
                "full-factorial does not take 'samples'; the count is the product of factor levels."
            )
        for factor in spec.factors:
            if isinstance(factor, ContinuousFactor) and factor.num_levels is None:
                raise DoeError(
                    f"factor '{factor.name}': continuous factors need 'num_levels' under full-factorial."
                )
    else:
        if spec.samples is None or spec.samples < 1:
            raise DoeError(f"design '{spec.design}' requires 'samples' >= 1.")


def load_spec(path: str | Path) -> DoeSpec:
    """Read a DoE spec YAML file, parse it, and validate it."""
    path = Path(path)
    if not path.is_file():
        raise DoeError(f"DoE spec file not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise DoeError(f"{path}: DoE spec must be a YAML mapping.")
    try:
        spec = DoeSpec(**raw)
    except ValidationError as exc:
        raise DoeError(f"{path}: invalid DoE spec:\n{exc}") from exc
    validate_spec(spec)
    return spec
