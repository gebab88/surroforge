"""Design-of-experiments generation for SurroForge Professional."""

from surroforge_professional.doe.spec import (
    CategoricalFactor,
    ContinuousFactor,
    DiscreteFactor,
    DoeError,
    DoeSpec,
    load_spec,
    validate_spec,
)

__all__ = [
    "CategoricalFactor",
    "ContinuousFactor",
    "DiscreteFactor",
    "DoeError",
    "DoeSpec",
    "load_spec",
    "validate_spec",
]
