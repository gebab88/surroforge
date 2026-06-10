"""Training configuration models (PRD 4.3).

A single YAML file is the complete and only specification of a training run. The
models below validate that file with Pydantic and expose a canonical
serialisation used for config hashing (PRD 4.5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from surroforge.errors import ConfigError

DEFAULT_METRICS = ["rmse", "r2", "mae"]


def _reject_null_extras(data: Any, declared: set[str]) -> Any:
    """Reject ``None``-valued hyperparameter extras (Story R1.2 review).

    Canonical serialization is ``exclude_none``, so a null extra would be silently
    dropped from the snapshot and ``config_hash`` while the trainer was constructed
    WITH it — the persisted run would no longer describe the trainer that trained.
    Rejecting loudly keeps the snapshot faithful; a user wanting a default omits the key.
    (Declared optional fields like ``model.reduction: null`` simply mean "absent" and
    stay allowed.)
    """
    if isinstance(data, dict):
        null_extras = sorted(k for k, v in data.items() if v is None and k not in declared)
        if null_extras:
            raise ValueError(
                f"null hyperparameter(s) {null_extras}: a null value would be dropped "
                "from the canonical config snapshot; omit the key to use the default"
            )
    return data


class SplitConfig(BaseModel):
    """Train/test split (PRD 4.3, optional with defaults)."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["random"] = "random"
    test_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42


class DataConfig(BaseModel):
    """Data source and column roles (PRD 4.3, required)."""

    model_config = ConfigDict(extra="forbid")

    source: str
    inputs: list[str] = Field(min_length=1)
    outputs: list[str] = Field(min_length=1)
    split: SplitConfig = Field(default_factory=SplitConfig)

    @field_validator("inputs", "outputs")
    @classmethod
    def _no_duplicates(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("column names must be unique")
        return value


class ReductionConfig(BaseModel):
    """Field reduction spec for a reducing method (Story R1.2).

    Exactly one of ``energy`` (POD energy share to retain) or ``rank`` (mode count,
    or the literal ``adaptive`` resolved at fit and written back to the snapshot)
    must be given. Validated here for early, precise errors; the trainer keeps its
    own equivalent validation as defense-in-depth (it is constructed directly by
    the Python API and the conformance suite, bypassing config).
    """

    model_config = ConfigDict(extra="forbid")

    basis: Literal["pod"] = "pod"
    energy: float | None = Field(default=None, gt=0.0, le=1.0)
    rank: int | Literal["adaptive"] | None = None

    @field_validator("rank", mode="before")
    @classmethod
    def _rank_positive_int_or_adaptive(cls, value: Any) -> Any:
        # mode="before" is load-bearing: under the int | Literal union Pydantic
        # coerces True -> 1 before an after-validator runs, so `rank: true` would
        # silently become rank=1. The raw value is the only place bool is visible.
        if isinstance(value, bool):
            raise ValueError("rank must be a positive integer or 'adaptive', got a boolean")
        if isinstance(value, int) and value < 1:
            raise ValueError(f"rank must be a positive integer or 'adaptive', got {value}")
        return value

    @field_validator("energy", mode="before")
    @classmethod
    def _energy_not_bool(cls, value: Any) -> Any:
        # Same lax-coercion trap as rank: `energy: true` would become 1.0.
        if isinstance(value, bool):
            raise ValueError("energy must be a number in (0, 1], got a boolean")
        return value

    @model_validator(mode="after")
    def _exactly_one_target(self) -> ReductionConfig:
        # No field-name prefix: the ConfigError formatter already prints the
        # dotted location (model.reduction).
        if self.energy is not None and self.rank is not None:
            raise ValueError("specify exactly one of energy or rank, not both")
        if self.energy is None and self.rank is None:
            raise ValueError("specify exactly one of energy or rank")
        return self


class InnerConfig(BaseModel):
    """Inner-surrogate method spec for a reducing method (Story R1.2).

    An ordinary method spec (``method`` + method hyperparameters). The method name
    is resolved through ``get_trainer`` by the trainer at fit time, never here —
    config stays a leaf (no ``plugins`` import).
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    method: str

    @model_validator(mode="before")
    @classmethod
    def _no_null_extras(cls, data: Any) -> Any:
        return _reject_null_extras(data, set(cls.model_fields))

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return dict(self.__pydantic_extra__ or {})


class ModelConfig(BaseModel):
    """Surrogate method plus method-specific hyperparameters (PRD 4.3).

    Hyperparameters are method-specific and therefore not enumerated here; any
    field other than ``method`` is collected and passed through to the trainer.
    The structured ``reduction``/``inner`` blocks (Story R1.2) are optional and,
    when present, surface through :attr:`hyperparameters` as plain dicts, so the
    engine's method-agnostic ``trainer_cls(**hyperparameters)`` factory is
    unchanged.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    method: str
    reduction: ReductionConfig | None = None
    inner: InnerConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_null_extras(cls, data: Any) -> Any:
        return _reject_null_extras(data, set(cls.model_fields))

    @property
    def hyperparameters(self) -> dict[str, Any]:
        hyper = dict(self.__pydantic_extra__ or {})
        if self.reduction is not None:
            hyper["reduction"] = self.reduction.model_dump(exclude_none=True)
        if self.inner is not None:
            hyper["inner"] = self.inner.model_dump()
        return hyper


class CrossValidationConfig(BaseModel):
    """k-fold cross-validation strategy (PRD 4.6)."""

    model_config = ConfigDict(extra="forbid")

    folds: int = Field(default=5, ge=2)
    shuffle: bool = True


class ValidationConfig(BaseModel):
    """Validation strategy and reported metrics (PRD 4.3)."""

    model_config = ConfigDict(extra="forbid")

    cross_validation: CrossValidationConfig
    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_METRICS))

    @field_validator("cross_validation", mode="before")
    @classmethod
    def _coerce_int(cls, value: Any) -> Any:
        # `cross_validation: 5` is shorthand for `{folds: 5}`.
        if isinstance(value, int) and not isinstance(value, bool):
            return {"folds": value}
        return value


class ExperimentConfig(BaseModel):
    """The full specification of one training run (PRD 4.3)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    experiment: str
    data: DataConfig
    model: ModelConfig
    validation: ValidationConfig

    def canonical_dict(self) -> dict[str, Any]:
        """Deterministic, sorted dict used for config hashing (PRD 4.5).

        ``exclude_none`` is load-bearing (Story R1.2): the declared-but-unset
        ``model.reduction``/``model.inner`` fields must never reach the wire, or
        every scalar config would dump ``reduction: null`` and change its
        ``config_hash`` — breaking the R2.4 byte-identity contract. No pre-R1.2
        config has a ``None``-valued *declared* field, and ``None``-valued
        hyperparameter extras are rejected at validation (``_reject_null_extras``),
        so nothing a trainer actually received is ever dropped here.
        """
        return self.model_dump(mode="json", exclude_none=True)

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.canonical_dict(), sort_keys=False)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML training configuration."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {path}: {exc}") from exc
    return parse_config(raw, origin=str(path))


def parse_config(raw: Any, *, origin: str = "<config>") -> ExperimentConfig:
    """Validate a raw mapping into an :class:`ExperimentConfig`."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{origin}: top level must be a mapping, got {type(raw).__name__}")
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(origin, exc)) from exc


def _format_validation_error(origin: str, exc: ValidationError) -> str:
    lines = [f"Invalid configuration in {origin}:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)
