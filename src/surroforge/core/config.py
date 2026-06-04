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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from surroforge.errors import ConfigError

DEFAULT_METRICS = ["rmse", "r2", "mae"]


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


class ModelConfig(BaseModel):
    """Surrogate method plus method-specific hyperparameters (PRD 4.3).

    Hyperparameters are method-specific and therefore not enumerated here; any
    field other than ``method`` is collected and passed through to the trainer.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    method: str

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return dict(self.__pydantic_extra__ or {})


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
        """Deterministic, sorted dict used for config hashing (PRD 4.5)."""
        return self.model_dump(mode="json")

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


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
