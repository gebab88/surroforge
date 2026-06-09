"""A persisted training run (PRD 4.5, 7.1, 7.2).

``Run`` is a thin, lazy view over a ``runs/<run-id>/`` directory: it reads the
lineage, metrics, config snapshot, and model artifacts on demand rather than
holding them in memory.
"""

from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from typing import Any

from surroforge.core.config import ExperimentConfig, parse_config
from surroforge.errors import RunNotFoundError
from surroforge.persistence import is_locally_produced, read_json, read_yaml

CONFIG_SNAPSHOT = "config.snapshot.yaml"
METRICS_FILE = "metrics.json"
LINEAGE_FILE = "lineage.json"
MODEL_PICKLE = "model.pkl"
MODEL_ONNX = "model.onnx"
LOG_FILE = "log.txt"

class ProvenanceWarning(UserWarning):
    """Loading a model artifact not confirmed as produced on this machine.

    A subclass (rather than a bare ``UserWarning``) so adapters can render it in
    their own style — the CLI prints a sober ``Warning:`` line — while library
    callers still filter it with the standard ``warnings`` machinery.
    """


# Artifacts we have already warned about this process, so the provenance warning
# fires once per artifact rather than on every predict/export call (Story 4.4).
_WARNED_ARTIFACTS: set[str] = set()


def _warn_if_foreign_artifact(model_path: Path, lineage: dict[str, Any]) -> None:
    """Warn once before unpickling a model not produced on this machine.

    Unpickling executes code embedded in the file, so loading a shared
    ``model.pkl`` is an arbitrary-code-execution risk (Story 4.4 / AR12).
    Locally-produced artifacts load silently. Emitted via ``warnings`` so it
    reaches both CLI and library callers and is *not* silenced by ``--quiet``
    (a security signal); users can still suppress it with ``-W ignore``.
    """
    if is_locally_produced(lineage):
        return
    key = str(model_path.resolve())
    if key in _WARNED_ARTIFACTS:
        return
    _WARNED_ARTIFACTS.add(key)
    warnings.warn(
        f"Model artifact '{model_path}' was not confirmed as produced on this "
        "machine. Loading a pickle executes code embedded in the file — only "
        "load model artifacts from sources you trust.",
        ProvenanceWarning,
        stacklevel=3,
    )


class Run:
    """Read access to a single run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name

    def exists(self) -> bool:
        return (self.run_dir / LINEAGE_FILE).is_file()

    def _require(self) -> None:
        if not self.exists():
            raise RunNotFoundError(f"Run '{self.run_id}' not found at {self.run_dir}")

    @cached_property
    def lineage(self) -> dict[str, Any]:
        self._require()
        return read_json(self.run_dir / LINEAGE_FILE)

    @cached_property
    def metrics(self) -> dict[str, Any]:
        self._require()
        return read_json(self.run_dir / METRICS_FILE)

    @cached_property
    def config_dict(self) -> dict[str, Any]:
        self._require()
        return read_yaml(self.run_dir / CONFIG_SNAPSHOT)

    @cached_property
    def config(self) -> ExperimentConfig:
        return parse_config(self.config_dict, origin=(self.run_dir / CONFIG_SNAPSHOT).as_posix())

    # Convenience accessors over lineage --------------------------------------

    @property
    def created_at(self) -> str:
        return self.lineage.get("created_at", "")

    @property
    def config_hash(self) -> str:
        return self.lineage.get("config_hash", "")

    @property
    def data_hash(self) -> str:
        return self.lineage.get("data_hash", "")

    @property
    def seed(self) -> int:
        return int(self.lineage.get("seed", 0))

    @property
    def method(self) -> str:
        return self.config.model.method

    @property
    def parent_run(self) -> str | None:
        return self.lineage.get("parent_run")

    @property
    def holdout_fingerprint(self) -> list[Any]:
        """Out-of-sample holdout predictions recorded for the reproducibility verdict.

        For a field run this is the reconstructed out-of-sample holdout *field*; the
        field-primary verdict compares it by field-energy-relative diff (Story R2.3).
        """
        return self.lineage.get("holdout_fingerprint", [])

    # Field-run diagnostics (Story R2.3) — empty/blank for scalar runs ----------

    @property
    def holdout_coefficients(self) -> list[Any]:
        """Canonicalized out-of-sample modal coefficients (field-run diagnostic)."""
        return self.lineage.get("holdout_coefficients", [])

    @property
    def basis_hash(self) -> str:
        """Canonicalized basis hash — diagnostic only, never a verdict gate."""
        return self.lineage.get("basis_hash", "")

    @property
    def reduction(self) -> dict[str, Any]:
        """Resolved reduction spec for a field run (empty dict for a scalar run)."""
        return self.lineage.get("reduction", {})

    @property
    def spec_hash(self) -> str:
        """Hashed spec (config + reduction + field tolerance); widening tolerance re-hashes."""
        return self.lineage.get("spec_hash", "")

    # Artifacts ----------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        return self.run_dir / MODEL_PICKLE

    @property
    def onnx_path(self) -> Path:
        return self.run_dir / MODEL_ONNX

    def has_onnx(self) -> bool:
        return self.onnx_path.is_file()

    def load_model(self) -> Any:
        from surroforge.exporters.pickle_exporter import load_pickle

        self._require()
        _warn_if_foreign_artifact(self.model_path, self.lineage)
        return load_pickle(self.model_path)
