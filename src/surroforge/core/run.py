"""A persisted training run (PRD 4.5, 7.1, 7.2).

``Run`` is a thin, lazy view over a ``runs/<run-id>/`` directory: it reads the
lineage, metrics, config snapshot, and model artifacts on demand rather than
holding them in memory.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

from surroforge.core.config import ExperimentConfig, parse_config
from surroforge.errors import RunNotFoundError
from surroforge.persistence import read_json, read_yaml

CONFIG_SNAPSHOT = "config.snapshot.yaml"
METRICS_FILE = "metrics.json"
LINEAGE_FILE = "lineage.json"
MODEL_PICKLE = "model.pkl"
MODEL_ONNX = "model.onnx"
LOG_FILE = "log.txt"


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
        """Out-of-sample holdout predictions recorded for the reproducibility verdict."""
        return self.lineage.get("holdout_fingerprint", [])

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
        return load_pickle(self.model_path)
