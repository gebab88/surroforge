"""High-level Python API (PRD 4.10).

Mirrors the CLI using the same concepts and vocabulary::

    from surroforge import Project, Trainer

    project = Project.open("path/to/project")
    trainer = Trainer.from_config("configs/kriging.yaml")
    run = trainer.fit(project.dataset("my_data"))
    print(run.metrics)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from surroforge.core.config import ExperimentConfig, load_config, parse_config
from surroforge.core.dataset import Dataset
from surroforge.core.project import Project
from surroforge.core.run import Run
from surroforge.engine import run_training
from surroforge.errors import DataError


class Trainer:
    """A training configuration bound to its surrogate method, ready to fit."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, path: str | Path) -> Trainer:
        return cls(load_config(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trainer:
        return cls(parse_config(data))

    def fit(self, target: Project | Dataset) -> Run:
        """Train and persist a run.

        ``target`` may be a :class:`Project` (the data source is taken from the
        config) or a :class:`Dataset` obtained from ``project.dataset(name)``,
        which overrides the config's ``data.source``.
        """
        if isinstance(target, Project):
            return run_training(target, self.config)
        if isinstance(target, Dataset):
            project = getattr(target, "project", None)
            if project is None:
                raise DataError(
                    "Dataset is not bound to a project. Obtain it via "
                    "project.dataset(name) so the run can be persisted."
                )
            return run_training(project, self.config, dataset=target)
        raise TypeError(
            f"Trainer.fit expects a Project or Dataset, got {type(target).__name__}."
        )


__all__ = ["Trainer"]
