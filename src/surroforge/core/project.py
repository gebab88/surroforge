"""Project facade: the on-disk workspace that holds data, configs, and runs.

Directory layout (PRD 7.1)::

    my-project/
    ├── surroforge.yaml      # project metadata + registered datasets
    ├── data/                # input data
    ├── configs/             # YAML training configurations
    ├── runs/                # one directory per training run
    └── reports/             # generated reports
"""

from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from surroforge.core.dataset import Dataset
from surroforge.core.run import Run
from surroforge.data.hashing import hash_dataframe
from surroforge.data.loader import SUPPORTED_SUFFIXES, infer_schema, load_table
from surroforge.errors import DataError, ProjectError, RunNotFoundError
from surroforge.persistence import format_run_id, parse_run_number, read_yaml, write_yaml

PROJECT_FILE = "surroforge.yaml"
SCHEMA_VERSION = 1


class Project:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    # Construction -------------------------------------------------------------

    @classmethod
    def init(cls, root: str | Path, name: str, description: str | None = None) -> Project:
        root = Path(root).resolve()
        project_file = root / PROJECT_FILE
        if project_file.exists():
            raise ProjectError(f"A SurroForge project already exists at {root}")
        for sub in ("data", "configs", "runs", "reports"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": name,
            "schema_version": SCHEMA_VERSION,
            "created_at": _now_iso(),
            "description": description or "",
            "datasets": [],
        }
        write_yaml(project_file, metadata)
        return cls(root)

    @classmethod
    def open(cls, root: str | Path) -> Project:
        project = cls(root)
        if not project.project_file.is_file():
            raise ProjectError(f"No SurroForge project at {project.root} (missing {PROJECT_FILE})")
        return project

    @classmethod
    def discover(cls, start: str | Path | None = None) -> Project:
        """Walk up from ``start`` (default: cwd) to find the enclosing project."""
        current = Path(start or Path.cwd()).resolve()
        for candidate in (current, *current.parents):
            if (candidate / PROJECT_FILE).is_file():
                return cls(candidate)
        raise ProjectError(
            f"No SurroForge project found in {current} or its parents. "
            "Run 'surroforge init <name>' first."
        )

    # Paths --------------------------------------------------------------------

    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    # Metadata -----------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        return read_yaml(self.project_file)

    @property
    def name(self) -> str:
        return self.metadata.get("name", self.root.name)

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        write_yaml(self.project_file, metadata)

    # Datasets -----------------------------------------------------------------

    def add_dataset(self, file: str | Path, name: str | None = None) -> Dataset:
        """Register a dataset: validate, hash, copy into ``data/``, record schema."""
        source = Path(file)
        if not source.is_file():
            raise DataError(f"Data file not found: {source}")
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise DataError(
                f"Unsupported data format '{source.suffix}'. "
                f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
            )
        name = name or source.stem
        metadata = self.metadata
        existing = {d["name"] for d in metadata.get("datasets", [])}
        if name in existing:
            raise DataError(f"A dataset named '{name}' is already registered.")

        frame = load_table(source)
        if frame.empty:
            raise DataError(f"Dataset '{name}' is empty.")

        # Copy into the project's data/ dir unless the file is already inside it.
        self.data_dir.mkdir(parents=True, exist_ok=True)
        dest = self.data_dir / f"{name}{source.suffix.lower()}"
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)

        dataset = Dataset(
            name=name,
            path=str(dest.relative_to(self.root)),
            hash=hash_dataframe(frame),
            n_rows=int(frame.shape[0]),
            n_cols=int(frame.shape[1]),
            schema=infer_schema(frame),
        )
        metadata.setdefault("datasets", []).append(dataset.to_dict())
        self._save_metadata(metadata)
        return self._bind(dataset)

    def datasets(self) -> list[Dataset]:
        return [self._bind(Dataset.from_dict(d)) for d in self.metadata.get("datasets", [])]

    def dataset(self, name: str) -> Dataset:
        for dataset in self.datasets():
            if dataset.name == name:
                return dataset
        known = [d.name for d in self.datasets()]
        raise DataError(f"No registered dataset '{name}'. Known datasets: {known}")

    def _bind(self, dataset: Dataset) -> Dataset:
        # Attach a back-reference so the high-level API can persist a run from a
        # dataset alone (PRD 4.10: ``trainer.fit(project.dataset(...))``).
        dataset.project = self  # type: ignore[attr-defined]
        return dataset

    def load_dataset_frame(self, dataset: Dataset) -> pd.DataFrame:
        return load_table(self.root / dataset.path)

    def resolve_source(self, source: str) -> tuple[pd.DataFrame, str, str]:
        """Resolve ``data.source`` to a dataframe, content hash, and label.

        ``source`` is first matched against registered dataset names; otherwise
        it is treated as a file path (relative to the project root or absolute).
        """
        names = {d.name for d in self.datasets()}
        if source in names:
            dataset = self.dataset(source)
            frame = self.load_dataset_frame(dataset)
            return frame, dataset.hash, f"dataset:{dataset.name}"

        path = Path(source)
        if not path.is_absolute():
            path = self.root / path
        if not path.is_file():
            raise DataError(
                f"data.source '{source}' is neither a registered dataset nor an "
                f"existing file (looked at {path})."
            )
        frame = load_table(path)
        return frame, hash_dataframe(frame), f"file:{source}"

    # Runs ---------------------------------------------------------------------

    def runs(self) -> list[Run]:
        if not self.runs_dir.is_dir():
            return []
        run_dirs = [
            d
            for d in self.runs_dir.iterdir()
            if d.is_dir() and parse_run_number(d.name) is not None
        ]
        run_dirs.sort(key=lambda d: parse_run_number(d.name) or 0)
        return [Run(d) for d in run_dirs]

    def run(self, run_id: str) -> Run:
        run = Run(self.runs_dir / run_id)
        if not run.exists():
            raise RunNotFoundError(f"Run '{run_id}' not found in project {self.name}.")
        return run

    def allocate_run_dir(self) -> Path:
        """Reserve the next sequential run directory and create it."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        numbers: list[int] = []
        for child in self.runs_dir.iterdir():
            number = parse_run_number(child.name)
            if child.is_dir() and number is not None:
                numbers.append(number)
        next_number = (max(numbers) + 1) if numbers else 1
        run_dir = self.runs_dir / format_run_id(next_number)
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
