"""High-level Python API (PRD 4.10)."""

from __future__ import annotations

import pytest
import yaml

from surroforge import Project, Trainer
from surroforge.core.dataset import Dataset
from surroforge.errors import DataError
from tests.conftest import poly_config


def test_fit_from_dataset(project):
    trainer = Trainer.from_dict(poly_config("polynomial"))
    run = trainer.fit(project.dataset("beam"))
    assert run.metrics["test"]["r2"] > 0.95


def test_fit_from_project(project):
    trainer = Trainer.from_dict(poly_config("polynomial"))
    run = trainer.fit(project)
    assert run.run_id == "run-001"


def test_from_config_file(project):
    config_path = project.configs_dir / "poly.yaml"
    config_path.write_text(yaml.safe_dump(poly_config("polynomial")))
    trainer = Trainer.from_config(config_path)
    run = trainer.fit(project)
    assert run.method == "polynomial"


def test_unbound_dataset_raises(project):
    trainer = Trainer.from_dict(poly_config("polynomial"))
    loose = Dataset(name="x", path="data/x.csv", hash="0", n_rows=1, n_cols=1)
    with pytest.raises(DataError):
        trainer.fit(loose)


def test_project_open_roundtrip(project):
    reopened = Project.open(project.root)
    assert reopened.name == "test-proj"
    assert [d.name for d in reopened.datasets()] == ["beam"]
