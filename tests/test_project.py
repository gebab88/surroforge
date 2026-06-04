"""Project lifecycle and dataset registration edge cases (PRD 7.1, 7.3)."""

from __future__ import annotations

import pandas as pd
import pytest

from surroforge.core.project import Project
from surroforge.errors import DataError, ProjectError, RunNotFoundError
from tests.conftest import make_samples


def test_init_creates_layout(tmp_path):
    project = Project.init(tmp_path / "p", name="p")
    for sub in ("data", "configs", "runs", "reports"):
        assert (project.root / sub).is_dir()
    assert project.project_file.is_file()
    assert project.name == "p"


def test_init_twice_raises(tmp_path):
    Project.init(tmp_path / "p", name="p")
    with pytest.raises(ProjectError):
        Project.init(tmp_path / "p", name="p")


def test_open_missing_raises(tmp_path):
    with pytest.raises(ProjectError):
        Project.open(tmp_path / "nope")


def test_discover_walks_up_from_subdirectory(tmp_path):
    project = Project.init(tmp_path / "p", name="p")
    nested = project.root / "configs" / "deep"
    nested.mkdir(parents=True)
    found = Project.discover(nested)
    assert found.root == project.root


def test_discover_without_project_raises(tmp_path):
    with pytest.raises(ProjectError):
        Project.discover(tmp_path)


def test_add_dataset_records_hash_and_schema(project):
    dataset = project.dataset("beam")
    assert dataset.n_rows == 80
    assert dataset.n_cols == 4
    assert set(dataset.schema) == {"length", "width", "load", "max_stress"}
    assert len(dataset.hash) == 64  # sha256 hex


def test_add_duplicate_dataset_name_raises(project, sample_csv):
    with pytest.raises(DataError):
        project.add_dataset(sample_csv, name="beam")


def test_add_unsupported_format_raises(project, tmp_path):
    bad = tmp_path / "data.txt"
    bad.write_text("not a table")
    with pytest.raises(DataError):
        project.add_dataset(bad)


def test_add_empty_dataset_raises(project, tmp_path):
    empty = tmp_path / "empty.csv"
    pd.DataFrame({"a": [], "b": []}).to_csv(empty, index=False)
    with pytest.raises(DataError):
        project.add_dataset(empty, name="empty")


def test_dataset_is_bound_to_project(project):
    dataset = project.dataset("beam")
    assert getattr(dataset, "project", None) is project


def test_unknown_dataset_raises(project):
    with pytest.raises(DataError):
        project.dataset("ghost")


def test_resolve_source_by_name_and_by_path(project):
    frame, h, label = project.resolve_source("beam")
    assert label == "dataset:beam"
    assert frame.shape[0] == 80

    rel = project.dataset("beam").path
    _, _, label2 = project.resolve_source(rel)
    assert label2.startswith("file:")


def test_resolve_source_unknown_raises(project):
    with pytest.raises(DataError):
        project.resolve_source("does/not/exist.csv")


def test_runs_empty_and_unknown_run(project):
    assert project.runs() == []
    with pytest.raises(RunNotFoundError):
        project.run("run-001")


def test_dataset_copied_into_project_data_dir(tmp_path):
    external = tmp_path / "external.csv"
    make_samples().to_csv(external, index=False)
    project = Project.init(tmp_path / "p", name="p")
    dataset = project.add_dataset(external, name="ext")
    assert (project.root / dataset.path).is_file()
    assert dataset.path.startswith("data/")
