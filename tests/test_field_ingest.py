"""Field-as-snapshot-matrix ingestion via output column globs (Story R1.1).

A field is "columns that move together": ``data.outputs: [temp_*]`` expands
deterministically (in dataset column order) into the concrete n_dof field vector at
the ``select_columns`` choke point — mesh-free (only column names are read), with the
engine/config unchanged. Literal output lists stay byte-for-byte identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.data.loader import expand_field_columns, select_columns
from surroforge.engine import run_training
from surroforge.errors import DataError
from surroforge.services import assert_reproducible, reproduce
from tests.test_reduced_field import low_rank_field


@pytest.fixture(autouse=True)
def _grant_license(granted_license):
    """reproduce is Professional since Wave 3; this module tests the Core math behind it."""


def _frame(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def _field_frame(n: int = 8, n_dof: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols: dict[str, np.ndarray] = {"a": rng.normal(size=n), "b": rng.normal(size=n)}
    for j in range(n_dof):
        cols[f"u{j}"] = rng.normal(size=n)
    return pd.DataFrame(cols)


# --- expand_field_columns: ordering + back-compat (AC1, AC4) -------------------------

def test_glob_expands_in_dataset_column_order():
    frame = _field_frame()
    assert expand_field_columns(frame, ["u*"]) == [f"u{j}" for j in range(6)]


def test_literal_outputs_unchanged_and_order_preserved():
    frame = _field_frame()
    # No metacharacters -> identity, and the user's order (not frame order) is preserved.
    assert expand_field_columns(frame, ["u3", "u0"]) == ["u3", "u0"]


def test_mixed_literal_and_pattern_interleaves():
    frame = _frame(a=[1.0], b=[2.0], p0=[3.0], u0=[4.0], u1=[5.0], u2=[6.0])
    # Literal kept in written order; the pattern expands in frame column order after it.
    assert expand_field_columns(frame, ["p0", "u*"]) == ["p0", "u0", "u1", "u2"]


def test_bracketed_index_column_is_literal_not_glob():
    # Scientific fields commonly index DOF with brackets (disp[0], stress[xx]). An entry
    # that exactly names such a column must be a literal, NOT misread as a [...] glob class
    # (which would parse '[xx]' as a character set and spuriously "match no columns").
    frame = _frame(p=[1.0], **{"disp[0]": [2.0], "disp[1]": [3.0], "stress[xx]": [4.0]})
    assert expand_field_columns(frame, ["disp[0]", "stress[xx]"]) == ["disp[0]", "stress[xx]"]
    # A bare prefix glob still expands the bracketed group as a field.
    assert expand_field_columns(frame, ["disp*"]) == ["disp[0]", "disp[1]"]


# --- expand_field_columns: guards name offenders (AC3) -------------------------------

def test_zero_match_pattern_raises_named():
    frame = _field_frame()
    with pytest.raises(DataError, match=r"temp_\*.*matched no columns"):
        expand_field_columns(frame, ["temp_*"])


def test_overlapping_pattern_and_literal_raise_named():
    frame = _field_frame()
    # u3 is reached by both the literal and the pattern -> duplicate/overlapping DOF.
    with pytest.raises(DataError, match=r"u3"):
        expand_field_columns(frame, ["u3", "u*"])


def test_two_overlapping_patterns_raise_named():
    frame = _field_frame()
    with pytest.raises(DataError, match=r"overlapping/duplicate"):
        expand_field_columns(frame, ["u*", "u[0-2]"])


# --- select_columns integration: mesh-free + guards on the expanded set (AC2, AC3) ---

def test_coordinates_ignored_mesh_free():
    # x/y/z coordinate columns exist but are referenced by neither inputs nor outputs:
    # they are never read into x/y (mesh-free — the field is just the temp_* columns).
    frame = _frame(
        a=[1.0, 2.0], b=[3.0, 4.0], x=[0.0, 1.0], y=[0.0, 1.0], z=[0.0, 1.0],
        temp_0=[5.0, 6.0], temp_1=[7.0, 8.0], temp_2=[9.0, 10.0],
    )
    xs, ys = select_columns(frame, ["a", "b"], ["temp_*"])
    assert xs.shape == (2, 2)
    assert ys.shape == (2, 3)  # coordinate columns excluded; only the field group


def test_numeric_guard_applies_to_expanded_set():
    frame = _field_frame()
    frame["u2"] = ["x"] * len(frame)  # a non-numeric member of the field group
    with pytest.raises(DataError, match="Non-numeric"):
        select_columns(frame, ["a", "b"], ["u*"])


def test_input_output_overlap_via_pattern_raises():
    frame = _field_frame()
    # A pattern that also captures an input column -> input/output overlap.
    with pytest.raises(DataError, match="both input and output"):
        select_columns(frame, ["u0", "a"], ["u*"])


def test_missing_literal_still_caught_after_expansion():
    frame = _field_frame()
    with pytest.raises(DataError, match="not found"):
        select_columns(frame, ["a", "b"], ["nope"])


# --- end-to-end + reproduce determinism (AC1, AC5) -----------------------------------

def _glob_field_project(tmp_path):
    p, field = low_rank_field(n_samples=80, n_dof=6, k_true=3, seed=1)
    cols: dict[str, np.ndarray] = {"p0": p[:, 0], "p1": p[:, 1]}
    for j in range(field.shape[1]):
        cols[f"f{j}"] = field[:, j]
    csv = tmp_path / "field.csv"
    pd.DataFrame(cols).to_csv(csv, index=False)
    proj = Project.init(tmp_path / "proj", name="rom")
    proj.add_dataset(csv, name="snaps")
    return proj


def _glob_config() -> dict:
    return {
        "experiment": "rom-glob",
        "data": {
            "source": "snaps",
            "inputs": ["p0", "p1"],
            "outputs": ["f*"],  # the field expressed as a glob, not f0..f5 enumerated
            "split": {"test_fraction": 0.25, "seed": 1},
        },
        "model": {
            "method": "reduced-field",
            "reduction": {"basis": "pod", "energy": 0.999},
            "inner": {"method": "polynomial", "degree": 2},
        },
        "validation": {"cross_validation": 3, "metrics": ["rmse", "r2", "mae"]},
    }


def test_reduced_field_trains_from_glob_outputs(tmp_path):
    proj = _glob_field_project(tmp_path)
    run = run_training(proj, parse_config(_glob_config()))
    assert run.metrics["n_outputs"] == 6  # f* resolved to the 6-DOF field
    assert run.metrics["test"]["r2"] > 0.95
    assert run.load_model().predict(np.zeros((3, 2))).shape == (3, 6)


def test_glob_run_reproduces(tmp_path):
    # The config snapshot stores the pattern verbatim; reproduce re-expands it
    # deterministically against the same dataset, so the field reproduces.
    proj = _glob_field_project(tmp_path)
    run = run_training(proj, parse_config(_glob_config()))
    result = reproduce(proj, run.run_id)
    assert result.verdict in {"reproduced", "reproduced_within_tolerance"}
    assert_reproducible(result)
