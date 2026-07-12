"""OpenFOAM field ingestion — solver-native, mesh-free (Story R5.1).

Reads ASCII ``internalField`` lists (scalar/vector, nonuniform/uniform) into the wide
``(n_samples × n_dof)`` snapshot matrix the unchanged ``reduced-field`` pipeline trains on.
Tests carry teeth: vector fixtures use distinct per-component values so a wrong interleave is
caught, malformed inputs assert the *named* ``DataError``, and the sweep layout is pinned to a
deterministic, hash-stable order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surroforge.data import (
    hash_dataframe,
    latest_time_field,
    read_mesh_field,
    read_openfoam_dimensioned_scalar,
    read_openfoam_field,
    read_openfoam_sweep,
)
from surroforge.data.loader import expand_field_columns
from surroforge.errors import DataError

_HEADER = (
    "FoamFile\n{{\n    version 2.0;\n    format ascii;\n"
    "    class vol{cls}Field;\n    object {obj};\n}}\n"
    "dimensions [0 0 0 0 0 0 0];\n\n"
)


def _scalar_file(path, values, obj="p"):
    body = "\n".join(f"{v:.10g}" for v in values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _HEADER.format(cls="Scalar", obj=obj)
        + f"internalField nonuniform List<scalar>\n{len(values)}\n(\n{body}\n)\n;\n\n"
        + "boundaryField\n{\n}\n"
    )
    return path


def _vector_file(path, rows, obj="U"):
    body = "\n".join(f"({a:.10g} {b:.10g} {c:.10g})" for a, b, c in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _HEADER.format(cls="Vector", obj=obj)
        + f"internalField nonuniform List<vector>\n{len(rows)}\n(\n{body}\n)\n;\n\n"
        + "boundaryField\n{\n}\n"
    )
    return path


def _transport(path, nu):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _HEADER.format(cls="dictionary", obj="transportProperties").replace("Field", "")
        + "transportModel Newtonian;\n"
        + f"nu [0 2 -1 0 0 0 0] {nu:.10g};\n"
    )
    return path


# --- single-field reader: scalar / vector / uniform (AC1, AC2) -----------------------


def test_scalar_roundtrip_exact(tmp_path):
    values = [0.0, 1.5, -2.25, 100.0]
    out = read_openfoam_field(_scalar_file(tmp_path / "p", values))
    assert np.array_equal(out, np.array(values))


def test_vector_flattens_cell_major(tmp_path):
    # Distinct per-component values: a component-major bug ([1 4 2 5 3 6]) would differ from
    # the required cell-major order ([1 2 3 4 5 6]) — the fixture makes the wiring observable.
    out = read_openfoam_field(_vector_file(tmp_path / "U", [(1, 2, 3), (4, 5, 6)]))
    assert np.array_equal(out, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_uniform_scalar(tmp_path):
    path = tmp_path / "p"
    path.write_text(_HEADER.format(cls="Scalar", obj="p") + "internalField uniform 3.5;\n")
    assert np.array_equal(read_openfoam_field(path), np.array([3.5]))


def test_uniform_vector(tmp_path):
    path = tmp_path / "U"
    path.write_text(_HEADER.format(cls="Vector", obj="U") + "internalField uniform (1 2 3);\n")
    assert np.array_equal(read_openfoam_field(path), np.array([1.0, 2.0, 3.0]))


def test_comments_are_stripped(tmp_path):
    path = tmp_path / "p"
    path.write_text(
        "/* a banner\n   block comment */\n"
        + _HEADER.format(cls="Scalar", obj="p")
        + "internalField nonuniform List<scalar>\n2\n(\n1.0  // first cell\n2.0\n)\n;\n"
    )
    assert np.array_equal(read_openfoam_field(path), np.array([1.0, 2.0]))


# --- single-field reader: malformations raise a named DataError (AC1, AC7) -----------


def test_count_mismatch_raises_named(tmp_path):
    path = tmp_path / "p"
    # Declares 3 scalars but lists 2 — a silent truncation a length-blind reader would miss.
    path.write_text(
        _HEADER.format(cls="Scalar", obj="p")
        + "internalField nonuniform List<scalar>\n3\n(\n1.0\n2.0\n)\n;\n"
    )
    with pytest.raises(DataError, match=r"declared 3 scalar.*parsed 2"):
        read_openfoam_field(path)


def test_missing_internalfield_raises(tmp_path):
    path = tmp_path / "p"
    path.write_text(_HEADER.format(cls="Scalar", obj="p") + "boundaryField\n{\n}\n")
    with pytest.raises(DataError, match="no internalField"):
        read_openfoam_field(path)


def test_unbalanced_parens_raises(tmp_path):
    path = tmp_path / "p"
    path.write_text(
        _HEADER.format(cls="Scalar", obj="p")
        + "internalField nonuniform List<scalar>\n3\n(\n1.0\n2.0\n3.0\n;\n"  # no closing ')'
    )
    with pytest.raises(DataError, match="unbalanced parentheses"):
        read_openfoam_field(path)


def test_non_numeric_value_raises(tmp_path):
    path = tmp_path / "p"
    path.write_text(
        _HEADER.format(cls="Scalar", obj="p")
        + "internalField nonuniform List<scalar>\n2\n(\n1.0\nNaNny\n)\n;\n"
    )
    with pytest.raises(DataError, match="non-numeric"):
        read_openfoam_field(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(DataError, match="not found"):
        read_openfoam_field(tmp_path / "does-not-exist")


# --- time resolution + dimensioned-scalar recovery (AC4) ----------------------------


def test_latest_time_field_picks_highest(tmp_path):
    case = tmp_path / "case"
    _scalar_file(case / "0" / "p", [0.0])
    _scalar_file(case / "100" / "p", [1.0])
    _scalar_file(case / "200" / "p", [2.0])  # "200" > "100" numerically, not lexically
    assert latest_time_field(case, "p") == case / "200" / "p"


def test_latest_time_field_no_times_raises(tmp_path):
    (tmp_path / "case" / "constant").mkdir(parents=True)
    with pytest.raises(DataError, match="No numeric time directories"):
        latest_time_field(tmp_path / "case", "p")


def test_dimensioned_scalar_recovered(tmp_path):
    path = _transport(tmp_path / "constant" / "transportProperties", nu=1.5e-5)
    assert read_openfoam_dimensioned_scalar(path, "nu") == pytest.approx(1.5e-5)


def test_dimensioned_scalar_missing_keyword_raises(tmp_path):
    path = _transport(tmp_path / "constant" / "transportProperties", nu=1e-5)
    with pytest.raises(DataError, match="rho.*not found"):
        read_openfoam_dimensioned_scalar(path, "rho")


# --- sweep ingestion: shape, order, hash determinism (AC3, AC6) ----------------------


def _scalar_sweep(root, n_cases=3, n_cells=4):
    """Write `n_cases` cases of a distinct scalar field + a {case: params} mapping."""
    params = {}
    for i in range(n_cases):
        case = root / f"case_{i:02d}"
        # Cell c of case i holds 100*i + c, so every value is unique and order is observable.
        _scalar_file(case / "10" / "p", [100 * i + c for c in range(n_cells)])
        params[case.name] = {"u_in": 1.0 + i, "aoa": float(i)}
    return params


def test_sweep_shape_columns_and_order(tmp_path):
    params = _scalar_sweep(tmp_path, n_cases=3, n_cells=4)
    frame = read_openfoam_sweep(tmp_path, field="p", params=params)

    assert list(frame.columns) == ["u_in", "aoa", "p_0000", "p_0001", "p_0002", "p_0003"]
    assert frame.shape == (3, 6)
    # Rows sorted by case name; field values placed cell-major in column order.
    assert frame["u_in"].tolist() == [1.0, 2.0, 3.0]
    assert frame.loc[1, ["p_0000", "p_0001", "p_0002", "p_0003"]].tolist() == [100, 101, 102, 103]


def test_sweep_order_is_deterministic_and_hash_stable(tmp_path):
    params = _scalar_sweep(tmp_path, n_cases=4, n_cells=3)
    first = read_openfoam_sweep(tmp_path, field="p", params=params)
    second = read_openfoam_sweep(tmp_path, field="p", params=params)
    pd.testing.assert_frame_equal(first, second)
    assert hash_dataframe(first) == hash_dataframe(second)


def test_sweep_field_columns_are_glob_addressable(tmp_path):
    # The emitted names must match the existing data.outputs glob expansion (Story R1.1),
    # so a config can say outputs: ['p_*'] with no second expansion path.
    params = _scalar_sweep(tmp_path, n_cases=2, n_cells=3)
    frame = read_openfoam_sweep(tmp_path, field="p", params=params)
    assert expand_field_columns(frame, ["p_*"]) == ["p_0000", "p_0001", "p_0002"]


def test_sweep_ragged_dof_raises_named(tmp_path):
    _scalar_file(tmp_path / "case_00" / "10" / "p", [1.0, 2.0, 3.0])
    _scalar_file(tmp_path / "case_01" / "10" / "p", [1.0, 2.0])  # different DOF count
    with pytest.raises(DataError, match=r"case_01.*2 DOF.*case_00.*has 3"):
        read_openfoam_sweep(tmp_path, field="p")


def test_sweep_vector_columns_suffixed(tmp_path):
    _vector_file(tmp_path / "case_00" / "10" / "U", [(1, 2, 3), (4, 5, 6)])
    frame = read_openfoam_sweep(tmp_path, field="U")
    assert list(frame.columns) == ["U_0000_x", "U_0000_y", "U_0000_z", "U_0001_x", "U_0001_y",
                                    "U_0001_z"]
    assert frame.iloc[0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_sweep_params_callable_reads_from_case(tmp_path):
    _scalar_file(tmp_path / "case_00" / "10" / "p", [1.0, 2.0])
    _transport(tmp_path / "case_00" / "constant" / "transportProperties", nu=2e-5)

    def params_for(case_dir):
        nu = read_openfoam_dimensioned_scalar(case_dir / "constant" / "transportProperties", "nu")
        return {"nu": nu}

    frame = read_openfoam_sweep(tmp_path, field="p", params=params_for)
    assert frame["nu"].tolist() == [pytest.approx(2e-5)]
    assert list(frame.columns) == ["nu", "p_0000", "p_0001"]


def test_sweep_empty_root_raises(tmp_path):
    with pytest.raises(DataError, match="No case directories"):
        read_openfoam_sweep(tmp_path, field="p")


# --- review hardening: malformed values, types, params, time dirs --------------------


def test_non_finite_value_raises(tmp_path):
    # float() accepts "nan"/"inf" silently; a diverged dump must not poison the matrix.
    path = tmp_path / "p"
    path.write_text(
        _HEADER.format(cls="Scalar", obj="p")
        + "internalField nonuniform List<scalar>\n2\n(\n1.0\ninf\n)\n;\n"
    )
    with pytest.raises(DataError, match="non-finite"):
        read_openfoam_field(path)


def test_ragged_vector_entry_raises(tmp_path):
    # Second cell has 2 components, not 3 — a blind flat parse would mis-group silently.
    path = tmp_path / "U"
    path.write_text(
        _HEADER.format(cls="Vector", obj="U")
        + "internalField nonuniform List<vector>\n2\n(\n(1 2 3)\n(4 5)\n)\n;\n"
    )
    with pytest.raises(DataError, match=r"vector entry has 2 component"):
        read_openfoam_field(path)


def test_unsupported_tensor_field_raises(tmp_path):
    path = tmp_path / "R"
    path.write_text(
        _HEADER.format(cls="SymmTensor", obj="R")
        + "internalField nonuniform List<symmTensor>\n1\n(\n(1 2 3 4 5 6)\n)\n;\n"
    )
    with pytest.raises(DataError, match=r"unsupported field type 'symmTensor'"):
        read_openfoam_field(path)


def test_uniform_unsupported_size_raises(tmp_path):
    path = tmp_path / "p"
    path.write_text(_HEADER.format(cls="Scalar", obj="p") + "internalField uniform (1 2);\n")
    with pytest.raises(DataError, match=r"unsupported uniform field with 2 component"):
        read_openfoam_field(path)


def test_nan_time_dir_is_ignored(tmp_path):
    # A dir literally named "nan" must not win max() over real numeric times.
    case = tmp_path / "case"
    _scalar_file(case / "nan" / "p", [9.0])
    _scalar_file(case / "100" / "p", [1.0])
    _scalar_file(case / "200" / "p", [2.0])
    assert latest_time_field(case, "p") == case / "200" / "p"


def test_field_columns_width_scales_past_10k():
    from surroforge.data.openfoam import _field_columns

    cols = _field_columns("p", 10001, 1)
    assert cols[0] == "p_00000" and cols[10000] == "p_10000"  # 5-wide, lexically sortable
    # Small fields keep the 4-wide pad (back-compat with existing column names).
    assert _field_columns("p", 4, 1) == ["p_0000", "p_0001", "p_0002", "p_0003"]


def test_sweep_param_keys_mismatch_raises(tmp_path):
    # A key present only in a later case must error, not be silently dropped.
    _scalar_file(tmp_path / "case_00" / "10" / "p", [1.0, 2.0])
    _scalar_file(tmp_path / "case_01" / "10" / "p", [3.0, 4.0])
    params = {"case_00": {"a": 1.0}, "case_01": {"a": 2.0, "b": 9.0}}
    with pytest.raises(DataError, match=r"case_01.*parameter keys.*case_00"):
        read_openfoam_sweep(tmp_path, field="p", params=params)


def test_sweep_params_callable_error_is_wrapped(tmp_path):
    _scalar_file(tmp_path / "case_00" / "10" / "p", [1.0, 2.0])

    def boom(case_dir):
        raise RuntimeError("kaboom")

    with pytest.raises(DataError, match=r"case_00.*kaboom"):
        read_openfoam_sweep(tmp_path, field="p", params=boom)


def test_sweep_params_callable_non_mapping_raises(tmp_path):
    _scalar_file(tmp_path / "case_00" / "10" / "p", [1.0, 2.0])
    with pytest.raises(DataError, match="expected a mapping"):
        read_openfoam_sweep(tmp_path, field="p", params=lambda case_dir: ["a", "b"])


def test_sweep_field_only_in_earlier_time_gives_clear_error(tmp_path):
    # p is in 100/, but the latest dir 200/ holds only U → clear, non-misleading error.
    _scalar_file(tmp_path / "case_00" / "100" / "p", [1.0, 2.0])
    _vector_file(tmp_path / "case_00" / "200" / "U", [(1, 2, 3), (4, 5, 6)])
    with pytest.raises(DataError, match="earlier time directories"):
        read_openfoam_sweep(tmp_path, field="p")


# --- AC6: the produced matrix trains through the unchanged reduced-field pipeline -----


def test_sweep_trains_through_reduced_field_pipeline(tmp_path, granted_license):
    from surroforge.core.config import parse_config
    from surroforge.core.project import Project
    from surroforge.engine import run_training
    from surroforge.services import assert_reproducible, reproduce
    from tests.test_reduced_field import low_rank_field

    p, field = low_rank_field(n_samples=60, n_dof=8, k_true=3, seed=1)
    params = {}
    for i in range(p.shape[0]):
        case = tmp_path / f"case_{i:03d}"
        _scalar_file(case / "10" / "p", field[i])
        params[case.name] = {"a": float(p[i, 0]), "b": float(p[i, 1])}

    frame = read_openfoam_sweep(tmp_path, field="p", params=params)
    csv = tmp_path / "snap.csv"
    frame.to_csv(csv, index=False)

    proj = Project.init(tmp_path / "proj", name="of")
    proj.add_dataset(csv, name="cfd")
    config = {
        "experiment": "of",
        "data": {
            "source": "cfd",
            "inputs": ["a", "b"],
            "outputs": ["p_*"],  # the reader's column group resolves via the existing glob
            "split": {"test_fraction": 0.25, "seed": 1},
        },
        "model": {
            "method": "reduced-field",
            "reduction": {"basis": "pod", "energy": 0.999},
            "inner": {"method": "polynomial", "degree": 2},
        },
        "validation": {"cross_validation": 3, "metrics": ["rmse", "r2"]},
    }
    run = run_training(proj, parse_config(config))
    assert run.metrics["n_outputs"] == 8  # p_* resolved to the 8-DOF field
    assert run.metrics["test"]["r2"] > 0.95
    assert_reproducible(reproduce(proj, run.run_id))


# --- meshio-backed binary/mesh reader behind [rom] (Story R5.2) ----------------------


def test_read_mesh_field_missing_meshio_gives_install_hint(tmp_path, monkeypatch):
    import sys

    # A None entry in sys.modules makes `import meshio` raise ImportError, so the test
    # is deterministic whether or not meshio is actually installed.
    monkeypatch.setitem(sys.modules, "meshio", None)
    with pytest.raises(DataError, match=r"surroforge\[rom\]"):
        read_mesh_field(tmp_path / "whatever.vtu", "T")


def test_read_mesh_field_point_data_roundtrip(tmp_path):
    meshio = pytest.importorskip("meshio")
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]
    mesh = meshio.Mesh(points, cells, point_data={"T": np.array([1.0, 2.0, 3.0])})
    path = tmp_path / "m.vtu"
    mesh.write(path)
    assert np.allclose(read_mesh_field(path, "T", location="point"), [1.0, 2.0, 3.0])


def test_read_mesh_field_unknown_field_raises(tmp_path):
    meshio = pytest.importorskip("meshio")
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]
    mesh = meshio.Mesh(points, cells, point_data={"T": [1.0, 2.0, 3.0]})
    path = tmp_path / "m.vtu"
    mesh.write(path)
    with pytest.raises(DataError, match=r"field 'p' not found"):
        read_mesh_field(path, "p", location="point")


def test_read_mesh_field_rejects_tensor(tmp_path):
    meshio = pytest.importorskip("meshio")
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]
    tensor = np.arange(3 * 9, dtype=float).reshape(3, 9)  # 9 components/point — not scalar/vector
    mesh = meshio.Mesh(points, cells, point_data={"R": tensor})
    path = tmp_path / "m.vtu"
    mesh.write(path)
    with pytest.raises(DataError, match="only scalar"):
        read_mesh_field(path, "R", location="point")
