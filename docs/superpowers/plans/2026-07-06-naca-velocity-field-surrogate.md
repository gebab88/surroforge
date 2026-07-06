# NACA Velocity-Field Surrogate (U) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a third surrogate `(AoA, |U∞|) → (Ux, Uy, Uz)` on the real NACA 0012 sweep and write pressure **and** velocity into the one ParaView VTKHDF time series.

**Architecture:** The Core reader `read_openfoam_sweep` already parses `List<vector>` fields into cell-major `U_NNNN_x/_y/_z` columns, so the example gains a second sweep read merged into the same dataset (one `data_hash`, `same_data` lineage), a `velocity-*` training block identical in shape to the pressure block, and a generalized `write_surrogate_vtkhdf` that takes a `{name: matrix}` mapping and writes scalars flat / vectors as `(n, 3)` datasets.

**Tech Stack:** Python, pandas/numpy, h5py (VTKHDF), SurroForge `reduced-field` (POD + inner), pytest.

**Spec:** `docs/superpowers/specs/2026-07-06-naca-velocity-field-surrogate-design.md`

**Amendment (2026-07-06):** the first verified run exposed that the machine-zero `U_*_z` columns make Core's uniform-average R² meaningless (test R² −861 despite per-column x/y R² 0.9994). User decision (see the spec's Amendment section): train on Ux/Uy only and re-pad z=0 at export. Task 3 implements the switch; Task 4's README numbers come from the post-switch run.

## Global Constraints

- Commits go directly to `main`; conventional-commit style (`feat:`/`test:`/`docs:`); **no `Co-Authored-By:` trailer, ever**. Do not push.
- Run everything from the repo root with the project venv: `./.venv/bin/python`, `./.venv/bin/pytest`, `./.venv/bin/ruff`.
- Code, comments, and docs in English (repo convention).
- **No changes to `run_sweep.py` or anything under `src/`** (spec non-goals — the Core reader already supports vectors).
- The verification data already exists on disk at `examples/naca0012_openfoam/run/` (110 converged cases incl. `U` fields). **OpenFOAM is NOT required** — the scripts only read files.
- `examples/naca0012_openfoam/run/` and `examples/naca0012_openfoam/ex09_real/` are **gitignored** — never `git add` them.
- `pytest -q` at the repo root becomes `-qq` (root `addopts` already has `-q`) and suppresses the final counts line; run plain `pytest` when you need the counts.

---

### Task 1: Generalize `write_surrogate_vtkhdf` to named fields (scalar + vector)

**Files:**
- Modify: `examples/naca0012_openfoam/surrogate_vtk.py` (imports + `write_surrogate_vtkhdf`, currently lines 147–241)
- Test: `tests/test_naca_vtk_export.py`

**Interfaces:**
- Consumes: existing private helpers in the same file (`_read_vtu_mesh`, `_internal_vtu_template`, `_step_offsets`, `VtuMesh`) — unchanged.
- Produces: `write_surrogate_vtkhdf(run_dir: Path, output: Path, samples: pd.DataFrame, fields: Mapping[str, np.ndarray]) -> Path`. Each mapping value is an `(n_cases, n_cells * n_components)` float matrix, vector components cell-major (`[x0 y0 z0 x1 …]` — exactly the column order `read_openfoam_sweep` produces). Scalars are written flat under `CellData/<name>`; vectors as `(n_steps·n_cells, 3)` datasets. The old `field_name="p"` kwarg is **gone** (clean break; the only consumers are `ingest_and_train.py` and this test). Task 2 calls it with `{"p": ..., "U": ...}`.

- [ ] **Step 1: Update the existing test and add the vector + error tests**

In `tests/test_naca_vtk_export.py`, replace the whole function `test_surrogate_vtkhdf_writes_single_time_series_file` (currently lines 69–113) with:

```python
def test_surrogate_vtkhdf_writes_single_time_series_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    module = _load_surrogate_vtk_module()
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "surrogate_vtk"
    samples = pd.DataFrame(
        {
            "case": ["case_000", "case_001"],
            "aoa": [-5.0, -4.0],
            "speed": [20.0, 20.0],
        }
    )

    for case in samples["case"]:
        template = run_dir / case / "VTK" / f"{case}_10" / "internal.vtu"
        template.parent.mkdir(parents=True)
        _write_minimal_vtu(template)

    output = out_dir / "surrogate_fields.vtkhdf"
    written = module.write_surrogate_vtkhdf(
        run_dir,
        output,
        samples,
        {
            "p": np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]),
            # 3 cells x 3 components per case row, cell-major [x0 y0 z0 x1 ...]
            "U": np.arange(18, dtype=float).reshape(2, 9),
        },
    )

    assert written == output
    assert pd.read_csv(out_dir / "samples.csv").to_dict("list") == samples.to_dict("list")
    with h5py.File(output, "r") as h5:
        vtk = h5["VTKHDF"]
        assert vtk.attrs["Type"] == b"UnstructuredGrid"
        assert vtk.attrs["Version"].tolist() == [2, 5]
        assert vtk["NumberOfPoints"][...].tolist() == [4, 4]
        assert vtk["NumberOfCells"][...].tolist() == [3, 3]
        assert vtk["NumberOfConnectivityIds"][...].tolist() == [9, 9]
        assert vtk["Offsets"][...].tolist() == [0, 3, 6, 9, 0, 3, 6, 9]
        assert vtk["CellData/p"][...].tolist() == [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        assert vtk["CellData/U"].shape == (6, 3)
        assert vtk["CellData/U"][...].tolist() == [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0],
        ]
        assert vtk["FieldData/TimeValue"][...].tolist() == [0.0, 1.0]
        assert vtk["Steps"].attrs["NSteps"] == 2
        assert vtk["Steps/Values"][...].tolist() == [0.0, 1.0]
        assert vtk["Steps/PointOffsets"][...].tolist() == [0, 4]
        assert vtk["Steps/ConnectivityIdOffsets"][...].tolist() == [0, 9]
        assert vtk["Steps/CellDataOffsets/p"][...].tolist() == [0, 3]
        assert vtk["Steps/CellDataOffsets/U"][...].tolist() == [0, 3]


def test_surrogate_vtkhdf_rejects_dof_count_not_matching_cells(tmp_path):
    pytest.importorskip("h5py")
    module = _load_surrogate_vtk_module()
    run_dir = tmp_path / "run"
    samples = pd.DataFrame({"case": ["case_000"], "aoa": [0.0], "speed": [20.0]})
    template = run_dir / "case_000" / "VTK" / "case_000_10" / "internal.vtu"
    template.parent.mkdir(parents=True)
    _write_minimal_vtu(template)

    # 4 DOF on a 3-cell mesh: neither n_cells (scalar) nor 3*n_cells (vector).
    with pytest.raises(ValueError, match="field 'U' has 4 values"):
        module.write_surrogate_vtkhdf(
            run_dir,
            tmp_path / "out.vtkhdf",
            samples,
            {"U": np.zeros((1, 4))},
        )
```

Leave `_load_surrogate_vtk_module`, `_vtk_payload`, and `_write_minimal_vtu` unchanged.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/pytest tests/test_naca_vtk_export.py -v`
Expected: both tests FAIL — the old signature can't take a `fields` mapping (`np.asarray` on a dict raises `TypeError`/`ValueError`, or the old `(n_cases, n_cells)` shape check fires).

- [ ] **Step 3: Implement the generalized writer**

In `examples/naca0012_openfoam/surrogate_vtk.py`:

(a) Add `Mapping` to the imports — the import block becomes:

```python
from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
```

(b) Replace the whole `write_surrogate_vtkhdf` function (signature through `return output`) with:

```python
def write_surrogate_vtkhdf(
    run_dir: Path,
    output: Path,
    samples: pd.DataFrame,
    fields: Mapping[str, np.ndarray],
) -> Path:
    """Write surrogate cell fields of all cases to one ParaView-readable VTKHDF file.

    ``fields`` maps each field name to an ``(n_cases, n_cells * n_components)`` matrix,
    vector components laid out cell-major (``[x0 y0 z0 x1 …]``) — exactly the column
    order ``read_openfoam_sweep`` produces. Scalar fields (1 component) are written
    flat under ``CellData/<name>``; vector fields (3 components) as an
    ``(n_steps·n_cells, 3)`` dataset, which ParaView reads as a vector array
    (magnitude colouring, glyphs, streamlines).
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "VTKHDF export requires h5py; install the optional dependency with "
            "`pip install 'surroforge[formats]'`."
        ) from exc

    if "case" not in samples.columns:
        raise ValueError("samples must include a 'case' column")
    if not fields:
        raise ValueError("fields must contain at least one named field matrix")
    matrices: dict[str, np.ndarray] = {}
    for name, values in fields.items():
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(samples):
            raise ValueError(
                f"field '{name}' must be an (n_cases, n_dof) matrix; got {matrix.shape} "
                f"for {len(samples)} sample row(s)"
            )
        matrices[name] = matrix

    components: dict[str, int] = {}
    meshes: list[VtuMesh] = []
    for row in samples.itertuples(index=False):
        case = str(row.case)
        mesh = _read_vtu_mesh(_internal_vtu_template(run_dir / case))
        n_cells = int(mesh.types.size)
        for name, matrix in matrices.items():
            n_dof = matrix.shape[1]
            if n_dof % n_cells or n_dof // n_cells not in (1, 3):
                raise ValueError(
                    f"{case}: field '{name}' has {n_dof} values, but the VTU template "
                    f"has {n_cells} cells; expected n_cells (scalar) or 3*n_cells "
                    "(vector) values"
                )
            inferred = n_dof // n_cells
            if components.setdefault(name, inferred) != inferred:
                raise ValueError(
                    f"{case}: field '{name}' implies {inferred} component(s) on this "
                    f"mesh, but an earlier case implied {components[name]}"
                )
        meshes.append(mesh)

    output.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output.parent / "samples.csv", index=False)

    n_steps = len(meshes)
    point_counts = np.asarray([mesh.points.shape[0] for mesh in meshes], dtype=np.int64)
    cell_counts = np.asarray([mesh.types.size for mesh in meshes], dtype=np.int64)
    connectivity_counts = np.asarray([mesh.connectivity.size for mesh in meshes], dtype=np.int64)
    times = np.arange(n_steps, dtype=np.float32)

    with h5py.File(output, "w") as h5:
        vtk = h5.create_group("VTKHDF")
        vtk.attrs["Type"] = np.bytes_("UnstructuredGrid")
        vtk.attrs["Version"] = np.asarray([2, 5], dtype=np.int64)

        vtk.create_dataset("NumberOfPoints", data=point_counts)
        vtk.create_dataset("NumberOfCells", data=cell_counts)
        vtk.create_dataset("NumberOfConnectivityIds", data=connectivity_counts)
        vtk.create_dataset(
            "Points",
            data=np.concatenate([mesh.points for mesh in meshes], axis=0).astype(np.float32),
        )
        vtk.create_dataset(
            "Connectivity",
            data=np.concatenate([mesh.connectivity for mesh in meshes]).astype(np.int64),
        )
        vtk.create_dataset(
            "Offsets",
            data=np.concatenate([mesh.offsets for mesh in meshes]).astype(np.int64),
        )
        vtk.create_dataset(
            "Types",
            data=np.concatenate([mesh.types for mesh in meshes]).astype(np.uint8),
        )

        cell_data = vtk.create_group("CellData")
        for name, matrix in matrices.items():
            if components[name] == 1:
                cell_data.create_dataset(name, data=matrix.reshape(-1))
            else:
                cell_data.create_dataset(name, data=matrix.reshape(-1, 3))
        vtk.create_group("PointData")
        field_data = vtk.create_group("FieldData")
        field_data.create_dataset("TimeValue", data=times)

        steps = vtk.create_group("Steps")
        steps.attrs["NSteps"] = np.int64(n_steps)
        steps.create_dataset("Values", data=times)
        steps.create_dataset("PartOffsets", data=np.zeros(n_steps, dtype=np.int64))
        steps.create_dataset("PointOffsets", data=_step_offsets(point_counts))
        steps.create_dataset("CellOffsets", data=np.zeros(n_steps, dtype=np.int64))
        steps.create_dataset("ConnectivityIdOffsets", data=_step_offsets(connectivity_counts))
        steps.create_group("PointDataOffsets")
        cell_data_offsets = steps.create_group("CellDataOffsets")
        for name in matrices:
            # Offsets count rows (cells) for scalar and vector datasets alike.
            cell_data_offsets.create_dataset(name, data=_step_offsets(cell_counts))
        field_data_offsets = steps.create_group("FieldDataOffsets")
        field_data_offsets.create_dataset("TimeValue", data=np.arange(n_steps, dtype=np.int64))
        field_data_sizes = steps.create_group("FieldDataSizes")
        field_data_sizes.create_dataset(
            "TimeValue", data=np.ones((n_steps, 2), dtype=np.int64)
        )

    return output
```

Everything before/after the function (module docstring, helpers) stays as is.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/pytest tests/test_naca_vtk_export.py -v`
Expected: 2 passed.

- [ ] **Step 5: Lint the touched files**

Run: `./.venv/bin/ruff check tests/test_naca_vtk_export.py examples/naca0012_openfoam/surrogate_vtk.py`
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add tests/test_naca_vtk_export.py examples/naca0012_openfoam/surrogate_vtk.py
git commit -m "feat(naca): write named scalar+vector surrogate fields to VTKHDF"
```

---

### Task 2: Velocity ingest + training block in `ingest_and_train.py`, verified on the real sweep

> **Amended:** implemented as specified (commit `30c4490`, review clean), then partly superseded by Task 3 — the ingest now drops `U_*_z` and the export re-pads z=0. Do not re-execute this task.

**Files:**
- Modify: `examples/naca0012_openfoam/ingest_and_train.py`

**Interfaces:**
- Consumes: `write_surrogate_vtkhdf(run_dir, output, samples, fields: Mapping[str, np.ndarray])` from Task 1 (call it with `{"p": ..., "U": ...}`); `read_openfoam_sweep(run_dir, field="U", params=...)` from Core (already supports `List<vector>`, producing cell-major columns `U_0000_x, U_0000_y, U_0000_z, U_0001_x, …`).
- Produces: console fit tables for three surrogates (the velocity table is transcribed into the README in Task 3); artifact `velocity_surrogate.pkl`; `surrogate_fields.vtkhdf` now containing cell arrays `p` (scalar) and `U` (vector).

- [ ] **Step 1: Update the module docstring**

Replace the docstring's surrogate list (currently lines 6–10, the sentence ending "…(``examples/09_naca0012_openfoam_surrogate.py``):" plus the two bullets) so it reads:

```
Run this after ``run_sweep.py`` has produced a sweep directory. It reads each case's
converged pressure and velocity fields with ``surroforge.data.read_openfoam_sweep`` and the
**real** lift and drag coefficients ``Cl``/``Cd`` that ``forceCoeffs`` integrated, then trains
the two surrogates of the synthetic CI example (``examples/09_naca0012_openfoam_surrogate.py``)
plus a velocity-field surrogate:

  * a **field** surrogate (``reduced-field`` POD + neural inner): (AoA, speed) -> pressure field;
  * a **velocity** surrogate (same construction): (AoA, speed) -> velocity field (Ux, Uy, Uz);
  * a **scalar** surrogate: (AoA, speed) -> [Cl, Cd] (lift and drag coefficients).
```

And in the later paragraph (currently lines 22–27), replace the sentence
"The best field surrogate is also evaluated on **every** case from ``samples.csv`` and written as one VTKHDF time-series file"
with
"The best pressure- and velocity-field surrogates are also evaluated on **every** case from ``samples.csv`` and written into one VTKHDF time-series file (cell arrays ``p`` and ``U``)".

- [ ] **Step 2: Read and merge the velocity sweep**

In `main()`, directly after the existing `snapshots = read_openfoam_sweep(...)` call (ends line 138), insert:

```python
    velocity = read_openfoam_sweep(
        run_dir, field="U",
        params=lambda case: {
            "aoa": float(doe.loc[case.name, "aoa"]),
            "speed": float(doe.loc[case.name, "speed"]),
        },
    )
    # Both frames are sorted by case name, but a case that has `p` while missing `U`
    # (or vice versa) would silently shift the rows — refuse instead of misaligning.
    if not snapshots[["aoa", "speed"]].equals(velocity[["aoa", "speed"]]):
        raise RuntimeError(
            "pressure and velocity sweeps disagree on their (AoA, speed) rows — "
            "some case has one field but not the other; re-run the sweep for that case"
        )
    velocity_cols = [c for c in velocity.columns if c.startswith("U_")]
    snapshots = pd.concat([snapshots, velocity[velocity_cols]], axis=1)
```

Then update the ingest print (currently line 157) to:

```python
    print(f"Ingested {len(snapshots)} real cases: {len(field_cols)} pressure DOFs "
          f"+ {len(velocity_cols)} velocity DOFs + Cl + Cd")
```

(`field_cols` is computed one line above and stays unchanged.)

- [ ] **Step 3: Add the velocity training block**

Directly after `_report(project, field_runs, "inner")` (currently line 194), insert:

```python
        print("\n=== Field surrogate: (AoA, speed) -> velocity field (POD + inner) ===")
        velocity_runs = {
            label: Trainer.from_dict({
                "experiment": f"velocity-{label}",
                "data": {"source": "sweep", "inputs": inputs, "outputs": ["U_*"], "split": split},
                "model": {"method": "reduced-field",
                          "reduction": {"basis": "pod", "energy": 0.999}, "inner": inner},
                "validation": validation,
            }).fit(project)
            for label, inner in _candidates([48, 48])
        }
        _report(project, velocity_runs, "inner")
```

- [ ] **Step 4: Export the velocity surrogate and write both fields to VTKHDF**

In the export section, directly after the `field_out` print (currently line 226), insert:

```python
        velocity_best = velocity_runs[_best_label(project, velocity_runs)]
        velocity_out = velocity_best.export("pickle", export_dir / "velocity_surrogate.pkl")
        print(f"  velocity surrogate -> {velocity_out.name} "
              "(pickle; reduced-field is pickle-only)")
```

Then replace the whole `=== Surrogate field VTK ===` block (currently lines 228–243) with:

```python
        print("\n=== Surrogate field VTK ===")
        prediction_cases = doe.loc[case_names, ["aoa", "speed"]].reset_index()
        prediction_inputs = prediction_cases[inputs].to_numpy(dtype=float)
        vtk_dir = export_dir / "surrogate_vtk"
        vtkhdf = write_surrogate_vtkhdf(
            run_dir,
            vtk_dir / "surrogate_fields.vtkhdf",
            prediction_cases,
            {
                "p": field_best.load_model().predict(prediction_inputs),
                "U": velocity_best.load_model().predict(prediction_inputs),
            },
        )
        print(
            f"  surrogate p+U fields -> {vtkhdf} "
            f"({len(prediction_cases)} time steps + samples.csv)"
        )
```

Note this removes the old `predicted_fields = field_best.load_model().predict(...)` line — the predictions now happen inline in the mapping.

- [ ] **Step 5: Static checks**

Run: `./.venv/bin/ruff check examples/naca0012_openfoam/ingest_and_train.py && ./.venv/bin/python -m py_compile examples/naca0012_openfoam/ingest_and_train.py`
Expected: no findings, exit 0.

- [ ] **Step 6: Real verification run (no OpenFOAM needed — reads existing files)**

```bash
./.venv/bin/python examples/naca0012_openfoam/ingest_and_train.py \
  examples/naca0012_openfoam/run --out examples/naca0012_openfoam/ex09_real \
  2>&1 | tee /private/tmp/claude-501/-Users-georg-Software-surroforge/35056efb-276a-40ab-95eb-ab773c0f6980/scratchpad/ingest_u_run.log
```

This may take several minutes (POD + 3×3 CV-trained models on 110 cases). Expected output:
- `Ingested 110 real cases: 10720 pressure DOFs + 32160 velocity DOFs + Cl + Cd`
- **three** `===` comparison tables (pressure field, velocity field, scalar), each with an `<- best` marker,
- an Export section with **four** artifact lines (`onnx`, `pt`, `field surrogate`, `velocity surrogate`),
- `surrogate p+U fields -> …/surrogate_fields.vtkhdf (110 time steps + samples.csv)`.

- [ ] **Step 7: Spot-check the VTKHDF file with h5py**

```bash
./.venv/bin/python - <<'EOF'
import h5py, numpy as np
path = "examples/naca0012_openfoam/ex09_real/surrogate_vtk/surrogate_fields.vtkhdf"
with h5py.File(path) as h5:
    vtk = h5["VTKHDF"]
    p = vtk["CellData/p"][...]
    U = vtk["CellData/U"][...]
    assert p.shape == (110 * 10720,), p.shape
    assert U.shape == (110 * 10720, 3), U.shape
    assert np.isfinite(p).all() and np.isfinite(U).all()
    assert vtk["Steps"].attrs["NSteps"] == 110
    assert vtk["Steps/CellDataOffsets/U"][...].tolist() == list(range(0, 110 * 10720, 10720))
    # Uz must be tiny everywhere (2-D case) while |U| is O(freestream) — a layout
    # mix-up (component-major instead of cell-major) would violate this instantly.
    assert np.abs(U[:, 2]).max() < 1.0 < np.abs(U[:, :2]).max()
print("VTKHDF OK")
EOF
```

Expected: `VTKHDF OK`.

- [ ] **Step 8: Record the three fit tables**

Copy the three comparison tables (method/inner, test R², test RMSE, best markers) from the run log verbatim into your task report — Task 3 transcribes the velocity numbers into the README and needs them exactly.

- [ ] **Step 9: Commit**

```bash
git add examples/naca0012_openfoam/ingest_and_train.py
git commit -m "feat(naca): train and export a velocity-field surrogate (U)"
```

(Only that one file — `run/` and `ex09_real/` are gitignored.)

---

### Task 3: Switch the velocity surrogate to Ux/Uy (re-pad z at export)

Rationale (spec Amendment): the 2-D case's 10 720 `U_*_z` columns are machine zero (column variance ≤ 2e-27). Core's R² uniform-averages all output columns, so those zero-variance columns drive the velocity test R² to ≈ −861 even though the x/y fit is excellent (per-column R² 0.9994, RMSE 0.51 m/s). Dropping z at ingest makes R²/RMSE meaningful; the export re-pads z as exact zero so ParaView still gets a 3-component vector field.

**Files:**
- Modify: `examples/naca0012_openfoam/ingest_and_train.py`

**Interfaces:**
- Consumes: the Task-1 writer `write_surrogate_vtkhdf(run_dir, output, samples, fields: Mapping[str, np.ndarray])` — unchanged; it still receives a full 3-component `U` matrix (padded).
- Produces: dataset `sweep` now holds 21 440 U DOFs (`U_NNNN_x, U_NNNN_y, …`, cell-major interleaved in frame order); `velocity_surrogate.pkl` predicts `(n, 21440)`; the VTKHDF `CellData/U` keeps shape `(110·10720, 3)` with `U[:, 2] == 0` exactly. Task 4 transcribes the post-switch fit tables.

- [ ] **Step 1: Add the numpy import**

`ingest_and_train.py` currently imports pandas but not numpy. In the import block, directly above `import pandas as pd`, add:

```python
import numpy as np
```

- [ ] **Step 2: Update the module docstring**

Replace the velocity bullet

```
  * a **velocity** surrogate (same construction): (AoA, speed) -> velocity field (Ux, Uy, Uz);
```

with:

```
  * a **velocity** surrogate (same construction): (AoA, speed) -> velocity field (Ux, Uy —
    the 2-D case's z component is machine zero, dropped at ingest, re-padded on export);
```

- [ ] **Step 3: Drop the z columns at ingest**

In `main()`, replace

```python
    velocity_cols = [c for c in velocity.columns if c.startswith("U_")]
```

with:

```python
    # The airFoil2D case is 2-D: every U_*_z value is machine zero (~1e-16), and Core's
    # uniform-average R² would be dominated by those zero-variance columns (per-column
    # R² of a constant output is arbitrarily negative). Drop z at ingest; the VTKHDF
    # export re-pads it as exact zero.
    velocity_cols = [
        c for c in velocity.columns if c.startswith("U_") and not c.endswith("_z")
    ]
```

The training block (`outputs: ["U_*"]`) needs no change — the wildcard now only matches x/y columns.

- [ ] **Step 4: Re-pad z in the VTKHDF export**

Replace the whole `=== Surrogate field VTK ===` block with:

```python
        print("\n=== Surrogate field VTK ===")
        prediction_cases = doe.loc[case_names, ["aoa", "speed"]].reset_index()
        prediction_inputs = prediction_cases[inputs].to_numpy(dtype=float)
        predicted_velocity_xy = velocity_best.load_model().predict(prediction_inputs)
        # Re-pad the dropped z component (exact zero in this 2-D case) so ParaView still
        # gets a 3-component vector field: cell-major [x0 y0 z0 x1 …], as the writer expects.
        predicted_velocity = np.zeros(
            (predicted_velocity_xy.shape[0], predicted_velocity_xy.shape[1] // 2 * 3)
        )
        predicted_velocity[:, 0::3] = predicted_velocity_xy[:, 0::2]
        predicted_velocity[:, 1::3] = predicted_velocity_xy[:, 1::2]
        vtk_dir = export_dir / "surrogate_vtk"
        vtkhdf = write_surrogate_vtkhdf(
            run_dir,
            vtk_dir / "surrogate_fields.vtkhdf",
            prediction_cases,
            {
                "p": field_best.load_model().predict(prediction_inputs),
                "U": predicted_velocity,
            },
        )
        print(
            f"  surrogate p+U fields -> {vtkhdf} "
            f"({len(prediction_cases)} time steps + samples.csv)"
        )
```

- [ ] **Step 5: Static checks**

Run: `./.venv/bin/ruff check examples/naca0012_openfoam/ingest_and_train.py && ./.venv/bin/python -m py_compile examples/naca0012_openfoam/ingest_and_train.py`
Expected: no findings, exit 0.

- [ ] **Step 6: Re-run the verification (no OpenFOAM needed)**

```bash
./.venv/bin/python examples/naca0012_openfoam/ingest_and_train.py \
  examples/naca0012_openfoam/run --out examples/naca0012_openfoam/ex09_real \
  > /private/tmp/claude-501/-Users-georg-Software-surroforge/35056efb-276a-40ab-95eb-ab773c0f6980/scratchpad/ingest_u_run.log 2>&1
```

Run it in the foreground with a 10-minute timeout (do not background it). Expected in the log:
- `Ingested 110 real cases: 10720 pressure DOFs + 21440 velocity DOFs + Cl + Cd`
- three fit tables; the velocity table now shows **meaningful positive R²** and NO `(all test R² < 0 — data-starved sweep…)` caveat line under it,
- four export lines and `surrogate p+U fields -> …/surrogate_fields.vtkhdf (110 time steps + samples.csv)`.

- [ ] **Step 7: Spot-check the VTKHDF file (z exactly zero)**

```bash
./.venv/bin/python - <<'EOF'
import h5py, numpy as np
path = "examples/naca0012_openfoam/ex09_real/surrogate_vtk/surrogate_fields.vtkhdf"
with h5py.File(path) as h5:
    vtk = h5["VTKHDF"]
    p = vtk["CellData/p"][...]
    U = vtk["CellData/U"][...]
    assert p.shape == (110 * 10720,), p.shape
    assert U.shape == (110 * 10720, 3), U.shape
    assert np.isfinite(p).all() and np.isfinite(U).all()
    assert vtk["Steps"].attrs["NSteps"] == 110
    assert vtk["Steps/CellDataOffsets/U"][...].tolist() == list(range(0, 110 * 10720, 10720))
    assert (U[:, 2] == 0).all()          # re-padded z is exact zero
    assert np.abs(U[:, :2]).max() > 1.0  # x/y carry the freestream-scale signal
print("VTKHDF OK")
EOF
```

Expected: `VTKHDF OK`.

- [ ] **Step 8: Record all three fit tables verbatim**

Copy the three post-switch comparison tables from the run log into your report — Task 4 transcribes **these** numbers (the pressure/scalar tables should reproduce the earlier run; if they shifted, Task 4 must use the new values everywhere).

- [ ] **Step 9: Commit**

```bash
git add examples/naca0012_openfoam/ingest_and_train.py
git commit -m "refactor(naca): train the velocity surrogate on Ux/Uy only (2-D case)"
```

---

### Task 4: README + CHANGELOG with the verified numbers, full gates

**Files:**
- Modify: `examples/naca0012_openfoam/README.md`
- Modify: `CHANGELOG.md` (the `examples/naca0012_openfoam/` bullet under `### Added — Examples`, currently lines 158–172)

**Interfaces:**
- Consumes: the three fit tables recorded in Task 3 step 8 (the post-switch run — NOT Task 2's tables). Every `⟨…⟩` slot below MUST be replaced with those actual values — leaving a slot unfilled is a task failure. If the pressure/scalar numbers shifted versus the README's current "Verified" table, update those cells too so the whole table reflects one run.
- Produces: documentation only; no code.

- [ ] **Step 1: Update README "What it does"**

Replace the **Outputs per case** bullet (currently lines 33–35) with:

```markdown
- **Outputs per case:** converged pressure and velocity fields `p`/`U` (read mesh-free by
  `surroforge.data.read_openfoam_sweep`) and the integrated lift and drag coefficients `Cl`/`Cd`
  written by the `forceCoeffs` function object.
```

Replace the **Surrogates** bullet (currently lines 36–38) with:

```markdown
- **Surrogates** (the two from example 09 plus a velocity field): a **field** model
  (`reduced-field` POD + neural inner) `(AoA, speed) → pressure field`, a second field model
  `(AoA, speed) → velocity field (Ux, Uy)` — the 2-D case's z component is machine zero, so it
  is dropped at ingest and re-padded as exact zero on export — and a **scalar** model
  `(AoA, speed) → [Cl, Cd]`. Core `mlp` always; the PyTorch `torch-mlp` backend joins if the
  `surroforge-torch` plugin is installed.
```

- [ ] **Step 2: Update README export paragraph**

Replace the paragraph starting "`ingest_and_train.py` also **exports** the trained surrogates:" (currently lines 59–66) with:

```markdown
`ingest_and_train.py` also **exports** the trained surrogates: the best coefficient model to
`coeffs_surrogate.onnx`, the torch-mlp coefficient model to `coeffs_surrogate.pt` (a native PyTorch
checkpoint, when the plugin is installed), and the pressure- and velocity-field surrogates to
`field_surrogate.pkl` / `velocity_surrogate.pkl` (`reduced-field` is pickle-only). It then
evaluates the best field surrogates on every original case in `samples.csv` and writes VTKHDF
output to `surrogate_vtk/surrogate_fields.vtkhdf`, plus a `samples.csv` mapping file for the
time-step to case/AoA/speed assignment. Open the `.vtkhdf` file in ParaView; the surrogate
pressure field is written as cell data `p` and the surrogate velocity as vector cell data `U`
(magnitude colouring, glyphs and streamlines work; `Uz` is identically zero in this 2-D case).
```

- [ ] **Step 3: Update README "Verified" section**

Replace the paragraph "Ingestion read **10 720 pressure DOFs/case …**" (currently lines 151–154) with:

```markdown
Ingestion read **10 720 pressure DOFs plus 21 440 velocity DOFs per case (`Ux`/`Uy` — the
machine-zero z columns of the 2-D case are dropped at ingest), plus the real `Cl` and
`Cd`** for all 110 cases; all three surrogates trained; all four model artifacts exported
(`coeffs_surrogate.onnx`/`.pt`, `field_surrogate.pkl`, `velocity_surrogate.pkl`); and the best
field surrogates wrote a **110-step VTKHDF pressure+velocity time series** under
`surrogate_vtk/surrogate_fields.vtkhdf` for ParaView.
```

In the fit-quality table (currently lines 167–171), add one row directly under the `Field (AoA, U) → p` row, filling the slots from Task 2's velocity table:

```markdown
| Field `(AoA, U) → U` (POD-NN) | `⟨best inner⟩` inner | ⟨test R²⟩ | ⟨test RMSE⟩ |
```

After the existing **Field (POD-NN)** bullet (currently lines 185–188), add a matching bullet reporting the velocity result — 3–4 sentences in the same style: which inner wins, the R² spread across the three inners (actual numbers from the run log), whether the ordering matches the pressure field, **and** the design note that the velocity surrogate trains on `Ux`/`Uy` only: the 2-D case's z columns are machine zero, and Core's uniform-average R² over zero-variance columns would be arbitrarily negative (the pre-switch full-vector run showed test R² ≈ −861 despite an excellent x/y fit), so z is dropped at ingest and re-padded as exact zero on export.

- [ ] **Step 4: Update README on-disk layout**

In the layout tree (currently lines 198–209), add `velocity_surrogate.pkl` after the `field_surrogate.pkl` line:

```
   ├─ coeffs_surrogate.onnx / .pt
   ├─ field_surrogate.pkl
   ├─ velocity_surrogate.pkl
```

- [ ] **Step 5: Update the CHANGELOG bullet**

In `CHANGELOG.md`, inside the `examples/naca0012_openfoam/` bullet under `### Added — Examples`:

(a) Replace "and `ingest_and_train.py` builds the same surrogates — `(AoA, speed) → field` and `(AoA, speed) → [Cl, Cd]` — from the genuine fields and" with "and `ingest_and_train.py` builds the same surrogates — `(AoA, speed) → field` (pressure **and** velocity) and `(AoA, speed) → [Cl, Cd]` — from the genuine fields and".

(b) Replace the final sentence "After training, `ingest_and_train.py` now evaluates the best field surrogate on every original `samples.csv` case and writes a single VTKHDF time series `surrogate_vtk/surrogate_fields.vtkhdf`, using the original VTU meshes as the geometry source and writing the surrogate pressure as cell field `p`. Verified against OpenFOAM v2512." with "After training, `ingest_and_train.py` now evaluates the best field surrogates on every original `samples.csv` case and writes a single VTKHDF time series `surrogate_vtk/surrogate_fields.vtkhdf`, using the original VTU meshes as the geometry source and writing the surrogate pressure as cell field `p` and the surrogate velocity as vector cell field `U`. Verified against OpenFOAM v2512."

- [ ] **Step 6: Full gates**

```bash
./.venv/bin/pytest tests/test_naca_vtk_export.py -v
./.venv/bin/pytest
./.venv/bin/ruff check src tests examples/naca0012_openfoam/ingest_and_train.py examples/naca0012_openfoam/surrogate_vtk.py
```

Expected: naca tests pass; the full root suite passes (plain `pytest` so the counts line shows); ruff clean.

- [ ] **Step 7: Commit**

```bash
git add examples/naca0012_openfoam/README.md CHANGELOG.md
git commit -m "docs(naca): document the velocity-field surrogate and verified U-fit numbers"
```
