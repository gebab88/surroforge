# NACA 0012 Example Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish `examples/naca0012_openfoam/` without changing behaviour: extract the VTK I/O into a new `surrogate_vtk.py`, delete the dead `.vtu`/`.pvd` export path, consolidate the docs into one tracked English README, and tidy the working tree — proven against the real 110-case sweep on disk.

**Architecture:** The live VTKHDF export chain moves verbatim (one indentation fix) from `ingest_and_train.py` into a new sibling module `surrogate_vtk.py` with no SurroForge imports; the script imports it back and keeps `main()` functionally identical. The dead single-`.vtu`-per-case path is deleted, not moved. README absorbs the machine-agnostic reference from the gitignored LOCAL_SETUP.md.

**Tech Stack:** Python 3.13 (`.venv`), numpy/pandas/h5py, ruff, existing OpenFOAM output on disk (no solver needed for any task).

**Spec:** `docs/superpowers/specs/2026-07-05-naca0012-example-polish-design.md`

## Global Constraints

- **Never push.** Commits stay local until the user has seen commits + changed files (CLAUDE.md rule).
- **No `Co-Authored-By:` trailer on any commit.** Repo-local identity is already configured (`Georg Babij <g.babij@gmail.com>`); plain `git commit` is correct.
- Commit directly to `main` (solo project).
- Use `.venv/bin/python` and `.venv/bin/ruff` (repo root as CWD unless a command says otherwise).
- Ruff config (pyproject): line-length 100, rules `E,F,I,UP,B,W`. `examples/` is **outside** the CI ruff scope (`src tests`), so run ruff on these files explicitly.
- Behaviour-preserving refactor: moved function bodies stay byte-identical except one indentation fix (noted in Task 1); `main()`'s console output and artifacts are unchanged.
- The naca directory is **not covered by CI** (pytest smoke-tests only `examples/[0-9]*.py`); the checks in these tasks are the only gates.
- Data prerequisite (verified present 2026-07-05): `examples/naca0012_openfoam/run/` with `samples.csv` + 110 `case_*` dirs, and `h5py 3.16.0` in `.venv`.
- Scratchpad for throwaway outputs: `/private/tmp/claude-501/-Users-georg-Software-surroforge/35056efb-276a-40ab-95eb-ab773c0f6980/scratchpad`

---

### Task 1: Extract `surrogate_vtk.py`, slim `ingest_and_train.py`

**Files:**
- Create: `examples/naca0012_openfoam/surrogate_vtk.py`
- Modify: `examples/naca0012_openfoam/ingest_and_train.py` (616 lines before; ~290 after)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `surrogate_vtk.write_surrogate_vtkhdf(run_dir: Path, output: Path, samples: pd.DataFrame, fields, *, field_name: str = "p") -> Path` — the only public name; imported by `ingest_and_train.py` and exercised by Task 2's acceptance run.

**Background for the implementer:** `ingest_and_train.py` currently contains two VTK export paths. `main()` only calls `write_surrogate_vtkhdf` (the VTKHDF time-series path). The older path — `write_surrogate_vtk_cases` → `write_surrogate_internal_vtu` → `_vtk_float32_payload`, plus `_write_pvd_collection` — is dead code and is **deleted**. The live chain moves to the new module.

- [ ] **Step 1: Create `examples/naca0012_openfoam/surrogate_vtk.py`**

The moved code is byte-identical to the current file except: (a) new docstring/imports, (b) one fix — in the original `write_surrogate_vtkhdf`, the closing `)` of the first `raise ValueError(...)` is misindented at 4 spaces (original line 381); here it is correctly indented at 8. Full content:

```python
"""VTK/VTKHDF I/O for surrogate fields on an OpenFOAM sweep.

Standalone helper for ``ingest_and_train.py`` — no SurroForge imports. It reads the
``internal.vtu`` mesh that ``foamToVTK`` wrote for each case (``run_sweep.py --vtk``) and
writes the surrogate-predicted cell fields of **all** cases into a single ParaView-readable
VTKHDF (UnstructuredGrid, format version 2.5) time-series file — one time step per case —
plus a ``samples.csv`` mapping time step -> case/parameters next to it.

Only ``write_surrogate_vtkhdf`` is public. The VTKHDF write needs ``h5py`` (imported
lazily); install it with ``pip install 'surroforge[formats]'``.
"""

from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


class VtuMesh(NamedTuple):
    points: np.ndarray
    connectivity: np.ndarray
    offsets: np.ndarray
    types: np.ndarray


def _data_array(parent: ET.Element, name: str) -> ET.Element | None:
    for child in parent.findall("DataArray"):
        if child.attrib.get("Name") == name:
            return child
    return None


def _vtk_dtype(vtk_type: str, byte_order: str) -> np.dtype:
    endian = ">" if byte_order == "BigEndian" else "<"
    if vtk_type == "Float32":
        return np.dtype(f"{endian}f4")
    if vtk_type == "Float64":
        return np.dtype(f"{endian}f8")
    if vtk_type == "Int32":
        return np.dtype(f"{endian}i4")
    if vtk_type == "Int64":
        return np.dtype(f"{endian}i8")
    if vtk_type == "UInt8":
        return np.dtype("u1")
    if vtk_type == "UInt32":
        return np.dtype(f"{endian}u4")
    if vtk_type == "UInt64":
        return np.dtype(f"{endian}u8")
    raise ValueError(f"unsupported VTK DataArray type {vtk_type!r}")


def _decode_vtk_data_array(
    data_array: ET.Element,
    *,
    byte_order: str,
    header_type: str,
) -> np.ndarray:
    dtype = _vtk_dtype(data_array.attrib["type"], byte_order)
    text = "".join((data_array.text or "").split())
    fmt = data_array.attrib.get("format", "ascii")
    if fmt == "binary":
        raw = base64.b64decode(text)
        header_dtype = _vtk_dtype(header_type, byte_order)
        header_nbytes = header_dtype.itemsize
        n_bytes = int(np.frombuffer(raw[:header_nbytes], dtype=header_dtype, count=1)[0])
        data = np.frombuffer(raw[header_nbytes : header_nbytes + n_bytes], dtype=dtype)
    elif fmt == "ascii":
        data = np.fromstring(data_array.text or "", dtype=dtype, sep=" ")
    else:
        raise ValueError(f"unsupported VTK DataArray format {fmt!r}")

    components = int(data_array.attrib.get("NumberOfComponents", "1"))
    if components > 1:
        if data.size % components:
            raise ValueError("VTK DataArray length is not divisible by NumberOfComponents")
        data = data.reshape((-1, components))
    return data


def _read_vtu_mesh(path: Path) -> VtuMesh:
    root = ET.parse(path).getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"{path}: no VTK Piece element found")

    n_points = int(piece.attrib["NumberOfPoints"])
    n_cells = int(piece.attrib["NumberOfCells"])
    byte_order = root.attrib.get("byte_order", "LittleEndian")
    header_type = root.attrib.get("header_type", "UInt32")

    points_array = piece.find("Points/DataArray")
    cells = piece.find("Cells")
    if points_array is None or cells is None:
        raise ValueError(f"{path}: VTU template must contain Points and Cells")
    connectivity_array = _data_array(cells, "connectivity")
    offsets_array = _data_array(cells, "offsets")
    types_array = _data_array(cells, "types")
    if connectivity_array is None or offsets_array is None or types_array is None:
        raise ValueError(f"{path}: VTU Cells must contain connectivity, offsets and types arrays")

    decode = {"byte_order": byte_order, "header_type": header_type}
    points = np.asarray(_decode_vtk_data_array(points_array, **decode), dtype=np.float32)
    if points.ndim == 1:
        if points.size % 3:
            raise ValueError(f"{path}: point coordinate array is not divisible by 3")
        points = points.reshape((-1, 3))
    if points.shape != (n_points, 3):
        raise ValueError(f"{path}: expected {n_points} 3D points, got shape {points.shape}")

    connectivity = np.asarray(
        _decode_vtk_data_array(connectivity_array, **decode).reshape(-1), dtype=np.int64
    )
    vtu_offsets = np.asarray(
        _decode_vtk_data_array(offsets_array, **decode).reshape(-1), dtype=np.int64
    )
    if vtu_offsets.size != n_cells:
        raise ValueError(f"{path}: expected {n_cells} cell offsets, got {vtu_offsets.size}")
    offsets = np.empty(n_cells + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = vtu_offsets

    types = np.asarray(_decode_vtk_data_array(types_array, **decode).reshape(-1), dtype=np.uint8)
    if types.size != n_cells:
        raise ValueError(f"{path}: expected {n_cells} cell types, got {types.size}")
    return VtuMesh(points=points, connectivity=connectivity, offsets=offsets, types=types)


def _internal_vtu_template(case_dir: Path) -> Path:
    templates = sorted((case_dir / "VTK").glob(f"{case_dir.name}_*/internal.vtu"))
    if not templates:
        raise FileNotFoundError(f"no internal.vtu template under {case_dir / 'VTK'}")
    return templates[-1]


def _step_offsets(counts: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        offsets[1:] = np.cumsum(counts[:-1], dtype=np.int64)
    return offsets


def write_surrogate_vtkhdf(
    run_dir: Path,
    output: Path,
    samples: pd.DataFrame,
    fields,
    *,
    field_name: str = "p",
) -> Path:
    """Write all surrogate fields to one ParaView-readable VTKHDF time-series file."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "VTKHDF export requires h5py; install the optional dependency with "
            "`pip install 'surroforge[formats]'`."
        ) from exc

    if "case" not in samples.columns:
        raise ValueError("samples must include a 'case' column")
    matrix = np.asarray(fields, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(samples):
        raise ValueError(
            f"surrogate fields must be an (n_cases, n_cells) matrix; got {matrix.shape} "
            f"for {len(samples)} sample row(s)"
        )

    meshes: list[VtuMesh] = []
    for row in samples.itertuples(index=False):
        case = str(row.case)
        mesh = _read_vtu_mesh(_internal_vtu_template(run_dir / case))
        if matrix.shape[1] != mesh.types.size:
            raise ValueError(
                f"{case}: surrogate field has {matrix.shape[1]} values, "
                f"but VTU template has {mesh.types.size} cells"
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
        cell_data.create_dataset(field_name, data=matrix.reshape(-1).astype(np.float32))
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
        cell_data_offsets.create_dataset(field_name, data=_step_offsets(cell_counts))
        field_data_offsets = steps.create_group("FieldDataOffsets")
        field_data_offsets.create_dataset("TimeValue", data=np.arange(n_steps, dtype=np.int64))
        field_data_sizes = steps.create_group("FieldDataSizes")
        field_data_sizes.create_dataset(
            "TimeValue", data=np.ones((n_steps, 2), dtype=np.int64)
        )

    return output
```

- [ ] **Step 2: Static-check the new module**

Run: `.venv/bin/python -m py_compile examples/naca0012_openfoam/surrogate_vtk.py && .venv/bin/ruff check examples/naca0012_openfoam/surrogate_vtk.py`
Expected: no py_compile output; ruff prints `All checks passed!`

- [ ] **Step 3: Replace the import block in `ingest_and_train.py`**

Edit (exact old → new; drops `base64`, `struct`, `xml.etree.ElementTree`, `numpy` — all now unused — and adds the sibling import; ruff-isort accepts `surrogate_vtk` in the third-party block, verified):

Old:
```python
import argparse
import base64
import shutil
import struct
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from surroforge import Project, Trainer
```

New:
```python
import argparse
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from surrogate_vtk import write_surrogate_vtkhdf

from surroforge import Project, Trainer
```

(`NamedTuple` and `pd` are still used: `CoeffReading` and `pd.read_csv`.)

- [ ] **Step 4: Delete the `VtuMesh` class from `ingest_and_train.py`**

Edit (exact old → new):

Old:
```python
class VtuMesh(NamedTuple):
    points: np.ndarray
    connectivity: np.ndarray
    offsets: np.ndarray
    types: np.ndarray


def read_coeffs(case_dir: Path) -> CoeffReading:
```

New:
```python
def read_coeffs(case_dir: Path) -> CoeffReading:
```

- [ ] **Step 5: Delete the whole VTK block from `ingest_and_train.py`**

The block is contiguous: it starts at the line `def _vtk_float32_payload(values) -> str:` and ends with the `    return output` line immediately before `def main(run_dir: Path, out_dir: Path | None = None) -> None:`. It sits between `_key(...)`'s `return` and `main`. Delete these eleven definitions (nothing else lives in that range):

`_vtk_float32_payload`, `_data_array`, `_vtk_dtype`, `_decode_vtk_data_array`, `_read_vtu_mesh`, `write_surrogate_internal_vtu`, `_internal_vtu_template`, `_write_pvd_collection`, `write_surrogate_vtk_cases`, `_step_offsets`, `write_surrogate_vtkhdf`.

After the deletion, exactly two blank lines separate `_key`'s `return (round(float(aoa), 6), round(float(speed), 6))` from `def main(...)`. `main()` itself is untouched — its call `write_surrogate_vtkhdf(run_dir, vtk_dir / "surrogate_fields.vtkhdf", prediction_cases, predicted_fields)` now resolves to the import from Step 3.

- [ ] **Step 6: Verify the slimmed script statically**

Run: `.venv/bin/python -m py_compile examples/naca0012_openfoam/ingest_and_train.py && .venv/bin/ruff check examples/naca0012_openfoam/ingest_and_train.py examples/naca0012_openfoam/surrogate_vtk.py`
Expected: ruff prints `All checks passed!`

Run: `grep -n 'def \|^class ' examples/naca0012_openfoam/ingest_and_train.py`
Expected: only `CoeffReading` (class), `read_coeffs`, `_nn`, `_candidates`, `_key`, `main`, `_best_label`, `_report` — none of the eleven deleted names.

Run: `wc -l examples/naca0012_openfoam/ingest_and_train.py examples/naca0012_openfoam/surrogate_vtk.py`
Expected: roughly 290 and 240 lines (±10 each).

- [ ] **Step 7: Import smoke test**

Run: `.venv/bin/python examples/naca0012_openfoam/ingest_and_train.py --help`
Expected: argparse usage text, exit 0. (This executes the module-level `from surrogate_vtk import write_surrogate_vtkhdf` — the sibling import works because Python puts the script's own directory on `sys.path[0]`.)

Run: `.venv/bin/python -c "import sys; sys.path.insert(0, 'examples/naca0012_openfoam'); from surrogate_vtk import write_surrogate_vtkhdf; print(write_surrogate_vtkhdf.__doc__.splitlines()[0])"`
Expected: `Write all surrogate fields to one ParaView-readable VTKHDF time-series file.`

- [ ] **Step 8: Commit**

```bash
git add examples/naca0012_openfoam/surrogate_vtk.py examples/naca0012_openfoam/ingest_and_train.py
git commit -m "refactor(examples): extract surrogate_vtk.py and drop the dead .vtu/.pvd path"
```

---

### Task 2: Behavioral acceptance against the real 110-case sweep

**Files:**
- No repo files touched. Output goes to the scratchpad (`ex09_real_check/`).

**Interfaces:**
- Consumes: the Task-1 script pair; on-disk data `examples/naca0012_openfoam/run/`.
- Produces: the two printed fit tables (field + scalar, three methods each) and the convergence-warning status — **record them verbatim in your task report**; Task 3 copies these numbers into the README.

- [ ] **Step 1: Run the full pipeline on the real data**

Run (takes several minutes — training 6 models on 110 cases × 10 720 DOFs; use a 600000 ms timeout):

```bash
.venv/bin/python examples/naca0012_openfoam/ingest_and_train.py \
  examples/naca0012_openfoam/run \
  --out /private/tmp/claude-501/-Users-georg-Software-surroforge/35056efb-276a-40ab-95eb-ab773c0f6980/scratchpad/ex09_real_check
```

Expected output, in order (exit 0):
- `Ingested 110 real cases: 10720 pressure DOFs + Cl + Cd`
- **No** `⚠ ... still-swinging C_L` line (all 110 cases settled in the verified run).
- `=== Field surrogate: ... ===` table with rows `mlp`, `torch-mlp`, `polynomial` — `torch-mlp` marked `<- best`.
- `=== Scalar surrogate: ... ===` table with the same three rows — `mlp` marked `<- best`.
- `=== Export ===` with three lines: `onnx -> coeffs_surrogate.onnx ...`, `pt -> coeffs_surrogate.pt ...`, `field surrogate -> field_surrogate.pkl ...`.
- `=== Surrogate field VTK ===` with `surrogate p fields -> .../surrogate_fields.vtkhdf (110 time steps + samples.csv)`.

Reference values from the previously verified run (fresh numbers should be in the same range; small drift is fine, a different best method is NOT): field best `torch-mlp` R² ≈ 0.986, RMSE ≈ 30.0; scalar best `mlp` R² ≈ 0.994, RMSE ≈ 0.0074.

**If the run fails or a best method flips:** stop, report the discrepancy, and do not proceed to Task 3.

- [ ] **Step 2: Check the artifacts exist**

Run: `ls -la /private/tmp/claude-501/-Users-georg-Software-surroforge/35056efb-276a-40ab-95eb-ab773c0f6980/scratchpad/ex09_real_check/ /private/tmp/claude-501/-Users-georg-Software-surroforge/35056efb-276a-40ab-95eb-ab773c0f6980/scratchpad/ex09_real_check/surrogate_vtk/`
Expected: `coeffs_surrogate.onnx`, `coeffs_surrogate.pt`, `field_surrogate.pkl`, `naca/` (project dir), and in `surrogate_vtk/`: `surrogate_fields.vtkhdf` + `samples.csv`.

- [ ] **Step 3: Structural check of the VTKHDF file (machine stand-in for opening it in ParaView)**

```bash
.venv/bin/python - <<'EOF'
import h5py

path = ("/private/tmp/claude-501/-Users-georg-Software-surroforge/"
        "35056efb-276a-40ab-95eb-ab773c0f6980/scratchpad/ex09_real_check/"
        "surrogate_vtk/surrogate_fields.vtkhdf")
with h5py.File(path) as f:
    g = f["VTKHDF"]
    assert g.attrs["Type"] == b"UnstructuredGrid", g.attrs["Type"]
    assert list(g.attrs["Version"]) == [2, 5], g.attrs["Version"]
    assert g["Steps"].attrs["NSteps"] == 110, g["Steps"].attrs["NSteps"]
    assert "p" in g["CellData"], list(g["CellData"])
    n_cells_total = int(g["NumberOfCells"][...].sum())
    assert g["CellData"]["p"].shape == (n_cells_total,), g["CellData"]["p"].shape
    print(f"OK: NSteps=110, cell data 'p' with {n_cells_total} total values")
EOF
```

Expected: `OK: NSteps=110, cell data 'p' with 1179200 total values` (110 × 10 720).

- [ ] **Step 4: Record the numbers**

Copy both printed fit tables (all three methods each, R² and RMSE) and the absence/presence of the unsettled-C_L warning into the task report, verbatim. No commit — nothing tracked changed.

---

### Task 3: Consolidate the README

**Files:**
- Modify: `examples/naca0012_openfoam/README.md` (full replacement, content below)

**Interfaces:**
- Consumes: Task 2's recorded fit tables — the six per-method R² values, the two best-row RMSE values, and the settled-C_L status.
- Produces: the consolidated README (needed by Task 4, which strips the now-duplicated content from LOCAL_SETUP.md).

**Deliberate content fix (not a drift accident):** the old README's quickstart called
`run_sweep.py` *without* `--vtk`, but `--vtk` is default-off and the surrogate VTKHDF export
reads the `case/VTK/internal.vtu` meshes it produces — following the old quickstart verbatim
ends in a `FileNotFoundError` at the export step. The content below adds `--vtk` to the
quickstart and documents it as required for the export.

**Number reconciliation (do this before writing):** The content below carries the numbers of the *previously* verified run. Compare every number in the "Fit quality" table and the three bullets after it against Task 2's tables and correct any digit that differs. The qualitative claims (which method is best per task, whether `torch-mlp` underfits the scalar task, whether the RMSE-best and R²-best scalar methods differ) must be checked against the fresh tables too — if a pattern no longer holds, rewrite that clause to match reality and flag it in your report. The convergence facts (90/110, which angles, ~520–670 iterations) come from the sweep itself and are **not** re-measured by Task 2 — leave them as written.

- [ ] **Step 1: Replace `examples/naca0012_openfoam/README.md` wholesale with:**

````markdown
# NACA 0012 — real OpenFOAM sweep → SurroForge surrogate

The **real-CFD companion** to [`examples/09_naca0012_openfoam_surrogate.py`](../09_naca0012_openfoam_surrogate.py).

Example 09 *fabricates* a representative sweep so it runs in CI with no solver. This directory
runs **actual** steady-state RANS with OpenFOAM, once per angle-of-attack/speed combination, and
trains the SurroForge surrogates on the genuine output. The downstream surrogate code is
identical — only the data source changes.

> ⚠️ **Not run in CI.** This needs a real OpenFOAM install, so it is not part of the smoke-tested
> examples. The scripts were authored and **verified against OpenFOAM v2512** (see *Verified* below).

## Files

- **`run_sweep.py`** — builds one `airFoil2D`-based case per DoE point and solves it with
  `simpleFoam`, optionally many cases in parallel (`--jobs`).
- **`ingest_and_train.py`** — reads the converged fields and force coefficients, trains and
  compares the surrogates, exports the models, and writes the surrogate fields for ParaView.
- **`surrogate_vtk.py`** — standalone VTK/VTKHDF I/O helper used by `ingest_and_train.py`: it
  reads the per-case `internal.vtu` mesh that `foamToVTK` wrote and writes one ParaView-readable
  VTKHDF time-series file of surrogate cell fields. No SurroForge imports — it works against any
  OpenFOAM sweep with the same layout.

## What it does

- **Mesh & physics:** the validated `airFoil2D` NACA 0012 C-mesh that ships with OpenFOAM
  (`$FOAM_TUTORIALS/incompressible/simpleFoam/airFoil2D`, ~10.7k cells), incompressible
  `simpleFoam` with the Spalart–Allmaras turbulence model.
- **2-D sweep (angle × speed):** the mesh is fixed; each angle of attack is imposed by **rotating
  the freestream velocity** `U = |U|·(cos α, sin α, 0)` (and the `forceCoeffs` lift/drag
  directions), and the freestream speed `|U|` is varied independently — a full DoE over every
  `angle × speed` combination, all on the one mesh (no re-meshing).
- **Outputs per case:** a converged pressure field `p` (read mesh-free by
  `surroforge.data.read_openfoam_sweep`) and the integrated lift and drag coefficients `Cl`/`Cd`
  written by the `forceCoeffs` function object.
- **Surrogates** (same two as example 09): a **field** model (`reduced-field` POD + neural inner)
  `(AoA, speed) → pressure field`, and a **scalar** model `(AoA, speed) → [Cl, Cd]`. Core `mlp`
  always; the PyTorch `torch-mlp` backend joins if the `surroforge-torch` plugin is installed.

## Run it

```bash
# 1. Source your OpenFOAM environment (so simpleFoam and $FOAM_TUTORIALS are available)
source /path/to/OpenFOAM/etc/bashrc          # e.g. /Users/<you>/Software/openfoam/etc/bashrc

# 2. Sweep angle of attack × speed (one simpleFoam run per combination); --vtk keeps each
#    case's converged mesh+field, which the surrogate VTKHDF export below needs
python examples/naca0012_openfoam/run_sweep.py --angles "-6 0 6 12 16 18" --speeds "20 40 60" --vtk

# 3. Ingest the fields + forceCoeffs and train the surrogates
python examples/naca0012_openfoam/ingest_and_train.py run
#    ...or save the whole run (SurroForge project + artifacts) for later inspection:
python examples/naca0012_openfoam/ingest_and_train.py run --out ./ex09_real
```

The VTKHDF export needs `h5py` (`pip install 'surroforge[formats]'` or
`pip install 'surroforge[all]'`).

`ingest_and_train.py` also **exports** the trained surrogates: the best coefficient model to
`coeffs_surrogate.onnx`, the torch-mlp coefficient model to `coeffs_surrogate.pt` (a native PyTorch
checkpoint, when the plugin is installed), and the field surrogate to `field_surrogate.pkl`
(`reduced-field` is pickle-only). It then evaluates the best field surrogate on every original
case in `samples.csv` and writes VTKHDF output to
`surrogate_vtk/surrogate_fields.vtkhdf`, plus a `samples.csv` mapping file for the time-step to
case/AoA/speed assignment. Open the `.vtkhdf` file in ParaView; the surrogate pressure field is
written as cell data `p`.

By default these outputs land in the sweep directory and the SurroForge project (datasets, runs,
lineage, POD basis, models) is built in a temp dir and discarded; pass **`--out DIR`** to save the
whole run instead — the project to `DIR/naca` plus the artifacts and `DIR/surrogate_vtk/` — so it
stays inspectable and reproducible afterwards (`surroforge --config-dir DIR/naca list`), exactly
like example 09's `--out`.

`run_sweep.py` options: `--angles` (deg) and `--speeds` (|U| m/s), each space/comma-separated — a
2-D DoE is run over every angle × speed; `--iters` (steady-state *cap*, default 1000), `--residual`
(residual-convergence tolerance, default `1e-8`), `--jobs` (cases solved concurrently), `--vtk`
(write each case's converged field to `case/VTK/` — **required** for the surrogate VTKHDF export,
which reuses those meshes), `--out` (sweep dir, default `run`), `--base` (an explicit `airFoil2D`
case; defaults to `$FOAM_TUTORIALS`).

## Case setup in detail

What `run_sweep.py` writes into every case (derived from the `airFoil2D` tutorial):

- **Geometry & mesh:** NACA 0012, chord `c = 1`, the validated `airFoil2D` C-mesh
  (~10 720 cells; 2-D, one cell layer with `frontAndBack` as `empty`). The mesh is **fixed** —
  the angle of attack is imposed by rotating the freestream, never by re-meshing.
- **Physics & solver:** `simpleFoam` — steady incompressible RANS (SIMPLE algorithm) with the
  Spalart–Allmaras model (`nuTilda`); Newtonian transport, `nu = 1e-5 m²/s`, `rho = 1`. Chord
  Reynolds number `Re = |U|·c/nu`: **2·10⁶ (20 m/s) … 1·10⁷ (100 m/s)**.
- **Boundary conditions (field `U`):**

  | Patch | Type |
  | --- | --- |
  | `inlet`, `outlet` | `freestreamVelocity` (`freestreamValue = internalField`) |
  | `walls` (airfoil) | `noSlip` |
  | `frontAndBack` | `empty` (2-D) |

  `internalField = uniform (|U|·cos α, |U|·sin α, 0)`; `p`, `nut`, `nuTilda` come from the base
  case's `0.orig`.
- **Force coefficients:** the `forceCoeffs1` function object rotates with the freestream:
  `liftDir = (−sin α, cos α, 0)`, `dragDir = (cos α, sin α, 0)`, with `patches (walls)`,
  `rhoInf 1`, `magUInf = |U|`, `lRef 1`, `Aref 1`, `CofR (0.25 0 0)`. `writeInterval 1` records
  `Cl`/`Cd` every iteration — the data behind the settledness check below.
- **Numerics:** `fvSchemes`/`fvSolution` from the base case; `SIMPLE/residualControl` is pinned
  to `--residual` (default `1e-8`) for `p`, `U`, `nuTilda` so the solver stops itself once every
  residual is below it; the linear-solver tolerances are tightened to `residual·1e-2` so that
  target is reachable; `endTime` is the `--iters` cap, with a single ASCII write at the end.

## Convergence is checked, not assumed

A surrogate is only as trustworthy as the CFD under it, so both stages guard against
half-converged data instead of silently using the last iteration:

- **Solver level (`run_sweep.py`):** `--residual` pins OpenFOAM's `SIMPLE/residualControl`, so
  `simpleFoam` stops itself once every residual is below it — `--iters` is only the cap. Each case
  reports the iteration it converged at (or "NOT converged" if it hit the cap), and a final summary
  lists every case that did not converge.
- **Quantity level (`ingest_and_train.py`):** even a residual-converged run can leave a marginally
  unsteady force, so `read_coeffs` checks the **peak-to-peak spread of `Cl` over the last 20
  writes**; any case whose lift is still swinging is flagged before training.

In the verified run below the two are **complementary, not identical**: the post-stall cases that
hit the iteration cap nonetheless show a *settled* `Cl` (the lift plateaus even while the last
residuals sit above `1e-8`), so the quantity-level check passes them. Residual non-convergence
did **not**, here, mean an unsteady force — which is also why the surrogate fit barely changes
whether those cases are included or not.

## Verified (OpenFOAM v2512, macOS/arm64)

The 2-D pipeline was run end-to-end on real `simpleFoam` with this DoE:

| Parameter | Values |
| --- | --- |
| Angle of attack | −5° … +16° in 1° steps (22 angles) |
| Freestream speed \|U\| | 20, 40, 60, 80, 100 m/s (5 speeds) |
| Cases | 22 × 5 = **110** (`case_000` … `case_109`) |

using these exact commands (repo root, OpenFOAM environment sourced):

```bash
python examples/naca0012_openfoam/run_sweep.py \
  --angles "-5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16" \
  --speeds "20 40 60 80 100" \
  --iters 1500 --jobs 16 --vtk --out examples/naca0012_openfoam/run

python examples/naca0012_openfoam/ingest_and_train.py \
  examples/naca0012_openfoam/run --out examples/naca0012_openfoam/ex09_real
```

Ingestion read **10 720 pressure DOFs/case plus the real `Cl` and `Cd`** for all 110 cases; both
surrogates trained; all three model artifacts exported (`coeffs_surrogate.onnx`/`.pt`,
`field_surrogate.pkl`); and the best field surrogate wrote a **110-step VTKHDF pressure-field
time series** under `surrogate_vtk/surrogate_fields.vtkhdf` for ParaView.

**Convergence.** 90/110 cases reached the `1e-8` residual target. The attached cases converge at
**~520–670 iterations** — which is why `--iters` defaults to 1000; an older 500 default clipped
them just short and falsely flagged them "not converged". The 20 cases that did not converge are
the post-stall angles **+12/+14/+15/+16° at all five speeds** — steady RANS cannot converge
post-stall no matter the cap (+13° happens to converge on this mesh). The quantity-level check
found **no still-swinging `Cl`**: even the residual-unconverged cases had a settled lift plateau,
so all 110 cases are usable training data (the DoE avoids the unsteady deep-stall cases ≤ −12° on
the negative side).

**Fit quality (0.25 held-out test, `hidden_layer_sizes [48, 48]`, POD `energy 0.999`):**

| Surrogate | Best method | test R² | test RMSE |
| --- | --- | --- | --- |
| Field `(AoA, U) → p` (POD-NN) | `torch-mlp` inner | 0.986 | 30.0 |
| Scalar `(AoA, U) → [Cl, Cd]` | `mlp` | 0.994 | 0.0074 |

- **Scalar:** `mlp` wins the example's RMSE-based "best" pick; the degree-2 `polynomial` is
  within a hair of it on R² — in *real* CFD `Cl` and `Cd` sit on different scales, so the
  RMSE-best and R²-best methods can differ. The `torch-mlp` backend *underfits* this 2-output
  regression (same hyperparameters as `mlp`; its full-batch training does not fit so few points
  without tuning).
- **Why the polynomial is so strong here (the honest caveat):** steady RANS **cannot converge the
  post-stall cases**, so the *converged* data is essentially **attached and low-order** (≈linear
  `Cl`, ≈quadratic `Cd`) — exactly what a quadratic fits best. A neural net's across-stall
  advantage — which the synthetic example 09 deliberately *can* show — **cannot be demonstrated
  on converged steady-RANS data**, because the nonlinear post-stall region is precisely the part
  that will not converge. Use a transient solver there, or keep the DoE in the attached regime.
- **Field (POD-NN):** the `torch-mlp` inner fits the pressure field best; the `polynomial` inner
  is close behind and the scikit-learn `mlp` inner is the weakest of the three — the reverse of
  the scalar task, so on this data neither NN backend dominates both tasks.

Takeaway: the **scalar coefficient surrogate `(AoA, speed) → [Cl, Cd]` is the robust, cheap win**,
and on converged steady-RANS data a **degree-2 polynomial is a strong baseline** — the post-stall
regime where a network pulls ahead is exactly what does not converge. The **field** POD-NN fits
the pressure field very well with a `torch-mlp` inner. Coverage and convergence matter more than
raw angular resolution.

**On-disk layout of the verified run** (all gitignored):

```
examples/naca0012_openfoam/
├─ run/                      # raw CFD, 110 cases (~514 MB)
│  ├─ case_000 … case_109/   #   each: 0/, <t_conv>/, constant/polyMesh,
│  │                         #     postProcessing/forceCoeffs1, VTK/, log.simpleFoam
│  └─ samples.csv            #   DoE table (case, aoa, speed)
└─ ex09_real/                # saved SurroForge run + artifacts (--out)
   ├─ coeffs_surrogate.onnx / .pt
   ├─ field_surrogate.pkl
   ├─ surrogate_vtk/         #   surrogate_fields.vtkhdf + samples.csv
   └─ naca/                  #   project: runs/, basis/, data/, surroforge.yaml
```

## How this maps onto SurroForge

`ingest_and_train.py` is deliberately thin: `read_openfoam_sweep(run, field="p", …)` turns the
sweep into the wide snapshot table the engine trains on, `read_coeffs(...)` parses `Cl`/`Cd` from
`forceCoeffs`, and the rest is ordinary `Trainer.from_dict(...)` — the same API as every other
example. The ParaView export lives in `surrogate_vtk.py`, which knows nothing about SurroForge.
Point `read_openfoam_sweep` at *any* OpenFOAM case sweep (different airfoil, more parameters than
angle and speed, a velocity field instead of pressure) and the flow is unchanged.

## Caveats

- `lRef`/`Aref` are the unit chord (the airFoil2D NACA 0012 has chord 1); adjust for other geometry.
  The exact `Cl` magnitude depends on these and on convergence.
- Convergence is checked automatically (see above), but near stall (high AoA) cases still need more
  iterations — raise `--iters` (or loosen `--residual`) for any case the sweep flags as unconverged.
````

- [ ] **Step 2: Verify the numbers match Task 2**

Re-read the written README's "Fit quality" table and bullets side-by-side with Task 2's recorded tables. Every R²/RMSE digit must come from the fresh run. Confirm the two contradictions from the spec are gone: the file must nowhere say "90/100", "`--iters 1000` → 90/100", or field R² "0.998".

Run: `grep -n '90/100\|0\.998' examples/naca0012_openfoam/README.md`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add examples/naca0012_openfoam/README.md
git commit -m "docs(examples): consolidate the NACA 0012 reference into the tracked README"
```

---

### Task 4: Shrink LOCAL_SETUP.md and tidy the tree

**Files:**
- Modify: `examples/naca0012_openfoam/LOCAL_SETUP.md` (gitignored, full replacement below)
- Delete (on disk, all untracked/ignored): `design-artifacts/`, `field_autoencoder_rmse10/`, `field_autoencoder_tuned/`, `field_energy_9999/`, three `.DS_Store` files

**Interfaces:**
- Consumes: Task 3's committed README (LOCAL_SETUP now points there instead of duplicating).
- Produces: nothing downstream. **This task creates no commit** — nothing tracked changes.

- [ ] **Step 1: Replace `examples/naca0012_openfoam/LOCAL_SETUP.md` wholesale with:**

```markdown
# Lokales OpenFOAM-Setup (diese Maschine)

> Maschinenspezifische Notiz für den **echten** CFD-Lauf (`examples/naca0012_openfoam/`).
> Alles Fachliche — Case-Setup, DoE, verifizierte Befehle und Ergebnisse — steht im
> getrackten [`README.md`](README.md); hier stehen nur die lokalen Pfade dieser Maschine.

## Installation

| Was | Wert |
| --- | --- |
| Installationsverzeichnis | `/Users/georg/Software/openfoam` |
| Source-Skript (`etc/bashrc`) | `/Users/georg/Software/openfoam/etc/bashrc` |
| Version | OpenFOAM **v2512** (api=2512, patch=0) |
| Build-Plattform | `darwin64ClangDPInt32Opt` (macOS/arm64) |
| Binaries | `simpleFoam`, `foamDictionary`, `foamToVTK` — gebaut & vorhanden |
| airFoil2D-Tutorial | `/Users/georg/Software/openfoam/tutorials/incompressible/simpleFoam/airFoil2D` |

Das ist exakt die **„Verified (OpenFOAM v2512, macOS/arm64)"**-Umgebung aus dem README.

## Umgebung sourcen

```bash
source /Users/georg/Software/openfoam/etc/bashrc
```

Prüfen, dass es geklappt hat:

```bash
command -v simpleFoam            # -> .../platforms/darwin64ClangDPInt32Opt/bin/simpleFoam
echo "$FOAM_TUTORIALS"           # -> .../openfoam/tutorials
```

Danach die Befehle aus dem README („Run it" bzw. „Verified") vom Repo-Root ausführen.
Auf dieser Maschine lohnt `--jobs 16` (18 Kerne; die Fälle sind unabhängig und
single-threaded → nahezu linearer Speed-up).
```

- [ ] **Step 2: Delete the stray dirs and Finder droppings**

All four root dirs were verified on 2026-07-05 to contain 0 bytes and no files (only empty subdirectories); git does not even see them. The `.DS_Store` files are gitignored.

```bash
rm -rf design-artifacts field_autoencoder_rmse10 field_autoencoder_tuned field_energy_9999
find examples/naca0012_openfoam -name .DS_Store -delete
```

**Do NOT touch:** `examples/naca0012_openfoam/run/`, `examples/naca0012_openfoam/ex09_real/`, `scratch/` (real user data), or any `__pycache__`.

- [ ] **Step 3: Verify**

Run: `ls | grep -E 'field_|design-artifacts' ; find examples/naca0012_openfoam -name .DS_Store ; git status --porcelain`
Expected: no output at all (dirs gone, no .DS_Store, working tree clean — confirming nothing tracked was touched).

---

### Task 5: Regression — full quality gates + example 09

**Files:** none touched.

**Interfaces:**
- Consumes: everything committed in Tasks 1 and 3.
- Produces: the final green light. No commit.

- [ ] **Step 1: Run the full CI script**

Run: `scripts/ci.sh` (600000 ms timeout — it runs ruff, mypy, pyright, import-linter, the full pytest suite, and the reproducibility check; it finds `.venv` on its own).
Expected: every **required** gate reports success in the summary; pytest ends `513 passed, 6 skipped` (exact counts may drift slightly — the criterion is **0 failed**). The refactor touched only `examples/` and docs, so nothing here should move.

- [ ] **Step 2: Run the synthetic top-level example 09**

Run: `.venv/bin/python examples/09_naca0012_openfoam_surrogate.py`
Expected: runs to completion, exit 0 (it fabricates its own sweep; no OpenFOAM needed). This is the spec's explicit "ex09 untouched" check.

- [ ] **Step 3: Final report**

Summarize for the user: the two commits (hashes + messages), files changed per commit, the acceptance-run fit tables, and the ci.sh summary line. **Do not push** — the user reviews first.
