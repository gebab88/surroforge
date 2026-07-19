# NACA 6412 — real SU2 sweep → SurroForge surrogate

This example runs **actual** steady-state incompressible RANS with SU2, once per
angle-of-attack/speed combination, and trains the SurroForge surrogates on the genuine
output. It is the SU2 twin of `examples/09_aerofoil_openfoam/`: the **same case** (NACA
6412, chord 1, `nu = 1e-4`, angle × speed DoE) and the **same downstream surrogate code**
— only the solver changes, SU2 instead of OpenFOAM.

> ⚠️ **Not run in CI.** This needs a real SU2 install, so it is not part of the
> smoke-tested examples. The scripts were authored and **verified against SU2 v8.5.0
> "Harrier"** (see *Verified* below).

## Files

- **`make_su2_mesh.py`** — generates the NACA 6412 C-grid directly in the native ASCII
  `.su2` format (no external mesher). Same geometry and topology as example 09's
  `make_blockmeshdict.py`; `run_sweep.py` calls it once per sweep.
- **`run_sweep.py`** — writes one SU2 config per DoE point and solves it with `SU2_CFD`,
  optionally many cases in parallel (`--jobs`).
- **`ingest_and_train.py`** — reads the converged point fields and force coefficients,
  trains and compares the surrogates, exports the models, and writes the surrogate fields
  for ParaView. No solver needed — it consumes what `run_sweep.py` produced.
- **`surrogate_vtk.py`** — standalone VTK/VTKHDF I/O helper used by `ingest_and_train.py`:
  it reads the per-case `flow.vtu` that SU2 wrote and writes one ParaView-readable VTKHDF
  time-series file of surrogate **point** fields. No SurroForge imports; it also parses
  SU2's raw-appended VTU encoding (which trips up `meshio`).

## What it does

- **Mesh & physics:** a generated NACA 6412 C-grid (chord 1, closed trailing edge, camber
  6 % at 40 %, thickness 12 %; ~26.8k quad cells, ~27.1k nodes), incompressible RANS
  (`INC_RANS`) with the Spalart–Allmaras turbulence model. The first wall spacing gives
  y⁺ ≤ 1, because SU2's SA model **integrates to the wall** by default (no wall functions
  — unlike example 09's `nutUSpaldingWallFunction`).
- **2-D sweep (angle × speed):** the mesh is fixed; each angle of attack is imposed by
  **rotating the freestream velocity** `INC_VELOCITY_INIT = |U|·(cos α, sin α, 0)`, and
  the freestream speed `|U|` is varied independently — a full DoE over every
  `angle × speed` combination on the one mesh (no re-meshing). SU2 derives the angle of
  attack from that velocity vector itself (`α = atan(v_y/v_x)`) and projects the forces
  into the rotated lift/drag directions automatically — the exact SU2 analogue of example
  09's rotated `U` plus rotated `forceCoeffs` directions.
- **Outputs per case:** converged pressure and velocity **point** fields `Pressure`/`Velocity`
  (read from `flow.vtu`, which SU2 writes natively — no separate conversion step) and the
  integrated lift and drag coefficients `CL`/`CD` written every iteration to `history.csv`.
- **Surrogates** (two field models plus the coefficients): a **field** model
  (`reduced-field` POD + neural inner) `(AoA, speed) → pressure field`, a second field
  model `(AoA, speed) → velocity field (Ux, Uy)` — SU2 pads the 2-D velocity with an
  exact-zero z component, so it is dropped at ingest and re-padded as exact zero on export
  — and a **scalar** model `(AoA, speed) → [Cl, Cd]`. `mlp` always runs; the PyTorch
  `torch-mlp` backend joins if `surroforge_professional[torch]` is installed.

## Run it

```bash
# 1. Put SU2_CFD on PATH (so run_sweep.py finds the solver)
export PATH="$PATH:/path/to/SU2/install/bin"    # e.g. /Users/<you>/Software/SU2/install/bin

# 2. Sweep angle of attack × speed (one SU2_CFD run per combination). The mesh is
#    generated once into <out>/mesh_naca6412.su2 if it is not already there.
python examples/10_aerofoil_su2/run_sweep.py --angles "-4 0 4 8 12" --speeds "20 40 60" --jobs 8

# 3. Ingest the fields + history.csv and train the surrogates
python examples/10_aerofoil_su2/ingest_and_train.py run
#    ...or save the whole run (SurroForge project + artifacts) for later inspection:
python examples/10_aerofoil_su2/ingest_and_train.py run --out ./ex10_real
```

The VTKHDF export needs `h5py` (`pip install 'surroforge[formats]'` or
`pip install 'surroforge[all]'`).

Note (editions): the neural-network candidates (`mlp`, `torch-mlp`) are Professional
methods — `pip install 'surroforge_professional[torch]'` and set `SURROFORGE_PRO_LICENSE`.
The polynomial candidates run on Core alone.

`ingest_and_train.py` also **exports** the trained surrogates (model export needs the
Professional license already required for the NN candidates): the best coefficient model to
`coeffs_surrogate.onnx`, the torch-mlp coefficient model to `coeffs_surrogate.pt` (a native
PyTorch checkpoint, when the plugin is installed), and the pressure- and velocity-field
surrogates to `field_surrogate.pkl` / `velocity_surrogate.pkl` (`reduced-field` is
pickle-only). It then evaluates the best field surrogates on every original case in
`samples.csv` and writes VTKHDF output to `surrogate_vtk/surrogate_fields.vtkhdf`, plus a
`samples.csv` mapping file for the time-step to case/AoA/speed assignment. Open the
`.vtkhdf` file in ParaView; the surrogate pressure field is written as point data `p` and
the surrogate velocity as vector point data `U` (magnitude colouring, glyphs and
streamlines work; `Uz` is identically zero in this 2-D case).

By default these outputs land in the sweep directory and the SurroForge project (datasets,
runs, lineage, POD basis, models) is built in a temp dir and discarded; pass **`--out DIR`**
to save the whole run instead — the project to `DIR/naca` plus the artifacts and
`DIR/surrogate_vtk/` — so it stays inspectable and reproducible afterwards
(`surroforge --config-dir DIR/naca list`).

`run_sweep.py` options: `--angles` (deg) and `--speeds` (|U| m/s), each space/comma-separated
— a 2-D DoE is run over every angle × speed; `--iters` (steady-state *cap*, default 2000),
`--residual` (residual-convergence target, default `1e-8`), `--jobs` (cases solved
concurrently), `--mesh` (an explicit `.su2` mesh; defaults to generating
`<out>/mesh_naca6412.su2`), `--out` (sweep dir, default `run`).

## Case setup in detail

What `run_sweep.py` writes into every case's `case.cfg` (an incompressible variant of SU2's
validated `naca0012` RANS test case):

- **Geometry & mesh:** NACA 6412, chord `c = 1`, closed trailing edge, the generated C-grid
  from `make_su2_mesh.py` (~26 752 quad cells, ~27 088 nodes; 2-D). Two markers: `airfoil`
  (the wall) and `farfield` (arc + wake top/bottom + downstream outlet). The mesh is
  **fixed** — the angle of attack is imposed by rotating the freestream, never by re-meshing.
- **Physics & solver:** `SU2_CFD` — steady incompressible RANS with the Spalart–Allmaras
  model. `rho = 1`, `mu = 1e-4` ⇒ `nu = 1e-4 m²/s`. Chord Reynolds number
  `Re = |U|·c/nu`: **2·10⁵ (20 m/s) … 1·10⁶ (100 m/s)**. `INC_NONDIM = DIMENSIONAL`, so
  `flow.vtu` carries dimensional pressure `[Pa]` and velocity `[m/s]` that scale with the
  freestream speed — the field the surrogate must learn (matching example 09's dimensional
  OpenFOAM fields; `INITIAL_VALUES` would normalise each case by its own reference values
  and erase that speed dependence).
- **Boundary conditions:**

  | Marker | SU2 type |
  | --- | --- |
  | `airfoil` | `MARKER_HEATFLUX` (adiabatic no-slip wall) |
  | `farfield` | `MARKER_FAR` (freestream, `INC_VELOCITY_INIT`) |

  `MARKER_MONITORING`/`MARKER_PLOTTING` are the airfoil, so `CL`/`CD` and the surface
  fields are integrated on it.
- **Force coefficients:** `REF_AREA = 1`, `REF_LENGTH = 1` (unit chord × unit span),
  `REF_ORIGIN_MOMENT = (0.25, 0, 0)`. SU2 references the coefficients to the freestream
  dynamic pressure `0.5·rho·|U|²` — the analogue of `forceCoeffs`' `rhoInf`/`magUInf` —
  so `CL`/`CD` are dimensionless and comparable across speeds. `history.csv` records them
  every iteration (the data behind the settledness check below).
- **Numerics:** `FDS` convective flux, `MUSCL` reconstruction with the `VENKATAKRISHNAN`
  slope limiter (kept active — freezing or dropping it destabilises this stretched C-grid),
  `EULER_IMPLICIT` time discretisation, `WEIGHTED_LEAST_SQUARES` gradients, FGMRES/ILU
  linear solver, CFL 10 with adaptation. `CONV_FIELD` is pinned to the RMS residuals of
  pressure, x-momentum and nu-tilde with `CONV_RESIDUAL_MINVAL = log10(--residual)` so SU2
  stops itself once every residual is below the target; `ITER` is the `--iters` cap.

## Convergence is checked, not assumed

A surrogate is only as trustworthy as the CFD under it, so both stages guard against
half-converged data instead of silently using the last iteration:

- **Solver level (`run_sweep.py`):** `--residual` pins SU2's residual criterion, so
  `SU2_CFD` stops itself once the RMS residuals are all below it — `--iters` is only the
  cap. Each case reports the iteration it converged at (or "NOT converged" if it hit the
  cap), and a final summary lists every case that did not.
- **Quantity level (`ingest_and_train.py`):** even a residual-plateaued run can leave a
  marginally unsteady force, so `read_coeffs` checks the **peak-to-peak spread of `CL` over
  the last 20 iterations**; any case whose lift is still swinging is flagged before training.

On this stretched C-grid the RMS residuals typically settle into a small limit cycle
around `1e-4…1e-5` (high-aspect trailing-edge/wake cells + the slope limiter) rather than
falling all the way to `1e-8`, so **most cases run to the `--iters` cap and report "NOT
converged"** by the strict residual test — even though their lift has plateaued to within
~`1e-4`. This is the SU2 analogue of example 09's post-stall situation: the quantity-level
`CL` settledness check, not the residual verdict, is the authority on whether a case's data
is usable. (Raise `--iters` if a case's `CL` is genuinely still swinging.)

## Reproducing a full sweep (SU2 v8.5.0)

The scripts were authored and exercised end-to-end against real `SU2_CFD` (v8.5.0
"Harrier", macOS/arm64). A representative full DoE:

| Parameter | Values |
| --- | --- |
| Angle of attack | −4° … +14° in 2° steps (10 angles) |
| Freestream speed \|U\| | 20, 40, 60, 80, 100 m/s (5 speeds) |
| Cases | 10 × 5 = **50** (`case_000` … `case_049`) |

Run it from the repo root with `SU2_CFD` on PATH:

```bash
python examples/10_aerofoil_su2/run_sweep.py \
  --angles "-4 -2 0 2 4 6 8 10 12 14" \
  --speeds "20 40 60 80 100" \
  --iters 2000 --jobs 16 --out examples/10_aerofoil_su2/run

python examples/10_aerofoil_su2/ingest_and_train.py \
  examples/10_aerofoil_su2/run --out examples/10_aerofoil_su2/ex10_naca6412_su2
```

Ingestion reads the pressure and velocity point fields from every `flow.vtu` (the 2-D
velocity's exact-zero z column is dropped at ingest and re-padded on export) plus the real
`CL`/`CD` from each `history.csv`, trains the three surrogates, exports the four model
artifacts (`coeffs_surrogate.onnx`/`.pt`, `field_surrogate.pkl`, `velocity_surrogate.pkl`),
and writes a VTKHDF pressure+velocity time series (one step per case) under
`surrogate_vtk/surrogate_fields.vtkhdf` for ParaView.

What to expect on this mesh (qualitatively — measure your own numbers): most cases run to
the `--iters` cap because the RMS residuals settle into a `~1e-4…1e-5` limit cycle rather
than reaching `1e-8`, while the `CL` settledness check (see above) is the authority on
usability; and the trained polar is physically sensible for a cambered NACA 6412 — nonzero
lift at zero incidence, a lift peak near the stall onset, and a Reynolds-number trend in
`CL` across speeds. Report the actual fit metrics from your own run
(`surroforge --config-dir <out>/naca list`); widen the DoE (finer angles, more speeds) to
firm up the field fits.

**On-disk layout the scripts produce** (all gitignored):

```
examples/10_aerofoil_su2/
├─ run/                       # raw CFD, 50 cases
│  ├─ case_000 … case_049/    #   each: case.cfg, flow.vtu, history.csv, log.SU2_CFD
│  ├─ mesh_naca6412.su2       #   the C-grid the sweep generated (~2 MB)
│  └─ samples.csv             #   DoE table (case, aoa, speed)
└─ ex10_naca6412_su2/         # saved SurroForge run + artifacts (--out)
   ├─ coeffs_surrogate.onnx / .pt
   ├─ field_surrogate.pkl
   ├─ velocity_surrogate.pkl
   ├─ surrogate_vtk/          #   surrogate_fields.vtkhdf + samples.csv
   └─ naca/                   #   project: runs/, basis/, data/, surroforge.yaml
```

## How this maps onto SurroForge

`ingest_and_train.py` is deliberately thin: `read_su2_sweep(run, doe)` turns the sweep into
the wide snapshot table the engine trains on (reading `Pressure`/`Velocity` from each
`flow.vtu` via `surrogate_vtk.read_point_field`), `read_coeffs(...)` parses `CL`/`CD` from
`history.csv`, and the rest is ordinary `Trainer.from_dict(...)` — the same API as every
other example. The ParaView export lives in `surrogate_vtk.py`, which knows nothing about
SurroForge. Change the airfoil, add parameters beyond angle and speed, or point the reader
at a different field, and the flow is unchanged.

## Caveats

- `REF_LENGTH`/`REF_AREA` are the unit chord (this NACA 6412 has chord 1, span 1); adjust
  for other geometry. The exact `CL` magnitude depends on these and on convergence.
- Convergence is checked automatically (see above), but near stall (high AoA) cases still
  need more iterations — raise `--iters` for any case the settledness check flags.
- The SA model integrates to the wall (y⁺ ≤ 1), so the generated mesh is deliberately finer
  at the wall than example 09's wall-function mesh; do not reuse example 09's spacing here.
