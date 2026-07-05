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

- **Scalar:** `mlp` wins on both metrics — the best R² (0.994) *and* the best RMSE (0.0074), the
  latter being the example's own "best"-pick criterion. The degree-2 `polynomial` is a close
  second on R² (0.987), but its RMSE (0.084) is an order of magnitude worse than `mlp`'s: `Cl`
  and `Cd` sit on very different scales in *real* CFD, which is enough to pull `polynomial`'s
  RMSE and R² ranking apart — a scale artifact for that one method, not a flipped overall winner.
  `torch-mlp` (R² 0.973) trails `mlp` here but is competitive on this 2-output regression, not
  underfitting.
- **Why the polynomial is so strong here (the honest caveat):** steady RANS **cannot converge the
  post-stall cases**, so the *converged* data is essentially **attached and low-order** (≈linear
  `Cl`, ≈quadratic `Cd`) — exactly what a quadratic fits best. A neural net's across-stall
  advantage — which the synthetic example 09 deliberately *can* show — **cannot be demonstrated
  on converged steady-RANS data**, because the nonlinear post-stall region is precisely the part
  that will not converge. Use a transient solver there, or keep the DoE in the attached regime.
- **Field (POD-NN):** the `torch-mlp` inner fits the pressure field best (R² 0.986); the
  `polynomial` inner is clearly behind (R² 0.921), and the scikit-learn `mlp` inner is the
  weakest of the three (R² 0.676) — the reverse of the scalar task, so on this data neither NN
  backend dominates both tasks.

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
