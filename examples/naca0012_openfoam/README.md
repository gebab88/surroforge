# NACA 0012 — real OpenFOAM sweep → SurroForge surrogate

The **real-CFD companion** to [`examples/09_naca0012_openfoam_surrogate.py`](../09_naca0012_openfoam_surrogate.py).

Example 09 *fabricates* a representative sweep so it runs in CI with no solver. This directory
runs **actual** steady-state RANS with OpenFOAM, once per angle of attack, and trains the
SurroForge surrogates on the genuine output. The downstream surrogate code is identical — only
the data source changes.

> ⚠️ **Not run in CI.** This needs a real OpenFOAM install, so it is not part of the smoke-tested
> examples. The scripts were authored and **verified against OpenFOAM v2512** (see *Verified* below).

## What it does

- **Mesh & physics:** the validated `airFoil2D` NACA 0012 C-mesh that ships with OpenFOAM
  (`$FOAM_TUTORIALS/incompressible/simpleFoam/airFoil2D`, ~10.7k cells), incompressible
  `simpleFoam` with the Spalart–Allmaras turbulence model.
- **Angle-of-attack sweep:** the mesh is fixed; each AoA is imposed by **rotating the freestream
  velocity** `U = |U|·(cos α, sin α, 0)` and the `forceCoeffs` lift/drag directions — standard
  practice that avoids re-meshing.
- **Outputs per case:** a converged pressure field `p` (read mesh-free by
  `surroforge.data.read_openfoam_sweep`) and the integrated lift coefficient `Cl` written by
  the `forceCoeffs` function object.
- **Surrogates** (same two as example 09): a **field** model (`reduced-field` POD + neural inner)
  AoA → pressure field, and a **scalar** model AoA → `Cl`. Core `mlp` always; the PyTorch
  `torch-mlp` backend joins if the `surroforge-torch` plugin is installed.

## Run it

```bash
# 1. Source your OpenFOAM environment (so simpleFoam and $FOAM_TUTORIALS are available)
source /path/to/OpenFOAM/etc/bashrc          # e.g. /Users/<you>/Software/openfoam/etc/bashrc

# 2. Sweep angle of attack (one simpleFoam run per angle)
python examples/naca0012_openfoam/run_sweep.py --angles "-6 -3 0 3 6 9 12 14 16 18" --iters 500

# 3. Ingest the fields + forceCoeffs and train the surrogates
python examples/naca0012_openfoam/ingest_and_train.py run
```

`ingest_and_train.py` also **exports** the trained surrogates into the sweep directory: the best
`Cl` model to `cl_surrogate.onnx` (and `cl_surrogate.pt`, a native PyTorch checkpoint, when
`torch-mlp` won), and the field surrogate to `field_surrogate.pkl` (`reduced-field` is pickle-only).

`run_sweep.py` options: `--angles` (deg, space/comma-separated), `--speed` (|U|, default 25.95 m/s),
`--iters` (steady-state *cap*, default 500), `--residual` (residual-convergence tolerance, default
`1e-8`), `--out` (sweep dir, default `run`), `--base` (an explicit `airFoil2D` case; defaults to
`$FOAM_TUTORIALS`).

## Convergence is checked, not assumed

A surrogate is only as trustworthy as the CFD under it, so both stages guard against
half-converged data instead of silently using the last iteration:

- **Solver level (`run_sweep.py`):** `--residual` pins OpenFOAM's `SIMPLE/residualControl`, so
  `simpleFoam` stops itself once every residual is below it — `--iters` is only the cap. Each case
  reports the iteration it converged at (or "NOT converged" if it hit the cap), and a final summary
  lists every case that did not converge.
- **Quantity level (`ingest_and_train.py`):** even a residual-converged run can leave a marginally
  unsteady force, so `read_cl` checks the **peak-to-peak spread of `Cl` over the last 20 writes**;
  any case whose lift is still swinging is flagged before training.

The two agree in practice — e.g. a near-stall +16° case both hits the iteration cap *and* shows a
still-swinging `Cl`, so it is reported by both stages.

## Verified (OpenFOAM v2512, macOS/arm64)

The pipeline was run end-to-end here (sweeps over [-6°, 18°], `simpleFoam` with `forceCoeffs`):

- Cases ran to completion; ingestion read **10 720 pressure DOFs/case + the real `Cl`** from
  `forceCoeffs`, and both surrogates trained.
- **Scalar `Cl` surrogate** (the lift curve, with stall) is robust and the network clearly wins —
  at 30 angles: `torch-mlp` R² ≈ **0.998**, `mlp` ≈ 0.99, vs. polynomial ≈ 0.96 (and at a coarse 10
  angles the gap is starker: NN ≈ 0.97–0.99 vs. poly ≈ 0.38). This is the headline real-CFD result.
- The **field** POD-NN surrogate is far more demanding on data and convergence. On a coarse,
  lightly-converged sweep (30 angles, 200 iters) it is **data-starved**: POD has few snapshots of a
  10.7k-DOF field and the neural inner *overfits* the modal coefficients (negative held-out R²; even
  the polynomial inner reaches only ≈ 0.4). For a production field model use a **denser,
  well-converged** sweep (more angles, `--iters 500`+). Example 09 — 61 clean synthetic angles —
  shows the converged-data ideal (neural field R² ≈ 1.0).

Takeaway: the **scalar quantity-of-interest surrogate (AoA → `Cl`) is the easy, robust win on real
CFD**; the full-field surrogate is worth it when you have a dense, converged sweep.

## How this maps onto SurroForge

`ingest_and_train.py` is deliberately thin: `read_openfoam_sweep(run, field="p", …)` turns the
sweep into the wide snapshot table the engine trains on, `read_cl(...)` parses `forceCoeffs`, and
the rest is ordinary `Trainer.from_dict(...)` — the same API as every other example. Point
`read_openfoam_sweep` at *any* OpenFOAM case sweep (different airfoil, more parameters than AoA,
a velocity field instead of pressure) and the surrogate flow is unchanged.

## Caveats

- `lRef`/`Aref` are the unit chord (the airFoil2D NACA 0012 has chord 1); adjust for other geometry.
  The exact `Cl` magnitude depends on these and on convergence.
- Convergence is checked automatically (see above), but near stall (high AoA) cases still need more
  iterations — raise `--iters` (or loosen `--residual`) for any case the sweep flags as unconverged.
