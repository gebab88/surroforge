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

# 2. Sweep angle of attack × speed (one simpleFoam run per combination)
python examples/naca0012_openfoam/run_sweep.py --angles "-6 0 6 12 16 18" --speeds "20 40 60"

# 3. Ingest the fields + forceCoeffs and train the surrogates
python examples/naca0012_openfoam/ingest_and_train.py run
```

`ingest_and_train.py` also **exports** the trained surrogates into the sweep directory: the best
coefficient model to `coeffs_surrogate.onnx`, the torch-mlp coefficient model to
`coeffs_surrogate.pt` (a native PyTorch checkpoint, when the plugin is installed), and the field
surrogate to `field_surrogate.pkl` (`reduced-field` is pickle-only).

`run_sweep.py` options: `--angles` (deg) and `--speeds` (|U| m/s), each space/comma-separated — a
2-D DoE is run over every angle × speed; `--iters` (steady-state *cap*, default 500), `--residual`
(residual-convergence tolerance, default `1e-8`), `--out` (sweep dir, default `run`), `--base` (an
explicit `airFoil2D` case; defaults to `$FOAM_TUTORIALS`).

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

The two agree in practice — e.g. a near-stall, high-AoA case both hits the iteration cap *and*
shows a still-swinging `Cl`, so it is reported by both stages.

## Verified (OpenFOAM v2512, macOS/arm64)

The 2-D pipeline was run end-to-end on real `simpleFoam`: cases over an `angle × speed` DoE ran to
completion, ingestion read **10 720 pressure DOFs/case + the real `Cl` and `Cd`** from
`forceCoeffs`, both surrogates trained, and all three artifacts exported
(`coeffs_surrogate.onnx`/`.pt`, `field_surrogate.pkl`).

What to expect from the *quality* of the fit (consistent with dense 1-D `Cl` sweeps run earlier,
where the scalar `torch-mlp` reached R² ≈ 0.998 vs. polynomial ≈ 0.96):

- **Scalar coefficient surrogate** `(AoA, speed) → [Cl, Cd]` is the robust real-CFD win — both are
  smooth, comparable-scale coefficients, and the network beats a quadratic across the stall. This is
  the headline result; it needs only a modest DoE.
- The **field** POD-NN surrogate is far more demanding: a 10.7k-DOF field needs a **dense,
  well-converged** sweep (enough `angle × speed` samples, `--iters 500`+), or POD has too few
  snapshots and the inner overfits (negative held-out R²). Example 09 — a clean synthetic DoE —
  shows the converged-data ideal (neural field R² ≈ 1.0).

Takeaway: the **scalar quantity-of-interest surrogate (`(AoA, speed) → [Cl, Cd]`) is the easy,
robust win on real CFD**; the full-field surrogate is worth it when you have a dense, converged
sweep.

## How this maps onto SurroForge

`ingest_and_train.py` is deliberately thin: `read_openfoam_sweep(run, field="p", …)` turns the
sweep into the wide snapshot table the engine trains on, `read_coeffs(...)` parses `Cl`/`Cd` from
`forceCoeffs`, and the rest is ordinary `Trainer.from_dict(...)` — the same API as every other
example. Point `read_openfoam_sweep` at *any* OpenFOAM case sweep (different airfoil, more
parameters than angle and speed, a velocity field instead of pressure) and the flow is unchanged.

## Caveats

- `lRef`/`Aref` are the unit chord (the airFoil2D NACA 0012 has chord 1); adjust for other geometry.
  The exact `Cl` magnitude depends on these and on convergence.
- Convergence is checked automatically (see above), but near stall (high AoA) cases still need more
  iterations — raise `--iters` (or loosen `--residual`) for any case the sweep flags as unconverged.
