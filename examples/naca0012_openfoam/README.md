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
2-D DoE is run over every angle × speed; `--iters` (steady-state *cap*, default 1000), `--residual`
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

In this verified run the two are **complementary, not identical**: the near/post-stall cases that
hit the iteration cap nonetheless show a *settled* `Cl` (the lift plateaus even while the last
residuals sit just above `1e-8`), so the quantity-level check passes them. Residual non-convergence
did **not**, here, mean an unsteady force — which is also why the surrogate fit barely changes
whether those cases are included or not.

## Verified (OpenFOAM v2512, macOS/arm64)

The 2-D pipeline was run end-to-end on real `simpleFoam`. A **100-case** DoE (25 angles −12…12°,
4 speeds 20/40/60/80 m/s, 1° steps) solved in a couple of minutes at `--jobs 16` (the post-stall
cases now run the full 1000-iter cap); ingestion read **10 720 pressure DOFs/case + the real `Cl`
and `Cd`** from `forceCoeffs`, both surrogates trained, and all three artifacts exported
(`coeffs_surrogate.onnx`/`.pt`, `field_surrogate.pkl`).

**Convergence — and why the iteration cap defaults to 1000.** At the default `--residual 1e-8` the
attached cases reach convergence only at **~520–670 iterations**, so the old `--iters 500` default
clipped most of them *just short* of the target and falsely flagged them "not converged" (only
37/100 passed). With **`--iters 1000` (the current default) 90/100 converge**; the 10 that do not
are the near/post-stall extremes (|AoA| ≳ 11–12°), which steady RANS cannot converge regardless.
Crucially, the *fit quality barely changes* between the two caps — the clipped cases were already
physically settled (their `Cl` had plateaued), so "not converged by residual" did **not** mean "bad
training data" here.

**Fit quality (verified numbers, degree-2 polynomial vs. `mlp` vs. `torch-mlp`, 0.25 held-out test):**

- **Scalar `(AoA, speed) → [Cl, Cd]`** — the **polynomial has the highest R²** (≈ 0.993 on the full
  sweep, ≈ 0.996 on the 90 converged cases); `mlp` is close (≈ 0.975 / 0.990) and wins the example's
  RMSE-based "best" pick — in *real* CFD `Cl` and `Cd` sit on different scales, so the RMSE-best and
  R²-best methods can differ. The `torch-mlp` backend *underfits* this 2-output regression here
  (R² ≈ 0.50): same hyperparameters as `mlp`, but its full-batch training does not fit on so few
  points without tuning.
- **Why the polynomial is hard to beat here (the honest caveat):** steady RANS **cannot converge the
  post-stall cases**, so the *converged* data is essentially **attached and low-order** (≈linear `Cl`,
  ≈quadratic `Cd`) — exactly what a quadratic fits best. A neural net's across-stall advantage — which
  the synthetic example 09 deliberately *can* show — **cannot be demonstrated on converged steady-RANS
  data**, because the nonlinear post-stall region is precisely the part that will not converge. Use a
  transient solver there, or keep the DoE in the attached regime.
- **Field `(AoA, speed) → pressure field` (POD-NN):** the `torch-mlp` inner fits the pressure field
  superbly (**R² ≈ 0.998**); the `polynomial` inner ≈ 0.97 and the scikit-learn `mlp` inner is weak
  here (≈ 0.66) — the reverse of the scalar, so on this data neither NN backend dominates both tasks.

Takeaway: the **scalar coefficient surrogate `(AoA, speed) → [Cl, Cd]` is the robust, cheap win**, and
on converged steady-RANS data a **degree-2 polynomial is a strong baseline** — the post-stall regime
where a network pulls ahead is exactly what does not converge. The **field** POD-NN fits the pressure
field very well with a `torch-mlp` inner. Coverage and convergence matter more than raw angular
resolution.

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
