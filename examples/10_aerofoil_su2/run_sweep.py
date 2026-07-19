#!/usr/bin/env python3
"""Drive a real SU2 NACA 6412 angle-of-attack sweep with the incompressible RANS solver.

This runs actual steady-state RANS (Spalart-Allmaras) on a generated NACA 6412 C-grid,
once per angle-of-attack/speed combination, and lays the results out exactly as
``ingest_and_train.py`` expects — the same physics as example 09 (chord 1, rho 1,
nu 1e-4, AoA imposed by rotating the freestream), solved with SU2 instead of OpenFOAM.

The mesh is fixed and generated once per sweep (``make_su2_mesh.py``); angle of attack
is imposed by **rotating the freestream velocity** ``INC_VELOCITY_INIT``. SU2 derives
the angle of attack from that vector itself (``atan(vy/vx)``) and projects the forces
into rotated lift/drag directions automatically — the exact SU2 analogue of example
09's rotated ``U`` plus rotated ``forceCoeffs`` directions.

Prerequisites — ``SU2_CFD`` on PATH (this script does not run in CI; it needs a real
SU2 install; verified against SU2 v8.5.0):

    python examples/10_aerofoil_su2/run_sweep.py --angles "0 4 8 12" --speeds "20 40" --jobs 8
    python examples/10_aerofoil_su2/ingest_and_train.py run

The cases are independent, so ``--jobs N`` solves N of them concurrently (each SU2_CFD
run is single-threaded here — near-linear speed-up up to your core count). Each case
directory ends up with ``case.cfg``, ``log.SU2_CFD``, ``history.csv`` (``CL``/``CD``
every iteration) and ``flow.vtu`` (the converged fields — SU2 writes VTK natively, so
there is no separate ``--vtk`` step as in example 09).

Convergence is checked, not assumed:
  * ``--residual`` (default ``1e-8``) sets SU2's residual criterion
    (``CONV_RESIDUAL_MINVAL = log10(residual)`` on pressure, x-momentum and nu-tilde) so
    a case that reaches it stops itself — ``--iters`` is only the *cap*.
  * After each case we scan ``log.SU2_CFD`` for "All convergence criteria satisfied.";
    a case that hit the iteration cap without converging is reported, and a final
    summary lists every such case so a half-converged sweep can't masquerade as clean.

Notes / honest caveats:
  * On this stretched C-grid the RMS residuals typically plateau around 1e-4..1e-5 in a
    small limit cycle (high-aspect trailing-edge/wake cells + slope limiter) while the
    force coefficients settle to within ~1e-4 by ~2000 iterations — so most cases will
    report "NOT converged" by the strict residual test and simply run to the cap. That
    is the same situation as example 09's post-stall cases: the quantity-level check in
    ``ingest_and_train.py`` (peak-to-peak C_L spread over the last 20 iterations) is the
    authority on whether a case's data is actually usable.
  * ``REF_AREA``/``REF_LENGTH`` are the unit chord; adjust if your geometry differs.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from make_su2_mesh import write_mesh

# --------------------------------------------------------------------------- #
# Templated SU2 configuration (only the velocity vector, the residual target and
# the iteration cap change per case; numerics follow SU2's validated
# incompressible NACA 0012 RANS test case, with the slope limiter kept active —
# freezing or dropping it destabilises this stretched C-grid).
# --------------------------------------------------------------------------- #

_CONFIG = """\
% NACA 6412 incompressible RANS (SA) — AoA {angle:.6g} deg, |U| {speed:.6g} m/s, nu 1e-4
SOLVER= INC_RANS
KIND_TURB_MODEL= SA
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
TIME_MARCHING= NO

% rho = 1, mu = 1e-4  ->  nu = 1e-4 m^2/s (Re = 2e5 .. 1e6 over 20..100 m/s), as in
% example 09. DIMENSIONAL output so flow.vtu carries dimensional p [Pa] and U [m/s]
% that scale with the freestream speed — the field the surrogate must learn, matching
% example 09's dimensional OpenFOAM fields (INITIAL_VALUES would write each case
% normalised by its own reference values, erasing that speed dependence). CL/CD are
% dimensionless either way: SU2 uses the freestream dynamic pressure 0.5*rho*|U|^2 as
% the force reference — the analogue of forceCoeffs' rhoInf/magUInf.
INC_DENSITY_INIT= 1.0
INC_VELOCITY_INIT= ( {ux:.8g} {uy:.8g} 0.0 )
INC_NONDIM= DIMENSIONAL
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= 1e-4
FREESTREAM_NU_FACTOR= 3.0

REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.00
REF_ORIGIN_MOMENT_Z= 0.00
REF_LENGTH= 1.0
REF_AREA= 1.0

MARKER_HEATFLUX= ( airfoil, 0.0 )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( airfoil )
MARKER_MONITORING= ( airfoil )

NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 10.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.8, 1.1, 1.0, 100.0 )
ITER= {iters}

VENKAT_LIMITER_COEFF= 0.01
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 15

CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
TIME_DISCRE_FLOW= EULER_IMPLICIT

CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
TIME_DISCRE_TURB= EULER_IMPLICIT

CONV_FIELD= ( RMS_PRESSURE, RMS_VELOCITY-X, RMS_NU_TILDE )
CONV_RESIDUAL_MINVAL= {minval:.6g}
CONV_STARTITER= 10

MESH_FILENAME= {mesh}
MESH_FORMAT= SU2
TABULAR_FORMAT= CSV
CONV_FILENAME= history
VOLUME_FILENAME= flow
OUTPUT_FILES= ( PARAVIEW )
OUTPUT_WRT_FREQ= 1000000
HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )
SCREEN_OUTPUT= ( INNER_ITER, RMS_PRESSURE, RMS_VELOCITY-X, RMS_NU_TILDE, LIFT, DRAG )
SCREEN_WRT_FREQ_INNER= 100
"""


def _find_su2() -> str:
    su2 = shutil.which("SU2_CFD")
    if su2 is None:
        sys.exit("SU2_CFD not on PATH — add your SU2 install's bin directory to PATH "
                 "(e.g. export PATH=\"$PATH:/path/to/SU2/install/bin\").")
    return su2


def _build_case(case: Path, mesh: Path, angle_deg: float, speed: float,
                iters: int, residual: float) -> None:
    alpha = math.radians(angle_deg)
    case.mkdir(parents=True, exist_ok=True)
    (case / "case.cfg").write_text(_CONFIG.format(
        angle=angle_deg, speed=speed,
        ux=speed * math.cos(alpha), uy=speed * math.sin(alpha),
        iters=iters, minval=math.log10(residual), mesh=mesh.resolve(),
    ))


def _run_su2(su2: str, case: Path) -> int | None:
    """Run SU2_CFD; return the iteration it converged at, or ``None`` if it hit the cap.

    Convergence is SU2's own verdict: it prints "All convergence criteria satisfied."
    (and stops) once every ``CONV_FIELD`` residual is below ``CONV_RESIDUAL_MINVAL``.
    """
    log_path = case / "log.SU2_CFD"
    with log_path.open("w") as log:
        result = subprocess.run([su2, "case.cfg"], cwd=case,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.exit(f"SU2_CFD failed in {case} (see {log_path})")
    if "All convergence criteria satisfied." not in log_path.read_text():
        return None
    last = (case / "history.csv").read_text().splitlines()[-1]
    return int(float(last.split(",")[2]))  # Inner_Iter column


def _process_case(su2: str, out: Path, mesh: Path, i: int, angle: float, speed: float,
                  iters: int, residual: float) -> int | None:
    """Build and solve one case (its own directory — independent, so safe in parallel)."""
    case = out / f"case_{i:03d}"
    if case.exists():
        shutil.rmtree(case)
    _build_case(case, mesh, angle, speed, iters, residual)
    return _run_su2(su2, case)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles", default="-4 0 4 8 12 16",
                        help="angles of attack in degrees, space- or comma-separated")
    parser.add_argument("--speeds", default="25.95",
                        help="freestream speeds |U| [m/s], space- or comma-separated "
                             "(a 2-D DoE is run over every angle × speed)")
    parser.add_argument("--iters", type=int, default=3000, help="steady-state iteration cap")
    parser.add_argument("--residual", type=float, default=1e-8,
                        help="residual convergence target (CONV_RESIDUAL_MINVAL = its log10; "
                             "the solver stops once met — see the module docstring for why "
                             "most cases run to the cap instead)")
    parser.add_argument("--mesh", default=None,
                        help=".su2 mesh to use (default: <out>/mesh_naca6412.su2, "
                             "generated by make_su2_mesh.py if missing)")
    parser.add_argument("--out", default="run", help="output sweep directory")
    parser.add_argument("--jobs", type=int, default=1,
                        help="number of cases to solve concurrently (each SU2_CFD run is "
                             "single-threaded; the cases are independent, so this is a "
                             "near-linear speed-up up to your core count)")
    args = parser.parse_args()

    angles = [float(a) for a in args.angles.replace(",", " ").split()]
    speeds = [float(s) for s in args.speeds.replace(",", " ").split()]
    if not angles:
        sys.exit('No angles given — pass e.g. --angles "-4 0 4 8 12 16".')
    if not speeds:
        sys.exit('No speeds given — pass e.g. --speeds "20 30 40".')
    if args.jobs < 1:
        sys.exit("--jobs must be >= 1.")
    if not 0 < args.residual < 1:
        sys.exit("--residual must be in (0, 1).")
    su2 = _find_su2()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.mesh is not None:
        mesh = Path(args.mesh)
        if not mesh.is_file():
            sys.exit(f"--mesh not found: {mesh}")
    else:
        mesh = out / "mesh_naca6412.su2"
        if not mesh.is_file():
            write_mesh(mesh)

    doe = [(a, s) for s in speeds for a in angles]  # 2-D grid, angle-major within each speed

    # Each case is an independent SU2_CFD run in its own directory, so the sweep is
    # embarrassingly parallel: --jobs cases solve concurrently. Results are printed as
    # they finish (any order); samples.csv is then written in DoE order.
    results: dict[int, int | None] = {}
    failed: list[int] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(_process_case, su2, out, mesh, i, a, s, args.iters, args.residual): i
            for i, (a, s) in enumerate(doe)
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            i = futures[fut]
            angle, speed = doe[i]
            label = f"[{done}/{len(doe)}] AoA {angle:+.1f}° U {speed:.4g} -> case_{i:03d}"
            try:
                conv = fut.result()
            except SystemExit as exc:  # _run_su2 aborts a failed case in its worker
                failed.append(i)
                print(f"{label} ... FAILED ({exc})", flush=True)
                continue
            results[i] = conv
            status = (f"converged at iter {conv}" if conv is not None
                      else f"NOT converged (hit {args.iters}-iter cap)")
            print(f"{label} ... {status}", flush=True)

    rows = ["case,aoa,speed"]
    unconverged: list[tuple[float, float]] = []
    for i, (angle, speed) in enumerate(doe):
        if i in failed:
            continue
        rows.append(f"case_{i:03d},{angle},{speed}")
        if results[i] is None:
            unconverged.append((angle, speed))
    (out / "samples.csv").write_text("\n".join(rows) + "\n")

    print(f"\nDone. {len(doe)} cases under {out}/ ({len(angles)} angles × {len(speeds)} speeds)",
          end="")
    if unconverged:
        listed = ", ".join(f"({a:+.1f}°, {s:.4g})" for a, s in unconverged)
        print(f"\n⚠  {len(unconverged)} case(s) did NOT reach the {args.residual:g} residual "
              f"target: {listed}\n"
              f"   Expected on this mesh (see docstring) — ingest_and_train.py checks whether "
              f"their C_L actually settled.", end="")
    if failed:
        listed = ", ".join(f"({doe[i][0]:+.1f}°, {doe[i][1]:.4g})" for i in failed)
        print(f"\n✗  {len(failed)} case(s) FAILED to run and were excluded: {listed}", end="")
    print(f"\n— now run:\n    python examples/10_aerofoil_su2/ingest_and_train.py {out}")


if __name__ == "__main__":
    main()
