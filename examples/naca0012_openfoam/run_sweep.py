#!/usr/bin/env python3
"""Drive a real OpenFOAM NACA 0012 angle-of-attack sweep with simpleFoam.

This is the *real-CFD* companion to ``examples/09_naca0012_openfoam_surrogate.py`` (which
fabricates a representative sweep so it can run in CI without a solver). Here we run actual
steady-state RANS (Spalart-Allmaras) on the validated ``airFoil2D`` mesh that ships with
OpenFOAM, once per angle of attack, and lay the results out exactly as
``surroforge.data.read_openfoam_sweep`` expects — so ``ingest_and_train.py`` then builds the
surrogate from genuine CFD output.

The airfoil mesh is fixed; angle of attack is imposed by **rotating the freestream velocity**
(and the ``forceCoeffs`` lift/drag directions) — standard practice that avoids re-meshing.

Prerequisites — source your OpenFOAM environment first so ``simpleFoam`` and ``$FOAM_TUTORIALS``
are available (this script does not run in CI; it needs a real OpenFOAM install):

    source /path/to/OpenFOAM/etc/bashrc
    python examples/naca0012_openfoam/run_sweep.py --angles "-4 0 4 8 12 16" --speeds "20 40"
    python examples/naca0012_openfoam/ingest_and_train.py run

Convergence is checked, not assumed:
  * ``--residual`` (default ``1e-8``) sets OpenFOAM's ``SIMPLE/residualControl`` so ``simpleFoam``
    stops itself once every residual drops below it — ``--iters`` is only the *cap*.
  * After each case we scan ``log.simpleFoam`` for "SIMPLE solution converged"; a case that hit the
    iteration cap without converging is reported, and a final summary lists every such case so a
    half-converged sweep can't masquerade as a clean one.

Notes / honest caveats:
  * ``lRef``/``Aref`` are set to the unit chord (the airFoil2D NACA 0012 has chord 1); adjust if
    your geometry differs. The exact C_L magnitude depends on these and on convergence.
  * ``--iters`` is the steady-state iteration budget; 500 is a reasonable default for this mesh.
    A converged case stops early; raise it (or loosen ``--residual``) if cases hit the cap.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Templated case files (only U and controlDict change per angle; the mesh,
# transport/turbulence properties and numerics come from the airFoil2D base).
# --------------------------------------------------------------------------- #

_U_FIELD = """\
FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({ux:.6g} {uy:.6g} 0);
boundaryField
{{
    inlet  {{ type freestreamVelocity; freestreamValue $internalField; }}
    outlet {{ type freestreamVelocity; freestreamValue $internalField; }}
    walls  {{ type noSlip; }}
    frontAndBack {{ type empty; }}
}}
"""

_CONTROL_DICT = """\
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {iters};
deltaT          1;
writeControl    timeStep;
writeInterval   {iters};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
runTimeModifiable true;

functions
{{
    forceCoeffs1
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patches         (walls);
        rho             rhoInf;
        rhoInf          1;
        liftDir         ({lift_x:.6g} {lift_y:.6g} 0);
        dragDir         ({drag_x:.6g} {drag_y:.6g} 0);
        CofR            (0.25 0 0);
        pitchAxis       (0 0 1);
        magUInf         {speed:.6g};
        lRef            1;
        Aref            1;
    }}
}}
"""


def _find_base(explicit: str | None) -> Path:
    if explicit:
        base = Path(explicit)
    else:
        tutorials = os.environ.get("FOAM_TUTORIALS")
        if not tutorials:
            sys.exit(
                "FOAM_TUTORIALS is not set — source your OpenFOAM environment first "
                "(. /path/to/OpenFOAM/etc/bashrc), or pass --base <airFoil2D case>."
            )
        base = Path(tutorials) / "incompressible" / "simpleFoam" / "airFoil2D"
    if not (base / "constant" / "polyMesh.orig").is_dir():
        sys.exit(f"Base case not found / has no polyMesh.orig: {base}")
    return base


def _set_residual_control(case: Path, residual: float) -> None:
    """Pin SIMPLE/residualControl, and tighten the linear solvers so it is actually reachable.

    A residualControl target below a field's *linear-solver* tolerance can never be met — the
    residual floors at the solver tolerance and every case would spuriously hit the iteration cap.
    So we also drop each field's linear-solver tolerance to two orders below ``residual``.
    """
    fvsolution = str(case / "system" / "fvSolution")
    solver_tol = residual * 1e-2
    for field in ("p", "U", "nuTilda"):
        for entry, value in (
            (f"SIMPLE/residualControl/{field}", residual),
            (f"solvers/{field}/tolerance", solver_tol),
        ):
            subprocess.run(
                ["foamDictionary", "-entry", entry, "-set", f"{value:.3g}", fvsolution],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )


def _build_case(base: Path, case: Path, angle_deg: float, speed: float,
                iters: int, residual: float) -> None:
    alpha = math.radians(angle_deg)
    case.mkdir(parents=True, exist_ok=True)
    # Reuse the validated mesh, transport/turbulence properties and numerics as-is.
    shutil.copytree(base / "constant" / "polyMesh.orig", case / "constant" / "polyMesh")
    for name in ("transportProperties", "turbulenceProperties"):
        shutil.copy(base / "constant" / name, case / "constant" / name)
    (case / "system").mkdir(exist_ok=True)
    for name in ("fvSchemes", "fvSolution"):
        shutil.copy(base / "system" / name, case / "system" / name)
    _set_residual_control(case, residual)
    # Initial fields: take p/nut/nuTilda from the base; rewrite U with the rotated freestream.
    shutil.copytree(base / "0.orig", case / "0")
    (case / "0" / "U").write_text(
        _U_FIELD.format(ux=speed * math.cos(alpha), uy=speed * math.sin(alpha))
    )
    # controlDict with a forceCoeffs whose lift/drag directions match this angle.
    (case / "system" / "controlDict").write_text(
        _CONTROL_DICT.format(
            iters=iters, speed=speed,
            lift_x=-math.sin(alpha), lift_y=math.cos(alpha),
            drag_x=math.cos(alpha), drag_y=math.sin(alpha),
        )
    )


def _run_simplefoam(case: Path) -> int | None:
    """Run simpleFoam; return the iteration it converged at, or ``None`` if it hit the cap.

    Convergence is OpenFOAM's own verdict: ``simpleFoam`` prints "SIMPLE solution converged in
    <N> iterations" (and stops) once every residual is below ``residualControl``.
    """
    if shutil.which("simpleFoam") is None:
        sys.exit("simpleFoam not on PATH — source your OpenFOAM environment first.")
    log_path = case / "log.simpleFoam"
    with log_path.open("w") as log:
        result = subprocess.run(["simpleFoam"], cwd=case, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.exit(f"simpleFoam failed in {case} (see {log_path})")
    for line in log_path.read_text().splitlines():
        if "solution converged in" in line:
            digits = [w for w in line.split() if w.isdigit()]
            return int(digits[0]) if digits else 0
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles", default="-4 0 4 8 12 16",
                        help="angles of attack in degrees, space- or comma-separated")
    parser.add_argument("--speeds", default="25.95",
                        help="freestream speeds |U| [m/s], space- or comma-separated "
                             "(a 2-D DoE is run over every angle × speed)")
    parser.add_argument("--iters", type=int, default=500, help="steady-state iteration cap")
    parser.add_argument("--residual", type=float, default=1e-8,
                        help="SIMPLE residualControl tolerance (solver stops once met)")
    parser.add_argument("--base", default=None, help="airFoil2D base case (else $FOAM_TUTORIALS)")
    parser.add_argument("--out", default="run", help="output sweep directory")
    args = parser.parse_args()

    angles = [float(a) for a in args.angles.replace(",", " ").split()]
    speeds = [float(s) for s in args.speeds.replace(",", " ").split()]
    if not angles:
        sys.exit('No angles given — pass e.g. --angles "-4 0 4 8 12 16".')
    if not speeds:
        sys.exit('No speeds given — pass e.g. --speeds "20 30 40".')
    doe = [(a, s) for s in speeds for a in angles]  # 2-D grid, angle-major within each speed
    base = _find_base(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = ["case,aoa,speed"]
    unconverged: list[tuple[float, float]] = []
    for i, (angle, speed) in enumerate(doe):
        case = out / f"case_{i:03d}"
        if case.exists():
            shutil.rmtree(case)
        print(f"[{i + 1}/{len(doe)}] AoA {angle:+.1f}° U {speed:.4g} -> {case} ...",
              end="", flush=True)
        _build_case(base, case, angle, speed, args.iters, args.residual)
        converged_at = _run_simplefoam(case)
        if converged_at is None:
            unconverged.append((angle, speed))
            print(f" NOT converged (hit {args.iters}-iter cap)", flush=True)
        else:
            print(f" converged at iter {converged_at}", flush=True)
        rows.append(f"{case.name},{angle},{speed}")
    (out / "samples.csv").write_text("\n".join(rows) + "\n")

    print(f"\nDone. {len(doe)} cases under {out}/ ({len(angles)} angles × {len(speeds)} speeds)",
          end="")
    if unconverged:
        listed = ", ".join(f"({a:+.1f}°, {s:.4g})" for a, s in unconverged)
        print(f"\n⚠  {len(unconverged)} case(s) did NOT converge: {listed}\n"
              f"   Raise --iters or loosen --residual; their coefficients/field may be unreliable.",
              end="")
    print(f"\n— now run:\n    python examples/naca0012_openfoam/ingest_and_train.py {out}")


if __name__ == "__main__":
    main()
