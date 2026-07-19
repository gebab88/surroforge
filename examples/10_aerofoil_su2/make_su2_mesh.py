#!/usr/bin/env python3
"""Generate a native SU2 C-grid around a NACA 6412 (chord 1, closed TE).

Same geometry and topology as example 09's ``make_blockmeshdict.py`` — exact NACA
4-digit profile (camber 6 % at 40 %, thickness 12 %), chord 1 along the x-axis, a
semicircular far-field arc of radius R centred on the trailing edge and a wake block
of W chords — but written directly in the ASCII ``.su2`` format (2-D, quads), with
two markers: ``airfoil`` (the wall) and ``farfield`` (arc + wake top/bottom + outlet).

One deliberate difference to example 09: the first wall spacing is y+ <= 1
(FIRST_WALL 2e-5 instead of 8e-4), because SU2's SA model integrates to the wall by
default — no wall functions, unlike the ``nutUSpaldingWallFunction`` OpenFOAM setup.

The grid is one structured C-mapping: the wrap direction runs outlet -> lower wake ->
TE -> lower surface -> LE -> upper surface -> TE -> upper wake -> outlet; the normal
direction runs wall/wake-centreline -> far field with geometric clustering. Duplicate
points (wake centreline, TE, LE) are merged so the wake line is interior.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

CAMBER_M = 0.06  # max camber (fraction of chord); 0 -> symmetric
CAMBER_P = 0.4   # position of max camber (fraction of chord)
R = 12.0         # far-field radius (chords), semicircle centred on the TE
W = 20.0         # wake length behind the TE (chords)
XSPLIT = 0.3     # chordwise anchor of the arc quarter-points (as in example 09)
N_FORE = 40      # cells LE -> XSPLIT (chordwise ratio 6: fine at the LE)
N_AFT = 56       # cells XSPLIT -> TE (chordwise ratio 0.25: fine at the TE)
N_NORMAL = 88    # cells wall -> far field
N_WAKE = 56      # cells TE -> outlet (first cell 0.005 at the TE)
FIRST_WALL = 2e-5   # first cell height at the wall (y+ <= 1 at 20..100 m/s, nu 1e-4)
FIRST_WAKE = 0.005  # first wake cell length at the TE


def halfthickness(x: float) -> float:
    """Closed-trailing-edge NACA 4-digit half thickness, 12 % (last coefficient -0.1036)."""
    return 0.6 * (
        0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x**2
        + 0.2843 * x**3 - 0.1036 * x**4
    )


def camber(x: float) -> tuple[float, float]:
    """NACA 4-digit camber line y_c and slope dy_c/dx at chord position x."""
    m, pp = CAMBER_M, CAMBER_P
    if m == 0.0:
        return 0.0, 0.0
    if x < pp:
        return m / pp**2 * (2 * pp * x - x**2), 2 * m / pp**2 * (pp - x)
    return (
        m / (1 - pp) ** 2 * ((1 - 2 * pp) + 2 * pp * x - x**2),
        2 * m / (1 - pp) ** 2 * (pp - x),
    )


def surface_point(x: float, upper: bool) -> tuple[float, float]:
    """Surface point at chord position x, thickness applied perpendicular to the camber line."""
    yt = halfthickness(x)
    yc, dyc = camber(x)
    theta = math.atan(dyc)
    if upper:
        return x - yt * math.sin(theta), yc + yt * math.cos(theta)
    return x + yt * math.sin(theta), yc - yt * math.cos(theta)


def solve_ratio(length: float, first: float, n: int) -> float:
    """Cell-to-cell ratio r so that n geometric cells starting at `first` span `length`."""
    lo, hi = 1.0 + 1e-9, 3.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        total = first * (mid**n - 1.0) / (mid - 1.0)
        if total < length:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def graded_fractions(n: int, total_ratio: float) -> list[float]:
    """n+1 cumulative fractions in [0, 1] with geometric cell sizes, last/first = total_ratio."""
    if abs(total_ratio - 1.0) < 1e-12:
        return [i / n for i in range(n + 1)]
    r = total_ratio ** (1.0 / (n - 1))
    sizes = [r**k for k in range(n)]
    total = sum(sizes)
    fractions = [0.0]
    acc = 0.0
    for s in sizes:
        acc += s
        fractions.append(acc / total)
    fractions[-1] = 1.0
    return fractions


def first_graded_fractions(length: float, first: float, n: int) -> list[float]:
    """n+1 cumulative fractions with geometric sizes fixed by the first cell length."""
    r = solve_ratio(length, first, n)
    sizes = [first * r**k for k in range(n)]
    total = sum(sizes)
    fractions = [0.0]
    acc = 0.0
    for s in sizes:
        acc += s
        fractions.append(acc / total)
    fractions[-1] = 1.0
    return fractions


def build_grid() -> tuple[list[list[tuple[float, float]]], int, int]:
    """Structured (wrap x normal) point grid; returns (grid, n_surface_stations, n_wrap)."""
    # Chordwise stations per side: LE -> split (ratio 6) + split -> TE (ratio 0.25),
    # exactly the example-09 blockMesh gradings (fine at both the LE and the TE).
    fore = graded_fractions(N_FORE, 6.0)
    aft = graded_fractions(N_AFT, 0.25)
    xs_side = [XSPLIT * f for f in fore] + [XSPLIT + (1.0 - XSPLIT) * f for f in aft[1:]]

    # Arc angles mirror the blockMesh block corners (TE 270 deg -> split 225 -> LE 180 ->
    # split 135 -> TE 90); within each segment the angle follows the station's chordwise
    # position, so the arc clustering tracks the wall clustering (parallel-edge grading).
    def theta_of(x: float, upper: bool) -> float:
        if upper:
            if x <= XSPLIT:
                return 180.0 + (135.0 - 180.0) * (x / XSPLIT)
            return 135.0 + (90.0 - 135.0) * ((x - XSPLIT) / (1.0 - XSPLIT))
        if x <= XSPLIT:
            return 225.0 + (180.0 - 225.0) * ((XSPLIT - x) / XSPLIT)
        return 270.0 + (225.0 - 270.0) * ((1.0 - x) / (1.0 - XSPLIT))

    # Wrap order along the surface: TE -> lower -> LE -> upper -> TE (the graded
    # fractions run LE->TE per side, so the lower side is traversed reversed).
    surface: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for x in xs_side[::-1]:
        rad = math.radians(theta_of(x, upper=False))
        surface.append((surface_point(x, upper=False),
                        (1.0 + R * math.cos(rad), R * math.sin(rad))))
    for x in xs_side[1:]:
        rad = math.radians(theta_of(x, upper=True))
        surface.append((surface_point(x, upper=True),
                        (1.0 + R * math.cos(rad), R * math.sin(rad))))
    # Exact closed TE / shared LE so the duplicate-point merge is exact.
    surface[0] = ((1.0, 0.0), (1.0, -R))
    surface[-1] = ((1.0, 0.0), (1.0, R))

    # Wake stations TE -> outlet, geometric from FIRST_WAKE at the TE.
    wake_fr = first_graded_fractions(W, FIRST_WAKE, N_WAKE)
    wake_xs = [1.0 + W * f for f in wake_fr]

    # Wrap station list: (inner point, outer point) in wrap order.
    stations: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for x in wake_xs[::-1]:                      # outlet -> TE, fanning down
        stations.append(((x, 0.0), (x, -R)))
    stations.extend(surface[1:-1])               # TE -> LE -> TE (ends live in the wake lists)
    stations[len(wake_xs) - 1] = surface[0]      # TE with the 270-deg outer point
    stations.append(surface[-1])                 # TE again, 90-deg outer point
    for x in wake_xs[1:]:                        # TE -> outlet, fanning up
        stations.append(((x, 0.0), (x, R)))

    # Normal distribution per station: geometric from FIRST_WALL along the straight
    # line inner -> outer (station lengths differ, so the ratio is solved per station).
    grid: list[list[tuple[float, float]]] = []
    for (ix, iy), (ox, oy) in stations:
        length = math.hypot(ox - ix, oy - iy)
        fractions = first_graded_fractions(length, FIRST_WALL, N_NORMAL)
        grid.append([(ix + (ox - ix) * f, iy + (oy - iy) * f) for f in fractions])
    n_surface = 2 * (N_FORE + N_AFT)             # surface cells along the wrap
    return grid, n_surface, len(stations) - 1


def build_mesh() -> tuple[str, dict[str, float]]:
    """Assemble the merged unstructured mesh and return (su2_text, stats)."""
    grid, n_surface, n_wrap = build_grid()
    n_normal = len(grid[0]) - 1

    ids: dict[tuple[float, float], int] = {}
    points: list[tuple[float, float]] = []
    grid_ids: list[list[int]] = []
    for column in grid:
        col_ids = []
        for x, y in column:
            key = (round(x, 10), round(y, 10))
            pid = ids.get(key)
            if pid is None:
                pid = len(points)
                ids[key] = pid
                points.append((x, y))
            col_ids.append(pid)
        grid_ids.append(col_ids)

    quads: list[tuple[int, int, int, int]] = []
    min_area = float("inf")
    for i in range(n_wrap):
        a, b = grid_ids[i], grid_ids[i + 1]
        for j in range(n_normal):
            quad = (a[j], b[j], b[j + 1], a[j + 1])
            quads.append(quad)
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = (points[p] for p in quad)
            area = 0.5 * (
                (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
                + (x2 - x0) * (y3 - y0) - (x3 - x0) * (y2 - y0)
            )
            min_area = min(min_area, area)
    if min_area <= 0.0:
        raise RuntimeError(f"mesh has non-positive cell areas (min {min_area:.3g})")

    i_te_lower = N_WAKE                      # wrap station of the TE (lower j-line)
    airfoil_edges = [
        (grid_ids[i][0], grid_ids[i + 1][0])
        for i in range(i_te_lower, i_te_lower + n_surface)
    ]
    farfield_edges = [(grid_ids[i][-1], grid_ids[i + 1][-1]) for i in range(n_wrap)]
    farfield_edges += [
        (grid_ids[0][j], grid_ids[0][j + 1]) for j in range(n_normal)
    ] + [
        (grid_ids[-1][j], grid_ids[-1][j + 1]) for j in range(n_normal)
    ]

    lines = ["NDIME= 2", f"NELEM= {len(quads)}"]
    lines += [f"9 {a} {b} {c} {d} {k}" for k, (a, b, c, d) in enumerate(quads)]
    lines.append(f"NPOIN= {len(points)}")
    lines += [f"{x:.16g} {y:.16g} {k}" for k, (x, y) in enumerate(points)]
    lines.append("NMARK= 2")
    lines.append("MARKER_TAG= airfoil")
    lines.append(f"MARKER_ELEMS= {len(airfoil_edges)}")
    lines += [f"3 {a} {b}" for a, b in airfoil_edges]
    lines.append("MARKER_TAG= farfield")
    lines.append(f"MARKER_ELEMS= {len(farfield_edges)}")
    lines += [f"3 {a} {b}" for a, b in farfield_edges]

    stats = {
        "points": len(points),
        "cells": len(quads),
        "min_area": min_area,
        "wall_ratio": solve_ratio(R - 1.0, FIRST_WALL, N_NORMAL),
        "wake_ratio": solve_ratio(W, FIRST_WAKE, N_WAKE),
    }
    return "\n".join(lines) + "\n", stats


def write_mesh(path: Path) -> Path:
    text, stats = build_mesh()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}: {stats['points']} points, {stats['cells']} quads "
          f"(min cell area {stats['min_area']:.3g})")
    print(f"wall-normal cell ratio ~{stats['wall_ratio']:.3f} (first {FIRST_WALL:g}), "
          f"wake ratio ~{stats['wake_ratio']:.3f} (first {FIRST_WAKE:g})")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="mesh_naca6412.su2",
                        help="output mesh file (native ASCII .su2)")
    args = parser.parse_args()
    write_mesh(Path(args.out))


if __name__ == "__main__":
    main()
