"""Modal-basis canonicalization + degeneracy handling (Story R2.2).

A POD/SVD basis carries gauge ambiguity: each mode's *sign* is free, equal-energy
modes may be *reordered*, and a near-degenerate cluster (equal singular values) may
be arbitrarily *rotated within its own subspace*. None of this changes the
reconstructed field ``Ŷ = C·Vᵀ``, but it changes the basis/coefficient bytes — so a
naive comparison false-diverges. These pure-numeric primitives pin the removable
ambiguity (sign, order) and detect the irremovable one (degenerate rotation), so the
field-primary reproducibility verdict (Story R2.3) never reports a spurious
``diverged``.

Design is frozen by ``_bmad-output/implementation-artifacts/spec-field-rom-fingerprint.md``
and proven on synthetic degenerate spectra in ``spikes/field_fingerprint_spike.py``:

- **Sign** by the max-abs basis entry; **order** by singular value (coefficient
  column norm) descending, tie-broken by peak-node index. Removes sign + reorder.
- **Degeneracy** (equal-σ clusters) is compared as a *subspace* via principal angles,
  **never per-mode** — per-mode canonicalization false-diverges on an intra-subspace
  rotation, which the field verdict (R2.3) and this subspace test both survive.

numpy only — no other SurroForge imports, no scipy, no new dependency.
"""

from __future__ import annotations

import numpy as np

#: Relative tolerance for grouping singular values into a degenerate cluster.
CLUSTER_RTOL = 1e-6
#: Principal-angle tolerance (radians) below which two subspaces are "the same".
#: A microradian is far below any physically meaningful subspace rotation, yet well
#: above the QR/SVD conditioning noise (~1e-8 rad) of the angle computation itself.
ANGLE_TOL = 1e-6


def _mode_signs(basis: np.ndarray) -> np.ndarray:
    """Return a ±1 vector that makes each mode's max-abs basis entry positive."""
    signs = np.ones(basis.shape[1], dtype=float)
    for k in range(basis.shape[1]):
        peak = int(np.argmax(np.abs(basis[:, k])))
        if basis[peak, k] < 0:
            signs[k] = -1.0
    return signs


def canonicalize_basis_signs(basis: np.ndarray) -> np.ndarray:
    """Fix each mode's sign so its max-abs entry is positive (#16); order untouched.

    This is the sign gauge a reducer pins on its already-σ-ordered basis so the basis
    is deterministic run-to-run. Returns a fresh array.
    """
    basis = np.array(basis, dtype=float, copy=True)
    basis *= _mode_signs(basis)
    return basis


def canonicalize(
    coeffs: np.ndarray, basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sign/order-canonicalize a coefficient/basis pair under one shared gauge.

    Fixes each mode's sign (max-abs basis entry positive) and orders modes by σ
    (coefficient column norm) descending, tie-broken by peak-node index. The same
    sign+permutation gauge is applied to ``coeffs`` and ``basis`` together, so the
    reconstructed field ``coeffs·basisᵀ`` is invariant.

    Undoes sign flips and mode reordering; it **cannot** undo an intra-subspace
    rotation of a degenerate cluster — that case is handled by :func:`degenerate_clusters`
    + a subspace comparison, never per-mode.

    Args:
        coeffs: ``(n_samples, n_modes)`` modal coefficients.
        basis: ``(n_dof, n_modes)`` spatial modes (columns).

    Returns:
        ``(coeffs_canon, basis_canon)`` — fresh arrays in canonical gauge.

    Raises:
        ValueError: if ``coeffs`` or ``basis`` is not 2-D, or their mode (column)
            counts differ (the shared sign+permutation gauge requires one mode count).
    """
    coeffs = np.array(coeffs, dtype=float, copy=True)
    basis = np.array(basis, dtype=float, copy=True)
    if coeffs.ndim != 2 or basis.ndim != 2:
        raise ValueError("canonicalize expects 2-D coeffs and basis")
    if coeffs.shape[1] != basis.shape[1]:
        raise ValueError(
            "coeffs and basis must share a mode count; got "
            f"{coeffs.shape[1]} coefficient columns and {basis.shape[1]} basis columns"
        )
    signs = _mode_signs(basis)
    basis *= signs
    coeffs *= signs
    sigma = np.linalg.norm(coeffs, axis=0)
    keys = [
        (-round(float(sigma[k]), 9), int(np.argmax(np.abs(basis[:, k]))))
        for k in range(basis.shape[1])
    ]
    order = sorted(range(basis.shape[1]), key=lambda k: keys[k])
    return coeffs[:, order], basis[:, order]


def degenerate_clusters(
    coeffs: np.ndarray, rtol: float = CLUSTER_RTOL
) -> tuple[list[list[int]], np.ndarray]:
    """Group modes whose singular value (coefficient column norm) is ~equal.

    Args:
        coeffs: ``(n_samples, n_modes)`` modal coefficients.
        rtol: relative tolerance for "equal" singular values.

    Returns:
        ``(clusters, sigma)`` where ``clusters`` is a list of index groups with more
        than one member (single modes are not degenerate), ordered/grouped by σ
        descending, and ``sigma`` is the per-mode singular-value vector.
    """
    sigma = np.linalg.norm(coeffs, axis=0)
    if sigma.size == 0:
        return [], sigma
    order = np.argsort(-sigma)
    clusters: list[list[int]] = []
    cur = [int(order[0])]
    for k in order[1:]:
        k = int(k)
        if abs(sigma[k] - sigma[cur[-1]]) <= rtol * max(sigma[cur[-1]], 1.0):
            cur.append(k)
        else:
            clusters.append(cur)
            cur = [k]
    clusters.append(cur)
    return [c for c in clusters if len(c) > 1], sigma


def max_principal_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Largest principal angle (radians) between the column-spans of ``a`` and ``b``.

    Zero iff the two matrices span the same subspace. Computed via QR orthonormalization
    then the SVD of ``Qaᵀ·Qb`` (whose singular values are the cosines of the principal
    angles); the result is clipped into ``[-1, 1]`` before ``arccos`` for FP safety.

    Both matrices must be ``(n_dof, k)`` with the **same** ``k`` columns living in the
    **same** ambient space, and ``k ≤ n_dof`` — otherwise the reduced QR basis would no
    longer span the columns and the angle would be silently understated (a false
    "reproduced"). Two zero-column subspaces are vacuously identical (angle ``0``).

    Raises:
        ValueError: if the inputs are not 2-D, have unequal column counts, live in
            different ambient dimensions, or have more columns than rows.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("max_principal_angle expects 2-D column matrices")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"subspaces must have equal column counts; got {a.shape[1]} and {b.shape[1]}"
        )
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "subspaces must share an ambient dimension; got "
            f"{a.shape[0]} and {b.shape[0]} rows"
        )
    if a.shape[1] == 0:
        return 0.0  # two 0-dimensional subspaces are vacuously identical
    if a.shape[1] > a.shape[0]:
        raise ValueError(
            f"more columns ({a.shape[1]}) than rows ({a.shape[0]}): the columns cannot "
            "form a basis, so reduced QR would understate the principal angle"
        )
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    cos_angles = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1.0, 1.0)
    return float(np.arccos(cos_angles.min()))


def subspace_reproduces(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    cluster: list[int],
    *,
    angle_tol: float = ANGLE_TOL,
) -> bool:
    """Whether two bases span the same degenerate-cluster subspace (within ``angle_tol``).

    Compares the cluster columns as a subspace (principal angle), not per-mode, so a
    benign intra-subspace rotation reads as reproduced. Callers needing the angle for
    diagnostics can call :func:`max_principal_angle` directly.
    """
    return max_principal_angle(basis_a[:, cluster], basis_b[:, cluster]) <= angle_tol
