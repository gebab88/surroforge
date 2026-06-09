"""Modal-basis canonicalization + degeneracy handling (Story R2.2).

These tests are the production version of the three claims proven on synthetic data
in ``spikes/field_fingerprint_spike.py`` (the R2.1 de-risk spike):

  1. Canonicalization is deterministic and fixes sign + order.
  2. A sign-flipped/reordered *equivalent* basis canonicalizes identically — a naive
     byte comparison would false-diverge.
  3. An intra-subspace rotation of a *degenerate* cluster leaves the field invariant;
     per-mode canonicalization FALSE-diverges, while the principal-angle subspace test
     correctly reads "reproduced".

Tolerances (1e-12 byte-equality, 1e-6 rad principal angle) sit well above ULP/QR
noise, so the asserts hold on any BLAS without a reproducibility guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from surroforge.core.canonicalize import (
    canonicalize,
    canonicalize_basis_signs,
    degenerate_clusters,
    max_principal_angle,
    subspace_reproduces,
)

# A spectrum with a DEGENERATE pair (the two 3.0s), exactly as the spike.
SIGMA = np.array([10.0, 7.0, 5.0, 3.0, 3.0, 1.0])
N_DOF, M_HOLDOUT = 200, 40
REL_TOL = 1e-9


def make_scenario(seed: int = 7):
    """Orthonormal basis V (n_dof × r) and coefficients C (m × r) scaled to SIGMA.

    Column k of C has norm SIGMA[k], so a rotation within the equal-σ cluster
    preserves each mode's norm exactly (a genuine degeneracy, not an approximation).
    """
    rng = np.random.default_rng(seed)
    r = SIGMA.size
    basis, _ = np.linalg.qr(rng.standard_normal((N_DOF, r)))
    frame, _ = np.linalg.qr(rng.standard_normal((M_HOLDOUT, r)))
    coeffs = frame * SIGMA
    return coeffs, basis


def signed_permutation(r: int, swaps, flips) -> np.ndarray:
    """An exact orthogonal gauge: column swaps + sign flips (V@G, C@G keep V·Cᵀ)."""
    g = np.eye(r)
    for i, j in swaps:
        g[:, [i, j]] = g[:, [j, i]]
    for k in flips:
        g[:, k] *= -1.0
    return g


def cluster_rotation(r: int, cluster, theta: float) -> np.ndarray:
    """A Givens rotation inside a degenerate cluster (orthogonal; field-invariant)."""
    g = np.eye(r)
    i, j = cluster
    c, s = np.cos(theta), np.sin(theta)
    g[i, i], g[i, j], g[j, i], g[j, j] = c, -s, s, c
    return g


def subspace_rotation(r: int, cluster, seed: int = 3) -> np.ndarray:
    """An arbitrary orthogonal mixing acting ONLY within ``cluster`` (field-invariant).

    Embeds a random ``k×k`` orthogonal block (``k = len(cluster)``) into the cluster
    indices, so a 3+-mode degenerate subspace is genuinely rotated, not just a Givens
    pair. The block is orthogonal and the cluster's coefficient columns are orthogonal
    with equal norm, so every rotated column keeps that norm — the degeneracy survives.
    """
    rng = np.random.default_rng(seed)
    block = list(cluster)
    q, _ = np.linalg.qr(rng.standard_normal((len(block), len(block))))
    g = np.eye(r)
    g[np.ix_(block, block)] = q
    return g


def reconstruct(coeffs, basis):
    return coeffs @ basis.T


def rel_diff(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))


# --- Claim 1: deterministic sign/order canonical form -----------------------

def test_canonicalize_is_deterministic():
    coeffs, basis = make_scenario()
    c1, v1 = canonicalize(coeffs, basis)
    c2, v2 = canonicalize(coeffs, basis)
    assert np.array_equal(c1, c2)
    assert np.array_equal(v1, v2)


def test_canonicalize_fixes_sign_and_order():
    coeffs, basis = make_scenario()
    c, v = canonicalize(coeffs, basis)
    # Each mode's max-abs basis entry is positive.
    for k in range(v.shape[1]):
        peak = int(np.argmax(np.abs(v[:, k])))
        assert v[peak, k] > 0
    # Modes are ordered by singular value (coefficient column norm) descending.
    sigma = np.linalg.norm(c, axis=0)
    assert np.all(np.diff(sigma) <= 1e-9)


def test_canonicalize_preserves_the_field():
    coeffs, basis = make_scenario()
    c, v = canonicalize(coeffs, basis)
    assert np.allclose(reconstruct(c, v), reconstruct(coeffs, basis), atol=1e-12)


def test_canonicalize_basis_signs_matches_full_canonicalize_signs():
    # The sign-only primitive (used by the reducer) agrees with the full canonicalize
    # on the sign gauge, and leaves an already-σ-ordered basis in place.
    _coeffs, basis = make_scenario()
    signed = canonicalize_basis_signs(basis)
    for k in range(signed.shape[1]):
        peak = int(np.argmax(np.abs(signed[:, k])))
        assert signed[peak, k] > 0
    # Sign-only: column count and ordering unchanged, magnitudes identical.
    assert signed.shape == basis.shape
    assert np.allclose(np.abs(signed), np.abs(basis))


# --- Claim 2: sign-flipped/reordered equivalent basis canonicalizes identically

def test_regauged_basis_canonicalizes_identically():
    coeffs, basis = make_scenario()
    g = signed_permutation(SIGMA.size, swaps=[(0, 2), (3, 4)], flips=[1, 4])
    child_c, child_v = coeffs @ g, basis @ g

    # Same physics: the reconstructed field is unchanged by the gauge.
    assert np.allclose(reconstruct(child_c, child_v), reconstruct(coeffs, basis))

    pc, pv = canonicalize(coeffs, basis)
    cc, cv = canonicalize(child_c, child_v)
    assert np.allclose(pc, cc, atol=1e-12)
    assert np.allclose(pv, cv, atol=1e-12)

    # A naive byte/raw comparison of the re-gauged basis WOULD false-diverge —
    # which is exactly why canonicalization earns its keep.
    assert not np.allclose(child_v, basis)


# --- Claim 3: degenerate-cluster rotation — the canonicalization trap --------

def test_degenerate_clusters_detects_the_equal_sigma_pair():
    coeffs, _basis = make_scenario()
    clusters, sigma = degenerate_clusters(coeffs)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2
    # The detected pair are the two σ≈3 modes.
    assert all(abs(sigma[k] - 3.0) < 1e-9 for k in clusters[0])


def test_degenerate_clusters_empty_when_no_degeneracy():
    rng = np.random.default_rng(1)
    frame, _ = np.linalg.qr(rng.standard_normal((M_HOLDOUT, 4)))
    coeffs = frame * np.array([8.0, 5.0, 2.0, 1.0])  # all distinct
    clusters, _sigma = degenerate_clusters(coeffs)
    assert clusters == []


def test_subspace_reproduces_but_per_mode_canonicalization_diverges():
    coeffs, basis = make_scenario()
    clusters, _sigma = degenerate_clusters(coeffs)
    cluster = clusters[0]  # the σ=3 degenerate pair
    g = cluster_rotation(SIGMA.size, cluster, theta=np.radians(35.0))
    child_c, child_v = coeffs @ g, basis @ g

    # The rotation is intra-subspace: the reconstructed field is invariant.
    assert np.allclose(reconstruct(child_c, child_v), reconstruct(coeffs, basis))

    # Ground-truth-aligned: the degenerate subspace is unchanged (principal angle ~0).
    assert subspace_reproduces(basis, child_v, cluster)
    assert max_principal_angle(basis[:, cluster], child_v[:, cluster]) < 1e-6

    # The TRAP: per-mode canonicalized coefficients false-diverge (a rotation cannot
    # be undone by sign/order canonicalization) — proving the subspace test is needed.
    pc, _ = canonicalize(coeffs, basis)
    cc, _ = canonicalize(child_c, child_v)
    assert rel_diff(pc, cc) > 1e-3


def test_max_principal_angle_zero_for_identical_subspace():
    # arccos near 1 amplifies the ~1e-16 cosine rounding of Qaᵀ·Qb to a ~1e-8 rad floor
    # (the noise ANGLE_TOL=1e-6 is designed to clear), so assert against that tolerance —
    # a <1e-9 bound passed on macOS/OpenBLAS but is unachievable on Windows (saw 2.98e-8).
    _coeffs, basis = make_scenario()
    assert max_principal_angle(basis[:, [3, 4]], basis[:, [3, 4]]) < 1e-6


def test_distinct_subspaces_do_not_reproduce():
    # Negative anchor: genuinely different subspaces must give a LARGE principal angle
    # and read as NOT reproduced. Without this, a silent always-~0 bug (e.g. a transpose
    # slip in qaᵀ·qb) would pass every other assertion in this file.
    _coeffs, basis = make_scenario()
    # Columns of an orthonormal basis: [0,1] and [2,3] span orthogonal subspaces.
    assert max_principal_angle(basis[:, [0, 1]], basis[:, [2, 3]]) > 1.0  # ≈ π/2

    cluster = [0, 1]
    child = basis.copy()
    child[:, cluster] = basis[:, [2, 3]]  # cluster block now spans a different subspace
    assert not subspace_reproduces(basis, child, cluster)


def test_subspace_reproduces_three_mode_degenerate_cluster():
    # A THREE-fold degenerate cluster (σ = 5 ×3), rotated by an arbitrary 3×3 orthogonal
    # mixing — the subspace path must still read "reproduced" where per-mode diverges.
    sigma = np.array([10.0, 5.0, 5.0, 5.0, 1.0])
    rng = np.random.default_rng(11)
    basis, _ = np.linalg.qr(rng.standard_normal((N_DOF, sigma.size)))
    frame, _ = np.linalg.qr(rng.standard_normal((M_HOLDOUT, sigma.size)))
    coeffs = frame * sigma

    clusters, detected = degenerate_clusters(coeffs)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert len(cluster) == 3
    assert all(abs(detected[k] - 5.0) < 1e-9 for k in cluster)

    g = subspace_rotation(sigma.size, cluster)
    child_c, child_v = coeffs @ g, basis @ g

    # Genuinely intra-subspace: field invariant, and the child still sees the triple.
    assert np.allclose(reconstruct(child_c, child_v), reconstruct(coeffs, basis))
    child_clusters, _ = degenerate_clusters(child_c)
    assert len(child_clusters) == 1 and len(child_clusters[0]) == 3

    # Subspace path: reproduced. Per-mode path: false-diverges.
    assert subspace_reproduces(basis, child_v, cluster)
    assert max_principal_angle(basis[:, cluster], child_v[:, cluster]) < 1e-6
    pc, _ = canonicalize(coeffs, basis)
    cc, _ = canonicalize(child_c, child_v)
    assert rel_diff(pc, cc) > 1e-3


def test_subspace_reproduces_empty_cluster_is_vacuously_true():
    # An empty cluster is two 0-dimensional subspaces: vacuously identical, no crash.
    _coeffs, basis = make_scenario()
    assert max_principal_angle(basis[:, []], basis[:, []]) == 0.0
    assert subspace_reproduces(basis, basis, [])


def test_canonicalize_raises_on_mode_count_mismatch():
    coeffs, basis = make_scenario()
    with pytest.raises(ValueError, match="mode count"):
        canonicalize(coeffs[:, :-1], basis)  # 5 coeff cols vs 6 basis cols


def test_max_principal_angle_raises_on_column_mismatch():
    _coeffs, basis = make_scenario()
    with pytest.raises(ValueError, match="equal column counts"):
        max_principal_angle(basis[:, [0, 1]], basis[:, [0, 1, 2]])


def test_max_principal_angle_raises_on_ambient_mismatch():
    _coeffs, basis = make_scenario()
    with pytest.raises(ValueError, match="ambient dimension"):
        max_principal_angle(basis[:, [0, 1]], basis[:50, [0, 1]])


def test_max_principal_angle_raises_when_more_columns_than_rows():
    rng = np.random.default_rng(5)
    wide = rng.standard_normal((2, 3))  # 3 columns, 2 rows
    with pytest.raises(ValueError, match="more columns"):
        max_principal_angle(wide, wide)
