"""Spike (Story R2.1) — de-risk the field-primary reproducibility fingerprint.

Goal: empirically settle ADR-ROM-001 BEFORE the v1 reproduce() contract freezes.
This is an exploratory prototype (not production code); it proves four claims on
synthetic data and prints PASS/FAIL per claim. The findings feed the contract
proposal in `_bmad-output/implementation-artifacts/spec-field-rom-fingerprint.md`
(Stories R2.2–R2.4).

Setup: a synthetic field over n_dof "nodes", reduced to r POD modes with a chosen
singular spectrum that contains a *degenerate* pair (two equal singular values), so
the degenerate-subspace failure mode is real, not hypothetical. A "run" is a holdout
coefficient matrix C (m × r) against a basis V (n_dof × r); the reconstructed field
is Ŷ = C·Vᵀ. Gauge transforms (sign flips, mode reorder, intra-subspace rotation)
are exact orthogonal maps G with V'=V·G, C'=C·G ⇒ Ŷ'=C·G·Gᵀ·Vᵀ=Ŷ: the field is
invariant, the basis/coefficients are not.

Claims:
  1. Under a *bound* basis, the reconstructed-field verdict and the canonicalized-
     coefficient fast-path AGREE (same/different) — so the fast-path is valid.
  2. A sign-flipped/reordered equivalent basis still reproduces under the field
     verdict AND the canonicalized-coefficient fast-path; only the *naive* byte
     comparison false-diverges.
  3. An intra-subspace rotation on degenerate modes makes the per-mode canonicalized
     fast-path FALSE-DIVERGE; the field verdict (and a principal-angle subspace
     comparison of the degenerate cluster) still reproduces. ⇒ field-primary is
     load-bearing, and the basis hash must be diagnostic, not a gate.
"""

from __future__ import annotations

import hashlib

import numpy as np

REL_TOL = 1e-9  # field-energy-relative verdict tolerance (folded into the hashed spec)
SIGMA = np.array([10.0, 7.0, 5.0, 3.0, 3.0, 1.0])  # modes 3,4 are DEGENERATE (equal)
N_DOF, M_HOLDOUT, N_PROBES = 200, 40, 64


# --- scenario ---------------------------------------------------------------

def make_scenario(seed: int = 7):
    rng = np.random.default_rng(seed)
    r = SIGMA.size
    # Orthonormal spatial basis V (n_dof × r) and orthonormal coefficient frame,
    # scaled to the chosen spectrum so per-mode energy == SIGMA (so a rotation
    # within the equal-σ cluster preserves each mode's norm exactly).
    V, _ = np.linalg.qr(rng.standard_normal((N_DOF, r)))
    Q, _ = np.linalg.qr(rng.standard_normal((M_HOLDOUT, r)))
    C = Q * SIGMA  # (m × r), column k has norm SIGMA[k]
    # Probe nodes: energy-weighted pick (top-N by reconstructed-field energy),
    # the simple R2.5 mechanism. Also report coverage.
    node_energy = ((V * SIGMA) ** 2).sum(axis=1)
    probes = np.sort(np.argsort(node_energy)[-N_PROBES:])
    coverage = node_energy[probes].sum() / node_energy.sum()
    return C, V, probes, coverage


def reconstruct(C, V):
    return C @ V.T


# --- gauges (exact orthogonal maps G; V'=V·G, C'=C·G leave Ŷ invariant) ------

def signed_permutation(r, swaps, flips):
    G = np.eye(r)
    for i, j in swaps:
        G[:, [i, j]] = G[:, [j, i]]
    for k in flips:
        G[:, k] *= -1.0
    return G


def cluster_rotation(r, cluster, theta):
    G = np.eye(r)
    i, j = cluster
    c, s = np.cos(theta), np.sin(theta)
    G[i, i], G[i, j], G[j, i], G[j, j] = c, -s, s, c
    return G


def degenerate_clusters(C, rtol=1e-6):
    """Group mode indices whose singular value (column norm) is ~equal."""
    sigma = np.linalg.norm(C, axis=0)
    order = np.argsort(-sigma)
    clusters, cur = [], [order[0]]
    for k in order[1:]:
        if abs(sigma[k] - sigma[cur[-1]]) <= rtol * max(sigma[cur[-1]], 1.0):
            cur.append(k)
        else:
            clusters.append(cur)
            cur = [k]
    clusters.append(cur)
    return [[int(k) for k in c] for c in clusters if len(c) > 1], sigma


# --- canonicalization (#16) + comparators -----------------------------------

def canonicalize(C, V):
    """Sign/order-canonicalize: fix each mode's sign (max-abs entry > 0), order by
    σ desc, tie-break by peak-node index. Undoes sign flips + reordering; CANNOT
    undo an intra-subspace rotation (the degeneracy trap this spike exposes)."""
    r = V.shape[1]
    Vf, Cf = V.copy(), C.copy()
    for k in range(r):
        j = int(np.argmax(np.abs(Vf[:, k])))
        if Vf[j, k] < 0:
            Vf[:, k] *= -1.0
            Cf[:, k] *= -1.0
    sigma = np.linalg.norm(Cf, axis=0)
    keys = [(-round(float(sigma[k]), 9), int(np.argmax(np.abs(Vf[:, k])))) for k in range(r)]
    order = sorted(range(r), key=lambda k: keys[k])
    return Cf[:, order], Vf[:, order]


def rel_diff(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))


def field_verdict(parent, child, probes):
    """Ground truth: reconstructed field at pinned probes, within REL_TOL."""
    pf = reconstruct(*parent)[:, probes]
    cf = reconstruct(*child)[:, probes]
    d = rel_diff(pf, cf)
    return ("reproduced" if d <= REL_TOL else "diverged"), d


def coeff_verdict_canonical(parent, child):
    """Fast/audit lens: compare canonicalized coefficients within REL_TOL."""
    pc, _ = canonicalize(*parent)
    cc, _ = canonicalize(*child)
    d = rel_diff(pc, cc)
    return ("reproduced" if d <= REL_TOL else "diverged"), d


def coeff_verdict_naive(parent, child):
    """The wrong way: raw coefficients + raw basis bytes (no canonicalization)."""
    d = max(rel_diff(parent[0], child[0]), rel_diff(parent[1], child[1]))
    return ("reproduced" if d <= REL_TOL else "diverged"), d


def basis_hash(V):
    return hashlib.sha256(np.ascontiguousarray(V, dtype=np.float64).tobytes()).hexdigest()[:12]


def max_principal_angle(A, B):
    """Largest principal angle (rad) between the column-spans of A and B."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1.0, 1.0)
    return float(np.arccos(s.min()))


# Principal angles via QR+SVD carry ~sqrt(eps) conditioning noise (~1e-8 rad);
# 1e-6 rad (a microradian) is far below any physically meaningful subspace rotation.
def subspace_verdict(parent, child, cluster, angle_tol=1e-6):
    Vp, Vc = parent[1][:, cluster], child[1][:, cluster]
    ang = max_principal_angle(Vp, Vc)
    return ("reproduced" if ang <= angle_tol else "diverged"), ang


# --- experiments ------------------------------------------------------------

def claim1(C, V, probes):
    parent = (C, V)
    same = (C.copy(), V.copy())
    rng = np.random.default_rng(99)
    diff = (C + 0.05 * rng.standard_normal(C.shape), V.copy())  # a genuinely different surrogate

    fs, dfs = field_verdict(parent, same, probes)
    cs, dcs = coeff_verdict_canonical(parent, same)
    fd, dfd = field_verdict(parent, diff, probes)
    cd, dcd = coeff_verdict_canonical(parent, diff)

    agree = (fs == cs) and (fd == cd)
    ok = agree and fs == "reproduced" and fd == "diverged"
    print("  [Claim 1] field verdict ≡ canonical-coeff fast-path under a bound basis")
    print(f"    identical child : field={fs}({dfs:.1e})  coeff={cs}({dcs:.1e})")
    print(f"    different child : field={fd}({dfd:.1e})  coeff={cd}({dcd:.1e})")
    print(f"    -> agree on same AND different: {agree}  => {'PASS' if ok else 'FAIL'}\n")
    return ok


def claim2(C, V, probes):
    parent = (C, V)
    G = signed_permutation(SIGMA.size, swaps=[(0, 2), (3, 4)], flips=[1, 4])
    child = (C @ G, V @ G)  # sign-flipped + reordered EQUIVALENT basis

    fv, dfv = field_verdict(parent, child, probes)
    cc, dcc = coeff_verdict_canonical(parent, child)
    nv, dnv = coeff_verdict_naive(parent, child)
    same_field = bool(np.allclose(reconstruct(*parent), reconstruct(*child)))
    ok = fv == "reproduced" and cc == "reproduced" and nv == "diverged" and same_field
    print("  [Claim 2] sign-flipped/reordered equivalent basis still reproduces")
    print(f"    reconstructed field identical : {same_field}")
    print(f"    field verdict           : {fv} ({dfv:.1e})")
    print(f"    canonical-coeff fast-path: {cc} ({dcc:.1e})")
    print(f"    naive byte comparison    : {nv} ({dnv:.1e})   <- false-diverge if used")
    print(f"    basis hash {basis_hash(parent[1])} -> {basis_hash(child[1])} "
          "(changes; diagnostic only)")
    print(f"    => {'PASS' if ok else 'FAIL'}\n")
    return ok


def claim3(C, V, probes):
    parent = (C, V)
    clusters, sigma = degenerate_clusters(C)
    cluster = clusters[0]  # the σ=3 degenerate pair
    G = cluster_rotation(SIGMA.size, cluster, theta=np.radians(35.0))
    child = (C @ G, V @ G)  # intra-subspace ROTATION on the degenerate modes

    fv, dfv = field_verdict(parent, child, probes)
    cc, dcc = coeff_verdict_canonical(parent, child)
    sv, ang = subspace_verdict(parent, child, cluster)
    same_field = bool(np.allclose(reconstruct(*parent), reconstruct(*child)))
    # The trap: per-mode canonicalization false-diverges; field + subspace fix it.
    ok = fv == "reproduced" and cc == "diverged" and sv == "reproduced" and same_field
    print(f"  [Claim 3] degenerate modes {cluster} (σ≈{sigma[cluster[0]]:.0f}) "
          "— the canonicalization trap")
    print(f"    reconstructed field identical : {same_field}")
    print(f"    field verdict (ground truth)  : {fv} ({dfv:.1e})")
    print(f"    per-mode canonical fast-path  : {cc} ({dcc:.1e})   <- FALSE-DIVERGE (the trap)")
    print(f"    subspace principal-angle test : {sv} (max angle {ang:.1e} rad)")
    print(f"    => {'PASS' if ok else 'FAIL'}\n")
    return ok


def main() -> int:
    C, V, probes, coverage = make_scenario()
    print("=" * 76)
    print("Spike R2.1 — field-primary reproducibility fingerprint (ADR-ROM-001)")
    print(f"  n_dof={N_DOF}  modes={SIGMA.size}  spectrum={SIGMA.tolist()}  "
          f"holdout={M_HOLDOUT}")
    print(f"  probes={N_PROBES} (energy-weighted)  probe_coverage_energy={coverage:.3f}")
    print(f"  verdict tolerance (field-energy-relative) = {REL_TOL:g}")
    print("=" * 76 + "\n")

    results = [claim1(C, V, probes), claim2(C, V, probes), claim3(C, V, probes)]

    print("-" * 76)
    if all(results):
        print("ALL CLAIMS HOLD — field-primary is the definition; canonicalized coefficients")
        print("are a valid fast/audit lens under a bound basis but FALSE-diverge on degenerate")
        print("modes; the basis hash changes under benign re-gauge ⇒ diagnostic, not a gate.")
        return 0
    print(f"SPIKE FAILED: {results.count(False)}/{len(results)} claims did not hold.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
