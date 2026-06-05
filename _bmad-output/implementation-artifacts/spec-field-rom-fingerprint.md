---
title: 'Field ROM reproducibility fingerprint — spike findings + reproduce() contract'
type: 'spike+contract'
created: '2026-06-05'
status: 'proposed'
baseline_commit: 'NO_VCS'
delivers: 'Story R2.1 (de-risk gate); feeds R2.2, R2.3, R2.4, R2.5'
context:
  - '_bmad-output/planning-artifacts/prd-field-rom-v1.md'
  - '_bmad-output/planning-artifacts/epics-field-rom-v1.md'
  - '_bmad-output/brainstorming/brainstorming-session-2026-06-04-231637.md (ADR-ROM-001)'
  - 'spikes/field_fingerprint_spike.py (runnable evidence)'
  - '_bmad-output/implementation-artifacts/spec-fix-reproduce-fingerprint.md (the scalar fingerprint this generalizes)'
---

## Intent

**Goal of the spike (Story R2.1):** settle the field/ROM reproducibility-fingerprint design *empirically* before the v1 `reproduce()` contract freezes — it is the highest-leverage and highest-risk change in Field ROM v1, and every persisted field run depends on it.

**What was de-risked:** ADR-ROM-001 (refined by First-Principles + a pre-mortem) proposed a layered fingerprint where the **observable field is the definition** and canonicalized coefficients are a fast/audit lens. The spike (`spikes/field_fingerprint_spike.py`) turns the three load-bearing assertions into runnable experiments on synthetic data with a deliberately **degenerate** spectrum (two equal singular values), and all three hold. This document records the evidence and converts it into a concrete, implementable `reproduce()` contract.

## Spike Results (reproducible: `python spikes/field_fingerprint_spike.py`, exit 0)

Scenario: `n_dof=200`, `r=6` modes, spectrum `[10,7,5,3,3,1]` (modes 3,4 degenerate), 40 holdout points, 64 energy-weighted probes (`probe_coverage_energy = 0.621`), field-energy-relative tolerance `1e-9`. Gauge transforms are exact orthogonal maps `G` (`V'=V·G`, `C'=C·G`), so the reconstructed field `Ŷ=C·Vᵀ` is invariant while basis/coefficients are not.

| Claim | Experiment | Result | Evidence |
|---|---|---|---|
| **1 — fast-path is valid under a bound basis** | identical vs. genuinely-different surrogate | field verdict and canonical-coeff fast-path **agree** (same→both reproduced; different→both diverged) | identical `0.0`/`0.0`; different `4.7e-2`/`5.4e-2` |
| **2 — sign/order re-gauge still reproduces** | signed-permutation gauge (`V@G, C@G`) | field **reproduced**, canonical-coeff **reproduced**, **naive byte comparison false-diverges**; basis hash changes | field `1.1e-16`, canon `0.0`, naive `1.6e0`; hash `e9e5…→4b00…` |
| **3 — degenerate-mode trap** | intra-subspace rotation (35°) on the σ=3 pair | field **reproduced**, **per-mode canonicalization FALSE-diverges**, principal-angle subspace test **reproduced** | field `9.8e-17`, per-mode canon `2.8e-1`, subspace angle `2.1e-8 rad` |

**Net:** the field verdict is gauge-invariant by construction (≤1e-16 across every benign re-gauge); the canonicalized-coefficient fast-path tracks it under a *bound* basis but **false-diverges on degeneracy** (the most common, symmetric, case); the basis hash changes under a benign re-gauge and therefore **cannot be a gate**.

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Contract Decisions (confirmed / adjusted vs. ADR-ROM-001)

**CONFIRMED by the spike:**

1. **Field equivalence is the definition (the verdict).** `reproduce()` reconstructs the out-of-sample holdout field at pinned, energy-weighted **probe nodes**, and the verdict is the relative field-energy difference vs. a tolerance. Gauge-invariant; needs no canonicalization to be correct.
2. **Canonicalized coefficients (#16) are a fast/audit lens, not the verdict.** Valid accelerator and per-mode "which mode moved" attribution **only** under a bound basis. Sign by max-abs entry, order by σ desc, tie-break by peak-node index.
3. **Degeneracy is handled by subspace comparison or the field fallback.** Detect equal-σ clusters (rtol); within a cluster compare the **subspace via principal angles** (`angle_tol ≈ 1e-6 rad`, a microradian — far below any physical rotation, above QR/SVD noise ~1e-8), never per-mode. The field verdict is the always-correct safety net.

**ADJUSTED by the spike (these change the ADR):**

4. **Basis hash → diagnostic, NEVER a gate.** ADR-ROM-001's original "key guard" bound the basis hash *into* the verdict. The pre-mortem flagged this and Claim 2 proves it: a benign re-gauge (or a LAPACK/SciPy upgrade) changes the basis bytes while the field is identical. The hash is recorded for diagnostics only ("basis re-gauged, field reproduced ⇒ benign"); the field verdict closes the original false-"reproduce"-on-a-different-basis trap *by construction* (you cannot match the field on a wrong basis).
5. **Tolerance is field-energy-relative and folded into the hashed spec.** Widening the tolerance is a *different run* (spec re-hash), not a looser compare. Per-mode tolerance bands (#72) are **deferred** to Phase 2.
6. **v1 ships one verdict only.** The cross-solver "physics vs. bitwise" two-verdict (#73) is deferred; v1's field verdict is the bitwise-equivalent-on-the-field contract. For a scalar run (identity basis) it collapses to today's byte behaviour — proven separately by R2.4's regression guard.

## Boundaries & Constraints

**Always:** The **field verdict is the ground truth** for any reduced run; coefficients/basis-hash are diagnostic. Canonicalization runs before any coefficient comparison. The scalar path stays **byte-identical** (identity basis ⇒ field = predictions; R2.4). Tolerance is a single field-energy-relative value folded into the hashed spec. Probe nodes are deterministic (energy-weighted) and recorded with their coverage. Reuse the existing out-of-sample holdout (no new RNG, no extra model fit), generalizing `spec-fix-reproduce-fingerprint.md`.

**Ask First:** Changing the public `ReproduceResult` shape beyond additive fields. Changing the scalar default tolerance (`1e-9`). Introducing the physics/bitwise two-verdict (#73) or per-mode bands (#72) — both deferred.

**Never:** Gate the verdict on the basis hash. Use per-mode coefficient canonicalization as the verdict (it false-diverges on degeneracy). Add a new heavy dependency (SVD/principal angles are numpy/scipy). Weaken a tolerance to silence a flaky case (loosening ⇒ a different hashed spec).

## I/O & Edge-Case Matrix

| Scenario | State | Verdict | Notes |
|---|---|---|---|
| Honest re-run | field within tol | **reproduced** | field + coeff agree (Claim 1) |
| Divergent surrogate | holdout field exceeds tol | **diverged** | field + coeff agree (Claim 1) |
| Re-gauged basis (sign/order) | same field, basis bytes differ | **reproduced** | naive byte compare would false-diverge (Claim 2) |
| Degenerate-mode rotation | same field, modes mixed | **reproduced** | per-mode canon would false-diverge; subspace/field correct (Claim 3) |
| LAPACK/SciPy upgrade | basis bytes change, field same | **reproduced** (basis re-gauged) | basis hash diagnostic, not a gate |
| Scalar run (no reduction) | identity basis | **byte-identical to today** | R2.4 regression guard |
| Out-of-probe energy high | probes miss a new active region | **warn** | record `probe_coverage_energy`; prefer global energy-norm (R2.5) |
| Tolerance widened | spec edited | **different run** | spec re-hash, not a looser compare |
| Data changed | source edited | data_hash mismatch → raises | unchanged from scalar path |

</frozen-after-approval>

## reproduce() Contract (proposed shape)

**Persisted (lineage.json / run dir), additive to the scalar fields:**
- `reduction`: resolved spec (`basis`, resolved `rank`, `energy_captured`) — part of the hashed spec.
- `basis.npz`: content-addressed canonicalized basis `V` (deduplicated across runs by `basis_hash`).
- `basis_hash`: canonicalized basis hash — **diagnostic metadatum, not gated**.
- `holdout_fingerprint`: the field-primary token = reconstructed holdout field at the pinned probes (canonicalized). Scalar run: identical to today's prediction sample.
- `probe_nodes` + `probe_coverage_energy`: deterministic energy-weighted probe indices + their share of total field energy.
- `reproduction_tolerance`: field-energy-relative, folded into the hashed spec.
- (diagnostic) canonicalized holdout coefficients for per-mode attribution.

**Verdict algorithm (`reproduce()` / `_verdict`):**
1. Re-train from the config snapshot (never unpickle — same ACE-free path as today).
2. Reconstruct the out-of-sample holdout field at the recorded `probe_nodes`; canonicalize.
3. **Verdict = relative field-energy diff ≤ `reproduction_tolerance`** → `reproduced` / `diverged`.
4. Diagnostics (non-gating): canonical-coefficient diff (per-mode "which mode moved"); `basis_hash` change → annotate "basis re-gauged" when the field still reproduces; equal-σ clusters compared by principal angle.
5. Scalar run (identity basis): steps 2–3 reduce to the existing byte/tolerance compare ⇒ no behaviour change.

## Tasks Derived (for R2.2–R2.5)

- **R2.2 — Canonicalization + degeneracy:** port `canonicalize()` (sign/order) and `degenerate_clusters()` + `max_principal_angle()` from the spike into `src/`; CI under pinned threads.
- **R2.3 — Field-primary verdict:** implement the verdict algorithm above in `services.reproduce`/`_verdict`; basis hash diagnostic; tolerance in hashed spec; additive `ReproduceResult` fields.
- **R2.4 — Scalar byte-identity guard:** regression test pinning identity-basis equivalence to the pre-ROM fingerprint.
- **R2.5 — Probe coverage:** record `probe_coverage_energy`; global energy-norm as definition; warn on out-of-probe energy.

## Verification

`python spikes/field_fingerprint_spike.py` → exit 0, all three claims PASS (numbers above). Re-runnable; deterministic (seeded `default_rng`). No new dependency (numpy only).

## Open Questions (carried)

- **OQ-3:** v1 uses the simple energy-weighted probe pick demonstrated here (coverage 0.621); Q-DEIM/QR probe optimisation (#49/#84) deferred — but coverage 0.621 argues for the **global energy-norm as the definition** with probes as the fast path (R2.5), not probes alone.
- **OQ-4:** concrete default for the field-energy-relative tolerance and exactly how it enters the spec hash (the spike used `1e-9`).

## References

- Feature PRD §4.5 · Epics R2 (`epics-field-rom-v1.md`) · ADR-ROM-001 (brainstorm doc) · scalar precedent (`spec-fix-reproduce-fingerprint.md`) · evidence (`spikes/field_fingerprint_spike.py`).
