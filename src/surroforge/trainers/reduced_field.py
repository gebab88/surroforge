"""Reduced-order field surrogate via POD + a per-mode inner surrogate (Story R1.3).

This is the *spine* of Field ROM v1 (brainstorm #36): the modal basis is hidden
*inside* an ordinary :class:`BaseTrainer`, so the engine, exporters, ``compare``,
and the conformance suite all see a normal ``(n, n_field_dof)`` trainer and need
zero ROM-specific changes. A field is "output columns that move together"
(snapshot-matrix / mesh-free, #19/#53): no coordinates or connectivity are read.

``fit`` reduces the field by POD (energy- or rank-truncated SVD of the centered
snapshots), then trains one inner surrogate per retained mode (#11) on the modal
coefficients. ``predict`` runs the inner surrogates to coefficients and
reconstructs the full field. The inner method is any registered trainer
(``model.inner.method``), so Kriging/RBF/polynomial work unchanged as the per-mode
regressor — each mode is a single output, squarely inside the SMT envelope.

Configuration (passed as ordinary hyperparameters; the engine is untouched)::

    model:
      method: reduced-field
      reduction: { basis: pod, energy: 0.999 }   # OR rank: 12 / rank: adaptive
      inner:     { method: kriging, corr: squar_exp }
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from surroforge.core.canonicalize import canonicalize, canonicalize_basis_signs
from surroforge.errors import TrainerError
from surroforge.trainers.base import BaseTrainer

DEFAULT_ENERGY = 0.999
DEFAULT_INNER_METHOD = "polynomial"


def _require_rom_support() -> None:
    """Fail with an actionable install hint if the ROM backend is unavailable (Story R4.4).

    Mirrors the ``[smt]``/``[onnx]`` lazy-import pattern (``kriging.fit`` / the polynomial
    ONNX export): the trainer always registers, so ``available_trainers()`` lists it, and a
    missing backend surfaces a clean :class:`TrainerError` ("pip install 'surroforge[rom]'")
    at fit time — never a raw ``ImportError``.

    ROM's reduction is the numpy/scipy linear-algebra stack, both already Core deps
    (PRD-ROM 4.10: v1 adds **no new heavy dependency**), so in a normal install this guard
    never fires. It is the documented ROM install seam and the gate the Phase-2 mesh readers
    (meshio/pyvista) will join. scipy is the PRD-named SVD backend; the v1 reduction itself
    uses numpy's SVD (frozen — the field fingerprint is pinned on it, R2.x), so this checks
    scipy's presence without touching the numeric path.
    """
    try:
        import scipy.linalg  # noqa: F401  -- ROM numeric backend (PRD-ROM 4.10)
    except ImportError as exc:
        raise TrainerError(
            "Reduced-field ROM requires the 'rom' extra. "
            "Install it with: pip install 'surroforge[rom]'"
        ) from exc


class ReducedFieldTrainer(BaseTrainer):
    """POD field reduction with a per-mode inner surrogate, behind the trainer contract."""

    name = "reduced-field"
    # The artifact is the basis + inner models (all picklable); v1 has no field ONNX.
    export_formats = frozenset({"pickle"})
    # POD/SVD and every built-in inner reproduce within this band: polynomial
    # declares 0.0 and kriging/rbf declare 1e-9 — equal to it. Like every
    # declared tolerance it presumes a bit-reproducible BLAS (Apple Accelerate
    # is not; such hosts are guarded in tests/CI — see tests/conftest.py). An
    # inner with a LOOSER declared tolerance would need this widened;
    # deliberately static until such an inner exists (the value is folded into
    # the field spec_hash, so changing how it is derived churns field-run
    # identity).
    reproduction_tolerance = 1e-9

    def __init__(
        self,
        reduction: dict[str, Any] | None = None,
        inner: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        if extra:
            raise TrainerError(
                f"Unknown hyperparameter(s) for reduced-field: {sorted(extra)}. "
                "Supported: reduction, inner."
            )
        super().__init__(reduction=reduction, inner=inner)

        # --- reduction spec -------------------------------------------------
        red = dict(reduction or {})
        basis = red.pop("basis", "pod")
        if basis != "pod":
            raise TrainerError(f"reduced-field v1 supports only 'pod' basis, got {basis!r}.")
        energy = red.pop("energy", None)
        rank = red.pop("rank", None)
        if red:
            raise TrainerError(
                f"Unknown reduction key(s): {sorted(red)}. Supported: basis, energy, rank."
            )
        if energy is not None and rank is not None:
            raise TrainerError("Specify exactly one of reduction.energy or reduction.rank.")
        if energy is None and rank is None:
            energy = DEFAULT_ENERGY
        if energy is not None and not (0.0 < float(energy) <= 1.0):
            raise TrainerError(f"reduction.energy must be in (0, 1], got {energy!r}.")
        if rank is not None and rank != "adaptive":
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise TrainerError(
                    f"reduction.rank must be a positive int or 'adaptive', got {rank!r}."
                )
        self._energy = float(energy) if energy is not None else None
        self._rank = rank  # int, "adaptive", or None

        # --- inner method spec ---------------------------------------------
        inner_spec = dict(inner or {})
        self._inner_method = str(inner_spec.pop("method", DEFAULT_INNER_METHOD))
        self._inner_hyper = inner_spec  # remaining keys passed to the inner trainer

        # --- fitted state ---------------------------------------------------
        self._basis: np.ndarray | None = None  # V (n_dof × r), orthonormal columns
        self._mean: np.ndarray | None = None  # (1 × n_dof) snapshot mean
        self._models: list[BaseTrainer] = []  # one inner trainer per retained mode
        self._n_dof: int | None = None
        self._energy_captured: float | None = None  # σ²-energy share of retained modes
        self._spectrum: np.ndarray | None = None  # full singular spectrum (Story R4.1)

    # -- fitting ------------------------------------------------------------

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        _require_rom_support()  # actionable hint if the ROM backend is missing (Story R4.4)
        from surroforge.plugins import get_trainer  # lazy: keep __init__/import cheap

        x = self._as_2d(x)
        field = self._as_2d(y)  # (n_samples, n_dof)
        self._n_dof = int(field.shape[1])

        # Center, then POD via economy SVD of the fluctuations: rows of Vt are the
        # spatial modes. Sign-canonicalize so the basis is deterministic (#16).
        self._mean = field.mean(axis=0, keepdims=True)
        fluctuations = field - self._mean
        _u, singular, vt = np.linalg.svd(fluctuations, full_matrices=False)
        self._spectrum = singular  # full spectrum, pre-truncation (rom inspect, R4.2)
        r = self._resolve_rank(singular)
        total = float((singular**2).sum())
        self._energy_captured = (
            float(np.cumsum(singular**2)[r - 1] / total) if total > 0.0 and r > 0 else 0.0
        )
        basis = _canonicalize_signs(vt[:r].T)  # (n_dof × r)
        self._basis = basis

        # One inner surrogate per mode, fit on that mode's coefficient column.
        coefficients = fluctuations @ basis  # (n_samples × r)
        self._models = []
        for k in range(r):
            model = get_trainer(self._inner_method)(**self._inner_hyper)
            model.fit(x, coefficients[:, k : k + 1])
            self._models.append(model)

    def _resolve_rank(self, singular: np.ndarray) -> int:
        available = int(singular.size)
        if available == 0:
            return 0
        if isinstance(self._rank, int) and not isinstance(self._rank, bool):
            return max(1, min(self._rank, available))
        # energy threshold (also the path for rank: 'adaptive' and the default)
        energy = self._energy if self._energy is not None else DEFAULT_ENERGY
        total = float((singular**2).sum())
        if total == 0.0:  # constant field: one (zero-energy) mode suffices
            return 1
        cumulative = np.cumsum(singular**2) / total
        return int(np.searchsorted(cumulative, energy) + 1)

    # -- prediction ---------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._basis is None or self._mean is None:
            raise TrainerError("Model is not fitted.")
        field = self.predict_coefficients(x) @ self._basis.T + self._mean
        return self._as_2d(field)

    def predict_coefficients(self, x: np.ndarray) -> np.ndarray:
        """Inner-surrogate modal coefficients for ``x`` (the pre-reconstruction step).

        The compact ``(n, r)`` answer ``predict`` reconstructs from — and the default
        prediction surface for field runs (#35, Story R4.1): services duck-type this
        method, so the ``(n, n_dof)`` BaseTrainer contract of ``predict`` is untouched.
        """
        if self._basis is None or not self._models:
            raise TrainerError("Model is not fitted.")
        x = self._as_2d(x)
        coefficients = np.empty((x.shape[0], len(self._models)), dtype=float)
        for k, model in enumerate(self._models):
            coefficients[:, k] = model.predict(x).reshape(-1)
        return coefficients

    def reduction_artifacts(self) -> dict[str, np.ndarray]:
        """Arrays of the PERSISTED model for the content-addressed store (Story R4.1).

        Consumed by the engine persist step on the FINAL model — this is a storage
        artifact of the shipped pickle (predict / `rom inspect`), explicitly NOT a
        fingerprint input: ``reproduce()`` never reads the store, and the holdout-model
        rule governs the diagnostics hooks, not storage (R3 retro action item 3; the
        fingerprint spec superseded basis.npz only as a VERDICT input).
        ``singular_values`` is the full pre-truncation spectrum (feeds R4.2).
        Returned arrays are read-only views, so accidental in-place mutation by
        a caller fails loudly (a deliberate write through ``.base`` remains
        possible — the views guard against mistakes, not malice).
        """
        spectrum = getattr(self, "_spectrum", None)  # absent on pre-R4.1 pickles
        if self._basis is None or self._mean is None or spectrum is None:
            raise TrainerError(
                "Model is not fitted, or it was trained before the reduction-"
                "artifact store existed (pre-R4.1 pickle) — re-train the run."
            )

        def _frozen(array: np.ndarray) -> np.ndarray:
            view = array.view()
            view.flags.writeable = False
            return view

        return {
            "basis": _frozen(self._basis),
            "mean": _frozen(self._mean),
            "singular_values": _frozen(spectrum),
        }

    # -- introspection (feeds `rom inspect`, Story R4.2; reproduce(), R2.3) -

    @property
    def rank(self) -> int:
        """Number of retained modes (0 until fitted)."""
        return 0 if self._basis is None else int(self._basis.shape[1])

    def reproducibility_diagnostics(self, x: np.ndarray) -> dict[str, Any]:
        """Field-primary reproducibility diagnostics for the engine to record (R2.3, R2.5).

        Returns the canonicalized basis hash (a **diagnostic**, never a verdict gate),
        the canonicalized out-of-sample modal coefficients (per-mode "which mode moved"
        attribution — degenerate clusters are compared as a subspace downstream, never
        per-mode), the resolved reduction spec, and ``probe_coverage_energy`` — the
        fraction of field energy the **global-energy-norm verdict verifies** (the
        retained-mode POD energy share; the remainder is the truncation floor, which the
        reproduce verdict cannot see). The engine duck-types this hook, so a scalar trainer
        that does not expose it leaves lineage unchanged (spine theorem #36;
        spec-field-rom-fingerprint.md).
        """
        if self._basis is None:
            raise TrainerError("Model is not fitted.")
        canon_coeffs, canon_basis = canonicalize(self.predict_coefficients(x), self._basis)
        basis_bytes = np.ascontiguousarray(canon_basis, dtype=np.float64).tobytes()
        # In v1 "the probes" are the full reconstructed field (the global energy-norm IS
        # the definition; node subsampling deferred), so coverage = the retained-mode
        # energy share. A constant / zero-energy field has NO modal energy to truncate —
        # the reconstruction (the mean) captures it exactly — so its coverage is 1.0, not
        # 0.0; 0.0 would mis-warn "0% verified" on a run that reproduces perfectly.
        coverage = float(self._energy_captured) if self._energy_captured else 1.0
        return {
            "basis_hash": hashlib.sha256(basis_bytes).hexdigest(),
            "holdout_coefficients": canon_coeffs,
            "reduction": self._resolved_reduction(),
            "probe_coverage_energy": coverage,
        }

    def _resolved_reduction(self) -> dict[str, Any]:
        """The reduction spec actually realized at fit time (recorded in lineage)."""
        return {"basis": "pod", "rank": self.rank, "energy_captured": self._energy_captured}

    def resolved_config_overrides(self) -> dict[str, Any] | None:
        """Config overrides resolved at fit time, for the canonical snapshot (Story R1.2).

        Only ``rank: adaptive`` is resolved into something the config did not already
        say: the engine deep-merges the returned dict into the config snapshot so the
        persisted run carries the concrete integer rank (the snapshot, not the input
        YAML, is canonical) and ``reproduce()`` re-trains deterministically. An
        energy- or explicit-rank-specified run returns ``None`` — its config is
        already canonical, and its snapshot/hashes must stay byte-identical (R2.3/R2.4).
        The engine duck-types this hook; it never names ``reduction.rank``.
        """
        if self._rank != "adaptive" or self._basis is None:
            return None
        return {"model": {"reduction": {"rank": self.rank}}}


def _canonicalize_signs(basis: np.ndarray) -> np.ndarray:
    """Fix each mode's sign so its max-abs entry is positive (#16).

    SVD leaves a per-mode sign gauge; pinning it makes the basis deterministic
    run-to-run and is the foundation the field-primary fingerprint builds on
    (spec-field-rom-fingerprint.md). Delegates to the shared canonicalization
    primitive (Story R2.2) so there is a single implementation of the sign gauge.
    """
    return canonicalize_basis_signs(basis)
