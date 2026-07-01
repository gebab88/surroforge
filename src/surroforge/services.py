"""High-level operations shared by the CLI and the Python API.

These compose the engine, persistence, plugins, and core domain into the verbs a
user actually runs: reproduce, compare, predict, export, report.
"""

from __future__ import annotations

import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from surroforge.comparison import build_comparison
from surroforge.core.canonicalize import degenerate_clusters, max_principal_angle
from surroforge.core.project import Project
from surroforge.core.run import FieldCoverageWarning, Run
from surroforge.data.loader import load_table
from surroforge.engine import run_training
from surroforge.errors import (
    DataError,
    ExportError,
    ProjectError,
    ReproducibilityError,
    TrainerError,
)
from surroforge.plugins import get_exporter, get_reporter
from surroforge.validators.metrics import FIELD_ENERGY_FLOOR

# --- Reproduce ---------------------------------------------------------------

# Single source of truth for the reproduction divergence tolerance: the max abs
# difference over the recorded out-of-sample holdout sample below which a re-run
# counts as reproduced. Anything larger is a divergence.
REPRODUCTION_TOLERANCE = 1e-9

# FIELD_ENERGY_FLOOR (the verdict denominator floor, Story R2.3) lives in
# validators/metrics.py since R3.1 — energy_l2 shares the verdict's norm, and
# validators must not import services, so the constant moved down. Imported above
# so the verdict and prior importers keep reading it from here.

# Below this fraction of field energy verified by the verdict, reproduce() emits a
# FieldCoverageWarning: "reproduced" then covers only part of the field, the truncated
# tail being unverifiable (Story R2.5). Advisory only — never a verdict gate, exactly like
# the basis hash. Single source of truth, never inlined per call site.
PROBE_COVERAGE_WARN_THRESHOLD = 0.99


@dataclass
class ReproduceResult:
    parent: Run
    reproduction: Run
    bit_identical: bool
    max_abs_diff: float
    data_hash_match: bool
    verdict: str
    #: The tolerance the run declared (Story 5.4); the verdict was judged against it.
    tolerance: float = REPRODUCTION_TOLERANCE
    #: Field runs only (Story R2.3): relative field-energy diff of the reconstructed
    #: holdout field. ``None`` for a scalar run, which is judged by ``max_abs_diff``.
    field_relative_diff: float | None = None
    #: Field runs only: whether the canonicalized basis hashes matched (diagnostic).
    basis_hash_match: bool | None = None
    #: Field runs only: the basis bytes changed but the field still reproduced — a benign
    #: re-gauge. The basis hash is never a verdict gate (spec-field-rom-fingerprint.md).
    basis_regauged: bool = False
    #: Field runs only: non-gating per-mode / degenerate-subspace coefficient attribution.
    coefficient_diagnostic: dict[str, Any] | None = None
    #: Field runs only (Story R2.5): fraction of field energy the verdict verifies (the
    #: retained-mode energy share). ``None`` for a scalar run. Advisory, never a gate.
    probe_coverage_energy: float | None = None


@dataclass
class RomInspectResult:
    """Energy spectrum and rank summary for a reduced-field run (Story R4.2, PRD §4.8 #30)."""

    is_reduced: bool
    rank: int = 0
    total_modes: int = 0
    #: Retained-mode energy share from lineage (holdout model, same as reproduce verdict).
    #: NOT re-derived from the final-model spectrum — they differ when holdout ≠ final set.
    energy_captured: float = 0.0
    probe_coverage_energy: float = 1.0
    per_mode_energy: list[float] = field(default_factory=list)
    cumulative_energy: list[float] = field(default_factory=list)
    basis: str = ""


def inspect_rom_run(run: Run) -> RomInspectResult:
    """Return the energy spectrum and rank for a reduced-field run (PRD §4.8 #30).

    Guard order (see Dev Notes, Story R4.2):
    1. Empty ``reduction`` → scalar/non-ROM run → ``is_reduced=False`` (no error).
    2. Empty ``basis_file`` → pre-R4.1 run, basis store not yet written → ``ProjectError``
       with "re-train" guidance.
    3. ``run.basis_path`` → validates layout + file existence (raises ``ProjectError``
       for missing or malformed-path files).
    4. Load npz, compute per-mode and cumulative energy fractions. The lineage is treated
       as untrusted disk data: a corrupt/foreign basis store (unreadable zip, missing
       ``singular_values``) or a malformed ``reduction`` block (missing/non-numeric
       ``rank``/``energy_captured``) is re-raised as ``ProjectError`` with re-train
       guidance, never as a bare ``KeyError``/``BadZipFile`` that would escape ``_guard``.

    ``energy_captured`` is read from lineage (holdout model), matching the reproduce
    verdict. Re-deriving it from the final-model spectrum yields a different value when
    the holdout/final train sets differ (e.g. under k-fold seed).
    """
    if not run.reduction:
        return RomInspectResult(is_reduced=False)

    if not run.basis_file:
        raise ProjectError(
            f"Run '{run.run_id}' has a reduction config but no basis store "
            "(trained before Story R4.1). Re-train the run to generate the basis file."
        )

    path = run.basis_path  # raises ProjectError for missing / malformed path

    try:
        with np.load(str(path)) as npz:
            sv = npz["singular_values"]
    except (zipfile.BadZipFile, OSError, ValueError, EOFError, KeyError) as exc:
        raise ProjectError(
            f"Run '{run.run_id}' basis store '{run.basis_file}' is unreadable or "
            f"malformed ({type(exc).__name__}). Re-train the run to regenerate the "
            "basis file."
        ) from exc

    try:
        rank = int(run.reduction["rank"])
        energy_captured = float(run.reduction["energy_captured"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ProjectError(
            f"Run '{run.run_id}' has a malformed reduction lineage "
            f"(missing or non-numeric rank/energy_captured: {exc!r}). "
            "Re-train the run to regenerate a consistent lineage."
        ) from exc

    total_sq = float((sv**2).sum())
    if total_sq > 0:
        per_mode_energy = (sv**2 / total_sq).tolist()
    else:
        per_mode_energy = [0.0] * len(sv)
    cumulative_energy = np.cumsum(per_mode_energy).tolist()

    return RomInspectResult(
        is_reduced=True,
        rank=rank,
        total_modes=len(sv),
        energy_captured=energy_captured,
        probe_coverage_energy=run.probe_coverage_energy,
        per_mode_energy=per_mode_energy,
        cumulative_energy=cumulative_energy,
        basis=str(run.reduction.get("basis", "pod")),
    )


def reproduce(project: Project, run_id: str, *, echo: Any = None) -> ReproduceResult:
    """Re-run a stored run from its config snapshot and compare to the original (PRD 4.5).

    For a **field run** (one carrying a reduction, Story R2.3) the verdict is the relative
    field-energy difference between the parent's and the re-run's reconstructed
    out-of-sample holdout field; the basis hash and canonicalized coefficients are a
    diagnostic lens, never a gate. A **scalar run** keeps the absolute byte/diff behaviour
    unchanged (the R2.4 regression guard).
    """
    parent = project.run(run_id)
    config = parent.config

    reproduction = run_training(project, config, parent_run=run_id, echo=echo)

    data_hash_match = parent.data_hash == reproduction.data_hash
    bit_identical = (
        parent.lineage.get("prediction_hash") == reproduction.lineage.get("prediction_hash")
    )

    # Quantify divergence over the recorded OUT-OF-SAMPLE holdout sample. In-sample
    # predictions are a poor signal: interpolating methods (Kriging) reproduce the
    # training points near-exactly even when the underlying model diverges.
    parent_fp = np.asarray(parent.holdout_fingerprint, dtype=float)
    repro_fp = np.asarray(reproduction.holdout_fingerprint, dtype=float)
    # "Comparable" requires both runs to carry a holdout fingerprint of the same
    # shape. A run trained before this fingerprint existed (empty sample) — or one
    # whose split shape changed — is not comparable, and is reported as
    # "unverifiable" rather than a false pass or a false divergence.
    comparable = parent_fp.size > 0 and repro_fp.size > 0 and parent_fp.shape == repro_fp.shape
    max_abs_diff = float(np.max(np.abs(parent_fp - repro_fp))) if comparable else float("inf")

    # Judge against the tolerance the method declared at train time (Story 5.4);
    # legacy runs recorded before this field fall back to the global default.
    tolerance = float(parent.lineage.get("reproduction_tolerance", REPRODUCTION_TOLERANCE))

    # Field run? A reducing trainer records a basis hash; a scalar run never does, so it
    # stays on the absolute-diff path byte-for-byte (R2.4).
    is_field = bool(parent.basis_hash)
    field_relative_diff: float | None = None
    basis_hash_match: bool | None = None
    basis_regauged = False
    coefficient_diagnostic: dict[str, Any] | None = None
    probe_coverage_energy: float | None = None

    if is_field:
        # The reconstructed field is gauge-invariant, so this never false-diverges on a
        # benign re-gauge (the verdict). Basis hash + coefficients are diagnostic only.
        field_relative_diff = (
            _field_energy_relative_diff(parent_fp, repro_fp) if comparable else float("inf")
        )
        diff_for_verdict = field_relative_diff
        parent_bh, repro_bh = parent.basis_hash, reproduction.basis_hash
        basis_hash_match = (parent_bh == repro_bh) if (parent_bh and repro_bh) else None
        coefficient_diagnostic = _coefficient_diagnostic(parent, reproduction)
        probe_coverage_energy = parent.probe_coverage_energy
    else:
        diff_for_verdict = max_abs_diff

    verdict = _verdict(data_hash_match, bit_identical, comparable, diff_for_verdict, tolerance)

    if is_field:
        # A changed basis hash with a reproduced field is a benign re-gauge, NEVER a
        # divergence — the field verdict closes that trap by construction. The hash is
        # never gated (spec-field-rom-fingerprint.md, decision #4).
        basis_regauged = _basis_regauged(basis_hash_match, verdict)
        # The verdict checks the reconstructed field, which represents only
        # ``probe_coverage_energy`` of the true field; the truncated tail is unverifiable.
        # Warn — never gate — so "reproduced" cannot masquerade as a full-field guarantee
        # (Story R2.5). A low coverage does not change the verdict. Only warn when the field
        # was actually compared: on ``unverifiable`` / ``data_changed`` nothing was checked,
        # so "verifies only X%" would be contradictory noise.
        if (
            probe_coverage_energy is not None
            and probe_coverage_energy < PROBE_COVERAGE_WARN_THRESHOLD
            and verdict in {"reproduced", "reproduced_within_tolerance", "diverged"}
        ):
            out_of_probe = 1.0 - probe_coverage_energy
            warnings.warn(
                f"Field run '{parent.run_id}' verdict verifies only "
                f"{probe_coverage_energy:.1%} of the field energy; {out_of_probe:.1%} "
                "lives in truncated modes that reproduce cannot check. Increase "
                "reduction.energy / rank for a stronger reproducibility guarantee.",
                FieldCoverageWarning,
                stacklevel=2,
            )

    return ReproduceResult(
        parent=parent,
        reproduction=reproduction,
        bit_identical=bit_identical,
        max_abs_diff=max_abs_diff,
        data_hash_match=data_hash_match,
        tolerance=tolerance,
        verdict=verdict,
        field_relative_diff=field_relative_diff,
        basis_hash_match=basis_hash_match,
        basis_regauged=basis_regauged,
        coefficient_diagnostic=coefficient_diagnostic,
        probe_coverage_energy=probe_coverage_energy,
    )


def _verdict(
    data_hash_match: bool,
    bit_identical: bool,
    comparable: bool,
    max_abs_diff: float,
    tolerance: float = REPRODUCTION_TOLERANCE,
) -> str:
    """Classify a reproduction outcome against the method's tolerance (PRD 4.5, Story 5.4)."""
    if not data_hash_match:
        return "data_changed"
    if not comparable:
        return "unverifiable"
    if bit_identical:
        return "reproduced"
    if max_abs_diff <= tolerance:
        return "reproduced_within_tolerance"
    return "diverged"


def _basis_regauged(basis_hash_match: bool | None, verdict: str) -> bool:
    """Whether the basis re-gauged but the field still reproduced (Story R2.3, AC2).

    ``True`` only when the canonicalized basis hashes **differ** and the field verdict is
    a reproduce — a benign re-gauge, never a divergence. The basis hash is never a gate
    (spec-field-rom-fingerprint.md, decision #4); this is a presentation annotation.
    """
    return basis_hash_match is False and verdict in {
        "reproduced",
        "reproduced_within_tolerance",
    }


def _field_energy_relative_diff(parent_fp: np.ndarray, repro_fp: np.ndarray) -> float:
    """Relative field-energy difference ``‖Δ‖_F / max(‖parent‖_F, floor)`` (Story R2.3).

    Gauge-invariant by construction — the reconstructed field is invariant under sign
    flips, mode reorder, and intra-subspace rotation of a degenerate cluster — so it
    never false-diverges on a benign re-gauge, unlike a per-mode or byte comparison.
    """
    denom = float(np.linalg.norm(parent_fp))
    return float(np.linalg.norm(parent_fp - repro_fp) / max(denom, FIELD_ENERGY_FLOOR))


def _coefficient_diagnostic(parent: Run, reproduction: Run) -> dict[str, Any] | None:
    """Non-gating per-mode "which mode moved" attribution (Story R2.3, diagnostic only).

    Reports a per-mode relative coefficient diff for non-degenerate modes; for a
    near-degenerate singular-value cluster it reports the principal-angle **subspace**
    diff instead (R2.2 primitives), because per-mode canonicalization false-diverges on
    a benign intra-subspace rotation. This never drives the verdict — the field does.
    Returns ``None`` when the canonical coefficients are absent or shape-mismatched.
    """
    # A diagnostic must never break the verdict: a ragged/non-numeric coefficient
    # lineage makes np.asarray raise, so fail soft to "no diagnostic".
    try:
        pc = np.asarray(parent.holdout_coefficients, dtype=float)
        rc = np.asarray(reproduction.holdout_coefficients, dtype=float)
    except (ValueError, TypeError):
        return None
    if pc.ndim != 2 or pc.size == 0 or pc.shape != rc.shape:
        return None
    clusters, _sigma = degenerate_clusters(pc)
    degenerate_modes = {m for cluster in clusters for m in cluster}
    denom = float(np.linalg.norm(pc)) or 1.0
    per_mode = {
        k: float(np.linalg.norm(pc[:, k] - rc[:, k]) / denom)
        for k in range(pc.shape[1])
        if k not in degenerate_modes
    }
    # Cluster indices come from `pc` (same shape as `rc`), so the slices are valid. A
    # cluster wider than the sample count cannot form a column subspace (principal
    # angles need cols ≤ rows), so skip it — this is a diagnostic, not a gate.
    subspace_angles = [
        max_principal_angle(pc[:, cluster], rc[:, cluster])
        for cluster in clusters
        if len(cluster) <= pc.shape[0]
    ]
    return {
        "per_mode_relative_diff": per_mode,
        "degenerate_clusters": clusters,
        "max_subspace_angle": max(subspace_angles) if subspace_angles else 0.0,
    }


def assert_reproducible(
    result: ReproduceResult, *, tolerance: float | None = None
) -> None:
    """Raise if a reproduction did not verifiably reproduce (used in CI/tests).

    Judges against the tolerance the run declared (``result.tolerance``, Story 5.4)
    unless an explicit ``tolerance`` override is supplied. Field runs are judged by the
    relative field-energy diff; scalar runs by the absolute max prediction diff (R2.3).
    """
    tol = result.tolerance if tolerance is None else tolerance
    if not result.data_hash_match:
        raise ReproducibilityError(
            "Underlying data changed since the original run (data hash mismatch)."
        )
    if result.verdict == "unverifiable":
        raise ReproducibilityError(
            "Cannot verify reproduction: the original run predates the holdout "
            "fingerprint (or its sample shape changed). Re-train to enable checking."
        )
    diff = (
        result.field_relative_diff
        if result.field_relative_diff is not None
        else result.max_abs_diff
    )
    # NaN-safe: ``not (diff <= tol)`` raises on NaN (a non-finite field), matching the
    # ``diverged`` verdict — a bare ``diff > tol`` would let NaN slip through as a pass.
    if not (diff <= tol):
        kind = (
            "field-energy-relative"
            if result.field_relative_diff is not None
            else "max abs prediction"
        )
        raise ReproducibilityError(
            f"Reproduction diverged: {kind} diff {diff:.3e} > tolerance {tol:.1e}."
        )


# --- Compare -----------------------------------------------------------------

def compare_runs(project: Project, run_ids: list[str], *, metric: str = "rmse") -> dict[str, Any]:
    """Compare runs by an extended metric / config / lineage view (PRD 4.1).

    Loads the runs and delegates to :func:`surroforge.comparison.build_comparison`, whose
    structured result the CLI renders and the Professional HTML comparison report consumes.
    """
    if not run_ids:
        raise DataError("Provide at least one run id to compare.")
    runs = [project.run(rid) for rid in run_ids]
    return build_comparison(runs, metric=metric)


# --- Predict -----------------------------------------------------------------

def predict_with_run(run: Run, frame: pd.DataFrame, *, field: bool = False) -> pd.DataFrame:
    """Predict outputs for new inputs using a trained run's model (PRD 4.1, 4.7).

    A reducing model (duck-typed ``predict_coefficients``) returns compact modal
    coefficients ``mode_0 … mode_{r-1}`` by default (#35 — KB, not GB, per query);
    ``field=True`` reconstructs the full field under the run's recorded
    ``output_columns``, so the written frame re-ingests under the run's own outputs
    pattern (#21). A scalar model has no coefficients to return — its prediction
    already is the full output — so ``field`` is accepted silently there and the
    behavior is byte-identical to pre-R4.1.
    """
    config = run.config
    inputs = config.data.inputs
    missing = [c for c in inputs if c not in frame.columns]
    if missing:
        raise DataError(f"Input file is missing required columns: {missing}")

    input_frame = frame[inputs]
    non_numeric = [
        c for c in inputs if not pd.api.types.is_numeric_dtype(input_frame[c])
    ]
    if non_numeric:
        raise DataError(
            f"Non-numeric input columns: {non_numeric}. "
            "Surrogate predictions require numeric input data."
        )
    if input_frame.isnull().to_numpy().any():
        raise DataError("Input file contains missing values in required input columns.")

    x = input_frame.to_numpy(dtype=float)
    model = run.load_model()
    if not field and hasattr(model, "predict_coefficients"):
        coefficients = np.asarray(model.predict_coefficients(x), dtype=float)
        if coefficients.ndim == 1:
            # Same normalization as the predict path: a duck-typed plugin may
            # return (n,) for a rank-1 reduction.
            coefficients = coefficients.reshape(-1, 1)
        if coefficients.ndim != 2 or coefficients.shape[0] != len(frame):
            # Duck-typed surface with no BaseTrainer conformance backing it —
            # fail the contract loudly instead of dying inside pandas.
            raise TrainerError(
                "predict_coefficients must return an (n, r) array for n input "
                f"rows; got shape {coefficients.shape} for {len(frame)} rows."
            )
        result = frame.copy()
        for k in range(coefficients.shape[1]):
            result[f"mode_{k}"] = coefficients[:, k]
        return result

    preds = np.asarray(model.predict(x), dtype=float)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    outputs = config.data.outputs
    recorded = run.output_columns  # empty for scalar/pre-R4.1 runs -> legacy chain
    if len(recorded) == preds.shape[1]:
        columns = recorded
    elif preds.shape[1] == len(outputs):
        columns = outputs
    else:
        columns = [f"output_{i}" for i in range(preds.shape[1])]
    result = frame.copy()
    for i, col in enumerate(columns):
        result[col] = preds[:, i]
    return result


def predict_from_file(
    run: Run, input_path: str | Path, *, field: bool = False
) -> pd.DataFrame:
    return predict_with_run(run, load_table(input_path), field=field)


# --- Export ------------------------------------------------------------------

def export_model(run: Run, fmt: str, output: str | Path | None = None) -> Path:
    """Export a run's model to ``fmt`` (pickle, onnx, or a plugin format) (PRD 4.4)."""
    exporter = get_exporter(fmt)()
    model = run.load_model()
    # Declaration-driven gate: the method itself says which formats it supports,
    # so an unsupported method/format pair fails clearly here (Story 5.4) rather
    # than deep inside an exporter or, worse, producing a broken artifact.
    if not model.supports_export(fmt):
        raise ExportError(
            f"Method '{model.name}' does not support export format '{fmt}'. "
            f"Supported: {', '.join(sorted(model.export_formats))}."
        )
    n_features = int(run.metrics.get("n_inputs", len(run.config.data.inputs)))
    if output is None:
        output = run.run_dir / f"model{exporter.extension}"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    return exporter.export(model, output, n_features=n_features)


# --- Report ------------------------------------------------------------------

def generate_report(
    project: Project, run: Run, fmt: str = "markdown", output: str | Path | None = None
) -> tuple[Path, str]:
    """Render a report for ``run`` and write it under ``reports/`` (PRD 4.9)."""
    reporter = get_reporter(fmt)()
    text = reporter.render(run)
    if output is None:
        output = project.reports_dir / f"{run.run_id}{reporter.extension}"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return output, text
