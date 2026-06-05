"""High-level operations shared by the CLI and the Python API.

These compose the engine, persistence, plugins, and core domain into the verbs a
user actually runs: reproduce, compare, predict, export, report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from surroforge.core.project import Project
from surroforge.core.run import Run
from surroforge.data.loader import load_table
from surroforge.engine import run_training
from surroforge.errors import DataError, ExportError, ReproducibilityError
from surroforge.plugins import get_exporter, get_reporter

# --- Reproduce ---------------------------------------------------------------

# Single source of truth for the reproduction divergence tolerance: the max abs
# difference over the recorded out-of-sample holdout sample below which a re-run
# counts as reproduced. Anything larger is a divergence.
REPRODUCTION_TOLERANCE = 1e-9


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


def reproduce(project: Project, run_id: str, *, echo: Any = None) -> ReproduceResult:
    """Re-run a stored run from its config snapshot and compare to the original (PRD 4.5)."""
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

    return ReproduceResult(
        parent=parent,
        reproduction=reproduction,
        bit_identical=bit_identical,
        max_abs_diff=max_abs_diff,
        data_hash_match=data_hash_match,
        tolerance=tolerance,
        verdict=_verdict(data_hash_match, bit_identical, comparable, max_abs_diff, tolerance),
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


def assert_reproducible(
    result: ReproduceResult, *, tolerance: float | None = None
) -> None:
    """Raise if a reproduction did not verifiably reproduce (used in CI/tests).

    Judges against the tolerance the run declared (``result.tolerance``, Story 5.4)
    unless an explicit ``tolerance`` override is supplied.
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
    if result.max_abs_diff > tol:
        raise ReproducibilityError(
            f"Reproduction diverged: max abs prediction diff "
            f"{result.max_abs_diff:.3e} > tolerance {tol:.1e}."
        )


# --- Compare -----------------------------------------------------------------

# Whether a smaller value is better, per metric (drives "best run" selection).
_LOWER_IS_BETTER = {"rmse": True, "mae": True, "r2": False}


def compare_runs(project: Project, run_ids: list[str], *, metric: str = "rmse") -> dict[str, Any]:
    """Tabulate test/CV metrics across runs and pick the best by ``metric`` (PRD 4.1)."""
    if not run_ids:
        raise DataError("Provide at least one run id to compare.")
    runs = [project.run(rid) for rid in run_ids]

    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = run.metrics
        cv = metrics["cross_validation"]
        names = run.config.validation.metrics
        rows.append(
            {
                "run_id": run.run_id,
                "method": run.method,
                "test": {m: metrics["test"].get(m) for m in names},
                "cv_mean": {m: cv[m]["mean"] for m in names if m in cv},
            }
        )

    best = _select_best(rows, metric)
    return {"metric": metric, "rows": rows, "best": best}


def _select_best(rows: list[dict[str, Any]], metric: str) -> str | None:
    candidates = [r for r in rows if r["test"].get(metric) is not None]
    if not candidates:
        return None
    lower_better = _LOWER_IS_BETTER.get(metric, True)
    chosen = (min if lower_better else max)(candidates, key=lambda r: r["test"][metric])
    return chosen["run_id"]


# --- Predict -----------------------------------------------------------------

def predict_with_run(run: Run, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict outputs for new inputs using a trained run's model (PRD 4.1)."""
    config = run.config
    inputs = config.data.inputs
    missing = [c for c in inputs if c not in frame.columns]
    if missing:
        raise DataError(f"Input file is missing required columns: {missing}")

    x = frame[inputs].to_numpy(dtype=float)
    model = run.load_model()
    preds = np.asarray(model.predict(x), dtype=float)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    outputs = config.data.outputs
    columns = outputs if preds.shape[1] == len(outputs) else [
        f"output_{i}" for i in range(preds.shape[1])
    ]
    result = frame.copy()
    for i, col in enumerate(columns):
        result[col] = preds[:, i]
    return result


def predict_from_file(run: Run, input_path: str | Path) -> pd.DataFrame:
    return predict_with_run(run, load_table(input_path))


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
