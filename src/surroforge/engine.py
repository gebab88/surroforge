"""Training orchestration — the engine shared by the CLI and the Python API.

One call to :func:`run_training` takes a project and a validated config and
produces a fully persisted, reproducible run: holdout metrics, cross-validation,
a final model fitted on all data, and a complete lineage record (PRD 4.1, 4.5).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from sklearn.model_selection import train_test_split

from surroforge.core.config import ExperimentConfig, parse_config
from surroforge.core.dataset import Dataset
from surroforge.core.project import Project
from surroforge.core.run import (
    CONFIG_SNAPSHOT,
    LINEAGE_FILE,
    LOG_FILE,
    METRICS_FILE,
    MODEL_ONNX,
    MODEL_PICKLE,
    Run,
)
from surroforge.data.loader import expand_field_columns, select_columns
from surroforge.errors import DataError, TrainerError
from surroforge.exporters.onnx_exporter import ONNXExporter
from surroforge.exporters.pickle_exporter import PickleExporter
from surroforge.persistence import capture_environment, write_json, write_npz, write_yaml
from surroforge.plugins import get_trainer
from surroforge.trainers.base import BaseTrainer
from surroforge.validators import compute_metrics, cross_validate, validate_metric_names


@dataclass
class _Log:
    lines: list[str] = field(default_factory=list)
    echo: Any = None  # optional callable for live output

    def __call__(self, message: str) -> None:
        self.lines.append(message)
        if self.echo is not None:
            self.echo(message)


# Cap on the number of out-of-sample rows stored as the reproducibility
# fingerprint, so lineage.json stays a compact metadata file on large datasets.
# The holdout split order is deterministic, so the cap is stable across runs.
_FINGERPRINT_MAX_ROWS = 512


# Optional trainer surfaces, gated structurally so the engine stays ROM-agnostic
# (spine theorem #36 — no reduced_field import here). A reducing trainer exposes
# these; every scalar trainer does not, so `isinstance` is False and that lineage
# is untouched. Stated as @runtime_checkable Protocols rather than bare hasattr
# so the duck-typed surface is both documented (the expected signature) and
# narrowed by the type checker after the guard.
@runtime_checkable
class _SupportsReproducibilityDiagnostics(Protocol):
    def reproducibility_diagnostics(self, x: np.ndarray) -> dict[str, Any]: ...


@runtime_checkable
class _SupportsResolvedConfigOverrides(Protocol):
    def resolved_config_overrides(self) -> dict[str, Any] | None: ...


@runtime_checkable
class _SupportsReductionArtifacts(Protocol):
    def reduction_artifacts(self) -> dict[str, np.ndarray]: ...


def run_training(
    project: Project,
    config: ExperimentConfig,
    *,
    dataset: Dataset | None = None,
    parent_run: str | None = None,
    echo: Any = None,
    export_onnx: bool = False,
) -> Run:
    """Execute one training run end-to-end and persist it. Returns the new :class:`Run`.

    ``export_onnx`` is opt-in (default ``False``): converting to ONNX pulls in and
    runs ``skl2onnx`` on every train, which dominates the cost of a cheap method
    while many runs never use the ONNX artifact. The pickle is always written, and
    ONNX can be produced on demand later via ``surroforge export`` / ``export_model``
    (both reconstruct it from the saved model), so nothing is lost by deferring it.
    """
    log = _Log(echo=echo)

    # 1. Resolve and validate data ------------------------------------------------
    if dataset is not None:
        frame = project.load_dataset_frame(dataset)
        data_hash = dataset.hash
        source_label = f"dataset:{dataset.name}"
    else:
        frame, data_hash, source_label = project.resolve_source(config.data.source)
    log(f"Data source: {source_label} ({frame.shape[0]} rows, {frame.shape[1]} columns)")

    x, y = select_columns(frame, config.data.inputs, config.data.outputs)
    validate_metric_names(config.validation.metrics)
    metrics = config.validation.metrics
    n_features = x.shape[1]
    log(f"Inputs: {config.data.inputs} -> Outputs: {config.data.outputs}")

    # 2. Resolve the method and build a fresh-trainer factory ---------------------
    trainer_cls = get_trainer(config.model.method)
    hyper = config.model.hyperparameters

    def factory() -> BaseTrainer:
        return trainer_cls(**hyper)

    log(f"Method: {config.model.method} {hyper if hyper else ''}".strip())

    # 3. Holdout split + train/test metrics --------------------------------------
    split = config.data.split
    x_train, x_test, y_train, y_test = cast(
        "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]",
        train_test_split(
            x, y, test_size=split.test_fraction, random_state=split.seed, shuffle=True
        ),
    )
    holdout_model = factory()
    holdout_model.fit(x_train, y_train)
    holdout_test_pred = holdout_model.predict(x_test)
    train_metrics = compute_metrics(y_train, holdout_model.predict(x_train), metrics)
    test_metrics = compute_metrics(y_test, holdout_test_pred, metrics)
    log(
        f"Holdout split: {x_train.shape[0]} train / {x_test.shape[0]} test "
        f"(test_fraction={split.test_fraction}, seed={split.seed})"
    )
    log(f"Test metrics: {_fmt(test_metrics)}")

    # 4. Cross-validation ---------------------------------------------------------
    cv_cfg = config.validation.cross_validation
    try:
        cv = cross_validate(
            factory, x, y,
            folds=cv_cfg.folds, shuffle=cv_cfg.shuffle, seed=split.seed, metrics=metrics,
        )
    except ValueError as exc:
        raise DataError(str(exc)) from exc
    log(f"{cv_cfg.folds}-fold CV: " + ", ".join(
        f"{m}={cv[m]['mean']:.4g}±{cv[m]['std']:.2g}" for m in metrics
    ))

    # 5. Final model fitted on all data (the saved artifact) ----------------------
    final_model = factory()
    final_model.fit(x, y)
    # Reproducibility fingerprint over OUT-OF-SAMPLE holdout predictions: the
    # holdout model never saw x_test, so model divergence actually moves these
    # values — unlike in-sample predictions, which interpolating methods (Kriging)
    # reproduce near-exactly regardless of divergence (PRD 4.5).
    holdout_fingerprint = np.ascontiguousarray(holdout_test_pred, dtype=np.float64)[
        :_FINGERPRINT_MAX_ROWS
    ]
    prediction_hash = _prediction_fingerprint(holdout_fingerprint)

    # Holdout *actuals* row-aligned with the fingerprint above (same split, same
    # [:_FINGERPRINT_MAX_ROWS] slice over the same x_test row order), so the
    # Professional report can plot predicted-vs-actual. Inert to every hash and to
    # reproduce(), which keys on holdout_fingerprint — this is report-only data.
    holdout_actual = np.ascontiguousarray(y_test, dtype=np.float64)[:_FINGERPRINT_MAX_ROWS]

    # Field-primary reproducibility diagnostics (Story R2.3): a reducing trainer exposes
    # an optional hook; scalar trainers do not, so their lineage is unchanged and the
    # engine stays ROM-agnostic (spine theorem #36 — no reduced_field import here). The
    # diagnostics come from the OUT-OF-SAMPLE holdout model on x_test, matching the field
    # fingerprint above. The basis hash + canonical coefficients are diagnostic, never a
    # verdict gate; the field itself is the verdict (spec-field-rom-fingerprint.md).
    field_extras: dict[str, Any] = {}
    if isinstance(holdout_model, _SupportsReproducibilityDiagnostics):
        diag = holdout_model.reproducibility_diagnostics(x_test)
        coeffs = np.ascontiguousarray(diag["holdout_coefficients"], dtype=np.float64)[
            :_FINGERPRINT_MAX_ROWS
        ]
        field_extras = {
            "basis_hash": diag["basis_hash"],
            "holdout_coefficients": coeffs.tolist(),
            "reduction": diag["reduction"],
            # Fraction of field energy the verdict verifies (Story R2.5). An SVD-derived
            # float, so — like energy_captured — it stays OUT of the spec hash. Field runs
            # only; a scalar trainer has no hook, so scalar lineage is unchanged (R2.4).
            "probe_coverage_energy": float(diag["probe_coverage_energy"]),
        }

    # Canonical config (Story R1.2): a trainer may resolve part of its spec at fit time
    # (e.g. `rank: adaptive` -> a concrete int) and expose it via the duck-typed
    # resolved_config_overrides() hook. The engine deep-merges that opaque dict into the
    # config and re-validates, so the SNAPSHOT — not the input YAML — is canonical and
    # config_hash/spec_hash reflect the resolved run. Without the hook (every scalar
    # trainer) or when it returns None (an energy-specified field run) the config is
    # untouched, keeping those snapshots and hashes byte-identical (R2.3/R2.4).
    #
    # The hook is queried on the HOLDOUT model — the model that defines the
    # reproducibility fingerprint and the R2.3 lineage diagnostics above — never the
    # final model: when an adaptive threshold straddles a mode the two resolve different
    # ranks, and a snapshot carrying the final model's rank would make reproduce()
    # re-train the holdout at the wrong truncation (a false "diverged" on a faithful
    # re-run) and contradict lineage `reduction.rank`. The residual: the persisted final
    # artifact may then hold more/fewer modes than the snapshot rank — a diagnostic
    # asymmetry, not a verdict one.
    canonical_config = config
    if isinstance(holdout_model, _SupportsResolvedConfigOverrides):
        overrides = holdout_model.resolved_config_overrides()
        if overrides:
            merged = _deep_merge(config.canonical_dict(), overrides)
            canonical_config = parse_config(merged, origin="<resolved config>")
            log(f"Resolved config overrides written to snapshot: {overrides}")

    metrics_doc = {
        "train": train_metrics,
        "test": test_metrics,
        "cross_validation": cv,
        "n_samples": int(x.shape[0]),
        "n_inputs": int(n_features),
        "n_outputs": int(y.shape[1]),
    }

    # 6. Persist ------------------------------------------------------------------
    run_dir = project.allocate_run_dir()
    run = Run(run_dir)
    log(f"Run id: {run.run_id}")

    write_yaml(run_dir / CONFIG_SNAPSHOT, canonical_config.canonical_dict())
    PickleExporter().export(final_model, run_dir / MODEL_PICKLE, n_features=n_features)

    # Content-addressed reduction-artifact store (Story R4.1): the FINAL model's
    # basis/mean/spectrum — the arrays `rom inspect` (R4.2) reads back; predict uses
    # the same arrays from inside the pickle — land in a project-level store,
    # deduplicated across runs. The address is sha256 over every stored array's
    # bytes, each framed with its key, dtype, and shape, in sorted key order (an npz
    # is a zip; its FILE bytes embed timestamps and never dedup — and unframed raw
    # bytes are ambiguous across shapes/dtypes). Deliberately the FINAL model, not
    # the holdout: this is a storage artifact of the shipped pickle, never a
    # fingerprint input — reproduce() does not read the store, and the lineage
    # basis_hash above remains the (different) holdout diagnostic.
    basis_file: str | None = None
    if isinstance(final_model, _SupportsReductionArtifacts):
        artifacts = final_model.reduction_artifacts()
        required = ("basis", "mean", "singular_values")
        missing = [key for key in required if key not in artifacts]
        if missing:
            raise TrainerError(
                f"reduction_artifacts() is missing required keys {missing}; "
                f"it must provide at least {list(required)}."
            )
        digest = hashlib.sha256()
        arrays: dict[str, np.ndarray] = {}
        for key in sorted(artifacts):
            # Contract before content: 'file' collides with np.savez's own
            # parameter, and the framing below relies on keys free of ':'/';'.
            if key == "file" or not re.fullmatch(r"[A-Za-z0-9_]+", key):
                raise TrainerError(
                    f"reduction_artifacts() key '{key}' is not a valid artifact "
                    "name. Use [A-Za-z0-9_]+ and avoid the reserved name 'file'."
                )
            array = np.ascontiguousarray(artifacts[key])
            if array.dtype.kind not in "biufc":
                raise TrainerError(
                    f"reduction_artifacts()['{key}'] has non-numeric dtype "
                    f"'{array.dtype}'. Artifacts must be numeric arrays."
                )
            digest.update(f"{key}:{array.dtype.str}:{array.shape};".encode())
            digest.update(array.tobytes())
            arrays[key] = array
        store_path = project.basis_dir / f"{digest.hexdigest()}.npz"
        basis_file = f"basis/{store_path.name}"
        if store_path.exists():
            log(f"Reduction basis reused (dedup hit): {basis_file}")
        else:
            write_npz(store_path, **arrays)
            log(f"Reduction basis stored: {basis_file}")

    onnx_exported = False
    if export_onnx and final_model.supports_export("onnx"):
        try:
            ONNXExporter().export(final_model, run_dir / MODEL_ONNX, n_features=n_features)
            onnx_exported = True
        except Exception as exc:  # ONNX is best-effort; never fail a run over it
            log(f"ONNX export skipped: {exc}")

    write_json(run_dir / METRICS_FILE, metrics_doc)

    lineage = {
        "run_id": run.run_id,
        "created_at": _now_iso(),
        "experiment": config.experiment,
        "method": config.model.method,
        "config_hash": _config_hash(canonical_config),
        "data_hash": data_hash,
        "data_source": source_label,
        "seed": split.seed,
        "prediction_hash": prediction_hash,
        "holdout_fingerprint": holdout_fingerprint.tolist(),
        "holdout_actual": holdout_actual.tolist(),
        # Recorded so reproduce() judges this run against the tolerance the method
        # declared at train time, surviving later changes to the trainer (Story 5.4).
        "reproduction_tolerance": float(final_model.reproduction_tolerance),
        "onnx_exported": onnx_exported,
        "parent_run": parent_run,
        **capture_environment(),
    }
    if field_extras:
        lineage.update(field_extras)
        # D2: fold the field tolerance into a dedicated spec hash so widening it is a
        # different run, without disturbing config_hash semantics.
        lineage["spec_hash"] = _spec_hash(canonical_config, final_model.reproduction_tolerance)
    if basis_file is not None:
        # Reducing trainers only (gated on the duck-typed reduction_artifacts
        # surface) — built-in scalar lineage stays byte-identical (R2.4).
        # POSIX form: lineage is cross-OS data, never a native path.
        lineage["basis_file"] = basis_file
        # The expanded snapshot column names, so `predict --field` writes a frame
        # that re-ingests under the run's own outputs pattern (Story R4.1, #21).
        lineage["output_columns"] = expand_field_columns(frame, canonical_config.data.outputs)
    write_json(run_dir / LINEAGE_FILE, lineage)
    (run_dir / LOG_FILE).write_text("\n".join(log.lines) + "\n")

    return run


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``base`` (dicts merge, else replace)."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _config_hash(config: ExperimentConfig) -> str:
    return hashlib.sha256(config.canonical_json().encode("utf-8")).hexdigest()


def _spec_hash(config: ExperimentConfig, tolerance: float) -> str:
    """Hash of (config + field tolerance) for a field run (Story R2.3, D2).

    Folds the field-energy-relative tolerance into the spec identity, so *widening the
    tolerance is a different run* (a fresh spec hash), not a looser compare on the same
    spec. The config already carries the user's reduction request, so the spec is fully
    captured. The *measured* reduction (resolved rank, ``energy_captured``) is recorded
    in lineage as a diagnostic but deliberately kept OUT of this hash: it is a SVD-derived
    float that varies at ULP level on non-reproducible BLAS (Apple Accelerate), and a
    reproducibility hash must not depend on that noise. Kept separate from ``config_hash``
    so the latter's semantics are undisturbed.
    """
    payload = config.canonical_json() + repr(float(tolerance))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prediction_fingerprint(preds: np.ndarray) -> str:
    """SHA-256 over a prediction sample (PRD 4.5).

    In :func:`run_training` this is computed over the out-of-sample holdout
    predictions, so two runs share a fingerprint only when they reproduce the
    same out-of-sample behaviour — not merely the same training-point values.
    """
    arr = np.ascontiguousarray(preds, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _fmt(metrics: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4g}" for k, v in metrics.items())


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
