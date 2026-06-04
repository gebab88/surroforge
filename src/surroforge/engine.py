"""Training orchestration — the engine shared by the CLI and the Python API.

One call to :func:`run_training` takes a project and a validated config and
produces a fully persisted, reproducible run: holdout metrics, cross-validation,
a final model fitted on all data, and a complete lineage record (PRD 4.1, 4.5).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from surroforge.core.config import ExperimentConfig
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
from surroforge.data.loader import select_columns
from surroforge.errors import DataError
from surroforge.exporters.onnx_exporter import ONNXExporter
from surroforge.exporters.pickle_exporter import PickleExporter
from surroforge.persistence import capture_environment, write_json, write_yaml
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


def run_training(
    project: Project,
    config: ExperimentConfig,
    *,
    dataset: Dataset | None = None,
    parent_run: str | None = None,
    echo: Any = None,
) -> Run:
    """Execute one training run end-to-end and persist it. Returns the new :class:`Run`."""
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
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=split.test_fraction, random_state=split.seed, shuffle=True
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

    write_yaml(run_dir / CONFIG_SNAPSHOT, config.model_dump(mode="json"))
    PickleExporter().export(final_model, run_dir / MODEL_PICKLE, n_features=n_features)

    onnx_exported = False
    if final_model.supports_onnx:
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
        "config_hash": _config_hash(config),
        "data_hash": data_hash,
        "data_source": source_label,
        "seed": split.seed,
        "prediction_hash": prediction_hash,
        "holdout_fingerprint": holdout_fingerprint.tolist(),
        "onnx_exported": onnx_exported,
        "parent_run": parent_run,
        **capture_environment(),
    }
    write_json(run_dir / LINEAGE_FILE, lineage)
    (run_dir / LOG_FILE).write_text("\n".join(log.lines) + "\n")

    return run


def _config_hash(config: ExperimentConfig) -> str:
    return hashlib.sha256(config.canonical_json().encode("utf-8")).hexdigest()


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
