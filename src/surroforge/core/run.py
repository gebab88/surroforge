"""A persisted training run (PRD 4.5, 7.1, 7.2).

``Run`` is a thin, lazy view over a ``runs/<run-id>/`` directory: it reads the
lineage, metrics, config snapshot, and model artifacts on demand rather than
holding them in memory.
"""

from __future__ import annotations

import re
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any

from surroforge.core.config import ExperimentConfig, parse_config
from surroforge.errors import ProjectError, RunNotFoundError
from surroforge.persistence import is_locally_produced, read_json, read_yaml

CONFIG_SNAPSHOT = "config.snapshot.yaml"
METRICS_FILE = "metrics.json"
LINEAGE_FILE = "lineage.json"
MODEL_PICKLE = "model.pkl"
MODEL_ONNX = "model.onnx"
LOG_FILE = "log.txt"

class ProvenanceWarning(UserWarning):
    """Loading a model artifact not confirmed as produced on this machine.

    A subclass (rather than a bare ``UserWarning``) so adapters can render it in
    their own style — the CLI prints a sober ``Warning:`` line — while library
    callers still filter it with the standard ``warnings`` machinery.
    """


class FieldCoverageWarning(UserWarning):
    """A field run whose reproduce verdict verifies only a low fraction of the field.

    The field-energy-relative verdict checks the reconstructed field, which represents
    ``probe_coverage_energy`` of the true field; the truncated tail modes are
    unverifiable. Advisory, **never** a verdict gate (Story R2.5) — like
    :class:`ProvenanceWarning`, the CLI renders it as a sober ``Warning:`` line while
    library callers filter it with the standard ``warnings`` machinery.
    """


# Artifacts we have already warned about this process, so the provenance warning
# fires once per artifact rather than on every predict/export call (Story 4.4).
_WARNED_ARTIFACTS: set[str] = set()


def _warn_if_foreign_artifact(model_path: Path, lineage: dict[str, Any]) -> None:
    """Warn once before unpickling a model not produced on this machine.

    Unpickling executes code embedded in the file, so loading a shared
    ``model.pkl`` is an arbitrary-code-execution risk (Story 4.4 / AR12).
    Locally-produced artifacts load silently. Emitted via ``warnings`` so it
    reaches both CLI and library callers and is *not* silenced by ``--quiet``
    (a security signal); users can still suppress it with ``-W ignore``.
    """
    if is_locally_produced(lineage):
        return
    key = str(model_path.resolve())
    if key in _WARNED_ARTIFACTS:
        return
    _WARNED_ARTIFACTS.add(key)
    warnings.warn(
        f"Model artifact '{model_path}' was not confirmed as produced on this "
        "machine. Loading a pickle executes code embedded in the file — only "
        "load model artifacts from sources you trust.",
        ProvenanceWarning,
        stacklevel=3,
    )


class Run:
    """Read access to a single run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name

    def exists(self) -> bool:
        return (self.run_dir / LINEAGE_FILE).is_file()

    def _require(self) -> None:
        if not self.exists():
            raise RunNotFoundError(f"Run '{self.run_id}' not found at {self.run_dir}")

    @cached_property
    def lineage(self) -> dict[str, Any]:
        self._require()
        return read_json(self.run_dir / LINEAGE_FILE)

    @cached_property
    def metrics(self) -> dict[str, Any]:
        self._require()
        return read_json(self.run_dir / METRICS_FILE)

    @cached_property
    def config_dict(self) -> dict[str, Any]:
        self._require()
        return read_yaml(self.run_dir / CONFIG_SNAPSHOT)

    @cached_property
    def config(self) -> ExperimentConfig:
        return parse_config(self.config_dict, origin=(self.run_dir / CONFIG_SNAPSHOT).as_posix())

    # Convenience accessors over lineage --------------------------------------

    @property
    def created_at(self) -> str:
        return self.lineage.get("created_at", "")

    @property
    def config_hash(self) -> str:
        return self.lineage.get("config_hash", "")

    @property
    def data_hash(self) -> str:
        return self.lineage.get("data_hash", "")

    @property
    def seed(self) -> int:
        return int(self.lineage.get("seed", 0))

    @property
    def method(self) -> str:
        return self.config.model.method

    @property
    def parent_run(self) -> str | None:
        return self.lineage.get("parent_run")

    @property
    def holdout_fingerprint(self) -> list[Any]:
        """Out-of-sample holdout predictions recorded for the reproducibility verdict.

        For a field run this is the reconstructed out-of-sample holdout *field*; the
        field-primary verdict compares it by field-energy-relative diff (Story R2.3).
        """
        return self.lineage.get("holdout_fingerprint", [])

    @property
    def holdout_actual(self) -> list[Any]:
        """Out-of-sample holdout actual targets, row-aligned with :attr:`holdout_fingerprint`.

        Persisted next to the prediction fingerprint so the Professional HTML report can
        plot predicted-vs-actual. Inert to the reproduce verdict (which keys on the
        fingerprint) and to every hash. Empty list for runs trained before this was
        recorded.
        """
        return self.lineage.get("holdout_actual", [])

    # Field-run diagnostics (Story R2.3) — empty/blank for scalar runs ----------

    @property
    def holdout_coefficients(self) -> list[Any]:
        """Canonicalized out-of-sample modal coefficients (field-run diagnostic)."""
        return self.lineage.get("holdout_coefficients", [])

    @property
    def basis_hash(self) -> str:
        """Canonicalized basis hash — diagnostic only, never a verdict gate."""
        return self.lineage.get("basis_hash", "")

    @property
    def reduction(self) -> dict[str, Any]:
        """Resolved reduction spec for a field run (empty dict for a scalar run)."""
        return self.lineage.get("reduction", {})

    @property
    def spec_hash(self) -> str:
        """Hashed spec (config + reduction + field tolerance); widening tolerance re-hashes."""
        return self.lineage.get("spec_hash", "")

    @property
    def probe_coverage_energy(self) -> float:
        """Fraction of field energy the reproduce verdict verifies (Story R2.5).

        Field runs only: the retained-mode POD energy share of the **holdout model** (the
        same out-of-sample model the fingerprint keys on, not the final all-data artifact);
        out-of-probe energy (``1 - this``) is the truncation floor the verdict cannot see.
        A constant / zero-energy field has no modal energy to truncate, so it reads ``1.0``.
        A scalar / non-reduced run has no reduction, so it covers its own field by identity
        and the key is absent — the accessor returns ``1.0`` (also the legacy/pre-R2.5
        default).
        """
        return float(self.lineage.get("probe_coverage_energy", 1.0))

    @property
    def basis_file(self) -> str:
        """Project-relative POSIX path of the content-addressed reduction artifact
        (Story R4.1). Empty for scalar and pre-R4.1 field runs. The filename stem is
        the sha256 of the FINAL model's array bytes — a storage identity, distinct
        from the holdout-diagnostic :attr:`basis_hash`."""
        return self.lineage.get("basis_file", "")

    @property
    def output_columns(self) -> list[str]:
        """Expanded field output column names recorded at train time (Story R4.1).

        Lets ``predict --field`` write a frame that re-ingests under the run's own
        outputs pattern. Empty for scalar and pre-R4.1 field runs (callers fall back
        to the legacy naming chain). Lineage is data from disk: a non-list value
        (corruption, hand-editing) also falls back to the legacy chain rather than
        being type-laundered into garbage column names."""
        value = self.lineage.get("output_columns", [])
        if not isinstance(value, list):
            return []
        return [str(column) for column in value]

    @property
    def basis_path(self) -> Path | None:
        """Absolute path of the stored basis npz, or ``None`` for runs without one
        (scalar and pre-R4.1 field runs).

        Resolves against the project root via the canonical ``<project>/runs/<run-id>``
        layout this class already assumes. The lineage value is data from disk —
        it is validated against the exact store layout the engine writes, so a
        corrupted or hand-edited lineage can never resolve outside the project.
        Raises :class:`ProjectError` when the lineage references a store file that
        no longer exists (deleted ``basis/`` dir, partial project copy)."""
        if not self.basis_file:
            return None
        if not re.fullmatch(r"basis/[0-9a-f]{64}\.npz", self.basis_file):
            raise ProjectError(
                f"Lineage basis_file '{self.basis_file}' does not match the "
                "expected 'basis/<sha256>.npz' store layout — the lineage may be "
                "corrupted or hand-edited. Re-train the run to restore it."
            )
        parents = self.run_dir.parents
        if len(parents) < 2:
            raise ProjectError(
                f"Cannot resolve the project root from run directory "
                f"'{self.run_dir}'. Expected the canonical "
                "'<project>/runs/<run-id>' layout."
            )
        path = parents[1] / self.basis_file
        if not path.is_file():
            raise ProjectError(
                f"Reduction artifact '{self.basis_file}' is missing from the "
                "project store. Re-train the run to restore it (the store is "
                "content-addressed: an identical retrain rewrites the same file)."
            )
        return path

    # Artifacts ----------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        return self.run_dir / MODEL_PICKLE

    @property
    def onnx_path(self) -> Path:
        return self.run_dir / MODEL_ONNX

    def has_onnx(self) -> bool:
        return self.onnx_path.is_file()

    def load_model(self) -> Any:
        from surroforge.exporters.pickle_exporter import load_pickle

        self._require()
        _warn_if_foreign_artifact(self.model_path, self.lineage)
        return load_pickle(self.model_path)

    def export(self, fmt: str, output: str | Path | None = None) -> Path:
        """Export this run's fitted model to ``fmt`` and return the path written.

        The object-level binding of :func:`surroforge.services.export_model`, so the
        Python API mirrors the ``surroforge export`` CLI (``run = trainer.fit(...);
        run.export("onnx", "model.onnx")``). ``fmt`` is any registered export format the
        run's method declares — ``pickle`` and ``onnx`` are built in, ``pt`` is added by
        the ``surroforge-torch`` plugin; an unsupported pair raises
        :class:`~surroforge.errors.ExportError`. ``output`` defaults to ``model<ext>``
        inside the run directory.
        """
        from surroforge.services import export_model

        return export_model(self, fmt, output)
