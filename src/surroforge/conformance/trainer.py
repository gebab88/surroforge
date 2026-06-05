"""Trainer conformance checks (PRD 4.11, Story 5.3).

A framework-agnostic suite a plugin author runs against their trainer class to
confirm it honours the Core contract before publishing. It drives the trainer
*directly* (not through the registry) so it works on an unregistered class, and
mirrors the engine's holdout-split → fit → out-of-sample-predict → fingerprint
flow so "fingerprint participation" means the same thing here as in real runs.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from surroforge.errors import ConformanceError, SurroForgeError
from surroforge.trainers.base import BaseTrainer

#: Determinism tolerance — mirrors ``services.REPRODUCTION_TOLERANCE`` so a
#: trainer that passes here also reproduces cleanly through ``reproduce()``.
DEFAULT_TOLERANCE = 1e-9


@dataclass
class CheckResult:
    """Outcome of a single conformance check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class ConformanceReport:
    """The collected results of running the suite against one trainer."""

    trainer: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        lines = [f"Conformance report for trainer '{self.trainer}':"]
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{mark}] {r.name}" + (f" — {r.detail}" if r.detail else ""))
        return "\n".join(lines)

    def raise_for_status(self) -> None:
        """Raise :class:`ConformanceError` if any check failed; else return."""
        if self.passed:
            return
        failed = ", ".join(r.name for r in self.failures)
        raise ConformanceError(
            f"Trainer '{self.trainer}' failed conformance checks: {failed}.\n{self.summary()}"
        )


def _make_data(seed: int, n_samples: int, n_features: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    # A smooth, single-output target every reasonable surrogate can fit; single
    # output keeps SMT methods (Kriging/RBF) inside their MVP envelope.
    y = (x**2).sum(axis=1, keepdims=True) + 0.5 * x[:, :1]
    return train_test_split(x, y, test_size=0.25, random_state=seed, shuffle=True)


def _fingerprint(preds: np.ndarray) -> str:
    """SHA-256 over float64 prediction bytes — same scheme as the engine's hash.

    The engine caps the hashed sample at ``_FINGERPRINT_MAX_ROWS`` rows; the suite
    does not, so the two hashes coincide only while the holdout fits under that cap
    (always true at the default ``n_samples``). The *algorithm* is identical.
    """
    arr = np.ascontiguousarray(preds, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def check_trainer(
    trainer_cls: type[BaseTrainer],
    *,
    hyperparameters: dict[str, Any] | None = None,
    seed: int = 0,
    n_samples: int = 60,
    n_features: int = 3,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ConformanceReport:
    """Run every contract check against ``trainer_cls`` and return a report.

    The call never raises on a *failed check* — failures are collected so the
    caller sees the whole picture. Call :meth:`ConformanceReport.raise_for_status`
    (or inspect :attr:`ConformanceReport.passed`) to turn failures into an error.
    """
    hyper = dict(hyperparameters or {})
    # Resolve a display name defensively: this runs before the record() machinery,
    # so it must never raise — even when handed a non-class (None, an int, an
    # instance). A bad argument is surfaced as a clean subclass_of_base_trainer FAIL.
    name = getattr(trainer_cls, "name", None)
    if not (isinstance(name, str) and name):
        name = getattr(trainer_cls, "__name__", None) or repr(trainer_cls)
    report = ConformanceReport(trainer=str(name))
    ctx: dict[str, Any] = {}

    def record(check_name: str, fn: Callable[[], str | None]) -> None:
        try:
            report.results.append(CheckResult(check_name, True, fn() or ""))
        except Exception as exc:
            report.results.append(
                CheckResult(check_name, False, f"{type(exc).__name__}: {exc}")
            )

    x_train, x_test, y_train, y_test = _make_data(seed, n_samples, n_features)

    def _subclass() -> str:
        if not (isinstance(trainer_cls, type) and issubclass(trainer_cls, BaseTrainer)):
            raise AssertionError("must subclass surroforge.trainers.base.BaseTrainer")
        if not isinstance(getattr(trainer_cls, "name", None), str) or not trainer_cls.name:
            raise AssertionError("class attribute 'name' must be a non-empty string")
        return f"name={trainer_cls.name!r}"

    def _construct() -> str:
        t = trainer_cls(**hyper)
        if not isinstance(getattr(t, "hyperparameters", None), dict):
            raise AssertionError("hyperparameters must be stored as a dict (call super().__init__)")
        return "constructs and stores hyperparameters"

    def _fit() -> None:
        t = trainer_cls(**hyper)
        if t.fit(x_train, y_train) is not None:
            raise AssertionError("fit must return None")
        ctx["trainer"] = t

    def _predict_2d() -> str:
        t = ctx.get("trainer")
        if t is None:
            raise AssertionError("skipped: trainer did not fit")
        raw = t.predict(x_test)
        preds = np.asarray(raw, dtype=np.float64)
        if preds.ndim != 2:
            raise AssertionError(f"predict must return a 2-D array, got ndim={preds.ndim}")
        if preds.shape[0] != x_test.shape[0]:
            raise AssertionError(
                f"predict returned {preds.shape[0]} rows for {x_test.shape[0]} inputs"
            )
        # The output-column count is part of the contract: the engine feeds these
        # straight into compute_metrics, which rejects a mismatch with a bare
        # ValueError. A trainer fitted on n_outputs columns must predict that many.
        n_outputs = y_train.shape[1]
        if preds.shape[1] != n_outputs:
            raise AssertionError(
                f"predict returned {preds.shape[1]} output column(s) for "
                f"{n_outputs}-output training data"
            )
        ctx["preds"] = np.ascontiguousarray(preds, dtype=np.float64)
        return f"shape={preds.shape}"

    def _picklable() -> str:
        t = ctx.get("trainer")
        if t is None:
            raise AssertionError("skipped: trainer did not fit")
        restored = pickle.loads(pickle.dumps(t))
        # Re-derive the original predictions here rather than reusing ctx["preds"]
        # (which is only set when predict_returns_2d fully passed) so the equality
        # check still runs — and means something — even for an off-contract predict.
        original = np.ascontiguousarray(t.predict(x_test), dtype=np.float64)
        rpreds = np.ascontiguousarray(restored.predict(x_test), dtype=np.float64)
        if rpreds.shape != original.shape:
            raise AssertionError("unpickled model predicts a different shape")
        diff = float(np.max(np.abs(rpreds - original))) if original.size else 0.0
        if diff > tolerance:
            raise AssertionError(
                f"unpickled model predicts differently (max abs diff {diff:.2e})"
            )
        return "round-trips through pickle and predicts identically"

    def _deterministic() -> str:
        # NOTE: this probes determinism on a single smooth target, so it cannot
        # provoke divergence in a trainer whose only stochasticity is multi-start
        # optimisation on a multimodal surface — document that limitation. We do
        # advance the global RNG between the two fits so a trainer that silently
        # leans on the ambient np.random state (no fixed seed) is exposed here.
        a = trainer_cls(**hyper)
        a.fit(x_train, y_train)
        np.random.random(8)  # perturb ambient global RNG between fits
        b = trainer_cls(**hyper)
        b.fit(x_train, y_train)
        pa = np.ascontiguousarray(a.predict(x_test), dtype=np.float64)
        pb = np.ascontiguousarray(b.predict(x_test), dtype=np.float64)
        if pa.shape != pb.shape:
            raise AssertionError("two independent fits produced different prediction shapes")
        # Honour the method's declared reproduction_tolerance (Story 5.4): a method
        # that legitimately reproduces only within a band must not be failed for it.
        # Never tighten below the suite's own floor.
        declared = getattr(trainer_cls, "reproduction_tolerance", tolerance)
        tol = tolerance
        if isinstance(declared, (int, float)) and not isinstance(declared, bool):
            tol = max(tolerance, float(declared))
        diff = float(np.max(np.abs(pa - pb)))
        if diff > tol:
            raise AssertionError(
                f"two independent fits diverged by {diff:.2e} (> tolerance {tol:.0e})"
            )
        return "bit-identical" if diff == 0.0 else f"reproducible within {diff:.2e}"

    def _fingerprint_participation() -> str:
        preds = ctx.get("preds")
        if preds is None:
            raise AssertionError("skipped: no valid 2-D predictions to fingerprint")
        if preds.size == 0:
            raise AssertionError("no predictions produced")
        if not np.all(np.isfinite(preds)):
            raise AssertionError("predictions contain NaN/inf and cannot be fingerprinted")
        fp = _fingerprint(preds)
        if len(fp) != 64:
            raise AssertionError("could not produce a SHA-256 fingerprint")
        return f"holdout fingerprint {fp[:12]}…"

    def _surroforge_error_only() -> str:
        # Misuse probe: predict before fit. A conforming trainer surfaces this
        # as a SurroForgeError — never a bare AttributeError/ValueError/etc.
        # Construct OUTSIDE the misuse try/except so a constructor failure is not
        # misreported as a predict-before-fit problem.
        try:
            inst = trainer_cls(**hyper)
        except Exception as exc:
            raise AssertionError(
                f"could not construct trainer with the given hyperparameters "
                f"({type(exc).__name__}: {exc}); pass valid hyperparameters"
            ) from None
        try:
            inst.predict(x_test)
        except SurroForgeError:
            return "predict-before-fit raises a SurroForgeError"
        except Exception as exc:
            raise AssertionError(
                f"predict-before-fit raised {type(exc).__name__}; "
                "trainers must fail with a SurroForgeError (e.g. TrainerError)"
            ) from None
        raise AssertionError("predict-before-fit did not raise (it must)")

    def _invalid_hyperparameter_error_only() -> str:
        # If a trainer *validates* its hyperparameters, that validation must raise
        # a SurroForgeError (the built-ins raise TrainerError). Two outcomes are
        # also acceptable: ignoring the unknown key (BaseTrainer's default), or a
        # plain TypeError from an explicit __init__ signature. Anything else (a
        # bare ValueError/KeyError/RuntimeError) is the anti-pattern we flag.
        bogus = "__surroforge_conformance_bogus_hyperparameter__"
        try:
            trainer_cls(**{**hyper, bogus: object()})
        except SurroForgeError:
            return "rejects an unknown hyperparameter with a SurroForgeError"
        except TypeError:
            return "rejects an unknown hyperparameter via its __init__ signature"
        except Exception as exc:
            raise AssertionError(
                f"unknown hyperparameter raised {type(exc).__name__}; a trainer that "
                "validates hyperparameters must use a SurroForgeError (e.g. TrainerError)"
            ) from None
        return "ignores unknown hyperparameters (no validation — acceptable)"

    def _declares_capabilities() -> str:
        # Story 5.4: a trainer declares its reproduction tolerance and export
        # capability, which the engine, reproduce(), and export now rely on.
        tol = getattr(trainer_cls, "reproduction_tolerance", None)
        if isinstance(tol, bool) or not isinstance(tol, (int, float)):
            raise AssertionError("reproduction_tolerance must be a number")
        if tol < 0 or not np.isfinite(tol):
            raise AssertionError(
                f"reproduction_tolerance must be finite and >= 0 (got {tol!r})"
            )
        fmts = getattr(trainer_cls, "export_formats", None)
        try:
            fmt_list = list(fmts)  # type: ignore[arg-type]
        except TypeError:
            raise AssertionError(
                "export_formats must be an iterable of format strings"
            ) from None
        if not all(isinstance(f, str) for f in fmt_list):
            raise AssertionError("export_formats must contain only strings")
        if "pickle" not in fmt_list:
            raise AssertionError(
                "every trainer must declare 'pickle' export (all trainers are picklable)"
            )
        kind = "deterministic" if tol == 0 else f"tolerant@{float(tol):.0e}"
        return f"{kind}, exports {sorted(fmt_list)}"

    record("subclass_of_base_trainer", _subclass)
    record("constructs_from_hyperparameters", _construct)
    record("fit_returns_none", _fit)
    record("predict_returns_2d", _predict_2d)
    record("picklable", _picklable)
    record("deterministic", _deterministic)
    record("fingerprint_participation", _fingerprint_participation)
    record("surroforge_error_only", _surroforge_error_only)
    record("invalid_hyperparameter_error_only", _invalid_hyperparameter_error_only)
    record("declares_capabilities", _declares_capabilities)
    return report
