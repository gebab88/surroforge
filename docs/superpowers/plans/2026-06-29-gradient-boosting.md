# Gradient Boosting Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `gradient-boosting` surrogate method to SurroForge Core, backed by scikit-learn's `HistGradientBoostingRegressor`, with single/multi-output fit, pickle + ONNX export, and conformance.

**Architecture:** A new Core builtin trainer `GradientBoostingTrainer(BaseTrainer)` in `src/surroforge/trainers/gradient_boosting.py`, mirroring `src/surroforge/trainers/mlp.py`. No new dependency (scikit-learn is already Core). Tree ensembles are scale-invariant, so there is no input scaler; multi-output uses `MultiOutputRegressor` (one booster per target); `early_stopping=False` makes the fit deterministic. Registered in the `_BUILTIN` table and exercised by the existing `check_trainer` conformance suite.

**Tech Stack:** Python 3.10+, scikit-learn (`HistGradientBoostingRegressor`, `MultiOutputRegressor`), pytest.

> **REVISION (2026-06-29, during execution): pickle-only.** ONNX export was dropped after
> Task 1. The installed `skl2onnx 1.20.0` + `onnx 1.21.0` cannot convert `HistGradientBoosting`
> (the `TreeEnsembleRegressor` converter emits numpy bool/uint32 attributes that onnx 1.21
> rejects; `mlp`/`polynomial` ONNX still works — only tree ensembles are affected). Therefore:
> `export_formats = frozenset({"pickle"})`; **Task 2 (ONNX Export) is VOID — skip it entirely**
> (no `to_onnx`, no ONNX tests); the Task 3 CHANGELOG says pickle, not ONNX. Every "pickle + ONNX"
> / `to_onnx` mention below is superseded. ONNX may return as a follow-up once the stack converts
> tree ensembles cleanly.

## Global Constraints

- Python `>=3.10`; ruff `line-length = 100`. The project lints `tests/` too — run `.venv/bin/python -m ruff check` on every file you create/modify and fix lint before committing.
- **No new dependency.** Use only scikit-learn (already Core). Import sklearn lazily inside methods (as `mlp` does), not at module top.
- Method name is exactly `gradient-boosting`. Class is `GradientBoostingTrainer`.
- Hyperparameters and defaults: `learning_rate=0.1` (`>0`), `max_iter=100` (`>=1`), `max_leaf_nodes=31` (`>=2` or `None`), `max_depth=None` (`>=1` or `None`), `min_samples_leaf=20` (`>=1`), `l2_regularization=0.0` (`>=0`), `random_state=0` (must be an `int`, not `bool`/`None`). Unknown hyperparameters raise `TrainerError`.
- `early_stopping=False` is set explicitly in the estimator.
- `reproduction_tolerance = 1e-6`; `export_formats = frozenset({"pickle", "onnx"})`.
- Domain errors are `TrainerError`/`ExportError` (from `surroforge.errors`), never raw sklearn errors at the API surface.
- **Do NOT write a "different seed diverges" test.** `HistGradientBoostingRegressor` has no row subsampling and bins on the full data when `n < 200_000`, so `random_state` does not change predictions in the small-DOE regime. Only assert same-seed reproducibility.
- Run all commands from the repo root `/Users/georg/Software/surroforge`. Python is `.venv/bin/python`. Commit with the git identity (the repo has none configured): `git -c user.name="Georg Babij" -c user.email="g.babij@googlemail.com" commit ...`.

---

## File Structure

- Create `src/surroforge/trainers/gradient_boosting.py` — `GradientBoostingTrainer`.
- Modify `src/surroforge/plugins/registry.py` — register `gradient-boosting` in `_BUILTIN[TRAINER_GROUP]`.
- Create `tests/test_gradient_boosting.py` — trainer tests (grows across Tasks 1–3).
- Modify `tests/test_conformance.py` — add `GradientBoostingTrainer` to the `BUILTINS` parametrize.
- Modify `CHANGELOG.md` — `[Unreleased]` method entry.

---

## Task 1: GradientBoostingTrainer Core + Registration

**Files:**
- Create: `src/surroforge/trainers/gradient_boosting.py`
- Modify: `src/surroforge/plugins/registry.py`
- Test: `tests/test_gradient_boosting.py`

**Interfaces:**
- Produces: `GradientBoostingTrainer(BaseTrainer)` with `name = "gradient-boosting"`, `reproduction_tolerance = 1e-6`, `export_formats = frozenset({"pickle", "onnx"})`, `__init__(learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, random_state=0, **extra)`, `fit(x, y) -> None`, `predict(x) -> np.ndarray` (2-D). Registered as `gradient-boosting`.

- [ ] **Step 1: Write the failing test file (core, no ONNX)**

Create `tests/test_gradient_boosting.py`:

```python
"""GradientBoostingTrainer — HistGradientBoosting surrogate via scikit-learn.

Mirrors test_mlp.py's structure: registration, the BaseTrainer contract (shapes,
multi-output, picklability), nonlinear-fit behaviour, hyperparameter validation,
and same-seed reproducibility. There is deliberately NO different-seed divergence
test: HistGradientBoosting bins on the full data for n < 200_000 and has no row
subsampling, so random_state does not change predictions in the DOE regime.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.metrics import r2_score

from surroforge.errors import TrainerError
from surroforge.plugins import available_trainers, get_trainer
from surroforge.trainers.gradient_boosting import GradientBoostingTrainer


def _nonlinear_problem(n=240, seed=0):
    """A nonlinear 2-D regression target a boosted ensemble fits well."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, (n, 2))
    y = (np.sin(2.0 * x[:, 0]) + x[:, 1] ** 2 - 0.5 * x[:, 0] * x[:, 1]).reshape(-1, 1)
    return x, y


# --- registration -----------------------------------------------------------

def test_registered_as_builtin():
    assert "gradient-boosting" in available_trainers()
    assert get_trainer("gradient-boosting") is GradientBoostingTrainer


# --- BaseTrainer contract ----------------------------------------------------

def test_predict_returns_2d_single_output():
    x, y = _nonlinear_problem()
    t = GradientBoostingTrainer()
    t.fit(x[:200], y[:200])
    preds = t.predict(x[200:])
    assert preds.ndim == 2 and preds.shape == (40, 1)


def test_multi_output_via_wrapper():
    x, y1 = _nonlinear_problem()
    y = np.column_stack([y1[:, 0], x[:, 0] ** 2 - x[:, 1]])  # 2 outputs
    t = GradientBoostingTrainer()
    t.fit(x[:200], y[:200])
    preds = t.predict(x[200:])
    assert preds.shape == (40, 2)


def test_single_output_emits_no_dataconversion_warning():
    """Pass a 1-D target for a single output so sklearn does not warn."""
    x, y = _nonlinear_problem(n=80)
    t = GradientBoostingTrainer(max_iter=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        t.fit(x, y)


def test_picklable_round_trip():
    import pickle

    x, y = _nonlinear_problem(n=120)
    t = GradientBoostingTrainer(max_iter=50)
    t.fit(x, y)
    restored = pickle.loads(pickle.dumps(t))
    np.testing.assert_array_equal(restored.predict(x), t.predict(x))


def test_learns_nonlinear_function():
    x, y = _nonlinear_problem()
    t = GradientBoostingTrainer()
    t.fit(x[:200], y[:200])
    score = r2_score(y[200:], t.predict(x[200:]))
    assert score > 0.9, f"R²={score:.3f} — gradient boosting failed to fit a nonlinear target"


def test_predict_before_fit_raises():
    with pytest.raises(TrainerError, match="not fitted"):
        GradientBoostingTrainer().predict(np.zeros((2, 2)))


# --- hyperparameter validation ----------------------------------------------

def test_unknown_hyperparameter_rejected():
    with pytest.raises(TrainerError) as exc:
        GradientBoostingTrainer(hidden_layer_sizes=(8,))  # an mlp param, not a boosting one
    msg = str(exc.value)
    assert "hidden_layer_sizes" in msg
    assert "Supported:" in msg and "learning_rate" in msg


def test_known_hyperparameters_accepted():
    t = GradientBoostingTrainer(learning_rate=0.05, max_iter=200, max_depth=3, max_leaf_nodes=None)
    assert t.learning_rate == 0.05
    assert t.max_iter == 200
    assert t.max_depth == 3
    assert t.max_leaf_nodes is None


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_leaf_nodes": 1}, "max_leaf_nodes"),
        ({"max_depth": 0}, "max_depth"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"l2_regularization": -1.0}, "l2_regularization"),
    ],
)
def test_bad_numeric_hyperparameters_rejected(kwargs, field):
    with pytest.raises(TrainerError, match=field):
        GradientBoostingTrainer(**kwargs)


@pytest.mark.parametrize("bad", [None, 0.5, "0", True])
def test_non_int_random_state_rejected(bad):
    with pytest.raises(TrainerError, match="random_state"):
        GradientBoostingTrainer(random_state=bad)


# --- reproducibility + capabilities -----------------------------------------

def test_same_seed_reproduces_within_tolerance():
    """Two independent fits on the same data reproduce within the declared band.
    (HistGradientBoosting is deterministic for the DOE regime, so this is tight.)"""
    x, y = _nonlinear_problem(n=160)
    a = GradientBoostingTrainer(random_state=0)
    a.fit(x, y)
    b = GradientBoostingTrainer(random_state=0)
    b.fit(x, y)
    diff = float(np.max(np.abs(a.predict(x) - b.predict(x))))
    assert diff <= GradientBoostingTrainer.reproduction_tolerance


def test_declares_capabilities():
    assert GradientBoostingTrainer.export_formats == frozenset({"pickle", "onnx"})
    tol = GradientBoostingTrainer.reproduction_tolerance
    assert isinstance(tol, float) and np.isfinite(tol) and tol > 0.0
    t = GradientBoostingTrainer()
    assert t.supports_export("onnx") and t.supports_export("pickle")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gradient_boosting.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'surroforge.trainers.gradient_boosting'`.

- [ ] **Step 3: Implement the trainer**

Create `src/surroforge/trainers/gradient_boosting.py`:

```python
"""Gradient-boosting surrogate — scikit-learn HistGradientBoostingRegressor (PRD 4.6).

The boosting counterpart to ``mlp`` and the second tree-free... no: a tree-ensemble
method. Backed by scikit-learn (already Core, BSD → NFR8), so it adds **no new
dependency**. Two deliberate differences from ``mlp``:

* **No input scaler.** Tree ensembles are scale-invariant, so the estimator is used
  directly rather than wrapped in a ``StandardScaler`` pipeline — simpler model, simpler
  ONNX graph.
* **Multi-output via a per-target wrapper.** ``HistGradientBoostingRegressor`` is
  single-output, so >1 target is fitted with ``MultiOutputRegressor`` (one independent
  booster per column); a single target uses the bare regressor on a 1-D vector (avoiding
  sklearn's ``DataConversionWarning``).

``early_stopping`` is disabled so there is no internal validation split and the fit is
deterministic for the small-DOE regime. Like ``mlp`` it exports to ONNX via ``skl2onnx``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from surroforge.errors import TrainerError
from surroforge.trainers.base import BaseTrainer

_SUPPORTED = {
    "learning_rate",
    "max_iter",
    "max_leaf_nodes",
    "max_depth",
    "min_samples_leaf",
    "l2_regularization",
    "random_state",
}


class GradientBoostingTrainer(BaseTrainer):
    """Gradient-boosting surrogate (scikit-learn ``HistGradientBoostingRegressor``).

    Hyperparameters (all optional, passed through to the estimator):
        learning_rate (float): shrinkage per boosting stage (default ``0.1``, ``>0``).
        max_iter (int): number of boosting stages (default ``100``, ``>=1``).
        max_leaf_nodes (int | None): leaves per tree (default ``31``, ``>=2`` or ``None``).
        max_depth (int | None): max tree depth (default ``None``, ``>=1`` or ``None``).
        min_samples_leaf (int): min samples per leaf (default ``20``, ``>=1``).
        l2_regularization (float): L2 penalty on leaf values (default ``0.0``, ``>=0``).
        random_state (int): seed (default ``0``) — fixed so the method is reproducible.
    """

    name = "gradient-boosting"
    # Tree ensemble with a fixed seed and early_stopping off: deterministic in-process.
    # Declares a small honest tolerance (like mlp) rather than 0.0 to absorb any
    # thread/platform float differences in histogram summation.
    reproduction_tolerance = 1e-6
    # Backed by scikit-learn → exports to ONNX via skl2onnx (HistGradientBoosting and the
    # MultiOutputRegressor wrapper both convert).
    export_formats = frozenset({"pickle", "onnx"})

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: int | None = 31,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        random_state: int = 0,
        **extra: Any,
    ) -> None:
        if extra:
            raise TrainerError(
                f"Unknown hyperparameter(s) for gradient-boosting: {sorted(extra)}. "
                f"Supported: {sorted(_SUPPORTED)}."
            )
        if learning_rate <= 0:
            raise TrainerError("gradient-boosting 'learning_rate' must be > 0.")
        if max_iter < 1:
            raise TrainerError("gradient-boosting 'max_iter' must be >= 1.")
        if max_leaf_nodes is not None and max_leaf_nodes < 2:
            raise TrainerError("gradient-boosting 'max_leaf_nodes' must be >= 2 or None.")
        if max_depth is not None and max_depth < 1:
            raise TrainerError("gradient-boosting 'max_depth' must be >= 1 or None.")
        if min_samples_leaf < 1:
            raise TrainerError("gradient-boosting 'min_samples_leaf' must be >= 1.")
        if l2_regularization < 0:
            raise TrainerError("gradient-boosting 'l2_regularization' must be >= 0.")
        # A fixed integer seed is what makes the method reproducible; None/RandomState/float
        # would silently break that (bool excluded though it subclasses int).
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise TrainerError(
                f"gradient-boosting 'random_state' must be an int; got {random_state!r}."
            )
        super().__init__(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self._model: Any = None

    def _build_regressor(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=self.random_state,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = self._as_2d(x)
        y = self._as_2d(y)
        if y.shape[1] == 1:
            # Single output: a 1-D target avoids sklearn's DataConversionWarning.
            model: Any = self._build_regressor()
            model.fit(x, y[:, 0])
        else:
            # HistGradientBoostingRegressor is single-output; wrap for multi-output.
            from sklearn.multioutput import MultiOutputRegressor

            model = MultiOutputRegressor(self._build_regressor())
            model.fit(x, y)
        # Assign only after a successful fit, so a failed fit can't leave a half-built
        # model that predict() would treat as fitted.
        self._model = model

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise TrainerError("Model is not fitted.")
        preds = self._model.predict(self._as_2d(x))
        return self._as_2d(preds)
```

- [ ] **Step 4: Register the method in the builtin table**

In `src/surroforge/plugins/registry.py`, add this line to the `TRAINER_GROUP` dict inside `_BUILTIN` (after the `"mlp": ...` entry):

```python
        "gradient-boosting": "surroforge.trainers.gradient_boosting:GradientBoostingTrainer",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gradient_boosting.py -q`
Expected: PASS. If `test_learns_nonlinear_function` is marginally below 0.9, do NOT lower the threshold — that would be a real underfit signal; report it. (With 200 training points and default `max_iter=100`, HistGradientBoosting comfortably exceeds 0.9 on this smooth target.)

- [ ] **Step 6: Lint and commit**

Run: `.venv/bin/python -m ruff check src/surroforge/trainers/gradient_boosting.py src/surroforge/plugins/registry.py tests/test_gradient_boosting.py` and fix any issues.

```bash
git add src/surroforge/trainers/gradient_boosting.py src/surroforge/plugins/registry.py tests/test_gradient_boosting.py
git -c user.name="Georg Babij" -c user.email="g.babij@googlemail.com" commit -m "feat(trainers): add gradient-boosting (HistGradientBoosting) method"
```

---

## Task 2: ONNX Export

**Files:**
- Modify: `src/surroforge/trainers/gradient_boosting.py`
- Test: `tests/test_gradient_boosting.py` (append)

**Interfaces:**
- Consumes: `GradientBoostingTrainer` (Task 1).
- Produces: `GradientBoostingTrainer.to_onnx(n_features: int) -> bytes`.

- [ ] **Step 1: Append the failing ONNX tests**

Append to `tests/test_gradient_boosting.py`:

```python
# --- ONNX export -------------------------------------------------------------

import importlib.util  # noqa: E402
import sys  # noqa: E402

from surroforge.errors import ExportError  # noqa: E402

_needs_onnx = pytest.mark.skipif(
    importlib.util.find_spec("skl2onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="requires the 'onnx' optional backend (skl2onnx + onnxruntime)",
)


@_needs_onnx
@pytest.mark.parametrize("n_outputs", [1, 2])
def test_onnx_round_trip(n_outputs):
    import onnxruntime as ort

    x, y1 = _nonlinear_problem(n=160)
    y = y1 if n_outputs == 1 else np.column_stack([y1[:, 0], x[:, 0] * x[:, 1]])
    t = GradientBoostingTrainer(max_iter=50)
    t.fit(x, y)

    blob = t.to_onnx(n_features=2)
    assert isinstance(blob, bytes) and blob

    sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x.astype(np.float32)})[0]
    ref = t.predict(x)
    np.testing.assert_allclose(onnx_out.reshape(ref.shape), ref, atol=1e-3)


def test_onnx_export_hint_when_backend_missing(monkeypatch):
    x, y = _nonlinear_problem(n=60)
    t = GradientBoostingTrainer(max_iter=30)
    t.fit(x, y)
    monkeypatch.setitem(sys.modules, "skl2onnx", None)
    with pytest.raises(ExportError, match=r"surroforge\[onnx\]"):
        t.to_onnx(n_features=2)


def test_onnx_export_before_fit_raises():
    with pytest.raises(TrainerError, match="not fitted"):
        GradientBoostingTrainer().to_onnx(n_features=2)


def test_onnx_export_rejects_wrong_feature_count():
    x, y = _nonlinear_problem(n=60)  # 2 input features
    t = GradientBoostingTrainer(max_iter=30)
    t.fit(x, y)
    with pytest.raises(ExportError, match="fitted on 2"):
        t.to_onnx(n_features=5)
```

- [ ] **Step 2: Run the ONNX tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gradient_boosting.py -k onnx -q`
Expected: FAIL — `to_onnx` is the `BaseTrainer` default that raises "does not support ONNX export" (the missing-backend / before-fit / wrong-feature tests fail because the method does not yet guard/translate).

- [ ] **Step 3: Implement `to_onnx`**

Append this method to `GradientBoostingTrainer` in `src/surroforge/trainers/gradient_boosting.py` (after `predict`):

```python
    def to_onnx(self, n_features: int) -> bytes:
        from surroforge.errors import ExportError

        if self._model is None:
            raise TrainerError("Model is not fitted.")
        # The model was fitted on a fixed input width; a mismatched n_features would
        # build a graph that silently mispredicts at inference, so refuse it.
        trained = int(self._model.n_features_in_)
        if n_features != trained:
            raise ExportError(
                f"ONNX export was asked for {n_features} input feature(s) but the model was "
                f"fitted on {trained}. Pass the training feature count."
            )
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ExportError(
                "ONNX export requires the 'onnx' extra. "
                "Install it with: pip install 'surroforge[onnx]'"
            ) from exc

        try:
            onnx_model = cast(Any, convert_sklearn(
                self._model,
                initial_types=[("input", FloatTensorType([None, n_features]))],
            ))
        except Exception as exc:  # skl2onnx raises various conversion errors
            raise ExportError(
                f"ONNX conversion of the gradient-boosting model failed: {exc}"
            ) from exc
        return onnx_model.SerializeToString()
```

- [ ] **Step 4: Run the ONNX tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gradient_boosting.py -k onnx -q`
Expected: PASS (the round-trip cases run only if `skl2onnx` + `onnxruntime` are installed; the guard tests run regardless).

**If `test_onnx_round_trip` fails because skl2onnx cannot convert `HistGradientBoostingRegressor` or `MultiOutputRegressor` in this environment** (a converter/version limitation, not a bug in this code): STOP and report BLOCKED with the exact skl2onnx error. Do not hack around it. The controller will decide between (a) pinning/locating a skl2onnx version that supports it, or (b) falling back the design to pickle-only export. Do not silently drop ONNX from `export_formats`.

- [ ] **Step 5: Lint and commit**

Run: `.venv/bin/python -m ruff check src/surroforge/trainers/gradient_boosting.py tests/test_gradient_boosting.py` and fix any issues.

```bash
git add src/surroforge/trainers/gradient_boosting.py tests/test_gradient_boosting.py
git -c user.name="Georg Babij" -c user.email="g.babij@googlemail.com" commit -m "feat(trainers): add ONNX export to gradient-boosting"
```

---

## Task 3: Conformance, Reduced-Field Inner, Docs

**Files:**
- Modify: `tests/test_conformance.py`
- Test: `tests/test_gradient_boosting.py` (append the reduced-field inner test)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `GradientBoostingTrainer` (Tasks 1–2); `check_trainer` (existing); `ReducedFieldTrainer` (existing).

- [ ] **Step 1: Add gradient-boosting to the conformance parametrize**

In `tests/test_conformance.py`, add the import alongside the other trainer imports:

```python
from surroforge.trainers.gradient_boosting import GradientBoostingTrainer
```

and add this entry to the `BUILTINS` list (after the `MLPTrainer` param):

```python
    pytest.param(GradientBoostingTrainer, id="gradient-boosting"),
```

- [ ] **Step 2: Run the conformance suite for the new method**

Run: `.venv/bin/python -m pytest "tests/test_conformance.py::test_builtin_trainer_is_conformant[gradient-boosting]" "tests/test_conformance.py::test_suite_runs_every_contract_check[gradient-boosting]" -q`
Expected: PASS. (The `deterministic` check does two same-config fits and compares within `reproduction_tolerance = 1e-6`; HistGradientBoosting reproduces, so it passes. No check requires seed-divergence.)

- [ ] **Step 3: Append the reduced-field inner test**

Append to `tests/test_gradient_boosting.py`:

```python
# --- works as a reduced-field inner ------------------------------------------

def test_reduced_field_inner_end_to_end():
    """gradient-boosting serves as a per-mode inner surrogate with no ROM-specific
    change, reconstructing a smooth low-rank field on a holdout."""
    from surroforge.trainers.reduced_field import ReducedFieldTrainer

    rng = np.random.default_rng(0)
    p = rng.uniform(-1.0, 1.0, (200, 2))
    phi, _ = np.linalg.qr(rng.standard_normal((30, 3)))  # 3 orthonormal spatial modes
    amps = np.column_stack([p[:, 0], p[:, 1] ** 2, p[:, 0] * p[:, 1]]) * np.array([5.0, 2.0, 1.0])
    field = amps @ phi.T

    t = ReducedFieldTrainer(
        reduction={"basis": "pod", "energy": 0.999},
        inner={"method": "gradient-boosting", "max_iter": 200},
    )
    t.fit(p[:150], field[:150])
    recon = t.predict(p[150:])
    assert recon.shape == field[150:].shape
    score = r2_score(field[150:], recon)
    assert score > 0.85, f"POD-boosting reconstruction R²={score:.3f}"
```

- [ ] **Step 4: Run the reduced-field inner test**

Run: `.venv/bin/python -m pytest tests/test_gradient_boosting.py::test_reduced_field_inner_end_to_end -q`
Expected: PASS. If R² is marginally below 0.85, do NOT lower the threshold — report it (boosting on three smooth POD amplitudes from 150 points should clear 0.85).

- [ ] **Step 5: Add the CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]`, add a new subsection immediately after the `### Added — Python API` block (before `### Added — Plugins`):

```markdown
### Added — Methods

- **`gradient-boosting` method** — a tree-ensemble surrogate backed by scikit-learn's
  `HistGradientBoostingRegressor`, the boosting counterpart to `mlp`. It is a Core builtin
  (scikit-learn is already Core → **no new dependency**), needs no input scaling, supports
  single- and multi-output targets (the latter via `MultiOutputRegressor`, one booster per
  target), runs with `early_stopping=False` for a deterministic small-DOE fit, is persisted
  via **pickle**, passes the `check_trainer` conformance suite, and works as a `reduced-field`
  inner. Select it with `model.method: gradient-boosting`; hyperparameters: `learning_rate`,
  `max_iter`, `max_leaf_nodes`, `max_depth`, `min_samples_leaf`, `l2_regularization`,
  `random_state`. (ONNX export is not offered: the current `skl2onnx`/`onnx` stack cannot
  convert tree-ensemble regressors.)
```

- [ ] **Step 6: Add gradient-boosting to the docs method list if one exists**

Run: `grep -rln "kriging" docs/ 2>/dev/null` to find any prose method list. If a file lists the methods (e.g. mentions `mlp`/`kriging`/`rbf` together), add `gradient-boosting` to that list in the same style. If no such list exists, skip this step and note it in your report.

- [ ] **Step 7: Run the full gradient-boosting + conformance suite, lint**

Run: `.venv/bin/python -m pytest tests/test_gradient_boosting.py tests/test_conformance.py -q`
Expected: PASS.
Run: `.venv/bin/python -m ruff check src tests` and fix any issues.

- [ ] **Step 8: Commit**

```bash
git add tests/test_conformance.py tests/test_gradient_boosting.py CHANGELOG.md docs/
git -c user.name="Georg Babij" -c user.email="g.babij@googlemail.com" commit -m "test(trainers): conformance + reduced-field inner for gradient-boosting; docs"
```

---

## Self-Review

**Spec coverage:**
- Core builtin `gradient-boosting`, HistGradientBoosting, no new dep → Task 1. ✓
- No scaler; multi-output via MultiOutputRegressor; single-output bare → Task 1 (`fit`). ✓
- Seven hyperparameters + validation; `early_stopping=False`; `random_state` int-only → Task 1. ✓
- `reproduction_tolerance = 1e-6`; `export_formats = {pickle, onnx}` → Task 1 (class attrs), Task 2 (to_onnx). ✓
- ONNX via skl2onnx, n_features guard, missing-backend hint → Task 2. ✓
- Registration in `_BUILTIN`; passes `check_trainer`; usable as reduced-field inner → Task 1 (register), Task 3 (conformance param + inner test). ✓
- Tests: validation, single/multi fit/predict, determinism (same-seed only), nonlinear behaviour, ONNX round-trip, registration/conformance → Tasks 1–3. ✓
- Docstring, CHANGELOG, docs method list → Tasks 1, 3. ✓
- Out-of-scope items (XGBoost/LightGBM, classic GB, early-stopping/max_bins tuning, new example) → not implemented. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `GradientBoostingTrainer`, `gradient-boosting`, `_build_regressor`, `fit`/`predict`/`to_onnx`, the seven hyperparameter names, `reproduction_tolerance`, `export_formats` are identical across all tasks and match the registry entry. The test helper `_nonlinear_problem` is defined in Task 1 and reused in Tasks 2–3 (same file). ✓

**Deliberate omission:** No "different seed diverges" test (HistGradientBoosting is seed-invariant on small data) — called out in Global Constraints so a reviewer does not flag its absence as a gap.
