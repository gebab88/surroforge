# Random Forest Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `random-forest` surrogate method to SurroForge Core, backed by scikit-learn's `RandomForestRegressor`, with native single/multi-output fit, pickle + ONNX export, and conformance.

**Architecture:** A new Core builtin trainer `RandomForestTrainer(BaseTrainer)` in `src/surroforge/trainers/random_forest.py`, mirroring `src/surroforge/trainers/gradient_boosting.py`. No new dependency (scikit-learn is already Core). Tree ensembles are scale-invariant → no input scaler. `RandomForestRegressor` is natively multi-output, so there is NO `MultiOutputRegressor` wrapper — `fit` passes 2-D `y` directly (1-D for a single output to avoid `DataConversionWarning`). ONNX export works for random forests in this stack. Registered in `_BUILTIN`, exercised by the existing `check_trainer` conformance suite.

**Tech Stack:** Python 3.10+, scikit-learn (`RandomForestRegressor`), skl2onnx (optional, ONNX export), pytest.

## Global Constraints

- Python `>=3.10`; ruff `line-length = 100`. The project lints `tests/` too — run `.venv/bin/python -m ruff check` on every file you create/modify and fix lint before committing.
- **No new dependency.** Use only scikit-learn (already Core). Import sklearn lazily inside methods (as `mlp`/`gradient_boosting` do), not at module top.
- Method name is exactly `random-forest`. Class is `RandomForestTrainer`.
- Hyperparameters and defaults: `n_estimators=100` (`>=1`), `max_depth=None` (`>=1` or `None`), `max_leaf_nodes=None` (`>=2` or `None`), `min_samples_leaf=1` (`>=1`), `max_features=1.0` (`"sqrt"`/`"log2"`/`None`, an `int >= 1`, or a `float` in `(0, 1]`), `random_state=0` (must be an `int`, not `bool`/`None`). Unknown hyperparameters raise `TrainerError`.
- `bootstrap` is NOT exposed (stays sklearn default `True`).
- `reproduction_tolerance = 1e-6`; `export_formats = frozenset({"pickle", "onnx"})`.
- Domain errors are `TrainerError`/`ExportError` (from `surroforge.errors`), never raw sklearn errors at the API surface.
- `RandomForestRegressor` is natively multi-output: pass 2-D `y` directly for >1 target, `y[:, 0]` (1-D) for a single output. Do NOT use `MultiOutputRegressor`.
- **Unlike `gradient-boosting`, `random_state` genuinely changes a random forest** (bootstrap + feature subsampling). So this method DOES get a "different seed diverges" test (the negative case with teeth), in addition to same-seed reproducibility.
- Run all commands from the repo root `/Users/georg/Software/surroforge`. Python is `.venv/bin/python`. Commit with the git identity (the repo has none configured): `git -c user.name="Georg Babij" -c user.email="g.babij@googlemail.com" commit ...`.

---

## File Structure

- Create `src/surroforge/trainers/random_forest.py` — `RandomForestTrainer`.
- Modify `src/surroforge/plugins/registry.py` — register `random-forest` in `_BUILTIN[TRAINER_GROUP]`.
- Create `tests/test_random_forest.py` — trainer tests (grows across Tasks 1–3).
- Modify `tests/test_conformance.py` — add `RandomForestTrainer` to the `BUILTINS` parametrize.
- Modify `CHANGELOG.md` — `[Unreleased]` method entry.
- Modify `docs/index.md` and `docs/reference/configuration.md` — add `random-forest` to the method lists.

---

## Task 1: RandomForestTrainer Core + Registration

**Files:**
- Create: `src/surroforge/trainers/random_forest.py`
- Modify: `src/surroforge/plugins/registry.py`
- Test: `tests/test_random_forest.py`

**Interfaces:**
- Produces: `RandomForestTrainer(BaseTrainer)` with `name = "random-forest"`, `reproduction_tolerance = 1e-6`, `export_formats = frozenset({"pickle", "onnx"})`, `__init__(n_estimators=100, max_depth=None, max_leaf_nodes=None, min_samples_leaf=1, max_features=1.0, random_state=0, **extra)`, `fit(x, y) -> None`, `predict(x) -> np.ndarray` (2-D). Registered as `random-forest`.

- [ ] **Step 1: Write the failing test file (core, no ONNX)**

Create `tests/test_random_forest.py`:

```python
"""RandomForestTrainer — random-forest surrogate via scikit-learn.

Mirrors test_gradient_boosting.py: registration, the BaseTrainer contract (shapes,
native multi-output, picklability), nonlinear-fit behaviour, hyperparameter
validation, and reproducibility. Unlike gradient-boosting, RandomForest's bootstrap
and feature subsampling make random_state genuinely matter, so this suite includes a
different-seed divergence test (the negative case with teeth).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.metrics import r2_score

from surroforge.errors import TrainerError
from surroforge.plugins import available_trainers, get_trainer
from surroforge.trainers.random_forest import RandomForestTrainer


def _nonlinear_problem(n=240, seed=0):
    """A nonlinear 2-D regression target a random forest fits well."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, (n, 2))
    y = (np.sin(2.0 * x[:, 0]) + x[:, 1] ** 2 - 0.5 * x[:, 0] * x[:, 1]).reshape(-1, 1)
    return x, y


# --- registration -----------------------------------------------------------

def test_registered_as_builtin():
    assert "random-forest" in available_trainers()
    assert get_trainer("random-forest") is RandomForestTrainer


# --- BaseTrainer contract ----------------------------------------------------

def test_predict_returns_2d_single_output():
    x, y = _nonlinear_problem()
    t = RandomForestTrainer()
    t.fit(x[:200], y[:200])
    preds = t.predict(x[200:])
    assert preds.ndim == 2 and preds.shape == (40, 1)


def test_native_multi_output():
    """RandomForestRegressor is natively multi-output (no MultiOutputRegressor wrapper)."""
    x, y1 = _nonlinear_problem()
    y = np.column_stack([y1[:, 0], x[:, 0] ** 2 - x[:, 1]])  # 2 outputs
    t = RandomForestTrainer()
    t.fit(x[:200], y[:200])
    preds = t.predict(x[200:])
    assert preds.shape == (40, 2)


def test_single_output_emits_no_dataconversion_warning():
    """fit() must extract a 1-D target for a single output so sklearn does not warn."""
    x, y = _nonlinear_problem(n=80)
    t = RandomForestTrainer(n_estimators=30)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        t.fit(x, y)


def test_picklable_round_trip():
    import pickle

    x, y = _nonlinear_problem(n=120)
    t = RandomForestTrainer(n_estimators=30)
    t.fit(x, y)
    restored = pickle.loads(pickle.dumps(t))
    np.testing.assert_array_equal(restored.predict(x), t.predict(x))


def test_learns_nonlinear_function():
    x, y = _nonlinear_problem()
    t = RandomForestTrainer()
    t.fit(x[:200], y[:200])
    score = r2_score(y[200:], t.predict(x[200:]))
    assert score > 0.9, f"R²={score:.3f} — random forest failed to fit a nonlinear target"


def test_predict_before_fit_raises():
    with pytest.raises(TrainerError, match="not fitted"):
        RandomForestTrainer().predict(np.zeros((2, 2)))


# --- hyperparameter validation ----------------------------------------------

def test_unknown_hyperparameter_rejected():
    with pytest.raises(TrainerError) as exc:
        RandomForestTrainer(hidden_layer_sizes=(8,))  # an mlp param, not a forest one
    msg = str(exc.value)
    assert "hidden_layer_sizes" in msg
    assert "Supported:" in msg and "n_estimators" in msg


def test_known_hyperparameters_accepted():
    t = RandomForestTrainer(n_estimators=50, max_depth=4, max_leaf_nodes=None, max_features="sqrt")
    assert t.n_estimators == 50
    assert t.max_depth == 4
    assert t.max_leaf_nodes is None
    assert t.max_features == "sqrt"


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"n_estimators": 0}, "n_estimators"),
        ({"max_depth": 0}, "max_depth"),
        ({"max_leaf_nodes": 1}, "max_leaf_nodes"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
    ],
)
def test_bad_numeric_hyperparameters_rejected(kwargs, field):
    with pytest.raises(TrainerError, match=field):
        RandomForestTrainer(**kwargs)


@pytest.mark.parametrize("good", ["sqrt", "log2", None, 1, 3, 0.5, 1.0])
def test_max_features_accepts_valid(good):
    RandomForestTrainer(max_features=good)  # must not raise


@pytest.mark.parametrize("bad", ["all", 0, -1, 1.5, 0.0, True])
def test_max_features_rejects_invalid(bad):
    with pytest.raises(TrainerError, match="max_features"):
        RandomForestTrainer(max_features=bad)


@pytest.mark.parametrize("bad", [None, 0.5, "0", True])
def test_non_int_random_state_rejected(bad):
    with pytest.raises(TrainerError, match="random_state"):
        RandomForestTrainer(random_state=bad)


# --- reproducibility + capabilities -----------------------------------------

def test_same_seed_reproduces_within_tolerance():
    """Two independent fits with the same seed reproduce within the declared band."""
    x, y = _nonlinear_problem(n=160)
    a = RandomForestTrainer(random_state=0)
    a.fit(x, y)
    b = RandomForestTrainer(random_state=0)
    b.fit(x, y)
    diff = float(np.max(np.abs(a.predict(x) - b.predict(x))))
    assert diff <= RandomForestTrainer.reproduction_tolerance


def test_different_seed_diverges_beyond_tolerance():
    """Negative case with teeth. Pins a wrong impl that ignores random_state (hardcodes
    it / never threads it into RandomForestRegressor): that impl would make these two
    fits identical, so this divergence assertion would fail and expose the bug."""
    x, y = _nonlinear_problem(n=160)
    a = RandomForestTrainer(random_state=0)
    b = RandomForestTrainer(random_state=1)
    a.fit(x, y)
    b.fit(x, y)
    diff = float(np.max(np.abs(a.predict(x) - b.predict(x))))
    assert diff > RandomForestTrainer.reproduction_tolerance


def test_declares_capabilities():
    assert RandomForestTrainer.export_formats == frozenset({"pickle", "onnx"})
    tol = RandomForestTrainer.reproduction_tolerance
    assert isinstance(tol, float) and np.isfinite(tol) and tol > 0.0
    t = RandomForestTrainer()
    assert t.supports_export("onnx") and t.supports_export("pickle")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_random_forest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'surroforge.trainers.random_forest'`.

- [ ] **Step 3: Implement the trainer**

Create `src/surroforge/trainers/random_forest.py`:

```python
"""Random-forest surrogate — scikit-learn RandomForestRegressor (PRD 4.6).

A bagged tree-ensemble method, completing the Phase 2 extended methods alongside
``mlp`` (NN), ``kriging`` (GP), and ``gradient-boosting``. Backed by scikit-learn
(already Core, BSD → NFR8), so it adds **no new dependency**. Two notes:

* **No input scaler.** Tree ensembles are scale-invariant, so the estimator is used
  directly — a simpler model.
* **Native multi-output.** Unlike ``gradient-boosting`` (which wraps a single-output
  booster in ``MultiOutputRegressor``), ``RandomForestRegressor`` predicts multiple
  targets directly, so a 2-D ``y`` is passed straight through; a single target uses a
  1-D vector (avoiding sklearn's ``DataConversionWarning``).

A fixed integer ``random_state`` makes the forest deterministic (bootstrap and feature
subsampling are seeded). Like ``mlp``/``gradient-boosting`` it exports to ONNX via
``skl2onnx`` (the RandomForest converter is unaffected by the tree-ensemble issue that
made ``gradient-boosting`` pickle-only).
"""

from __future__ import annotations

from numbers import Real
from typing import Any, cast

import numpy as np

from surroforge.errors import TrainerError
from surroforge.trainers.base import BaseTrainer

_SUPPORTED = {
    "n_estimators",
    "max_depth",
    "max_leaf_nodes",
    "min_samples_leaf",
    "max_features",
    "random_state",
}


def _valid_max_features(value: Any) -> bool:
    if value is None or value in ("sqrt", "log2"):
        return True
    if isinstance(value, bool):  # bool subclasses int; not a valid max_features
        return False
    if isinstance(value, int):
        return value >= 1
    if isinstance(value, float):
        return 0.0 < value <= 1.0
    return False


class RandomForestTrainer(BaseTrainer):
    """Random-forest surrogate (scikit-learn ``RandomForestRegressor``).

    Hyperparameters (all optional, passed through to the estimator):
        n_estimators (int): number of trees (default ``100``, ``>=1``).
        max_depth (int | None): max tree depth (default ``None``, ``>=1`` or ``None``).
        max_leaf_nodes (int | None): leaves per tree (default ``None``, ``>=2`` or ``None``).
        min_samples_leaf (int): min samples per leaf (default ``1``, ``>=1``).
        max_features: features considered per split — ``"sqrt"``/``"log2"``/``None``,
            an ``int >= 1``, or a ``float`` in ``(0, 1]`` (default ``1.0``).
        random_state (int): seed (default ``0``) — fixed so the method is reproducible.
    """

    name = "random-forest"
    # Bootstrap + feature subsampling are seeded, so a fixed random_state is
    # deterministic. Declares a small honest tolerance (like mlp/gradient-boosting)
    # rather than 0.0 to absorb thread/platform float differences in tree averaging.
    reproduction_tolerance = 1e-6
    # Backed by scikit-learn → exports to ONNX via skl2onnx. The RandomForest converter
    # emits clean integer attributes (unlike HistGradientBoosting), so ONNX works here.
    export_formats = frozenset({"pickle", "onnx"})

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_leaf_nodes: int | None = None,
        min_samples_leaf: int = 1,
        max_features: Any = 1.0,
        random_state: int = 0,
        **extra: Any,
    ) -> None:
        if extra:
            raise TrainerError(
                f"Unknown hyperparameter(s) for random-forest: {sorted(extra)}. "
                f"Supported: {sorted(_SUPPORTED)}."
            )
        if n_estimators < 1:
            raise TrainerError("random-forest 'n_estimators' must be >= 1.")
        if max_depth is not None and max_depth < 1:
            raise TrainerError("random-forest 'max_depth' must be >= 1 or None.")
        if max_leaf_nodes is not None and max_leaf_nodes < 2:
            raise TrainerError("random-forest 'max_leaf_nodes' must be >= 2 or None.")
        if min_samples_leaf < 1:
            raise TrainerError("random-forest 'min_samples_leaf' must be >= 1.")
        if not _valid_max_features(max_features):
            raise TrainerError(
                "random-forest 'max_features' must be 'sqrt', 'log2', None, an int >= 1, "
                f"or a float in (0, 1]; got {max_features!r}."
            )
        # A fixed integer seed is what makes the method reproducible; None/RandomState/float
        # would silently break that (bool excluded though it subclasses int).
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise TrainerError(
                f"random-forest 'random_state' must be an int; got {random_state!r}."
            )
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._model: Any = None

    def _build_regressor(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = self._as_2d(x)
        y = self._as_2d(y)
        # RandomForestRegressor is natively multi-output; pass 2-D y directly. For a
        # single output, a 1-D target avoids sklearn's DataConversionWarning.
        target = y[:, 0] if y.shape[1] == 1 else y
        model = self._build_regressor()
        model.fit(x, target)
        # Assign only after a successful fit, so a failed fit can't leave a half-built
        # model that predict() would treat as fitted.
        self._model = model

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise TrainerError("Model is not fitted.")
        preds = self._model.predict(self._as_2d(x))
        return self._as_2d(preds)
```

Note: `Real` is imported for symmetry with other trainers' numeric handling but the
validation above uses concrete `int`/`float` checks; if ruff flags `Real` as unused,
remove the `from numbers import Real` import.

- [ ] **Step 4: Register the method in the builtin table**

In `src/surroforge/plugins/registry.py`, add this line to the `TRAINER_GROUP` dict inside `_BUILTIN` (after the `"gradient-boosting": ...` entry):

```python
        "random-forest": "surroforge.trainers.random_forest:RandomForestTrainer",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_random_forest.py -q`
Expected: PASS. If `test_learns_nonlinear_function` is below 0.9, do NOT lower the threshold — report it (a default 100-tree forest scores ~0.99 on this target).

- [ ] **Step 6: Lint and commit**

Run: `.venv/bin/python -m ruff check src/surroforge/trainers/random_forest.py src/surroforge/plugins/registry.py tests/test_random_forest.py` and fix any issues (including removing the unused `Real` import if flagged).

```bash
git add src/surroforge/trainers/random_forest.py src/surroforge/plugins/registry.py tests/test_random_forest.py
git -c user.name="Georg Babij" -c user.email="g.babij@googlemail.com" commit -m "feat(trainers): add random-forest method"
```

---

## Task 2: ONNX Export

**Files:**
- Modify: `src/surroforge/trainers/random_forest.py`
- Test: `tests/test_random_forest.py` (append)

**Interfaces:**
- Consumes: `RandomForestTrainer` (Task 1).
- Produces: `RandomForestTrainer.to_onnx(n_features: int) -> bytes`.

- [ ] **Step 1: Append the failing ONNX tests**

Append to `tests/test_random_forest.py`:

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
    t = RandomForestTrainer(n_estimators=40)
    t.fit(x, y)

    blob = t.to_onnx(n_features=2)
    assert isinstance(blob, bytes) and blob

    sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x.astype(np.float32)})[0]
    ref = t.predict(x)
    np.testing.assert_allclose(onnx_out.reshape(ref.shape), ref, atol=1e-3)


def test_onnx_export_hint_when_backend_missing(monkeypatch):
    x, y = _nonlinear_problem(n=60)
    t = RandomForestTrainer(n_estimators=20)
    t.fit(x, y)
    monkeypatch.setitem(sys.modules, "skl2onnx", None)
    with pytest.raises(ExportError, match=r"surroforge\[onnx\]"):
        t.to_onnx(n_features=2)


def test_onnx_export_before_fit_raises():
    with pytest.raises(TrainerError, match="not fitted"):
        RandomForestTrainer().to_onnx(n_features=2)


def test_onnx_export_rejects_wrong_feature_count():
    x, y = _nonlinear_problem(n=60)  # 2 input features
    t = RandomForestTrainer(n_estimators=20)
    t.fit(x, y)
    with pytest.raises(ExportError, match="fitted on 2"):
        t.to_onnx(n_features=5)
```

- [ ] **Step 2: Run the ONNX tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_random_forest.py -k onnx -q`
Expected: FAIL — `to_onnx` is the `BaseTrainer` default that raises "does not support ONNX export" (so the guard/round-trip tests fail).

- [ ] **Step 3: Implement `to_onnx`**

Append this method to `RandomForestTrainer` in `src/surroforge/trainers/random_forest.py` (after `predict`):

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
                f"ONNX conversion of the random-forest model failed: {exc}"
            ) from exc
        return onnx_model.SerializeToString()
```

- [ ] **Step 4: Run the ONNX tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_random_forest.py -k onnx -q`
Expected: PASS (round-trip cases run only if `skl2onnx` + `onnxruntime` are installed; guard tests run regardless). For multi-output, onnxruntime may print a benign C-level shape-metadata note to stderr — the reshape-to-`ref.shape` + `allclose` assertion is what verifies correctness; the note is not a Python warning and does not fail the test.

- [ ] **Step 5: Lint and commit**

Run: `.venv/bin/python -m ruff check src/surroforge/trainers/random_forest.py tests/test_random_forest.py` and fix any issues.

```bash
git add src/surroforge/trainers/random_forest.py tests/test_random_forest.py
git -c user.name="Georg Babij" -c user.email="g.babij@googlemail.com" commit -m "feat(trainers): add ONNX export to random-forest"
```

---

## Task 3: Conformance, Reduced-Field Inner, Docs

**Files:**
- Modify: `tests/test_conformance.py`
- Test: `tests/test_random_forest.py` (append the reduced-field inner test)
- Modify: `CHANGELOG.md`
- Modify: `docs/index.md`, `docs/reference/configuration.md`

**Interfaces:**
- Consumes: `RandomForestTrainer` (Tasks 1–2); `check_trainer` (existing); `ReducedFieldTrainer` (existing).

- [ ] **Step 1: Add random-forest to the conformance parametrize**

In `tests/test_conformance.py`, add the import alongside the other trainer imports:

```python
from surroforge.trainers.random_forest import RandomForestTrainer
```

and add this entry to the `BUILTINS` list (after the `GradientBoostingTrainer` param):

```python
    pytest.param(RandomForestTrainer, id="random-forest"),
```

- [ ] **Step 2: Run the conformance suite for the new method**

Run: `.venv/bin/python -m pytest "tests/test_conformance.py::test_builtin_trainer_is_conformant[random-forest]" "tests/test_conformance.py::test_suite_runs_every_contract_check[random-forest]" -q`
Expected: PASS. (The `deterministic` check does two same-config fits and compares within `reproduction_tolerance = 1e-6`; a seeded random forest reproduces, so it passes.)

- [ ] **Step 3: Append the reduced-field inner test**

Append to `tests/test_random_forest.py`:

```python
# --- works as a reduced-field inner ------------------------------------------

def test_reduced_field_inner_end_to_end():
    """random-forest serves as a per-mode inner surrogate with no ROM-specific
    change, reconstructing a smooth low-rank field on a holdout."""
    from surroforge.trainers.reduced_field import ReducedFieldTrainer

    rng = np.random.default_rng(0)
    p = rng.uniform(-1.0, 1.0, (200, 2))
    phi, _ = np.linalg.qr(rng.standard_normal((30, 3)))  # 3 orthonormal spatial modes
    amps = np.column_stack([p[:, 0], p[:, 1] ** 2, p[:, 0] * p[:, 1]]) * np.array([5.0, 2.0, 1.0])
    field = amps @ phi.T

    t = ReducedFieldTrainer(
        reduction={"basis": "pod", "energy": 0.999},
        inner={"method": "random-forest", "n_estimators": 200},
    )
    t.fit(p[:150], field[:150])
    recon = t.predict(p[150:])
    assert recon.shape == field[150:].shape
    score = r2_score(field[150:], recon)
    assert score > 0.85, f"POD-random-forest reconstruction R²={score:.3f}"
```

- [ ] **Step 4: Run the reduced-field inner test**

Run: `.venv/bin/python -m pytest tests/test_random_forest.py::test_reduced_field_inner_end_to_end -q`
Expected: PASS. If R² is below 0.85, do NOT lower the threshold — report it (a 200-tree forest reconstructs this rank-3 field at R²≈0.95).

- [ ] **Step 5: Add the CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Added — Methods`, add a second bullet (after the `gradient-boosting` entry):

```markdown
- **`random-forest` method** — a bagged tree-ensemble surrogate backed by scikit-learn's
  `RandomForestRegressor`, completing the Phase 2 extended methods (alongside `mlp`,
  `kriging`, and `gradient-boosting`). It is a Core builtin (scikit-learn is already Core →
  **no new dependency**), needs no input scaling, is **natively multi-output** (no
  `MultiOutputRegressor` wrapper), exports to **pickle and ONNX** (skl2onnx — the
  RandomForest converter is unaffected by the tree-ensemble issue that kept
  `gradient-boosting` pickle-only), passes the `check_trainer` conformance suite, and works
  as a `reduced-field` inner. A fixed integer `random_state` makes it reproducible. Select it
  with `model.method: random-forest`; hyperparameters: `n_estimators`, `max_depth`,
  `max_leaf_nodes`, `min_samples_leaf`, `max_features`, `random_state`.
```

- [ ] **Step 6: Add random-forest to the docs method lists**

In `docs/index.md`, find the methods table (it has a row for `gradient-boosting`) and add, in the same style, after that row:

```markdown
| Random forest | `random-forest` | scikit-learn | (core) |
```

In `docs/reference/configuration.md`, find the method list that includes `gradient-boosting` and add `random-forest` to it in the same style (after `gradient-boosting`).

- [ ] **Step 7: Run the full random-forest + conformance suite, lint**

Run: `.venv/bin/python -m pytest tests/test_random_forest.py tests/test_conformance.py -q`
Expected: PASS.
Run: `.venv/bin/python -m ruff check src tests` and fix any issues.

- [ ] **Step 8: Commit**

```bash
git add tests/test_conformance.py tests/test_random_forest.py CHANGELOG.md docs/
git -c user.name="Georg Babij" -c user.email="g.babij@googlemail.com" commit -m "test(trainers): conformance + reduced-field inner for random-forest; docs"
```

---

## Self-Review

**Spec coverage:**
- Core builtin `random-forest`, RandomForestRegressor, no new dep → Task 1. ✓
- No scaler; native multi-output (2-D y direct, 1-D single); no MultiOutputRegressor → Task 1 (`fit`). ✓
- Six hyperparameters + validation (incl. multi-type `max_features`); `random_state` int-only → Task 1. ✓
- `reproduction_tolerance = 1e-6`; `export_formats = {pickle, onnx}` → Task 1 (attrs), Task 2 (to_onnx). ✓
- Seed matters → same-seed reproduces AND different-seed diverges tests → Task 1. ✓
- ONNX via skl2onnx, n_features guard, missing-backend hint, single+multi round-trip → Task 2. ✓
- Registration in `_BUILTIN`; passes `check_trainer`; usable as reduced-field inner → Task 1 (register), Task 3 (conformance param + inner test). ✓
- Tests: validation, single/native-multi fit/predict, no-DataConversionWarning, learns-nonlinear, reproducibility (both directions), ONNX, registration/conformance, reduced-field inner → Tasks 1–3. ✓
- Docstring, CHANGELOG, docs method lists → Tasks 1, 3. ✓
- Out-of-scope items (bootstrap=False/OOB, n_jobs, ExtraTrees, new example) → not implemented. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `RandomForestTrainer`, `random-forest`, `_build_regressor`, `_valid_max_features`, `fit`/`predict`/`to_onnx`, the six hyperparameter names, `reproduction_tolerance`, `export_formats` are identical across all tasks and match the registry entry. The test helper `_nonlinear_problem` is defined in Task 1 and reused in Tasks 2–3 (same file). ✓
