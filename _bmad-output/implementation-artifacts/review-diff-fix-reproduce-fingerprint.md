# Diff under review — fix-reproduce-fingerprint (NO_VCS, reconstructed)

Reproducibility fingerprint changed from in-sample (final model on training inputs)
to out-of-sample (holdout model on the held-out test split), with an emitted verdict
and a named tolerance. 5 files.

## src/surroforge/engine.py

```diff
@@ holdout split (step 3) @@
     holdout_model = factory()
     holdout_model.fit(x_train, y_train)
-    train_metrics = compute_metrics(y_train, holdout_model.predict(x_train), metrics)
-    test_metrics = compute_metrics(y_test, holdout_model.predict(x_test), metrics)
+    holdout_test_pred = holdout_model.predict(x_test)
+    train_metrics = compute_metrics(y_train, holdout_model.predict(x_train), metrics)
+    test_metrics = compute_metrics(y_test, holdout_test_pred, metrics)

@@ final model (step 5) @@
     final_model = factory()
     final_model.fit(x, y)
-    prediction_hash = _prediction_fingerprint(final_model, x)
+    # Reproducibility fingerprint over OUT-OF-SAMPLE holdout predictions: the
+    # holdout model never saw x_test, so model divergence actually moves these
+    # values — unlike in-sample predictions, which interpolating methods (Kriging)
+    # reproduce near-exactly regardless of divergence (PRD 4.5).
+    holdout_fingerprint = np.ascontiguousarray(holdout_test_pred, dtype=np.float64)
+    prediction_hash = _prediction_fingerprint(holdout_fingerprint)

@@ lineage dict @@
         "seed": split.seed,
         "prediction_hash": prediction_hash,
+        "holdout_fingerprint": holdout_fingerprint.tolist(),
         "onnx_exported": onnx_exported,

@@ fingerprint helper @@
-def _prediction_fingerprint(model: BaseTrainer, x: np.ndarray) -> str:
-    """SHA-256 over the model's predictions on the training inputs. ..."""
-    preds = np.ascontiguousarray(model.predict(x), dtype=np.float64)
-    return hashlib.sha256(preds.tobytes()).hexdigest()
+def _prediction_fingerprint(preds: np.ndarray) -> str:
+    """SHA-256 over a prediction sample (PRD 4.5). Computed over out-of-sample
+    holdout predictions in run_training, so two runs share a fingerprint only when
+    they reproduce the same out-of-sample behaviour."""
+    arr = np.ascontiguousarray(preds, dtype=np.float64)
+    return hashlib.sha256(arr.tobytes()).hexdigest()
```

## src/surroforge/core/run.py

```diff
     @property
     def parent_run(self) -> str | None:
         return self.lineage.get("parent_run")
+
+    @property
+    def holdout_fingerprint(self) -> list[Any]:
+        """Out-of-sample holdout predictions recorded for the reproducibility verdict."""
+        return self.lineage.get("holdout_fingerprint", [])
```

## src/surroforge/services.py

```diff
 # --- Reproduce ---
+REPRODUCTION_TOLERANCE = 1e-9
+
 @dataclass
 class ReproduceResult:
     parent: Run
     reproduction: Run
     bit_identical: bool
     max_abs_diff: float
     data_hash_match: bool
+    verdict: str

 def reproduce(project, run_id, *, echo=None) -> ReproduceResult:
     ...
     data_hash_match = parent.data_hash == reproduction.data_hash
     bit_identical = (parent.lineage.get("prediction_hash")
                      == reproduction.lineage.get("prediction_hash"))
-    # Quantify any divergence on a shared input set.
-    frame, _, _ = project.resolve_source(config.data.source)
-    x = frame[config.data.inputs].to_numpy(dtype=float)
-    parent_pred = parent.load_model().predict(x)
-    repro_pred = reproduction.load_model().predict(x)
-    max_abs_diff = float(np.max(np.abs(parent_pred - repro_pred))) if x.size else 0.0
+    # Divergence over the recorded OUT-OF-SAMPLE holdout sample.
+    parent_fp = np.asarray(parent.holdout_fingerprint, dtype=float)
+    repro_fp = np.asarray(reproduction.holdout_fingerprint, dtype=float)
+    if parent_fp.shape != repro_fp.shape:
+        max_abs_diff = float("inf")
+    elif parent_fp.size:
+        max_abs_diff = float(np.max(np.abs(parent_fp - repro_fp)))
+    else:
+        max_abs_diff = 0.0
     return ReproduceResult(parent=parent, reproduction=reproduction,
         bit_identical=bit_identical, max_abs_diff=max_abs_diff,
-        data_hash_match=data_hash_match)
+        data_hash_match=data_hash_match,
+        verdict=_verdict(data_hash_match, bit_identical, max_abs_diff))
+
+def _verdict(data_hash_match, bit_identical, max_abs_diff) -> str:
+    if not data_hash_match:
+        return "data_changed"
+    if bit_identical:
+        return "reproduced"
+    if max_abs_diff <= REPRODUCTION_TOLERANCE:
+        return "reproduced_within_tolerance"
+    return "diverged"

-def assert_reproducible(result, *, tolerance: float = 1e-9) -> None:
+def assert_reproducible(result, *, tolerance: float = REPRODUCTION_TOLERANCE) -> None:
     if not result.data_hash_match:
         raise ReproducibilityError("Underlying data changed ...")
     if result.max_abs_diff > tolerance:
         raise ReproducibilityError("Reproduction diverged: ...")
```

## src/surroforge/cli/main.py

```diff
-def _warn(message: str) -> None:
-    typer.secho(message, fg=typer.colors.YELLOW, err=True)
-
-
 def _echo_cb():
     ...

 @app.command()
 @_guard
 def reproduce(run_id: str = typer.Argument(...)) -> None:
     result = services.reproduce(project, run_id, echo=_echo_cb())
     new_id = result.reproduction.run_id
-    if not result.data_hash_match:
-        _warn(f"Data changed since {run_id}: reproduction {new_id} used different data.")
-    if result.bit_identical:
-        _success(f"Reproduced {run_id} as {new_id} — bit-identical model.")
-    elif result.max_abs_diff <= 1e-9:
-        _success(f"... numerically identical (max diff {result.max_abs_diff:.2e}).")
-    else:
-        _warn(f"... predictions diverged (max diff {result.max_abs_diff:.2e}). ...")
+    if result.verdict == "reproduced":
+        _success(f"Reproduced {run_id} as {new_id} — bit-identical model.")
+    elif result.verdict == "reproduced_within_tolerance":
+        _success(f"... numerically identical (max diff {result.max_abs_diff:.2e}).")
+    else:
+        # data_changed or diverged → emit the verdict as a hard failure (exit 1).
+        services.assert_reproducible(result)
```

## tests/test_reproduce.py

```diff
+import copy
+import numpy as np
 import pytest
-from surroforge.services import assert_reproducible, reproduce
+from surroforge.services import REPRODUCTION_TOLERANCE, assert_reproducible, reproduce

 def test_reproduce_detects_data_change(project):
     ...
     assert not result.data_hash_match
+    assert result.verdict == "data_changed"
     with pytest.raises(ReproducibilityError):
         assert_reproducible(result)

+def test_reproduce_emits_verdict(project):
+    cfg = parse_config(poly_config("polynomial"))
+    original = run_training(project, cfg)
+    result = reproduce(project, original.run_id)
+    assert result.verdict == "reproduced"

+def test_fingerprint_is_sensitive_to_model_divergence(project):
+    # Two Kriging models, SAME split (same seed), different correlation kernels.
+    base = poly_config("kriging")
+    cfg_a = copy.deepcopy(base); cfg_a["model"] = {"method": "kriging", "corr": "squar_exp"}
+    cfg_b = copy.deepcopy(base); cfg_b["model"] = {"method": "kriging", "corr": "abs_exp"}
+    a = run_training(project, parse_config(cfg_a))
+    b = run_training(project, parse_config(cfg_b))
+    fa = np.asarray(a.holdout_fingerprint, dtype=float)
+    fb = np.asarray(b.holdout_fingerprint, dtype=float)
+    assert fa.size > 0 and fa.shape == fb.shape
+    assert float(np.max(np.abs(fa - fb))) > REPRODUCTION_TOLERANCE
+    assert a.lineage["prediction_hash"] != b.lineage["prediction_hash"]
```
