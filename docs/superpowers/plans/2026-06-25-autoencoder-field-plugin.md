# Autoencoder Field Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a SurroForge plugin that registers an `autoencoder-field` trainer for nonlinear field compression plus parameter-to-latent regression.

**Architecture:** Create a separate plugin package under `plugins/surroforge-autoencoder`. The trainer implements `BaseTrainer`, trains a PyTorch autoencoder on standardized field snapshots, then fits an existing SurroForge inner trainer on latent coordinates. Prediction composes inner latent prediction with the decoder.

**Tech Stack:** Python, PyTorch, NumPy, pytest, SurroForge trainer entry points.

---

## File Structure

- Create `plugins/surroforge-autoencoder/pyproject.toml`: package metadata and `surroforge.trainers` entry point.
- Create `plugins/surroforge-autoencoder/README.md`: install, config, and behavior notes.
- Create `plugins/surroforge-autoencoder/src/surroforge_autoencoder/__init__.py`: package exports.
- Create `plugins/surroforge-autoencoder/src/surroforge_autoencoder/trainer.py`: `AutoencoderFieldTrainer` and PyTorch module helpers.
- Create `plugins/surroforge-autoencoder/tests/test_autoencoder_field.py`: TDD coverage for validation, training, prediction, pickling, and entry-point metadata.

### Task 1: Plugin Skeleton And Registration Metadata

**Files:**
- Create: `plugins/surroforge-autoencoder/pyproject.toml`
- Create: `plugins/surroforge-autoencoder/README.md`
- Create: `plugins/surroforge-autoencoder/src/surroforge_autoencoder/__init__.py`
- Test: `plugins/surroforge-autoencoder/tests/test_autoencoder_field.py`

- [ ] **Step 1: Write failing metadata test**

```python
def test_pyproject_declares_autoencoder_field_entry_point():
    data = tomllib.loads(Path("plugins/surroforge-autoencoder/pyproject.toml").read_text())
    trainers = data["project"]["entry-points"]["surroforge.trainers"]
    assert trainers == {
        "autoencoder-field": "surroforge_autoencoder.trainer:AutoencoderFieldTrainer"
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /Users/georg/Software/surroforge/.venv/bin/python -m pytest -p no:cacheprovider plugins/surroforge-autoencoder/tests/test_autoencoder_field.py::test_pyproject_declares_autoencoder_field_entry_point -q`

Expected: FAIL because the plugin files do not exist yet.

- [ ] **Step 3: Add minimal plugin package files**

Create `pyproject.toml` with the `autoencoder-field` entry point, a README with one config example, and `__init__.py` exporting `AutoencoderFieldTrainer`.

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command. Expected: PASS.

### Task 2: Constructor Validation

**Files:**
- Modify: `plugins/surroforge-autoencoder/src/surroforge_autoencoder/trainer.py`
- Test: `plugins/surroforge-autoencoder/tests/test_autoencoder_field.py`

- [ ] **Step 1: Write failing validation tests**

```python
def test_constructor_rejects_invalid_hyperparameters():
    with pytest.raises(TrainerError, match="Unknown hyperparameter"):
        AutoencoderFieldTrainer(bogus=1)
    with pytest.raises(TrainerError, match="latent_dim"):
        AutoencoderFieldTrainer(latent_dim=0)
    with pytest.raises(TrainerError, match="activation"):
        AutoencoderFieldTrainer(activation="swish")
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL because `AutoencoderFieldTrainer` is not implemented.

- [ ] **Step 3: Implement constructor and validation**

Implement `AutoencoderFieldTrainer.__init__` with validated hyperparameters:
`latent_dim`, `encoder_hidden_layer_sizes`, `decoder_hidden_layer_sizes`,
`activation`, `alpha`, `max_iter`, `tol`, `learning_rate_init`, `random_state`,
and `inner`.

- [ ] **Step 4: Run validation tests**

Expected: PASS.

### Task 3: Fit/Predict Behavior

**Files:**
- Modify: `plugins/surroforge-autoencoder/src/surroforge_autoencoder/trainer.py`
- Test: `plugins/surroforge-autoencoder/tests/test_autoencoder_field.py`

- [ ] **Step 1: Write failing fit/predict tests**

```python
def test_predict_before_fit_raises_trainer_error():
    trainer = AutoencoderFieldTrainer(max_iter=20)
    with pytest.raises(TrainerError, match="not fitted"):
        trainer.predict(np.zeros((2, 2)))

def test_autoencoder_field_predicts_smooth_holdout_field():
    x, y = smooth_field_fixture()
    trainer = AutoencoderFieldTrainer(
        latent_dim=2,
        encoder_hidden_layer_sizes=(16,),
        decoder_hidden_layer_sizes=(16,),
        activation="tanh",
        max_iter=800,
        learning_rate_init=5.0e-3,
        random_state=1,
        inner={"method": "mlp", "hidden_layer_sizes": (32, 32), "max_iter": 2000},
    )
    trainer.fit(x[:90], y[:90])
    pred = trainer.predict(x[90:])
    assert pred.shape == y[90:].shape
    assert r2_score(y[90:], pred) > 0.85
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL because `fit` and `predict` are not implemented.

- [ ] **Step 3: Implement PyTorch autoencoder fit and predict**

Add `_AutoencoderModule`, field standardization buffers, deterministic torch seed
handling, full-batch Adam training, latent encoding, inner trainer fit, and decoder
prediction.

- [ ] **Step 4: Run fit/predict tests**

Expected: PASS.

### Task 4: Persistence And API Integration

**Files:**
- Modify: `plugins/surroforge-autoencoder/src/surroforge_autoencoder/trainer.py`
- Test: `plugins/surroforge-autoencoder/tests/test_autoencoder_field.py`

- [ ] **Step 1: Write failing pickle and trainer API tests**

```python
def test_pickle_round_trip_preserves_predictions():
    x, y = smooth_field_fixture()
    trainer = AutoencoderFieldTrainer(latent_dim=2, max_iter=400, random_state=2)
    trainer.fit(x[:90], y[:90])
    before = trainer.predict(x[90:95])
    loaded = pickle.loads(pickle.dumps(trainer))
    np.testing.assert_allclose(loaded.predict(x[90:95]), before, rtol=1.0e-6, atol=1.0e-6)

def test_public_metadata_matches_surroforge_contract():
    assert AutoencoderFieldTrainer.name == "autoencoder-field"
    assert AutoencoderFieldTrainer.export_formats == frozenset({"pickle"})
    assert AutoencoderFieldTrainer.reproduction_tolerance > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL until fitted state is picklable and metadata is declared.

- [ ] **Step 3: Make fitted state picklable and declare metadata**

Keep all fitted objects on module-level classes and store primitive hyperparameters
only in trainer state.

- [ ] **Step 4: Run persistence/API tests**

Expected: PASS.

### Task 5: Plugin Verification

**Files:**
- Modify: none unless verification reveals a defect.

- [ ] **Step 1: Run plugin tests without installation**

Run: `PYTHONPATH=src:plugins/surroforge-autoencoder/src /Users/georg/Software/surroforge/.venv/bin/python -m pytest -p no:cacheprovider plugins/surroforge-autoencoder/tests -q`

Expected: all plugin tests pass.

- [ ] **Step 2: Install plugin editable and verify registry**

Run: `/Users/georg/Software/surroforge/.venv/bin/python -m pip install -e plugins/surroforge-autoencoder`

Run: `/Users/georg/Software/surroforge/.venv/bin/python - <<'PY'
from surroforge.plugins import available_trainers, get_trainer
from surroforge_autoencoder import AutoencoderFieldTrainer
assert "autoencoder-field" in available_trainers()
assert get_trainer("autoencoder-field") is AutoencoderFieldTrainer
print("registry=ok")
PY`

Expected: `registry=ok`.

- [ ] **Step 3: Run related existing tests**

Run: `PYTHONPATH=src:plugins/surroforge-torch/src:plugins/surroforge-autoencoder/src /Users/georg/Software/surroforge/.venv/bin/python -m pytest -p no:cacheprovider tests/test_reduced_field.py::test_non_pod_basis_raises plugins/surroforge-torch/tests/test_torch_mlp.py::test_registered_as_plugin plugins/surroforge-autoencoder/tests -q`

Expected: all selected tests pass.

## Self-Review

- Spec coverage: package skeleton, trainer registration, validation, training, prediction, persistence, and verification are covered.
- Placeholder scan: no TBD/TODO placeholders are present.
- Type consistency: the trainer name is consistently `autoencoder-field`; the class is consistently `AutoencoderFieldTrainer`; latent dimension is consistently `latent_dim`.
