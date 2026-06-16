# surroforge-torch

A **PyTorch** neural-network surrogate (`torch-mlp`) for [SurroForge](../../README.md).

This is an **external plugin** — a separate distribution that registers its method through
the `surroforge.trainers` entry point. SurroForge Core stays PyTorch-free; you opt in by
installing this package. It is the first real exercise of SurroForge's documented plugin
seam (the Golden Path: framework plugins use entry points only, never Core's built-in
table).

## Install

```bash
pip install ./plugins/surroforge-torch          # from a SurroForge checkout
pip install './plugins/surroforge-torch[onnx]'  # + ONNX export & round-trip (onnx + onnxruntime)
```

This pulls in PyTorch (BSD-3-licensed, so it keeps your install permissive).

## Switch between scikit-learn and PyTorch

The switch is one config line — `torch-mlp` mirrors the built-in `mlp`'s hyperparameter
surface, so the same YAML block trains either backend:

```yaml
model:
  method: mlp          # scikit-learn (Core, no extra install)
  # method: torch-mlp  # PyTorch (this plugin) — same hyperparameters
  hidden_layer_sizes: [64, 64]
  activation: relu     # relu | tanh | logistic | identity
  solver: lbfgs        # lbfgs | adam | sgd
  alpha: 1.0e-4
  max_iter: 2000
  random_state: 0
```

```python
from surroforge import Trainer
run = Trainer.from_dict({
    "experiment": "torch-demo",
    "data": {"source": "snaps", "inputs": ["p0", "p1"], "outputs": ["y"]},
    "model": {"method": "torch-mlp", "hidden_layer_sizes": [64, 64], "activation": "tanh"},
    "validation": {"cross_validation": 5, "metrics": ["rmse", "r2"]},
}).fit(project)
```

Because any SurroForge trainer is automatically usable as a `reduced-field` inner,
`model.inner.method: torch-mlp` gives you POD-NN for free.

## Guarantees it keeps

- **Reproducible.** Weights are seeded from the method's own integer `random_state` (the
  engine does not seed trainers), training is full-batch, and the global torch RNG is left
  untouched — so a same-seed refit reproduces within the declared
  `reproduction_tolerance = 1e-4` (an honest float32 band), while a changed seed diverges
  far beyond it.
- **Exportable.** `export_formats == {"pickle", "onnx"}`; ONNX export uses torch's built-in
  exporter and bakes input/output standardisation into the graph, so the exported model
  matches `predict`.
- **Conformant.** `TorchMLPTrainer` passes SurroForge's shipped `check_trainer` suite
  unmodified.

## Develop / test

```bash
pip install -e './plugins/surroforge-torch[onnx,dev]'
pytest plugins/surroforge-torch/tests -q
ruff check plugins/surroforge-torch
mypy plugins/surroforge-torch/src
```
