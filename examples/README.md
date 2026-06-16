# SurroForge examples

Runnable, self-contained scripts that exercise the public API end to end. Each
one synthesises its own data in a temporary directory and cleans up on exit, so
you can run any of them straight from a checkout with no setup:

```bash
pip install -e .          # Core; add [smt] for Kriging/RBF, [onnx] for ONNX export
python examples/01_quickstart.py
```

| Script | Shows | Needs |
| --- | --- | --- |
| [`01_quickstart.py`](01_quickstart.py) | Greenfield end to end: init → register → train → predict → export → report → reproduce | Core only |
| [`02_method_comparison.py`](02_method_comparison.py) | Train several methods, tabulate metrics, pick the best with `compare_runs` | Core (Kriging/RBF added if `smt` present) |
| [`03_reproduction.py`](03_reproduction.py) | Re-run from recorded lineage and verify with `reproduce` / `assert_reproducible` | Core only |
| [`04_field_rom.py`](04_field_rom.py) | Reduce a high-dimensional field with POD and surrogate the coefficients (`reduced-field`) | Core only |
| [`05_openfoam_field_rom.py`](05_openfoam_field_rom.py) | Ingest an OpenFOAM case sweep (parse `internalField`) into a field ROM | Core only |
| [`06_mlp_surrogate.py`](06_mlp_surrogate.py) | Fit a strongly nonlinear response with the `mlp` neural-network method, beat a polynomial, export to ONNX | Core (ONNX export adds `[onnx]`) |
| [`07_pod_nn_field.py`](07_pod_nn_field.py) | POD-NN: reduce a field with POD, learn the modal coefficients with an `mlp` inner, beat a polynomial inner | Core only |
| [`08_torch_mlp.py`](08_torch_mlp.py) | Switch the NN backend with one config line — `mlp` (scikit-learn) ↔ `torch-mlp` (PyTorch), same hyperparameters | Core; the PyTorch half needs the `surroforge-torch` plugin |

The Python scripts mirror the documented workflows — the same steps are shown
from the shell in [docs/quickstart.md](../docs/quickstart.md) and the
[scenarios](../docs/scenarios/).

## CLI configs

[`configs/`](configs/) holds YAML training configs for the command-line flow.
Register a dataset, then point `train` at a config:

```bash
surroforge init study && cd study
surroforge data add ../samples.csv --name beam
surroforge train --config ../examples/configs/kriging.yaml
```

- [`configs/kriging.yaml`](configs/kriging.yaml) — a scalar Kriging surrogate (needs `[smt]`).
- [`configs/field-rom.yaml`](configs/field-rom.yaml) — a POD field surrogate (Core only).

> These examples are smoke-tested in CI (`tests/test_examples.py`) — each script
> must run to completion, so they stay in step with the API.
