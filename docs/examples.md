# Examples

Runnable, self-contained scripts live in the [`examples/`](https://github.com/gebab88/surroforge/tree/main/examples)
directory of the repository. Each one synthesises its own data in a temporary
directory and cleans up on exit, so you can run any of them straight from a
checkout:

```bash
pip install -e .          # Core; add [smt] for Kriging/RBF, [onnx] for ONNX export
python examples/01_quickstart.py
```

| Script | Shows | Needs |
| --- | --- | --- |
| [`01_quickstart.py`](https://github.com/gebab88/surroforge/blob/main/examples/01_quickstart.py) | Greenfield end to end: init → register → train → predict → export → report → reproduce | Core only |
| [`02_method_comparison.py`](https://github.com/gebab88/surroforge/blob/main/examples/02_method_comparison.py) | Train several methods, tabulate metrics, pick the best with `compare_runs` | Core (Kriging/RBF added if `smt` present) |
| [`03_reproduction.py`](https://github.com/gebab88/surroforge/blob/main/examples/03_reproduction.py) | Re-run from recorded lineage and verify with `reproduce` / `assert_reproducible` | Core only |
| [`04_field_rom.py`](https://github.com/gebab88/surroforge/blob/main/examples/04_field_rom.py) | Reduce a high-dimensional field with POD and surrogate the coefficients (`reduced-field`) | Core only |
| [`05_openfoam_field_rom.py`](https://github.com/gebab88/surroforge/blob/main/examples/05_openfoam_field_rom.py) | Ingest an OpenFOAM case sweep (parse `internalField`) into a field ROM | Core only |
| [`06_mlp_surrogate.py`](https://github.com/gebab88/surroforge/blob/main/examples/06_mlp_surrogate.py) | Fit a strongly nonlinear response with the `mlp` neural-network method, beat a polynomial, export to ONNX | Core (ONNX export adds `[onnx]`) |
| [`07_pod_nn_field.py`](https://github.com/gebab88/surroforge/blob/main/examples/07_pod_nn_field.py) | POD-NN: reduce a field with POD, learn the modal coefficients with an `mlp` inner, beat a polynomial inner | Core only |
| [`08_torch_mlp.py`](https://github.com/gebab88/surroforge/blob/main/examples/08_torch_mlp.py) | Switch the NN backend with one config line — `mlp` (scikit-learn) ↔ `torch-mlp` (PyTorch), same hyperparameters | Core; the PyTorch half needs the `surroforge-torch` plugin |

The Python scripts are the API twin of the shell workflows in the
[Quickstart](quickstart.md) and the [Scenarios](scenarios/greenfield.md). They
are smoke-tested in CI, so they stay in step with the API.

For a **real** OpenFOAM run (not smoke-tested, since it needs a solver), see
[`examples/09_naca0012_openfoam/`](https://github.com/gebab88/surroforge/tree/main/examples/09_naca0012_openfoam)
— it drives `simpleFoam` over an angle-of-attack × speed sweep and trains the SurroForge
surrogates on the genuine results with the same surrogate code as the examples above.

## Command-line configs

[`examples/configs/`](https://github.com/gebab88/surroforge/tree/main/examples/configs)
holds YAML training configs for the CLI flow — register a dataset, then point
`train` at a config:

```bash
surroforge init study && cd study
surroforge data add ../samples.csv --name beam
surroforge train --config ../examples/configs/kriging.yaml
```

- [`configs/kriging.yaml`](https://github.com/gebab88/surroforge/blob/main/examples/configs/kriging.yaml) — a scalar Kriging surrogate (needs `[smt]`).
- [`configs/field-rom.yaml`](https://github.com/gebab88/surroforge/blob/main/examples/configs/field-rom.yaml) — a POD field surrogate (Core only).
