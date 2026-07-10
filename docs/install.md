# Install

SurroForge requires **Python 3.10 or newer** and is distributed on PyPI.

```bash
pip install surroforge            # core
pip install 'surroforge[smt]'     # + Kriging / RBF backends (SMT)
pip install 'surroforge[onnx]'    # + ONNX conversion backend (export is Professional)
pip install 'surroforge[formats]' # + Parquet / HDF5 input
pip install 'surroforge[rom]'     # + Field ROM (POD field reduction)
pip install 'surroforge[all]'     # everything an MVP user might want
```

After installing, the `surroforge` command is on your `PATH`:

```bash
surroforge version
surroforge --help
```

## Optional backends

Core runs standalone with the polynomial method and CSV input; the trained model
is stored in the run directory, so predict needs no export (model export is a
Professional feature). The optional extras unlock more capability and are imported
lazily — a method or format that needs a missing backend fails with a clear,
actionable install hint rather than a traceback.

| Extra | Unlocks | Packages |
|---|---|---|
| `smt` | Kriging and RBF methods | `smt` |
| `onnx` | ONNX conversion backend (model export itself is Professional) | `skl2onnx`, `onnx`, `onnxruntime` |
| `formats` | Parquet and HDF5 input | `pyarrow`, `h5py` |
| `rom` | Field ROM (`reduced-field`, POD field reduction) | — (uses core `numpy`/`scipy`) |
| `all` | all of the above | — |

!!! note "Licensing of dependencies"
    Core depends only on permissively licensed packages (Apache/MIT/BSD); there
    are no GPL/LGPL dependencies, and this is enforced in CI.

## From source

```bash
git clone https://github.com/gebab88/surroforge
cd surroforge
pip install -e ".[all,dev]"   # editable install with optional backends + dev tools
pytest -q                     # run the test suite
```

## Next

Head to the [Quickstart](quickstart.md) to go from a data table to a validated,
reproducible surrogate model.
