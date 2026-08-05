# SurroForge

**Surrogate models for FEM/CFD simulation data — scriptable, versionable, reproducible.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Core license](https://img.shields.io/badge/core%20license-Apache--2.0-green)
![Status](https://img.shields.io/badge/status-beta-orange)

SurroForge is a command-line tool (with a matching Python API) for building,
validating, and managing surrogate models from tabular simulation data. It takes
you from a solver's output table to a validated, exportable surrogate model — with
the data, configuration, and model cleanly versioned and reproducible months or
years later.

> **The promise:** from solver output to a validated surrogate model in under an
> hour — with all data, configurations, and models cleanly versioned and
> reproducible after months or years.

## Why SurroForge

- **A single YAML file is the whole run.** The configuration is the complete,
  Git-versionable specification of a training run — no hidden state.
- **Reproducibility is built in, not bolted on.** Every run records its config, a
  content hash of the data, the seed, exact library versions, and a prediction
  fingerprint.
- **The CLI and the Python API are equals.** Anything you can do at the command
  line you can do from Python with the same vocabulary.
- **Extensible by design.** Methods, exporters, reporters, and commands register
  through entry points — the built-ins use exactly that mechanism, and so do the
  commercial editions.

## Repository layout

This is a monorepo covering all three editions (see [Editions](#editions)):

| Path | Contents | License |
|---|---|---|
| [`src/surroforge/`](src/surroforge/) | **Core** — the open-source kernel | Apache 2.0 ([LICENSE](LICENSE)) |
| [`plugins/surroforge_professional/`](plugins/surroforge_professional/) | **Professional** edition plugin | Proprietary |
| [`plugins/surroforge_enterprise/`](plugins/surroforge_enterprise/) | **Enterprise** edition plugin | Proprietary |
| [`tools/licensing/`](tools/licensing/) | Vendor-side license tooling — never packaged | Vendor-internal |

The editions are ordinary plugins: they attach through the same public entry-point
groups any third-party package would use. Core never imports them and works fully
without them.

## Install

SurroForge requires **Python 3.10+**. Install the Core from a checkout:

```bash
git clone <repository-url> surroforge && cd surroforge
pip install -e .                  # core
pip install -e '.[smt]'           # + Kriging / RBF backends
pip install -e '.[onnx]'          # + ONNX conversion backend (export is Professional)
pip install -e '.[formats]'       # + Parquet / HDF5 input
pip install -e '.[rom]'           # + mesh / binary OpenFOAM readers (meshio)
pip install -e '.[all]'           # everything
```

> **Note:** the PyPI package name is not yet claimed for this project, so
> `pip install surroforge` currently resolves to an **unrelated** package. Until the
> public release, install from source as shown above.

The commercial editions install the same way and register themselves automatically:

```bash
pip install -e plugins/surroforge_professional            # licensed methods, export, HTML, DoE
pip install -e 'plugins/surroforge_professional[torch]'   # + the PyTorch MLP
pip install -e 'plugins/surroforge_enterprise[torch]'     # autoencoder field surrogate
```

## Quickstart

```bash
# 1. Create a project
surroforge init my-study
cd my-study

# 2. Register a dataset (schema validated, content hashed)
surroforge data add ../samples.csv --name beam
surroforge data add-openfoam ../sweep/ --field U --name wing   # or an OpenFOAM case sweep

# 3. Describe a training run in YAML (configs/kriging.yaml below), then train
surroforge train --config configs/kriging.yaml

# 4. Inspect, compare, reproduce
surroforge list
surroforge validate run-001
surroforge info run-001
surroforge compare --runs run-001,run-002
surroforge compare --runs run-001,run-002 --format html -o compare.html   # Professional
surroforge reproduce run-001                                              # Professional

# 5. Use the model
surroforge predict run-001 --input ../new_points.csv --output preds.csv
surroforge export run-001 --format pickle                                 # Professional
surroforge report run-001
surroforge report run-001 --format html -o report.html                    # Professional
```

Two more command groups round this out: `surroforge rom inspect <run-id>` shows the
energy spectrum, rank, and probe coverage of a field-ROM run, and `surroforge doe run
<spec.yaml>` generates a sample table plus manifest from a DoE spec (Professional).

### Example configuration

A single YAML file is the complete specification of a training run:

```yaml
experiment: beam-kriging
data:
  source: beam            # a registered dataset name, or a file path
  inputs: [length, width, load]
  outputs: [max_stress]
  split:
    test_fraction: 0.2
    seed: 42
model:
  method: kriging         # see the method table below
  corr: squar_exp         # model.<extra> = method-specific hyperparameters
validation:
  cross_validation: 5     # k-fold; or {folds: 5, shuffle: true}
  metrics: [rmse, r2, mae]
```

Available metrics are `rmse`, `mae`, and `r2` for scalar outputs, plus `energy_l2`
and `truncation_error` for fields. Input tables may be CSV, Parquet, or HDF5.

## Python API

The API mirrors the CLI one-to-one:

```python
from surroforge import Project, Trainer

project = Project.open("my-study")
trainer = Trainer.from_config("configs/kriging.yaml")
run = trainer.fit(project.dataset("beam"))
print(run.metrics)

run.export("pickle", "model.pkl")   # Professional
```

### How the two surfaces stay in step

That parity is structural, not a duplicated implementation. The verbs a user
actually runs live in **`surroforge.services`**, a layer that composes the engine,
persistence, plugins, and core domain. The CLI is a thin adapter over it: parse
arguments, call a service, render the result. So `surroforge compare` and
`compare_runs()` are not two implementations that happen to agree — they are one
implementation reached through two doors, and a licensed feature is gated once, in
the layer beneath both.

| Service function | CLI equivalent |
|---|---|
| `reproduce()`, `assert_reproducible()` | `surroforge reproduce` |
| `compare_runs()`, `generate_comparison_report()` | `surroforge compare` |
| `predict_with_run()`, `predict_from_file()` | `surroforge predict` |
| `export_model()` | `surroforge export` |
| `generate_report()` | `surroforge report` |
| `inspect_rom_run()` | `surroforge rom inspect` |

Reach for them directly when you want a CLI verb's result as a Python object rather
than as terminal output:

```python
from surroforge.services import compare_runs, generate_report, predict_from_file

ranking = compare_runs(project, ["run-001", "run-002"], metric="rmse")
frame = predict_from_file(project.run("run-001"), "new_points.csv")
path, text = generate_report(project, project.run("run-001"))
```

`Run.export()` above is exactly this: an object-level binding of
`services.export_model`. The layering is enforced rather than documented — an
[import-linter](.importlinter) contract forbids `core`, `engine`, and `services`
from ever importing the `cli` or `api` adapters, so the dependency can only point
one way.

## Methods

| Method | `model.method` | Backend | Edition |
|---|---|---|---|
| Kriging (Gaussian process) | `kriging` | SMT | Core (`[smt]`) |
| Radial basis functions | `rbf` | SMT | Core (`[smt]`) |
| Polynomial regression | `polynomial` | scikit-learn | Core |
| Field ROM (POD + inner method) | `reduced-field` | scipy + inner | Core |
| Neural network (MLP) | `mlp` | scikit-learn | Professional |
| Random forest | `random-forest` | scikit-learn | Professional |
| Gradient boosting | `gradient-boosting` | scikit-learn | Professional |
| PyTorch MLP | `torch-mlp` | PyTorch | Professional (`[torch]`) |
| Autoencoder field surrogate | `autoencoder-field` | PyTorch | Enterprise |

Cross-validation is k-fold. Kriging and RBF are single-output (an SMT limitation);
multi-output is covered by the NN methods and the field-ROM paths.

Export formats are declared per method: every method exports to `pickle`, while
`onnx` is available for `polynomial`, `mlp`, `random-forest`, and `torch-mlp` (which
also exports `pt`). An unsupported method/format pair fails with an explicit error
rather than a silent fallback.

## Editions

Core is a complete, genuinely usable tool on its own. The commercial value sits in
four layers: method breadth, trust (reproducibility and lineage), deployment and
integration (model export, agent control), and presentation (HTML reports, DoE).

| Capability | Core | Professional | Enterprise |
|---|:--:|:--:|:--:|
| Kriging, RBF, polynomial, field ROM | ✓ | ✓ | ✓ |
| Cross-validation, RMSE / R² / MAE, multi-output | ✓ | ✓ | ✓ |
| Train / validate / predict, terminal compare | ✓ | ✓ | ✓ |
| OpenFOAM ingestion, run store, Markdown reports | ✓ | ✓ | ✓ |
| Python API and plugin interfaces | ✓ | ✓ | ✓ |
| Random forest, gradient boosting, neural networks | — | ✓ | ✓ |
| Field ROM with an NN inner surrogate | — | ✓ | ✓ |
| `reproduce`, lineage display and export | — | ✓ | ✓ |
| Model export (ONNX, pickle, PyTorch) | — | ✓ | ✓ |
| Interactive HTML reports, DoE, extended comparisons | — | ✓ | ✓ |
| Machine-readable `--json` output for scriptable agents | — | ○ | ○ |
| Autoencoder field surrogate | — | — | ✓ |
| MCP server — drive SurroForge from an AI agent | — | — | ○ |
| Active learning, multi-fidelity, PDF audit reports | — | — | ○ |

✓ = shipping today · ○ = specified in [PRD.md](PRD.md), not yet implemented

Licensed features fail with a clear message rather than a traceback when no license
is present, and `train --onnx` degrades gracefully instead of failing the run.

### AI-agent control (planned)

The [PRD](PRD.md) specifies two ways to drive SurroForge from an AI agent. A global
`--json` flag turns every command's output into a stable JSON object for agents that
already have a shell (Professional). An **MCP server** — `surroforge mcp serve` —
exposes the verbs as Model Context Protocol tools over stdio, so an MCP-capable
client discovers them with typed schemas, approves them individually rather than
being handed blanket shell access, and receives run artefacts as addressable
resources (Enterprise). Both sit on the `services` layer described above and add no
domain logic of their own. See PRD sections 4.12 and 4.13.

### License activation

A license is an Ed25519-signed `SFL1` token, verified **entirely offline** against
the vendor key embedded in the Professional package — there is no online activation
and no phone-home:

```bash
surroforge license install SFL1.…      # writes ~/.surroforge/professional.lic
surroforge license install license.txt # …or install from a file
surroforge license status              # holder, tier, expiry, validity (exit 0 only when valid)
```

`SURROFORGE_PRO_LICENSE` works as an alternative channel and takes precedence over
the file. Tiers are ordered — an enterprise license unlocks the professional tier too.

## Reproducibility

Every run records its full configuration, a SHA-256 hash of the data, the random
seed, and the exact Python/library versions in `runs/<id>/lineage.json`. A
prediction fingerprint lets `surroforge reproduce` verify that a re-run produced a
bit-identical model.

Recording happens on **every** run and stays in Core, so no data is lost without a
license. Verifying and viewing it (`reproduce`, and the provenance shown in `info`,
reports, and `compare`) is a Professional feature.

## Examples

[`examples/`](examples/) holds runnable scripts that exercise the public API end to
end. Scripts `01`–`08` synthesise their own data in a temporary directory, so they
run straight from a checkout with no setup:

```bash
python examples/01_quickstart.py
```

They cover the greenfield flow, method comparison, POD field ROM, and OpenFOAM
ingestion on Core alone; the reproduction, MLP, PyTorch, and POD-NN scripts exercise
Professional features and either need a license or skip that step gracefully. Each
script's requirements are listed in [`examples/README.md`](examples/README.md).

Two **real-solver** studies are not part of CI, since they need a solver installed —
both run the same NACA aerofoil sweep over angle of attack × freestream speed and
train the identical surrogate code on genuine results, so they double as a
solver-to-solver comparison:

- [`examples/09_aerofoil_openfoam/`](examples/09_aerofoil_openfoam/) — `simpleFoam`, verified against OpenFOAM v2512
- [`examples/10_aerofoil_su2/`](examples/10_aerofoil_su2/) — `SU2_CFD`, verified against SU2 v8.5.0

## Extending SurroForge

Methods, exporters, reporters, and CLI commands are discovered through entry-point
groups — `surroforge.trainers`, `surroforge.exporters`, `surroforge.reporters`,
`surroforge.comparison_reporters`, `surroforge.commands`, and
`surroforge.authorizers`. The built-in methods use exactly this mechanism, and a
plugin may declare a `required_tier` that the registry enforces through the
registered authorizers.

## Documentation

Full documentation lives in [`docs/`](docs/) as a MkDocs Material site — install,
quickstart, the three scenarios (greenfield, method comparison, reproduction), and
the CLI / configuration / Python API reference. Read it locally with:

```bash
pip install -e '.[docs]' && mkdocs serve
```

The product requirements are in [PRD.md](PRD.md); developer-facing context and
conventions are in [MEMORY.md](MEMORY.md); release history is in
[CHANGELOG.md](CHANGELOG.md).

## License

The Core in `src/surroforge/` is licensed under the Apache License 2.0 — see
[LICENSE](LICENSE). The Professional and Enterprise plugins under `plugins/` are
proprietary and require a license key.
