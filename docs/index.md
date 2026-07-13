# SurroForge

**Surrogate models for FEM/CFD simulation data — scriptable, versionable, reproducible.**

SurroForge is a command-line tool (with a matching Python API) for building,
validating, and managing surrogate models from tabular simulation data. It takes
you from a solver's output table to a validated, exportable surrogate model — with
the data, configuration, and model cleanly versioned and reproducible months or
years later.

This site documents the **Core** (Apache 2.0) edition: the open-source kernel,
Phase 1 (MVP).

## The promise

> From solver output to a validated surrogate model in under an hour — with all
> data, configurations, and models cleanly versioned and reproducible after months
> or years.

## Why SurroForge

- **A single YAML file is the whole run.** The configuration is the complete,
  Git-versionable specification of a training run — no hidden state. See the
  [configuration reference](reference/configuration.md).
- **Reproducibility is built in, not bolted on.** Every run records its config, a
  content hash of the data, the seed, and exact library versions, plus an
  out-of-sample prediction fingerprint. `surroforge reproduce` re-runs from the
  config and returns a verdict (Professional; the recording itself happens on every
  run). See [Reproduction](scenarios/reproduction.md).
- **The CLI and the Python API are equals.** Anything you can do at the command
  line you can do from Python with the same vocabulary. See the
  [Python API](reference/python-api.md).
- **Extensible by design.** New methods, exporters, and reporters register through
  entry points — the built-ins use exactly that mechanism.

## Methods

| Method | `model.method` | Backend | Edition / Extra |
|---|---|---|---|
| Kriging (Gaussian process) | `kriging` | SMT | Core (`surroforge[smt]`) |
| Radial Basis Functions | `rbf` | SMT | Core (`surroforge[smt]`) |
| Polynomial regression | `polynomial` | scikit-learn | Core |
| Field ROM (POD + inner method) | `reduced-field` | scipy + inner | Core |
| Neural network (MLP) | `mlp` | scikit-learn | Professional (licensed) |
| Gradient boosting | `gradient-boosting` | scikit-learn | Professional (licensed) |
| Random forest | `random-forest` | scikit-learn | Professional (licensed) |
| PyTorch MLP | `torch-mlp` | PyTorch | Professional (`surroforge_professional[torch]`) |
| Autoencoder field surrogate | `autoencoder-field` | PyTorch | Enterprise (`surroforge_enterprise[torch]`) |

Licensed methods need a signed license key — install it with `surroforge license install`
(or set `SURROFORGE_PRO_LICENSE`); an enterprise licence unlocks the professional tier as well.
Cross-validation is k-fold; the standard metrics are RMSE, R², and MAE.

The Professional HTML report includes interactive charts — parity, residual, per-fold
cross-validation, and a train/test/CV-mean comparison — rendered with inlined uPlot, so
the report stays a single self-contained file.

## Where to next

<div class="grid cards" markdown>

- **[Install](install.md)** — pip, optional backends, Python version.
- **[Quickstart](quickstart.md)** — solver table → validated model, end to end.
- **[Scenarios](scenarios/greenfield.md)** — greenfield, method comparison, reproduction.
- **[Reference](reference/cli.md)** — CLI, configuration, and Python API.

</div>
