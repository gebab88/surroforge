# Scenario: Greenfield

**Goal:** you have a fresh table of simulation results and want a validated
surrogate model quickly.

This is the primary onboarding path, covered step by step in the
[Quickstart](../quickstart.md). In short:

```bash
surroforge init beam-study && cd beam-study
surroforge data add ../samples.csv --name beam
surroforge train --config configs/kriging.yaml
surroforge validate run-001
```

You finish with a saved model, train/test/CV metrics, and a complete lineage
record — ready to predict and report, and, with a Professional license, to
reproduce and export.

!!! tip "What 'validated' means here"
    SurroForge always reports **out-of-sample** quality: a held-out test split and
    k-fold cross-validation, using RMSE, R², and MAE. The same out-of-sample
    predictions feed the reproducibility fingerprint, so "it validated" and "it
    reproduces" rest on the same evidence.

## Choosing a method

If you are not sure which method fits, train more than one and let the numbers
decide — that is the [Method comparison](method-comparison.md) scenario.

| Method | Good first choice when… |
|---|---|
| `polynomial` | the response is smooth and low-order; you want ONNX export (Professional). |
| `kriging` | you want an interpolating GP with uncertainty-friendly behaviour. |
| `rbf` | scattered data, smooth interpolation, few hyperparameters. |

## Defaults that work

You only have to specify `data` (source, inputs, outputs) and `model.method`.
Everything else has sensible, documented defaults: a 0.2 test fraction at seed 42,
5-fold shuffled cross-validation, and the RMSE/R²/MAE metric set. Override any of
them in the YAML — see the [configuration reference](../reference/configuration.md).
