# Scenario: Reproduction

**Goal:** confirm — months or years later — that a stored run still produces the
same model, given only its configuration and data.

!!! info "Professional feature"
    `reproduce` and the provenance views (`info`, report, `compare` lineage) need a
    Professional license. The recording itself happens on every run — Core writes
    `runs/<id>/lineage.json` regardless — so runs trained before a license was
    installed remain reproducible afterwards.

## How it works

Every run automatically records, in `runs/<id>/lineage.json`:

- the full **config snapshot** (the exact spec used),
- a **content hash of the data** (values + column names + dtypes, order-sensitive),
- the **random seed**,
- the exact **Python and library versions**,
- an out-of-sample **prediction fingerprint** (a SHA-256 over the holdout model's
  predictions on the held-out test split), and the method's declared
  **reproduction tolerance**.

`reproduce` re-runs training from the config snapshot and compares the new
fingerprint to the original. It never unpickles the stored model, so it carries no
arbitrary-code-execution risk.

```bash
surroforge reproduce run-001
```

## Verdicts

| Verdict | Meaning |
|---|---|
| **reproduced** | bit-identical fingerprint — the model is exactly the same. |
| **reproduced_within_tolerance** | predictions agree within the method's declared tolerance (iterative methods such as Kriging/RBF). |
| **diverged** | the fingerprint moved beyond tolerance — investigate environment or data. |
| **data_changed** | the underlying data hash no longer matches the original. |
| **unverifiable** | the run predates the fingerprint (or its shape changed); re-train to enable checking. |

A deterministic method (e.g. polynomial) declares a `0.0` tolerance and is expected
to be bit-identical; an iterative method declares the numerical band within which
two independent fits count as reproduced. The verdict follows the tolerance the
method declared **at train time**, recorded in lineage — so it survives later
changes to the method.

!!! tip "Why out-of-sample?"
    The fingerprint is taken over predictions on the **held-out** split, not the
    training points. Interpolating methods reproduce training points near-exactly
    even when the underlying model diverges, so an in-sample fingerprint would give
    a false "reproduced". Out-of-sample predictions actually move when the model
    changes.

## Reproducible by construction

For bit-level reproducibility across machines, pin BLAS/OpenMP threads before the
interpreter starts (multi-threaded reductions reorder floating-point summation):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 surroforge reproduce run-001
```

SurroForge's own CI runs the reproducibility tests under pinned threads for exactly
this reason.

## From Python

```python
from surroforge import Project
from surroforge.services import reproduce, assert_reproducible

project = Project.open("beam-study")
result = reproduce(project, "run-001")
print(result.verdict, result.max_abs_diff, result.tolerance)
assert_reproducible(result)   # raises ReproducibilityError unless verifiably reproduced
```
