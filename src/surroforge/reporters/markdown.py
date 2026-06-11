"""Markdown report (PRD 4.9, Core edition).

Sections: data overview, model specification, validation metrics, the two-source
error split for field runs (PRD 4.6), and the configuration/lineage details needed
to reproduce the run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jinja2 import Template

from surroforge.core.run import Run
from surroforge.reporters.base import BaseReporter
from surroforge.validators.metrics import two_source_split

_TEMPLATE = Template(
    """# SurroForge Report — {{ run_id }}

**Experiment:** {{ experiment }}
**Method:** `{{ method }}`
**Created:** {{ created_at }}
{% if parent_run %}**Reproduction of:** {{ parent_run }}
{% endif %}
## Data Overview

- **Source:** {{ data_source }}
- **Samples:** {{ n_samples }}
- **Inputs ({{ n_inputs }}):** {{ inputs }}
- **Outputs ({{ n_outputs }}):** {{ outputs }}
- **Data hash:** `{{ data_hash }}`

## Model Specification

- **Method:** {{ method }}
{% if hyperparameters %}- **Hyperparameters:**
{% for key, value in hyperparameters.items() %}    - `{{ key }}`: {{ value }}
{% endfor %}{% else %}- **Hyperparameters:** (defaults)
{% endif %}
## Validation Metrics

| Metric | Train | Test | CV mean | CV std |
|---|---|---|---|---|
{% for m in metric_names %}| {{ m }} | {{ fmt(train[m]) }} | {{ fmt(test[m]) }} | {{ fmt(cv[m].mean) }} | {{ fmt(cv[m].std) }} |
{% endfor %}
Cross-validation: {{ cv_folds }}-fold.
{% if two_source %}
## Two-Source Error Split

| Error source | Train | Test | CV mean | CV std |
|---|---|---|---|---|
| total field error (energy_l2) | {{ fmt(two_source.train.total) }} | {{ fmt(two_source.test.total) }} | {{ fmt(two_source.cv_mean.total) }} | {{ fmt(two_source.cv_std.total) }} |
| truncation floor (basis) | {{ fmt(two_source.train.truncation) }} | {{ fmt(two_source.test.truncation) }} | {{ fmt(two_source.cv_mean.truncation) }} | {{ fmt(two_source.cv_std.truncation) }} |
| surrogate error (in-span ≡ coefficient-space) | {{ fmt(two_source.train.surrogate) }} | {{ fmt(two_source.test.surrogate) }} | {{ fmt(two_source.cv_mean.surrogate) }} | {{ fmt(two_source.cv_std.surrogate) }} |

The POD basis is orthonormal, so the in-span surrogate error is also the
coefficient-space error in the basis the predictions actually inhabit (Parseval) --
this table is the coefficient-vs-field view. A dominant truncation floor means the
basis is too small: add modes. A dominant surrogate error means the inner models
are underfitting: add data.
{% if two_source.reduction %}Reduction: rank {{ two_source.reduction.rank }}, energy captured {{ fmt(two_source.reduction.energy_captured) }}.
Basis variance across folds (honest CV): per-fold truncation floors {{ two_source.truncation_per_fold }} (std {{ fmt(two_source.cv_std.truncation) }}).
{% endif %}{% endif %}
## Configuration

- **Config hash:** `{{ config_hash }}`
- **Random seed:** {{ seed }}
- **Test fraction:** {{ test_fraction }}

```yaml
{{ config_yaml }}```

## Reproducibility

- **Python:** {{ python_version }}
- **SurroForge:** {{ surroforge_version }}
- **Prediction fingerprint:** `{{ prediction_hash }}`
- **Dependencies:**
{% for dep in dependencies %}    - {{ dep }}
{% endfor %}
*Reproduce this run with* `surroforge reproduce {{ run_id }}`.
"""
)


class MarkdownReporter(BaseReporter):
    name = "markdown"
    extension = ".md"

    def render(self, run: Run) -> str:
        config = run.config
        metrics = run.metrics
        lineage = run.lineage
        cv = metrics["cross_validation"]

        def cv_view(metric: str) -> dict[str, float]:
            return {"mean": cv[metric]["mean"], "std": cv[metric]["std"]}

        def fmt(value: Any) -> str:
            return f"{value:.5g}"

        # Two-source error split (Story R3.2): derived at render time from the r3-1
        # field metrics — per fold first, then aggregated (Jensen: never split the
        # aggregated means). Rendered only when EVERY consumed record carries both
        # metrics (train, test, and equal-length non-empty per-fold lists); a scalar,
        # legacy, or incomplete run degrades to the exact pre-R3.2 report instead of
        # crashing mid-render.
        two_source: dict[str, Any] | None = None
        names = config.validation.metrics
        field_names = ("energy_l2", "truncation_error")
        if (
            set(field_names) <= set(names)
            and all(m in metrics[s] for s in ("train", "test") for m in field_names)
            and all(m in cv and cv[m].get("per_fold") for m in field_names)
            and len(cv["energy_l2"]["per_fold"]) == len(cv["truncation_error"]["per_fold"])
        ):
            fold_splits = [
                two_source_split(e, t)
                for e, t in zip(
                    cv["energy_l2"]["per_fold"],
                    cv["truncation_error"]["per_fold"],
                    strict=True,
                )
            ]
            components = ("total", "truncation", "surrogate")
            two_source = {
                "train": two_source_split(
                    metrics["train"]["energy_l2"], metrics["train"]["truncation_error"]
                ),
                "test": two_source_split(
                    metrics["test"]["energy_l2"], metrics["test"]["truncation_error"]
                ),
                "cv_mean": {
                    c: float(np.mean([s[c] for s in fold_splits])) for c in components
                },
                "cv_std": {
                    c: float(np.std([s[c] for s in fold_splits])) for c in components
                },
                # Honest-basis CV (Story R3.3): each fold's floor is measured against
                # that fold's freshly derived basis; the per-fold spread is the
                # prediction-observable basis variance the table's aggregate hides.
                "truncation_per_fold": ", ".join(
                    fmt(t) for t in cv["truncation_error"]["per_fold"]
                ),
                # Absent on runs without the R2.3 diagnostics hook — the template
                # then omits the context line rather than failing.
                "reduction": lineage.get("reduction"),
            }

        context = {
            "run_id": run.run_id,
            "experiment": config.experiment,
            "method": config.model.method,
            "created_at": lineage.get("created_at", ""),
            "parent_run": lineage.get("parent_run"),
            "data_source": lineage.get("data_source", config.data.source),
            "data_hash": run.data_hash,
            "n_samples": metrics.get("n_samples", "?"),
            "n_inputs": metrics.get("n_inputs", len(config.data.inputs)),
            "n_outputs": metrics.get("n_outputs", len(config.data.outputs)),
            "inputs": ", ".join(config.data.inputs),
            "outputs": ", ".join(config.data.outputs),
            "hyperparameters": config.model.hyperparameters,
            "metric_names": config.validation.metrics,
            "train": metrics["train"],
            "test": metrics["test"],
            "cv": {m: cv_view(m) for m in config.validation.metrics},
            "cv_folds": cv.get("folds", config.validation.cross_validation.folds),
            "two_source": two_source,
            "config_hash": run.config_hash,
            "seed": run.seed,
            "test_fraction": config.data.split.test_fraction,
            "config_yaml": config.to_yaml(),
            "python_version": lineage.get("python_version", ""),
            "surroforge_version": lineage.get("surroforge_version", ""),
            "prediction_hash": lineage.get("prediction_hash", ""),
            "dependencies": lineage.get("dependencies", []),
            "fmt": fmt,
        }
        return _TEMPLATE.render(**context)
