"""Markdown report (PRD 4.9, Core edition).

Sections: data overview, model specification, validation metrics, and the
configuration/lineage details needed to reproduce the run.
"""

from __future__ import annotations

from jinja2 import Template

from surroforge.core.run import Run
from surroforge.reporters.base import BaseReporter

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
            "config_hash": run.config_hash,
            "seed": run.seed,
            "test_fraction": config.data.split.test_fraction,
            "config_yaml": config.to_yaml(),
            "python_version": lineage.get("python_version", ""),
            "surroforge_version": lineage.get("surroforge_version", ""),
            "prediction_hash": lineage.get("prediction_hash", ""),
            "dependencies": lineage.get("dependencies", []),
            "fmt": lambda v: f"{v:.5g}",
        }
        return _TEMPLATE.render(**context)
