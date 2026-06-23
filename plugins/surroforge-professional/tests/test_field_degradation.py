from __future__ import annotations

import numpy as np
import pandas as pd

from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import run_training
from surroforge_professional.reports.html import HtmlReporter


def _field_run(tmp_path):
    rng = np.random.default_rng(42)
    n_dof, n = 28, 64
    modes = np.linalg.qr(rng.standard_normal((n_dof, 3)))[0]
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    amps = np.column_stack([5.0 * x[:, 0], 2.0 * x[:, 1] ** 2, 2.0 * x[:, 0] * x[:, 1]])
    y = amps @ modes.T
    cols: dict[str, np.ndarray] = {"p0": x[:, 0], "p1": x[:, 1]}
    for j in range(n_dof):
        cols[f"f{j}"] = y[:, j]
    csv = tmp_path / "field.csv"
    pd.DataFrame(cols).to_csv(csv, index=False)
    proj = Project.init(tmp_path / "proj", name="rom")
    proj.add_dataset(csv, name="snaps")
    cfg = parse_config(
        {
            "experiment": "rom-html",
            "data": {
                "source": "snaps",
                "inputs": ["p0", "p1"],
                "outputs": ["f*"],
                "split": {"test_fraction": 0.25, "seed": 1},
            },
            "model": {
                "method": "reduced-field",
                "reduction": {"basis": "pod", "rank": 2},
                "inner": {"method": "polynomial", "degree": 2},
            },
            "validation": {"cross_validation": 3, "metrics": ["energy_l2", "truncation_error", "rmse", "r2"]},
        }
    )
    return run_training(proj, cfg)


def test_complete_field_run_renders_two_source_section(licensed, tmp_path):
    run = _field_run(tmp_path)
    text = HtmlReporter().render(run)

    assert "Field-ROM Details" in text
    assert "total field error" in text
    assert "truncation floor" in text
    assert "surrogate error" in text
    assert "Reduction rank" in text


def test_incomplete_field_records_omit_field_section(licensed, tmp_path, monkeypatch):
    run = _field_run(tmp_path)
    metrics = dict(run.metrics)
    metrics["cross_validation"] = dict(metrics["cross_validation"])
    metrics["cross_validation"]["truncation_error"] = {"mean": 0.1, "std": 0.01}
    # Use cached_property override if monkeypatch.setattr cannot replace Run.metrics.
    run.__dict__["metrics"] = metrics

    text = HtmlReporter().render(run)

    assert "Field-ROM Details" not in text
    assert "Validation Details" in text
