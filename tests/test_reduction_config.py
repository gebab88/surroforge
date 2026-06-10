"""`model.reduction` end-to-end: adaptive-rank write-back to the canonical snapshot (R1.2).

The config snapshot, not the input YAML, is canonical: a `rank: adaptive` run resolves
the rank at fit and writes the resolved integer back into `config.snapshot.yaml`, and
`config_hash`/`spec_hash` are computed from that resolved snapshot. Scalar runs and
energy-field runs are byte-for-byte unchanged (no hook / hook returns None).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import run_training
from surroforge.services import assert_reproducible, reproduce
from tests.test_reduced_field import low_rank_field


def _field_project(tmp_path) -> Project:
    p, field = low_rank_field(n_samples=80, n_dof=6, k_true=3, seed=1)
    cols: dict[str, np.ndarray] = {"p0": p[:, 0], "p1": p[:, 1]}
    for j in range(field.shape[1]):
        cols[f"f{j}"] = field[:, j]
    csv = tmp_path / "field.csv"
    pd.DataFrame(cols).to_csv(csv, index=False)
    proj = Project.init(tmp_path / "proj", name="rom")
    proj.add_dataset(csv, name="snaps")
    return proj


def _field_config(reduction: dict) -> dict:
    return {
        "experiment": "rom-reduction",
        "data": {
            "source": "snaps",
            "inputs": ["p0", "p1"],
            "outputs": [f"f{j}" for j in range(6)],
            "split": {"test_fraction": 0.25, "seed": 1},
        },
        "model": {
            "method": "reduced-field",
            "reduction": reduction,
            "inner": {"method": "polynomial", "degree": 2},
        },
        "validation": {"cross_validation": 3, "metrics": ["rmse", "r2", "mae"]},
    }


# --- AC4: an energy run is unchanged (no write-back) ----------------------------------

def test_energy_run_snapshot_keeps_energy_spec_no_writeback(tmp_path):
    proj = _field_project(tmp_path)
    cfg = parse_config(_field_config({"basis": "pod", "energy": 0.999}))
    run = run_training(proj, cfg)
    snap = run.config_dict["model"]["reduction"]
    assert snap == {"basis": "pod", "energy": 0.999}  # energy stays canonical; no rank key
    # The hash is computed from the SAME (unmodified) config — byte-identical to the
    # hash of the input config, proving canonical_config is config on this path.
    assert run.config_hash == _sha(cfg)
    assert run.metrics["test"]["r2"] > 0.95


def _sha(cfg) -> str:
    import hashlib

    return hashlib.sha256(cfg.canonical_json().encode("utf-8")).hexdigest()


# --- AC1/AC5: an explicit rank trains with exactly that rank ---------------------------

def test_explicit_rank_run_trains_with_exactly_that_rank(tmp_path):
    proj = _field_project(tmp_path)
    run = run_training(proj, parse_config(_field_config({"basis": "pod", "rank": 3})))
    assert run.reduction["rank"] == 3  # lineage diagnostic (R2.3)
    assert run.config_dict["model"]["reduction"]["rank"] == 3  # snapshot unchanged


# --- AC3: rank adaptive resolves at fit and is written back to the snapshot -----------

def test_adaptive_rank_written_back_as_int_to_snapshot(tmp_path):
    proj = _field_project(tmp_path)
    cfg = parse_config(_field_config({"rank": "adaptive"}))
    run = run_training(proj, cfg)

    snap_rank = run.config_dict["model"]["reduction"]["rank"]
    assert isinstance(snap_rank, int) and not isinstance(snap_rank, bool)
    assert snap_rank >= 1
    assert snap_rank == run.reduction["rank"]  # snapshot agrees with the lineage diagnostic
    # config_hash is computed from the RESOLVED snapshot, not the input config.
    assert run.config_hash == _sha(run.config)
    assert run.config_hash != _sha(cfg)


def test_adaptive_rank_straddling_spectrum_stays_consistent_and_reproduces(tmp_path):
    # Review regression (High): the adaptive threshold can resolve DIFFERENT ranks on
    # the 75% holdout split vs the full data ("straddling" spectrum). The snapshot must
    # carry the HOLDOUT model's rank — the model that defines the fingerprint and the
    # lineage diagnostic — or reproduce() re-trains the holdout at the wrong truncation
    # and falsely diverges. Fixture chosen empirically so the ranks genuinely differ
    # (seed=2, n_dof=6, 0.06 noise: holdout resolves 3, full data resolves 4).
    p, field = low_rank_field(n_samples=80, n_dof=6, k_true=3, seed=2)
    rng = np.random.default_rng(1002)
    noisy = field + 0.06 * rng.standard_normal(field.shape)
    cols: dict[str, np.ndarray] = {"p0": p[:, 0], "p1": p[:, 1]}
    for j in range(noisy.shape[1]):
        cols[f"f{j}"] = noisy[:, j]
    csv = tmp_path / "field.csv"
    pd.DataFrame(cols).to_csv(csv, index=False)
    proj = Project.init(tmp_path / "proj", name="rom")
    proj.add_dataset(csv, name="snaps")

    run = run_training(proj, parse_config(_field_config({"rank": "adaptive"})))
    snap_rank = run.config_dict["model"]["reduction"]["rank"]
    final_rank = run.load_model().rank
    # The fixture really straddles: the persisted final artifact resolved a different
    # rank than the holdout (the documented residual of holdout-canonical write-back).
    assert final_rank != snap_rank
    # Snapshot agrees with the holdout-derived lineage diagnostic — never the final model.
    assert snap_rank == run.reduction["rank"]
    # And the faithful re-run reproduces instead of falsely diverging.
    result = reproduce(proj, run.run_id)
    assert result.verdict in {"reproduced", "reproduced_within_tolerance"}
    assert_reproducible(result)


def test_adaptive_rank_run_reproduces_deterministically(tmp_path):
    # The snapshot's resolved rank makes re-training deterministic: reproduce() reads
    # rank=<int> (not adaptive), the trainer resolves it trivially, and the reproduction's
    # snapshot + config_hash match the parent's (metadata consistent, AC3).
    proj = _field_project(tmp_path)
    run = run_training(proj, parse_config(_field_config({"rank": "adaptive"})))
    result = reproduce(proj, run.run_id)
    assert result.verdict in {"reproduced", "reproduced_within_tolerance"}
    assert_reproducible(result)  # must not raise
    assert (
        result.reproduction.config_dict["model"]["reduction"]["rank"]
        == run.config_dict["model"]["reduction"]["rank"]
    )
    assert result.reproduction.config_hash == run.config_hash
