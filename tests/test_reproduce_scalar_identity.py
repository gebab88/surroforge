"""Scalar-path byte-identity regression guard (Story R2.4).

The field-primary reproducibility fingerprint (Story R2.3) generalized ``reproduce()``
to judge a *field run* by its reconstructed field. This file pins the other half of the
contract: a **scalar run** (no reduction / identity basis ⇒ the field *is* the
prediction) must stay **byte-for-byte** what it was before ROM existed.

It is a *regression guard*, in the spirit of the existing boundary/profiling gates: it
passes against today's (correct) code and FAILS the moment a future change to the field
code path perturbs a scalar run — a field key leaking into scalar lineage, a changed
fingerprint algorithm, or ``reproduce()`` routing a scalar run through the field metric.

The inherited pre-ROM scalar contract lives in ``tests/test_reproduce.py`` (it passes
verbatim — Story R2.3 AC5). This file does not duplicate those assertions; it deepens
them to the **byte** level (raw-lineage key absence + recomputed-fingerprint-byte
equality + scalar-shaped ``ReproduceResult``). The CI ``reproducibility (pinned threads)``
job runs both files together.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from surroforge.core.config import parse_config
from surroforge.data.loader import select_columns
from surroforge.engine import _FINGERPRINT_MAX_ROWS, run_training
from surroforge.errors import ReproducibilityError
from surroforge.plugins import get_trainer
from surroforge.services import ReproduceResult, assert_reproducible, reproduce
from tests.conftest import poly_config

# Lineage keys the field-primary fingerprint adds for a *field* run only; a scalar run
# must carry NONE of them (the field code path is inert without a reducing trainer).
# ``probe_coverage_energy`` (Story R2.5) is field-gated too, so it belongs in this guard.
_FIELD_LINEAGE_KEYS = (
    "basis_hash",
    "holdout_coefficients",
    "reduction",
    "spec_hash",
    "probe_coverage_energy",
)
# The pre-ROM scalar lineage keys that the reproducibility verdict depends on.
_SCALAR_LINEAGE_KEYS = (
    "prediction_hash",
    "holdout_fingerprint",
    "reproduction_tolerance",
    "config_hash",
)


# --- AC1: the scalar lineage carries no field keys (the field path is inert) ----------

def test_scalar_lineage_carries_no_field_keys(project):
    # Assert on the RAW lineage dict, not the Run accessors: the accessors default-mask
    # absence (basis_hash -> "", holdout_coefficients -> [], reduction -> {},
    # spec_hash -> ""), so asserting on them would be a weaker test. Membership in the
    # persisted lineage.json is the real pin.
    run = run_training(project, parse_config(poly_config("polynomial")))
    for key in _FIELD_LINEAGE_KEYS:
        assert key not in run.lineage, f"scalar run grew a field lineage key: {key!r}"
    # The pre-ROM scalar key set must still be intact.
    for key in _SCALAR_LINEAGE_KEYS:
        assert key in run.lineage, f"scalar run lost a pre-ROM lineage key: {key!r}"


# --- AC1: the scalar fingerprint bytes == the pre-ROM algorithm, recomputed -----------

def test_scalar_fingerprint_is_byte_identical_to_pre_rom_algorithm(project):
    # Recompute the out-of-sample holdout fingerprint exactly as the pre-ROM engine does
    # (engine.run_training): same deterministic split, the HOLDOUT model fit on x_train
    # (not the final model on all data), float64 + C-contiguous + the 512-row cap. Equality
    # pins byte-identity WITHOUT a fragile golden hash constant. Polynomial is closed-form
    # and bit-stable run-to-run even on Apple Accelerate (test_reproduce_is_bit_identical
    # carries no BLAS guard), so this needs no requires_reproducible_blas marker.
    cfg = parse_config(poly_config("polynomial"))
    run = run_training(project, cfg)

    frame, _, _ = project.resolve_source(cfg.data.source)
    x, y = select_columns(frame, cfg.data.inputs, cfg.data.outputs)
    split = cfg.data.split
    x_train, x_test, y_train, _y_test = train_test_split(
        x, y, test_size=split.test_fraction, random_state=split.seed, shuffle=True
    )
    holdout_model = get_trainer(cfg.model.method)(**cfg.model.hyperparameters)
    holdout_model.fit(x_train, y_train)
    recomputed = np.ascontiguousarray(holdout_model.predict(x_test), dtype=np.float64)[
        :_FINGERPRINT_MAX_ROWS
    ]

    # holdout_fingerprint is the 2-D (n_test, 1) array (predict returns (n, 1) via _as_2d;
    # the engine stores it .tolist() as a list-of-lists), so the JSON round-trip preserves
    # the 2-D shape. Keep both sides 2-D and C-contiguous — flattening would break the
    # .tobytes() comparison even though the values match.
    stored = np.ascontiguousarray(run.holdout_fingerprint, dtype=np.float64)
    assert stored.ndim == 2 and stored.shape == recomputed.shape
    assert recomputed.tobytes() == stored.tobytes()
    assert hashlib.sha256(recomputed.tobytes()).hexdigest() == run.lineage["prediction_hash"]


# --- AC1: the scalar ReproduceResult is scalar-shaped (no field routing) --------------

def test_scalar_reproduce_result_is_scalar_shaped(project):
    run = run_training(project, parse_config(poly_config("polynomial")))
    result = reproduce(project, run.run_id)

    # Every field-only field stays at its scalar-identical default — reproduce() never
    # routes a scalar run through the field metric.
    assert result.field_relative_diff is None
    assert result.basis_hash_match is None
    assert result.basis_regauged is False
    assert result.coefficient_diagnostic is None

    # For the deterministic polynomial the verdict is driven by bit_identical (the
    # prediction_hash matches, so _verdict returns "reproduced" at services.py:149 BEFORE
    # the max_abs_diff <= tolerance branch); max_abs_diff is 0.0 because the recomputed
    # holdout bytes are identical. Do not infer the verdict came from the tolerance compare.
    assert result.verdict == "reproduced"
    assert result.max_abs_diff == 0.0
    assert_reproducible(result)  # must not raise


# --- AC1 (teeth): a divergent scalar run raises via the absolute-diff branch ----------

def test_scalar_divergence_raises_via_absolute_diff_branch():
    # Scalar-flavoured teeth (project-context Testing [MUST]): a genuine over-tolerance
    # scalar divergence must raise — and through the max_abs_diff branch, NOT the field
    # branch, proving the scalar raise path is unchanged. field_relative_diff is None, so
    # assert_reproducible judges by max_abs_diff (1.0) vs the default tolerance (1e-9):
    # not (1.0 <= 1e-9) -> raise. The message names "max abs prediction", confirming the
    # scalar branch drove the raise (a field run would say "field-energy-relative").
    diverged = ReproduceResult(
        parent=None,
        reproduction=None,
        bit_identical=False,
        max_abs_diff=1.0,
        data_hash_match=True,
        verdict="diverged",
    )
    assert diverged.field_relative_diff is None
    with pytest.raises(ReproducibilityError, match="max abs prediction"):
        assert_reproducible(diverged)


# --- AC1: the one documented scalar deviation (NaN-safe assert) stays verdict-consistent --

def test_scalar_nan_diff_raises_consistently_with_verdict():
    # R2.4's single intentional deviation: R2.3's review made assert_reproducible NaN-safe
    # via ``not (diff <= tol)`` (services.py:247). For a FINITE scalar diff this is identical
    # to the pre-ROM ``diff > tol``; only a NaN diff changes — it now RAISES, matching the
    # ``diverged`` verdict ``_verdict`` already returns for NaN (pre-ROM ``nan > tol`` was
    # False, so the verdict said diverged while assert_reproducible silently passed — an
    # inconsistency). This pins that deviation so a revert to ``diff > tol`` is caught and
    # the documented behaviour is tamper-evident. (A NaN diff is practically unreachable for
    # a real finite scalar run; the pin guards the consistency contract, not a real run.)
    from surroforge.services import _verdict

    nan = float("nan")
    # The verdict classifies a NaN scalar diff as diverged — true both pre- and post-ROM.
    assert _verdict(True, False, True, nan, tolerance=1e-9) == "diverged"

    nan_result = ReproduceResult(
        parent=None,
        reproduction=None,
        bit_identical=False,
        max_abs_diff=nan,
        data_hash_match=True,
        verdict="diverged",
    )  # field_relative_diff defaults to None -> assert_reproducible judges by max_abs_diff
    assert nan_result.field_relative_diff is None
    with pytest.raises(ReproducibilityError, match="max abs prediction"):
        assert_reproducible(nan_result)
