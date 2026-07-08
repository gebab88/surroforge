"""The shipped conformance suite, run against the Professional sklearn trainers.

torch-mlp's conformance run lives in test_torch_mlp.py (needs the [torch] extra).
"""

from __future__ import annotations

import pytest

from surroforge.conformance import check_trainer
from surroforge_professional.trainers.gradient_boosting import GradientBoostingTrainer
from surroforge_professional.trainers.mlp import MLPTrainer
from surroforge_professional.trainers.random_forest import RandomForestTrainer

PRO_TRAINERS = [
    pytest.param(MLPTrainer, id="mlp"),
    pytest.param(GradientBoostingTrainer, id="gradient-boosting"),
    pytest.param(RandomForestTrainer, id="random-forest"),
]


@pytest.mark.parametrize("trainer_cls", PRO_TRAINERS)
def test_professional_trainer_is_conformant(trainer_cls):
    report = check_trainer(trainer_cls)
    assert report.passed, report.summary()
    report.raise_for_status()  # must not raise
