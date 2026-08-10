from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crucible.agents.neural_ppo import featurize_step  # noqa: E402
from crucible.factorial import FactorialEpisode, generate_affordance_quartet  # noqa: E402
from experiments.train_factorial_bc import (  # noqa: E402
    collect_matched_demonstrations,
    validate_disjoint_seed_sets,
)


def test_matched_bc_datasets_have_exactly_equal_training_volume():
    cue = collect_matched_demonstrations("cue", seeds=[0, 1], target_transitions=32)
    mechanism = collect_matched_demonstrations("mechanism", seeds=[0, 1], target_transitions=32)
    assert len(cue) == len(mechanism) == 32


def test_bc_train_and_eval_world_seeds_must_be_disjoint():
    validate_disjoint_seed_sets([0, 1], [2, 3])
    with pytest.raises(ValueError, match="must be disjoint"):
        validate_disjoint_seed_sets([0, 1], [1, 2])


def test_neural_features_distinguish_positive_and_negative_public_markers():
    cell = generate_affordance_quartet(0).cell(0, 1)
    episode = FactorialEpisode(cell)
    episode.reset()
    from crucible.factorial import compile_factorial_certificate

    for action in compile_factorial_certificate(cell, use_detector=True):
        if episode.detector_queries == 3:
            break
        episode.step(action)
    features, _ = featurize_step(episode.obs)
    marker_signal = features.obj_flt[:, -1]
    assert 1.0 in marker_signal
    assert -1.0 in marker_signal
