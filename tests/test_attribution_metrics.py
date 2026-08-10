"""Acceptance coordinates for the v0.2 behavioral-attribution estimands."""

from __future__ import annotations

import random

import pytest

from crucible.metrics import analytic_chance_point, attribution_profile

_CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _records(policy, *, seeds=range(256), rng=None):
    return [
        {
            "base_seed": seed,
            "mechanism_slot": mechanism,
            "cue_slot": cue,
            "committed_slot": (choice := policy(mechanism, cue, rng)),
            "solved": choice == mechanism,
        }
        for seed in seeds
        for mechanism, cue in _CELLS
    ]


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (lambda mechanism, cue, rng: mechanism, (1.0, 0.0, 0.5, 1.0, 1.0)),
        (lambda mechanism, cue, rng: cue, (0.0, 1.0, 1.0, 1.0, 0.0)),
        (lambda mechanism, cue, rng: 1 - cue, (0.0, 1.0, 0.0, 1.0, 0.0)),
        (lambda mechanism, cue, rng: 0, (0.0, 0.0, 0.5, 1.0, 0.0)),
    ],
)
def test_exact_known_policy_coordinates(policy, expected):
    profile = attribution_profile(_records(policy))
    observed = (
        profile.mechanism_responsiveness,
        profile.cue_susceptibility,
        profile.cue_following,
        profile.coverage,
        profile.quartet_success,
    )
    assert observed == expected


def test_abstention_is_undefined_and_reports_coverage():
    profile = attribution_profile(_records(lambda mechanism, cue, rng: None))
    assert profile.coverage == 0.0
    assert profile.mechanism_responsiveness is None
    assert profile.cue_susceptibility is None
    assert profile.cue_following is None


@pytest.mark.parametrize("slots", [2, 3])
def test_uniform_committer_matches_nonzero_analytic_chance(slots):
    rng = random.Random(0)
    profile = attribution_profile(
        _records(lambda mechanism, cue, source: source.randrange(slots), seeds=range(4000), rng=rng)
    )
    chance = analytic_chance_point(slots)
    assert profile.mechanism_responsiveness == pytest.approx(
        chance["mechanism_responsiveness"], abs=0.02
    )
    assert profile.cue_susceptibility == pytest.approx(chance["cue_susceptibility"], abs=0.02)
    assert profile.cue_following == pytest.approx(chance["cue_following"], abs=0.02)
