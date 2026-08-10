"""Acceptance tests for the v0.2 attribution metrics.

These fail until the v0.2 metrics land. Required API:

    crucible.metrics.attribution_profile(records) -> AttributionProfile

``records`` is an iterable of dicts with keys: base_seed, mechanism_slot,
cue_slot, committed_slot (int or None), solved (bool).

AttributionProfile exposes mechanism_responsiveness, cue_susceptibility,
cue_following, coverage, quartet_success, and an ``n`` mapping of denominators.
Metrics with an empty denominator are None, never 0.0.

Chance point for a policy committing uniformly over k slots, with the
mechanism confined to slots {0, 1}:

    mechanism_responsiveness = 1 / k**2
    cue_susceptibility       = 1 - 1 / k
    cue_following            = 1 / k
"""

from __future__ import annotations

import random

import pytest

from crucible.metrics import attribution_profile

CELLS = [(0, 0), (0, 1), (1, 0), (1, 1)]
SEEDS = range(256)
TOL = 0.02


def _records(policy, seeds=SEEDS, rng=None):
    out = []
    for seed in seeds:
        for mech, cue in CELLS:
            committed = policy(mech, cue, rng)
            out.append(
                {
                    "base_seed": seed,
                    "mechanism_slot": mech,
                    "cue_slot": cue,
                    "committed_slot": committed,
                    "solved": committed == mech,
                }
            )
    return out


def _oracle(mech, cue, rng):
    return mech


def _cue_follower(mech, cue, rng):
    return cue


def _anti_cue(mech, cue, rng):
    return 1 - cue


def _fixed_slot(mech, cue, rng):
    return 0


def _abstainer(mech, cue, rng):
    return None


def _uniform(k):
    def policy(mech, cue, rng):
        return rng.randrange(k)

    return policy


# --- scripted controls sit at their exact coordinates ------------------------


def test_oracle_is_fully_mechanism_responsive():
    p = attribution_profile(_records(_oracle))
    assert p.mechanism_responsiveness == 1.0
    assert p.cue_susceptibility == 0.0
    assert p.coverage == 1.0
    assert p.quartet_success == 1.0


def test_cue_follower_is_fully_cue_driven():
    p = attribution_profile(_records(_cue_follower))
    assert p.mechanism_responsiveness == 0.0
    assert p.cue_susceptibility == 1.0
    assert p.cue_following == 1.0
    assert p.coverage == 1.0
    assert p.quartet_success == 0.0


def test_fixed_slot_is_flat_on_both_axes():
    p = attribution_profile(_records(_fixed_slot))
    assert p.mechanism_responsiveness == 0.0
    assert p.cue_susceptibility == 0.0
    assert p.cue_following == 0.5
    assert p.quartet_success == 0.0


def test_directional_metric_separates_anti_cue_from_cue_follower():
    """Non-directional susceptibility cannot tell these apart; cue_following can."""
    follower = attribution_profile(_records(_cue_follower))
    anti = attribution_profile(_records(_anti_cue))
    assert follower.cue_susceptibility == anti.cue_susceptibility == 1.0
    assert follower.cue_following == 1.0
    assert anti.cue_following == 0.0


# --- abstention is undefined, not zero ---------------------------------------


def test_abstainer_has_zero_coverage_and_undefined_axes():
    p = attribution_profile(_records(_abstainer))
    assert p.coverage == 0.0
    assert p.mechanism_responsiveness is None
    assert p.cue_susceptibility is None
    assert p.cue_following is None


def test_partial_abstention_reports_its_denominator():
    records = _records(_oracle)
    for record in records[::2]:
        record["committed_slot"] = None
        record["solved"] = False
    p = attribution_profile(records)
    assert p.coverage == pytest.approx(0.5)
    assert p.n["mechanism_responsiveness"] < len(records) // 4


# --- the chance point ---------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3])
def test_uniform_committer_sits_at_analytic_chance_point(k):
    rng = random.Random(0)
    p = attribution_profile(_records(_uniform(k), seeds=range(4000), rng=rng))
    assert p.mechanism_responsiveness == pytest.approx(1 / k**2, abs=TOL)
    assert p.cue_susceptibility == pytest.approx(1 - 1 / k, abs=TOL)
    assert p.cue_following == pytest.approx(1 / k, abs=TOL)
    assert p.coverage == 1.0


def test_chance_point_is_not_the_origin():
    """A uniform committer must not be confusable with a flat policy."""
    rng = random.Random(1)
    chance = attribution_profile(_records(_uniform(3), seeds=range(4000), rng=rng))
    flat = attribution_profile(_records(_fixed_slot))
    assert chance.mechanism_responsiveness > flat.mechanism_responsiveness + TOL
    assert chance.cue_susceptibility > flat.cue_susceptibility + TOL
