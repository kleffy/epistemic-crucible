from crucible.factorial import generate_affordance_quartet, run_scripted_control
from crucible.factorial_metrics import compute_factorial_metrics
from crucible.metrics import attribution_profile


def test_unconditional_cue_metric_counts_all_abstention_as_unchanged():
    outcomes = [
        run_scripted_control(cell, "abstain")
        for cell in generate_affordance_quartet(0).cells.values()
    ]
    report = compute_factorial_metrics(outcomes, bootstrap_samples=0)
    assert report.cue_susceptibility.value is None
    assert report.cue_susceptibility_all.value == 0.0
    assert report.cue_susceptibility_all.denominator == 2
    profile = attribution_profile(
        {
            "base_seed": outcome.base_seed,
            "mechanism_slot": outcome.mechanism_slot,
            "cue_slot": outcome.cue_slot,
            "committed_slot": outcome.committed_slot,
            "solved": outcome.solved,
        }
        for outcome in outcomes
    )
    assert profile.cue_susceptibility_all == 0.0
    assert profile.n["cue_susceptibility_all"] == 1
