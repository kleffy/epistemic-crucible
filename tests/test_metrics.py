"""Unit tests for crucible/metrics.py against hand-constructed trace fixtures."""

from __future__ import annotations

import json
import pathlib

from crucible.metrics import (
    MetricResult,
    concept_reuse_proxy,
    counterfactual_accuracy,
    curriculum_progression,
    failure_diversity,
    filter_records,
    full_report,
    intervention_efficiency,
    intervention_validity,
    load_trace,
    shortcut_exposure_score,
    shortcut_sensitivity,
    task_success_rate,
    transfer_success,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _outcome(
    goal_achieved: bool,
    family: str = "affordance",
    agent: str = "random",
    split: str = "train",
    seed: int = 0,
    episode: int = 0,
    steps: int = 10,
    interventions: int = 2,
    illegal_rate: float = 0.0,
    energy_remaining: int = 100,
    unique_effects: list | None = None,
) -> dict:
    return {
        "kind": "outcome",
        "episode": episode,
        "seed": seed,
        "family": family,
        "split": split,
        "agent": agent,
        "goal_achieved": goal_achieved,
        "steps": steps,
        "interventions": interventions,
        "energy_remaining": energy_remaining,
        "illegal_rate": illegal_rate,
        "unique_effects": unique_effects or [],
    }


def _step(
    action_kind: str = "apply",
    effects: list | None = None,
    legal: bool = True,
    family: str = "affordance",
    agent: str = "random",
    split: str = "train",
    seed: int = 0,
    episode: int = 0,
    step: int = 0,
    energy: int = 100,
    done: bool = False,
    action_args: dict | None = None,
) -> dict:
    return {
        "kind": "step",
        "episode": episode,
        "seed": seed,
        "family": family,
        "split": split,
        "agent": agent,
        "step": step,
        "action": {"kind": action_kind, "args": action_args or {}},
        "effects": effects or [],
        "legal": legal,
        "energy": energy,
        "done": done,
    }


# ---------------------------------------------------------------------------
# Load trace
# ---------------------------------------------------------------------------


def test_load_trace_splits_record_types(tmp_path: pathlib.Path):
    lines = [
        json.dumps(_step(step=0)),
        json.dumps(_outcome(True)),
        json.dumps(_step(step=1)),
    ]
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(lines))
    steps, outcomes = load_trace(p)
    assert len(steps) == 2
    assert len(outcomes) == 1


def test_load_trace_empty(tmp_path: pathlib.Path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    steps, outcomes = load_trace(p)
    assert steps == []
    assert outcomes == []


# ---------------------------------------------------------------------------
# filter_records
# ---------------------------------------------------------------------------


def test_filter_records_by_agent():
    records = [
        _outcome(True, agent="random"),
        _outcome(True, agent="heuristic"),
        _outcome(False, agent="random"),
    ]
    result = filter_records(records, agent="random")
    assert len(result) == 2
    assert all(r["agent"] == "random" for r in result)


def test_filter_records_by_family():
    records = [
        _outcome(True, family="affordance"),
        _outcome(True, family="causal_gate"),
    ]
    result = filter_records(records, family="affordance")
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Task Success Rate
# ---------------------------------------------------------------------------


def test_task_success_rate_basic():
    outcomes = [
        _outcome(True),
        _outcome(True),
        _outcome(True),
        _outcome(False),
        _outcome(False),
    ]
    result = task_success_rate(outcomes)
    assert isinstance(result, MetricResult)
    assert result.value == 0.6
    assert result.count == 5


def test_task_success_rate_filters():
    outcomes = [
        _outcome(True, family="affordance"),
        _outcome(False, family="causal_gate"),
        _outcome(True, family="affordance"),
    ]
    result = task_success_rate(outcomes, family="affordance")
    assert result.value == 1.0
    assert result.count == 2


def test_task_success_rate_empty():
    result = task_success_rate([])
    assert result.value == 0.0
    assert result.count == 0


def test_task_success_rate_split_filter():
    outcomes = [
        _outcome(True, split="train"),
        _outcome(False, split="test"),
    ]
    result = task_success_rate(outcomes, split="train")
    assert result.value == 1.0


# ---------------------------------------------------------------------------
# Transfer Success
# ---------------------------------------------------------------------------


def test_transfer_success_delta():
    outcomes = (
        [_outcome(True, split="train", seed=i, episode=i) for i in range(8)]
        + [_outcome(False, split="train", seed=8, episode=8)]
        + [_outcome(False, split="train", seed=9, episode=9)]
        + [_outcome(True, split="test", seed=10, episode=10)]
        + [_outcome(True, split="test", seed=11, episode=11)]
        + [_outcome(False, split="test", seed=12, episode=12)]
        + [_outcome(False, split="test", seed=13, episode=13)]
        + [_outcome(False, split="test", seed=14, episode=14)]
    )
    result = transfer_success(outcomes, family="affordance", agent="random")
    value = result.value
    assert isinstance(value, dict)
    key = "affordance/random"
    assert key in value
    assert value[key]["delta"] is not None
    # train_tsr = 0.8, test_tsr = 0.4 → delta = -0.4
    assert abs(value[key]["delta"] - (-0.4)) < 0.01


def test_transfer_success_no_test_data():
    outcomes = [_outcome(True, split="train", seed=i, episode=i) for i in range(5)]
    result = transfer_success(outcomes, family="affordance", agent="random")
    key = "affordance/random"
    assert result.value[key]["test_tsr"] is None
    assert result.value[key]["delta"] is None


# ---------------------------------------------------------------------------
# Shortcut Sensitivity
# ---------------------------------------------------------------------------


def test_shortcut_sensitivity_affordance():
    train = [
        _outcome(True, family="affordance", split="train", seed=i, episode=i) for i in range(5)
    ]
    test = [
        _outcome(False, family="affordance", split="test", seed=i + 10, episode=i + 10)
        for i in range(5)
    ]
    outcomes = train + test
    result = shortcut_sensitivity(outcomes)
    key = "affordance/random"
    assert key in result.value
    # train_tsr=1.0, test_tsr=0.0 → SS=1.0
    assert result.value[key] == 1.0


def test_shortcut_sensitivity_non_shortcut_family():
    outcomes = [
        _outcome(True, family="causal_gate", split="train", seed=i, episode=i) for i in range(3)
    ]
    result = shortcut_sensitivity(outcomes)
    # causal_gate is not a shortcut family → not in results
    assert not any("causal_gate" in k for k in result.value.keys())


def test_shortcut_sensitivity_increases_under_perturbation():
    _train = [
        _outcome(True, family="affordance", split="train", seed=i, episode=i) for i in range(5)
    ]
    # Low perturbation: test TSR=0.8
    low_test = [
        _outcome(True, family="affordance", split="test", seed=i + 10, episode=i + 10)
        for i in range(4)
    ] + [_outcome(False, family="affordance", split="test", seed=14, episode=14)]
    outcomes_low = _train + low_test
    # High perturbation: test TSR=0.0
    high_test = [
        _outcome(False, family="affordance", split="test", seed=i + 10, episode=i + 10)
        for i in range(5)
    ]
    outcomes_high = _train + high_test
    ss_low = shortcut_sensitivity(outcomes_low).value.get("affordance/random", 0)
    ss_high = shortcut_sensitivity(outcomes_high).value.get("affordance/random", 0)
    assert ss_high > ss_low


# ---------------------------------------------------------------------------
# Intervention Validity
# ---------------------------------------------------------------------------


def test_intervention_validity_all():
    steps = [_step(action_kind="apply", effects=["opened gate"]) for _ in range(5)]
    result = intervention_validity(steps)
    assert result.value == 1.0
    assert result.count == 5


def test_intervention_validity_partial():
    steps = [
        _step(action_kind="apply", effects=["opened gate"]),
        _step(action_kind="apply", effects=[]),
        _step(action_kind="combine", effects=["produced token"]),
        _step(action_kind="combine", effects=[]),
    ]
    result = intervention_validity(steps)
    assert result.value == 0.5
    assert result.count == 4


def test_intervention_validity_no_steps():
    steps = [_step(action_kind="move"), _step(action_kind="wait")]
    result = intervention_validity(steps)
    assert result.count == 0
    assert result.value == 0.0


def test_intervention_validity_empty():
    result = intervention_validity([])
    assert result.count == 0


# ---------------------------------------------------------------------------
# Intervention Efficiency
# ---------------------------------------------------------------------------


def test_intervention_efficiency_no_spec():
    result = intervention_efficiency([], [])
    assert result.value is None
    assert result.count == 0
    assert "note" in result.metadata


def test_intervention_efficiency_with_spec():
    outcomes = [_outcome(True, steps=4)]
    # Oracle has 2 actions; agent took 4 steps → IE = min(2/4, 1.0) = 0.5
    cert = type("C", (), {"action_sequence": [{}, {}]})()
    spec_mock = type("S", (), {"solution_certificate": cert})()

    def get_spec(family, seed, split):
        return spec_mock

    result = intervention_efficiency([], outcomes, get_spec)
    assert result.count == 1
    assert result.value == 0.5


def test_intervention_efficiency_oracle_longer_than_agent():
    # Oracle 10 steps, agent 2 → ratio capped at 1.0
    outcomes = [_outcome(True, steps=2)]
    spec_mock = type(
        "S", (), {"solution_certificate": type("C", (), {"action_sequence": list(range(10))})()}
    )()

    def get_spec(family, seed, split):
        return spec_mock

    result = intervention_efficiency([], outcomes, get_spec)
    assert result.value == 1.0


# ---------------------------------------------------------------------------
# Counterfactual Accuracy
# ---------------------------------------------------------------------------


def test_counterfactual_accuracy_no_predict():
    steps = [_step(action_kind="apply")]
    result = counterfactual_accuracy(steps, [], mode="prediction")
    assert result.count == 0
    assert result.value is None
    assert "note" in result.metadata


def test_counterfactual_accuracy_behavioral_no_spec():
    result = counterfactual_accuracy([], [], get_spec=None, mode="behavioral")
    assert result.value is None
    assert "note" in result.metadata


def test_counterfactual_accuracy_behavioral_with_spec():
    correct_obj = "cf_0_train_block0"
    outcomes = [_outcome(True, family="counterfactual", seed=0, split="train", episode=0)]
    steps = [
        _step(
            action_kind="apply",
            family="counterfactual",
            seed=0,
            split="train",
            episode=0,
            action_args={"tool_id": "cf_0_train_source", "target_id": correct_obj},
        )
    ]
    spec_mock = type("S", (), {"goal": type("G", (), {"classify_correct_obj_id": correct_obj})()})()

    def get_spec(family, seed, split):
        return spec_mock

    result = counterfactual_accuracy(steps, outcomes, get_spec=get_spec, mode="behavioral")
    assert result.count == 1
    assert result.value == 1.0


# ---------------------------------------------------------------------------
# Failure Diversity
# ---------------------------------------------------------------------------


def test_failure_diversity_distinct():
    outcomes = [
        _outcome(False, steps=40, interventions=0, episode=0),  # timeout + no_interaction
        _outcome(False, steps=5, interventions=1, illegal_rate=0.8, episode=1),  # high_illegal
    ]
    result = failure_diversity([], outcomes)
    assert result.count == 2
    v = result.value
    assert isinstance(v, dict)
    assert v["distinct_modes"] >= 2


def test_failure_diversity_empty():
    result = failure_diversity([], [])
    assert result.count == 0
    v = result.value
    assert v["distinct_modes"] == 0
    assert all(count == 0 for key, count in v.items() if key != "distinct_modes")


def test_failure_diversity_no_failures():
    outcomes = [_outcome(True, episode=i) for i in range(3)]
    result = failure_diversity([], outcomes)
    assert result.count == 0


def test_failure_diversity_no_effects_mode():
    outcomes = [_outcome(False, unique_effects=[], interventions=2, episode=0)]
    result = failure_diversity([], outcomes)
    assert result.value["no_effects"] == 1


# ---------------------------------------------------------------------------
# Curriculum Progression
# ---------------------------------------------------------------------------


def test_curriculum_progression_flat():
    # All successes → slope ≈ 0
    outcomes = [_outcome(True, seed=i, episode=i) for i in range(10)]
    result = curriculum_progression(outcomes, window=5)
    key = "affordance/random"
    assert key in result.value
    slope = result.value[key]["slope"]
    assert abs(slope) < 0.01


def test_curriculum_progression_improving():
    # First 5 seeds fail, next 5 succeed → positive slope
    outcomes = [_outcome(False, seed=i, episode=i) for i in range(5)] + [
        _outcome(True, seed=i + 5, episode=i + 5) for i in range(5)
    ]
    result = curriculum_progression(outcomes, window=5)
    key = "affordance/random"
    slope = result.value[key]["slope"]
    assert slope > 0


def test_curriculum_progression_empty():
    result = curriculum_progression([])
    assert result.count == 0
    assert result.value == {}


# ---------------------------------------------------------------------------
# Full report and gaming risk
# ---------------------------------------------------------------------------


def test_metrics_handle_empty_traces():
    # All metric functions must handle empty inputs without raising.
    task_success_rate([])
    transfer_success([])
    shortcut_sensitivity([])
    intervention_validity([])
    intervention_efficiency([], [])
    counterfactual_accuracy([], [])
    failure_diversity([], [])
    curriculum_progression([])


def test_full_report_keys(tmp_path: pathlib.Path):
    # Write a minimal trace with one step and one outcome.
    lines = [
        json.dumps(_step()),
        json.dumps(_outcome(False)),
    ]
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(lines))
    report = full_report(p)
    expected = {
        "task_success_rate",
        "transfer_success",
        "shortcut_sensitivity",
        "intervention_validity",
        "intervention_efficiency",
        "counterfactual_accuracy",
        "concept_reuse_proxy",
        "failure_diversity",
        "curriculum_progression",
    }
    assert set(report.keys()) == expected


def test_gaming_risk_non_empty(tmp_path: pathlib.Path):
    lines = [json.dumps(_step()), json.dumps(_outcome(False))]
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(lines))
    report = full_report(p)
    for name, result in report.items():
        assert result.gaming_risk, f"MetricResult.gaming_risk is empty for {name}"


# ---------------------------------------------------------------------------
# Shortcut Exposure Score
# ---------------------------------------------------------------------------


def test_shortcut_exposure_score_high_for_color_cue_agent():
    clean_outcomes = [_outcome(True, episode=i) for i in range(5)]
    perturbed_outcomes = [_outcome(False, episode=i + 10) for i in range(5)]
    result = shortcut_exposure_score(clean_outcomes, perturbed_outcomes)
    assert result.value == 1.0
    assert result.count == 10
    assert result.gaming_risk


def test_shortcut_exposure_score_zero_for_causal_agent():
    outcomes = [_outcome(True, episode=i) for i in range(5)]
    result = shortcut_exposure_score(outcomes, list(outcomes))
    assert result.value == 0.0


def test_shortcut_exposure_score_empty():
    result = shortcut_exposure_score([], [])
    assert result.count == 0
    assert result.value == 0.0


def test_shortcut_exposure_score_filters():
    clean = [_outcome(True, family="affordance", agent="random", episode=i) for i in range(4)]
    perturbed = [
        _outcome(False, family="affordance", agent="random", episode=i + 10) for i in range(4)
    ]
    # Filter to specific family/agent — result should still be 1.0
    result = shortcut_exposure_score(clean, perturbed, family="affordance", agent="random")
    assert result.value == 1.0


# ---------------------------------------------------------------------------
# Concept reuse proxy
# ---------------------------------------------------------------------------


def test_concept_reuse_proxy_partitions_on_prior_evidence():
    """Test-split episodes with informative interventions succeed; others fail."""
    # Two episodes WITH informative evidence (both solved).
    ep_a = [
        _step(action_kind="apply", effects=["opened gate"], split="test", episode=0),
        _outcome(True, split="test", episode=0),
    ]
    ep_b = [
        _step(action_kind="inspect", effects=["inspect:tool0:..."], split="test", episode=1),
        _outcome(True, split="test", episode=1),
    ]
    # Two episodes WITHOUT informative evidence (both failed): one with no
    # intervention, one whose APPLY only produced "no_effect".
    ep_c = [
        _step(action_kind="move", effects=[], split="test", episode=2),
        _outcome(False, split="test", episode=2),
    ]
    ep_d = [
        _step(action_kind="apply", effects=["no_effect"], split="test", episode=3),
        _outcome(False, split="test", episode=3),
    ]
    # A train-split episode that must be ignored entirely.
    ep_train = [
        _step(action_kind="apply", effects=["opened gate"], split="train", episode=4),
        _outcome(True, split="train", episode=4),
    ]

    steps, outcomes = [], []
    for ep in (ep_a, ep_b, ep_c, ep_d, ep_train):
        for rec in ep:
            (steps if rec["kind"] == "step" else outcomes).append(rec)

    result = concept_reuse_proxy(steps, outcomes)
    assert result.count == 4, "only the four test-split episodes are counted"
    v = result.value
    assert v["with_evidence_tsr"] == 1.0
    assert v["without_evidence_tsr"] == 0.0
    assert v["concept_reuse"] == 1.0
    assert v["n_with_evidence"] == 2 and v["n_without_evidence"] == 2


def test_concept_reuse_proxy_no_test_episodes_is_none():
    steps = [_step(split="train")]
    outcomes = [_outcome(True, split="train")]
    result = concept_reuse_proxy(steps, outcomes)
    assert result.value is None
    assert result.count == 0
