"""Integrity tests for the v0.2 crossed-intervention protocol."""

from __future__ import annotations

import copy

import pytest

from crucible.actions import Action, ActionKind
from crucible.agents.base import enumerate_candidate_actions
from crucible.agents.prompting import build_label_map, build_user_message, describe_goal
from crucible.factorial import (
    EpistemicAction,
    FactorialEpisode,
    MacroActionKind,
    compile_factorial_certificate,
    execute_compiled_actions,
    generate_affordance_challenge,
    generate_affordance_quartet,
    run_scripted_control,
    shortest_legal_moves,
    validate_affordance_quartet,
)
from crucible.factorial_metrics import (
    compute_challenge_metrics,
    compute_factorial_metrics,
    paired_arm_contrast,
    paired_challenge_arm_contrast,
)
from crucible.grammar import build_world_from_spec, validate_task
from crucible.ledger import EpisodeLedger, content_hash
from crucible.observations import observe
from crucible.utils.serialization import to_dict


def _without_color(observation: dict) -> dict:
    normalized = copy.deepcopy(observation)
    for obj in normalized["objects"].values():
        obj.pop("color", None)
    return normalized


def _render_cell(cell) -> str:
    spec = cell.task_spec
    observation = observe(build_world_from_spec(spec, compact=True))
    candidates = enumerate_candidate_actions(observation, spec.grid_size)
    message, _, _ = build_user_message(
        observation, describe_goal(spec.goal), spec.grid_size, candidates
    )
    return message


def test_quartet_crosses_only_mechanism_and_red_cue_carriers():
    quartet = generate_affordance_quartet(7)
    assert quartet.base_world_id == "affq_7"
    assert validate_affordance_quartet(quartet) == []
    assert {cell.condition_id for cell in quartet.cells.values()} == {
        "m0_c0",
        "m0_c1",
        "m1_c0",
        "m1_c1",
    }
    for cell in quartet.cells.values():
        assert cell.task_spec.split is None
        assert validate_task(cell.task_spec) == []


def test_quartet_worlds_have_identical_positions_and_nuisances():
    quartet = generate_affordance_quartet(91)
    observations = [
        observe(build_world_from_spec(cell.task_spec)) for cell in quartet.cells.values()
    ]
    assert all(_without_color(obs) == _without_color(observations[0]) for obs in observations)


def test_quartet_labels_and_initial_candidate_order_are_identical():
    quartet = generate_affordance_quartet(11)
    labels = []
    candidates = []
    for cell in quartet.cells.values():
        obs = observe(build_world_from_spec(cell.task_spec, compact=True))
        labels.append(build_label_map(obs))
        candidates.append([to_dict(action) for action in enumerate_candidate_actions(obs)])
    assert all(label_map == labels[0] for label_map in labels)
    assert all(action_list == candidates[0] for action_list in candidates)


@pytest.mark.parametrize("seed", range(16))
def test_mechanism_axis_prompt_is_byte_identical_including_colour(seed: int):
    quartet = generate_affordance_quartet(seed)
    for cue in (0, 1):
        assert _render_cell(quartet.cell(0, cue)) == _render_cell(quartet.cell(1, cue))


@pytest.mark.parametrize("seed", range(16))
def test_cue_axis_public_observation_matches_after_colour_mask(seed: int):
    quartet = generate_affordance_quartet(seed)
    for mechanism in (0, 1):
        left = observe(build_world_from_spec(quartet.cell(mechanism, 0).task_spec, compact=True))
        right = observe(build_world_from_spec(quartet.cell(mechanism, 1).task_spec, compact=True))
        assert _without_color(left) == _without_color(right)


@pytest.mark.parametrize("use_detector", [False, True])
def test_compiled_certificate_uses_only_legal_actions_and_solves(use_detector: bool):
    for seed in range(10):
        for cell in generate_affordance_quartet(seed).cells.values():
            actions = compile_factorial_certificate(cell, use_detector=use_detector)
            outcome = execute_compiled_actions(cell, actions)
            assert outcome.solved is True
            assert outcome.committed_slot == cell.mechanism_carrier_slot
            assert outcome.interventions <= cell.task_spec.constraints.max_interventions
            assert outcome.steps <= cell.task_spec.constraints.max_steps
            assert outcome.detector_queries == (3 if use_detector else 0)
            assert all(
                event["payload"].get("legal", True)
                for event in outcome.trace
                if event["kind"] == "action"
            )


def test_first_tool_to_gate_application_is_terminal_even_when_wrong():
    cell = generate_affordance_quartet(3).cell(1, 0)
    outcome = run_scripted_control(cell, "fixed_slot")
    assert outcome.committed_slot == 0
    assert outcome.solved is False
    assert outcome.done_reason == "committed"


def test_primary_macro_actions_compile_to_legal_environment_steps():
    cell = generate_affordance_quartet(8).cell(1, 0)
    episode = FactorialEpisode(cell)
    episode.reset()
    result = episode.macro_step(EpistemicAction(MacroActionKind.QUERY, 0))
    assert result["evidence"] in {"signal-positive", "signal-negative"}
    assert all(event["payload"].get("legal", True) for event in episode.trace_records())
    assert EpistemicAction(MacroActionKind.QUERY, 0) not in episode.macro_actions()
    episode.macro_step(EpistemicAction(MacroActionKind.COMMIT, cell.mechanism_carrier_slot))
    assert episode.outcome().solved is True


def test_invalid_response_termination_is_abstention_not_evaluator_choice():
    episode = FactorialEpisode(generate_affordance_quartet(8).cell(1, 0))
    episode.reset()
    outcome = episode.terminate_without_commit("invalid_response")
    assert outcome.committed_slot is None
    assert outcome.commit_mode is None
    assert outcome.solved is False
    assert outcome.done_reason == "invalid_response"


def test_focal_and_neutral_global_slots_are_balanced_across_seeds():
    neutral_counts = [0, 0, 0]
    mechanism_counts = [0, 0, 0]
    cue_counts = [0, 0, 0]
    for seed in range(300):
        quartet = generate_affordance_quartet(seed)
        neutral_counts[int(quartet.cell(0, 0).task_spec.metadata["neutral_slot"])] += 1
        for cell in quartet.cells.values():
            mechanism_counts[cell.mechanism_carrier_slot] += 1
            cue_counts[cell.cue_carrier_slot] += 1
    assert neutral_counts == [100, 100, 100]
    assert mechanism_counts == [400, 400, 400]
    assert cue_counts == [400, 400, 400]


def test_conductivity_detector_is_property_specific():
    for cell in generate_affordance_quartet(5).cells.values():
        for spec in cell.task_spec.object_specs:
            if spec.role.startswith("tool_slot_"):
                assert not any((spec.solubility, spec.magnetism, spec.charge, spec.fragility))
                assert spec.affinity is None
        episode = FactorialEpisode(cell)
        episode.reset()
        for slot in range(3):
            result = episode.macro_step(EpistemicAction(MacroActionKind.QUERY, slot))
            assert (result["evidence"] == "signal-positive") == (
                slot == cell.mechanism_carrier_slot
            )


@pytest.mark.parametrize(
    ("control", "mechanism", "cue", "coverage", "quartet"),
    [
        ("mechanism_oracle", 1.0, 0.0, 1.0, 1.0),
        ("detector_policy", 1.0, 0.0, 1.0, 1.0),
        ("cue_follower", 0.0, 1.0, 1.0, 0.0),
        ("fixed_slot", 0.0, 0.0, 1.0, 0.0),
    ],
)
def test_scripted_controls_occupy_expected_metric_regions(
    control: str,
    mechanism: float,
    cue: float,
    coverage: float,
    quartet: float,
):
    outcomes = [
        run_scripted_control(cell, control)
        for seed in range(3)
        for cell in generate_affordance_quartet(seed).cells.values()
    ]
    report = compute_factorial_metrics(outcomes, bootstrap_samples=100)
    assert report.mechanism_responsiveness.value == mechanism
    assert report.cue_susceptibility.value == cue
    assert report.coverage.value == coverage
    assert report.quartet_success.value == quartet
    assert report.mechanism_responsiveness.denominator == 6
    assert report.cue_susceptibility.denominator == 6


def test_focal_uniform_control_matches_two_slot_reference_region():
    outcomes = [
        run_scripted_control(cell, "focal_uniform")
        for seed in range(512)
        for cell in generate_affordance_quartet(seed).cells.values()
    ]
    report = compute_factorial_metrics(outcomes, bootstrap_samples=0)
    assert report.mechanism_tracking.value == pytest.approx(0.5, abs=0.05)
    assert report.mechanism_responsiveness.value == pytest.approx(0.25, abs=0.05)
    assert report.cue_following.value == pytest.approx(0.5, abs=0.05)
    assert report.cue_susceptibility.value == pytest.approx(0.5, abs=0.05)


def test_paired_arm_contrast_preserves_seed_pairing():
    reference = [
        run_scripted_control(cell, "cue_follower")
        for seed in range(8)
        for cell in generate_affordance_quartet(seed).cells.values()
    ]
    treatment = [
        run_scripted_control(cell, "detector_policy")
        for seed in range(8)
        for cell in generate_affordance_quartet(seed).cells.values()
    ]
    mechanism_delta = paired_arm_contrast(
        reference, treatment, "mechanism_tracking", bootstrap_samples=100
    )
    cue_delta = paired_arm_contrast(reference, treatment, "cue_following", bootstrap_samples=100)
    assert mechanism_delta.value == 0.5
    assert cue_delta.value == -0.5
    assert mechanism_delta.denominator == cue_delta.denominator == 8


def test_challenge_metrics_use_seed_clustered_intervals_and_denominators():
    outcomes = [
        run_scripted_control(cell, "detector_policy")
        for seed in range(3)
        for cell in generate_affordance_challenge(seed).cells.values()
    ]
    report = compute_challenge_metrics(outcomes, bootstrap_samples=100)
    assert report.mechanism_accuracy.value == 1.0
    assert report.cue_susceptibility.value == 0.0
    assert report.cue_following.value == pytest.approx(1 / 3)
    assert report.coverage.value == 1.0
    assert report.detector_query_rate.value == 1.0
    assert report.all_cells_success.value == 1.0
    assert report.mechanism_accuracy.denominator == 18
    assert report.cue_susceptibility.denominator == 9
    assert report.coverage.denominator == 3
    assert report.complete_challenges == 3


def test_challenge_abstention_is_undefined_not_zero():
    outcomes = [
        run_scripted_control(cell, "abstain")
        for cell in generate_affordance_challenge(0).cells.values()
    ]
    report = compute_challenge_metrics(outcomes, bootstrap_samples=0)
    assert report.coverage.value == 0.0
    assert report.mechanism_accuracy.value is None
    assert report.cue_susceptibility.value is None
    assert report.cue_following.value is None


def test_paired_challenge_contrast_preserves_seed_pairing():
    reference = [
        run_scripted_control(cell, "cue_follower")
        for seed in range(8)
        for cell in generate_affordance_challenge(seed).cells.values()
    ]
    treatment = [
        run_scripted_control(cell, "detector_policy")
        for seed in range(8)
        for cell in generate_affordance_challenge(seed).cells.values()
    ]
    mechanism = paired_challenge_arm_contrast(
        reference,
        treatment,
        "mechanism_accuracy",
        bootstrap_samples=100,
    )
    cue = paired_challenge_arm_contrast(
        reference,
        treatment,
        "cue_following",
        bootstrap_samples=100,
    )
    assert mechanism.value == pytest.approx(2 / 3)
    assert cue.value == pytest.approx(-2 / 3)
    assert mechanism.denominator == cue.denominator == 8


def test_aligned_crossed_success_and_query_rate_make_controls_legible():
    cue = compute_factorial_metrics(
        (
            run_scripted_control(cell, "cue_follower")
            for cell in generate_affordance_quartet(0).cells.values()
        ),
        bootstrap_samples=0,
    )
    mechanism = compute_factorial_metrics(
        (
            run_scripted_control(cell, "detector_policy")
            for cell in generate_affordance_quartet(0).cells.values()
        ),
        bootstrap_samples=0,
    )
    assert cue.aligned_success.value == 1.0
    assert cue.crossed_success.value == 0.0
    assert cue.detector_query_rate.value == 0.0
    assert mechanism.aligned_success.value == 1.0
    assert mechanism.crossed_success.value == 1.0
    assert mechanism.detector_query_rate.value == 1.0


def test_abstention_is_coverage_failure_not_zero_responsiveness():
    outcomes = [
        run_scripted_control(cell, "abstain")
        for cell in generate_affordance_quartet(0).cells.values()
    ]
    report = compute_factorial_metrics(outcomes, bootstrap_samples=0)
    assert report.coverage.value == 0.0
    assert report.mechanism_responsiveness.value is None
    assert report.cue_susceptibility.value is None
    assert report.cue_susceptibility_all.value == 0.0
    assert report.cue_susceptibility_all.denominator == 2
    assert report.choice_accuracy.value == 0.0
    assert report.choice_accuracy.denominator == 4


def test_detector_budget_rejects_a_fourth_query_without_mutation():
    cell = generate_affordance_quartet(4).cell(0, 0)
    actions = compile_factorial_certificate(cell, use_detector=True)
    episode = FactorialEpisode(cell)
    episode.reset()
    detector_actions = [action for action in actions if action.kind == ActionKind.APPLY][0:3]
    # Execute the compiled navigation and detector queries up to the third query.
    for action in actions:
        if episode.detector_queries == 3:
            break
        episode.step(action)
    assert len(detector_actions) == 3
    tool = next(obj.obj_id for obj in cell.task_spec.object_specs if obj.role == "tool_slot_0")
    detector = next(obj.obj_id for obj in cell.task_spec.object_specs if obj.role == "detector")
    # Navigate to a legal target, then the wrapper rejects the over-budget query.
    target_pos = episode.env.world.objects[tool].visible.pos
    assert target_pos is not None
    row, col = target_pos
    destinations = [
        pos
        for pos in ((row, col), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        if 0 <= pos[0] < episode.env.world.grid_size and 0 <= pos[1] < episode.env.world.grid_size
    ]
    for move in shortest_legal_moves(episode.env.world, destinations):
        episode.step(move)
    _, _, done, info = episode.step(
        Action(ActionKind.APPLY, {"tool_id": detector, "target_id": tool})
    )
    assert done is True
    assert info["effects"] == ["protocol_budget_exceeded"]


def test_ledger_roundtrip_is_lossless_and_preserves_semantics():
    ledger = EpisodeLedger()
    ledger.record_message("user", "state")
    ledger.append("action", action={"kind": "wait", "args": {}}, legal=True)
    ledger.append("effect", effects=["waited"])
    ledger.record_message("assistant", "ACTION: 0")
    restored = EpisodeLedger.from_jsonl(ledger.to_jsonl())
    assert restored.events == ledger.events
    assert restored.hash == ledger.hash
    assert restored.semantic_projection() == ledger.semantic_projection()
    assert "EPISODE_LEDGER_JSONL" in ledger.compact_message()["content"]


def test_full_and_compact_ledger_renderings_preserve_public_evidence():
    episode = FactorialEpisode(generate_affordance_quartet(12).cell(0, 1))
    episode.reset()
    episode.macro_step(EpistemicAction(MacroActionKind.QUERY, 0))
    compact = episode.ledger.compact_message()["content"]
    restored = EpisodeLedger.from_jsonl(compact.split("\n", 1)[1])
    assert content_hash(restored.semantic_projection()) == content_hash(
        episode.ledger.semantic_projection()
    )
    assert any(evidence in compact for evidence in ("signal-positive", "signal-negative"))
