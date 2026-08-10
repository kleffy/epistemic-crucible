"""Counterfactual pair generation and prediction scoring tests."""

from crucible.counterfactuals import (
    CounterfactualDeltaKind,
    PredictionQuery,
    changed_state_paths,
    generate_counterfactual_pair,
    score_prediction,
)
from crucible.grammar import TaskFamily, generate_task
from crucible.splits import SplitLabel


def test_counterfactual_pairs_generate_for_all_task_families():
    for family in TaskFamily:
        spec = generate_task(family, seed=23, split=SplitLabel.TRAIN)
        pair = generate_counterfactual_pair(spec)

        assert pair.factual_hash != pair.counterfactual_hash
        assert pair.delta.kind == CounterfactualDeltaKind.LATENT_PROPERTY
        assert pair.delta.field_path


def test_counterfactual_pair_differs_by_exactly_declared_delta():
    spec = generate_task(TaskFamily.AFFORDANCE, seed=24, split=SplitLabel.TRAIN)
    pair = generate_counterfactual_pair(spec)

    assert changed_state_paths(pair.factual, pair.counterfactual) == [pair.delta.field_path]


def test_visible_preserving_latent_delta_has_identical_public_state():
    spec = generate_task(TaskFamily.COUNTERFACTUAL, seed=25, split=SplitLabel.TEST)
    pair = generate_counterfactual_pair(spec)

    assert pair.delta.public_visibility is False
    assert pair.factual_public_hash == pair.counterfactual_public_hash
    assert changed_state_paths(pair.factual, pair.counterfactual, public=True) == []


def test_delta_metadata_contains_required_fields():
    spec = generate_task(TaskFamily.CONTRADICTION, seed=26, split=SplitLabel.TRAIN)
    pair = generate_counterfactual_pair(spec)

    assert pair.delta.kind == CounterfactualDeltaKind.LATENT_PROPERTY
    assert pair.delta.object_id is not None
    assert pair.delta.field_path.endswith(".hidden.affinity")
    assert pair.delta.before_value is None
    assert pair.delta.after_value == "organic"
    assert pair.delta.public_visibility is False


def test_visible_feature_delta_changes_public_state():
    spec = generate_task(TaskFamily.TOOL_SUBSTITUTION, seed=27, split=SplitLabel.TEST)
    pair = generate_counterfactual_pair(spec, CounterfactualDeltaKind.VISIBLE_FEATURE)

    assert pair.delta.kind == CounterfactualDeltaKind.VISIBLE_FEATURE
    assert pair.delta.public_visibility is True
    assert pair.factual_public_hash != pair.counterfactual_public_hash
    assert changed_state_paths(pair.factual, pair.counterfactual, public=True) == [
        pair.delta.field_path
    ]


def test_prediction_query_scoring_matches_actual_effect():
    query = PredictionQuery(
        query_id="q1",
        world_hash="world",
        action={"kind": "apply", "args": {"tool_id": "tool", "target_id": "gate"}},
    )

    correct = score_prediction(query, ["opened gate"], ["opened gate"])
    incorrect = score_prediction(query, ["no_effect"], ["opened gate"])

    assert correct.correct is True
    assert incorrect.correct is False
