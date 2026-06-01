"""Hidden rule generation, consistency, and public-boundary tests."""

from crucible.grammar import TaskFamily, build_world_from_spec, generate_task
from crucible.observations import observe
from crucible.rules import (
    EffectKind,
    LatentRule,
    RuleEffect,
    RuleTrigger,
    validate_rule_set,
)
from crucible.splits import SplitLabel
from crucible.utils.serialization import to_dict
from crucible.world import generate_world


def test_known_task_world_has_deterministic_rule_ids():
    spec = generate_task(TaskFamily.AFFORDANCE, seed=42, split=SplitLabel.TRAIN)
    world = build_world_from_spec(spec)

    assert [rule.rule_id for rule in world.rules] == [
        "affinity_gate_modifier",
        "conductivity_gate_rule",
        "magnetism_gate_rule",
        "source_activation_rule",
        "solubility_transform_rule",
        "property_reveal_rule",
        "source_production_rule",
        "hazard_consequence_rule",
    ]


def test_generated_rule_set_passes_consistency_validation():
    world = generate_world(seed=13)
    assert validate_rule_set(world.rules, world) == []


def test_conflicting_same_trigger_rules_are_rejected():
    world = generate_world(seed=14)
    base = world.rules[0]
    conflicting = LatentRule(
        rule_id="conflicting_rule",
        trigger=base.trigger,
        visible_preconditions=base.visible_preconditions,
        hidden_preconditions=base.hidden_preconditions,
        effects=(RuleEffect(kind=EffectKind.OPEN_GATE),),
        description="Deliberately conflicts with the base rule for testing.",
        priority=base.priority + 1,
    )

    errors = validate_rule_set([base, conflicting], world)

    assert any("contradictory effects" in error for error in errors)


def test_rules_and_pending_effects_are_not_publicly_serialized_or_observed():
    world = generate_world(seed=15)
    public_world = to_dict(world, public=True)
    obs = observe(world)

    assert "rules" not in public_world
    assert "pending_effects" not in public_world
    assert "hidden" not in str(public_world)
    assert "conductivity" not in str(public_world)
    assert "rules" not in str(obs)
    assert "pending_effects" not in str(obs)


def test_duplicate_rule_ids_are_rejected():
    world = generate_world(seed=16)
    duplicate = LatentRule(
        rule_id=world.rules[0].rule_id,
        trigger=RuleTrigger.APPLY,
        visible_preconditions=("x",),
        hidden_preconditions=("y",),
        effects=(RuleEffect(kind=EffectKind.BLOCKED),),
        description="Duplicate ID for validation.",
    )

    errors = validate_rule_set([world.rules[0], duplicate], world)

    assert any("duplicate rule_id" in error for error in errors)
