from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from crucible.actions import Action, ActionKind
from crucible.objects import (
    CrucibleObject,
    HiddenObjectProps,
    ObjectColor,
    ObjectShape,
    ObjectSize,
    ObjectState,
    ObjectTexture,
    ObjectType,
    VisibleObjectState,
)
from crucible.relations import Relation, RelationKind
from crucible.world import WorldState, derive_relations


class RuleTrigger(str, Enum):
    APPLY = "apply"
    COMBINE = "combine"
    DELAYED = "delayed"


class EffectKind(str, Enum):
    OPEN_GATE = "open_gate"
    ACTIVATE_OBJECT = "activate_object"
    TRANSFORM_OBJECT = "transform_object"
    REVEAL_PROPERTY = "reveal_property"
    PRODUCE_OBJECT = "produce_object"
    DAMAGE_OBJECT = "damage_object"
    DRAIN_ENERGY = "drain_energy"
    BLOCKED = "blocked"


@dataclass
class RuleEffect:
    kind: EffectKind
    target_id: str | None = None
    target_state: ObjectState | None = None
    marker: str | None = None
    energy_delta: int = 0
    produced_type: ObjectType | None = None
    delay_steps: int = 0


@dataclass
class PendingEffect:
    rule_id: str
    due_step: int
    effect: RuleEffect
    participants: dict[str, str] = field(default_factory=dict)


@dataclass
class CausalProvenance:
    rule_id: str
    trigger: RuleTrigger
    participants: dict[str, str]
    hidden_preconditions: tuple[str, ...]
    public_effects: list[str]
    scheduled_effects: list[PendingEffect] = field(default_factory=list)
    blocked_by: str | None = None


@dataclass
class LatentRule:
    rule_id: str
    trigger: RuleTrigger
    visible_preconditions: tuple[str, ...]
    hidden_preconditions: tuple[str, ...]
    effects: tuple[RuleEffect, ...]
    description: str
    priority: int = 100


def generate_rule_set(world: WorldState) -> list[LatentRule]:
    """Return the deterministic latent mechanism library for a world."""
    del world
    return _rule_templates()


def _rule_templates() -> list[LatentRule]:
    rules = [
        LatentRule(
            rule_id="affinity_gate_modifier",
            trigger=RuleTrigger.APPLY,
            visible_preconditions=("actor.type in {tool,key}", "target.type == gate"),
            hidden_preconditions=(
                "actor.conductivity == true",
                "target.affinity is not none",
                "actor.affinity != target.affinity",
            ),
            effects=(RuleEffect(kind=EffectKind.BLOCKED),),
            description="A gate with a hidden affinity rejects incompatible conductive actors.",
            priority=10,
        ),
        LatentRule(
            rule_id="conductivity_gate_rule",
            trigger=RuleTrigger.APPLY,
            visible_preconditions=("actor.type in {tool,key}", "target.type == gate"),
            hidden_preconditions=("actor.conductivity == true", "affinity compatible"),
            effects=(RuleEffect(kind=EffectKind.OPEN_GATE),),
            description="A conductive tool or key opens a compatible closed gate.",
            priority=20,
        ),
        LatentRule(
            rule_id="magnetism_gate_rule",
            trigger=RuleTrigger.APPLY,
            visible_preconditions=("actor.type == key", "target.type == gate"),
            hidden_preconditions=("actor.magnetism == true", "actor.charge == true"),
            effects=(RuleEffect(kind=EffectKind.OPEN_GATE),),
            description="An active charged magnetic key opens a closed gate.",
            priority=30,
        ),
        LatentRule(
            rule_id="source_activation_rule",
            trigger=RuleTrigger.APPLY,
            visible_preconditions=("actor.type == source", "target.type == key"),
            hidden_preconditions=("target.magnetism == true",),
            effects=(
                RuleEffect(
                    kind=EffectKind.ACTIVATE_OBJECT,
                    target_state=ObjectState.ACTIVE,
                    marker="charged",
                ),
            ),
            description="An active source charges and activates a magnetic key.",
            priority=40,
        ),
        LatentRule(
            rule_id="solubility_transform_rule",
            trigger=RuleTrigger.APPLY,
            visible_preconditions=("actor.type == source", "target.type == block"),
            hidden_preconditions=("target.solubility == true",),
            effects=(
                RuleEffect(
                    kind=EffectKind.TRANSFORM_OBJECT,
                    target_state=ObjectState.DESTROYED,
                    marker="transformed",
                ),
            ),
            description="An active source transforms a soluble block.",
            priority=50,
        ),
        LatentRule(
            rule_id="property_reveal_rule",
            trigger=RuleTrigger.APPLY,
            visible_preconditions=("actor.type == detector", "target.type == any"),
            hidden_preconditions=("target has latent signal",),
            effects=(RuleEffect(kind=EffectKind.REVEAL_PROPERTY),),
            description="A detector writes a public signal marker without naming hidden fields.",
            priority=60,
        ),
        LatentRule(
            rule_id="source_production_rule",
            trigger=RuleTrigger.COMBINE,
            visible_preconditions=("source + catalyst",),
            hidden_preconditions=("source.state == active",),
            effects=(
                RuleEffect(
                    kind=EffectKind.PRODUCE_OBJECT,
                    produced_type=ObjectType.TOKEN,
                    marker="produced",
                ),
            ),
            description="An active source combined with a catalyst produces a token.",
            priority=70,
        ),
        LatentRule(
            rule_id="hazard_consequence_rule",
            trigger=RuleTrigger.APPLY,
            visible_preconditions=("actor.type == hazard or target.type == hazard",),
            hidden_preconditions=("hazard interaction occurred",),
            effects=(
                RuleEffect(
                    kind=EffectKind.DRAIN_ENERGY,
                    energy_delta=-10,
                    delay_steps=2,
                ),
                RuleEffect(
                    kind=EffectKind.DAMAGE_OBJECT,
                    target_state=ObjectState.DESTROYED,
                    delay_steps=2,
                ),
            ),
            description="Hazard interactions schedule a delayed consequence.",
            priority=80,
        ),
    ]
    return sorted(rules, key=lambda r: (r.priority, r.rule_id))


def attach_rule_set(world: WorldState) -> WorldState:
    """Attach and validate the deterministic latent rules for a world."""
    rules = generate_rule_set(world)
    errors = validate_rule_set(rules, world)
    if errors:
        raise ValueError(f"Invalid latent rule set: {errors}")
    world.rules = rules
    return world


def validate_rule_set(rules: list[LatentRule], world: WorldState | None = None) -> list[str]:
    """Return validation errors for contradictory or malformed latent rules."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    effect_by_preconditions: dict[tuple, tuple[str, ...]] = {}

    for rule in rules:
        if not rule.rule_id:
            errors.append("rule_id must be non-empty")
        if rule.rule_id in seen_ids:
            errors.append(f"duplicate rule_id {rule.rule_id!r}")
        seen_ids.add(rule.rule_id)
        if not rule.effects:
            errors.append(f"{rule.rule_id}: effects must be non-empty")

        key = (
            rule.trigger.value,
            rule.visible_preconditions,
            rule.hidden_preconditions,
        )
        effect_signature = tuple(repr(effect) for effect in rule.effects)
        previous = effect_by_preconditions.get(key)
        if previous is not None and previous != effect_signature:
            errors.append(f"contradictory effects for trigger/preconditions on {rule.rule_id!r}")
        effect_by_preconditions[key] = effect_signature

    return errors


def evaluate_action_rules(
    world: WorldState,
    action: Action,
) -> tuple[list[str], list[CausalProvenance]]:
    """Apply latent rules for an action, mutating world state when a rule fires."""
    if not world.rules:
        attach_rule_set(world)

    if action.kind == ActionKind.APPLY:
        return _evaluate_apply(world, action)
    if action.kind == ActionKind.COMBINE:
        return _evaluate_combine(world, action)
    return ["no_effect"], []


def resolve_pending_effects(world: WorldState) -> tuple[list[str], list[CausalProvenance]]:
    """Resolve delayed effects due at the world's current step."""
    if not world.pending_effects:
        return [], []

    remaining: list[PendingEffect] = []
    public_effects: list[str] = []
    provenance: list[CausalProvenance] = []

    for pending in world.pending_effects:
        if pending.due_step > world.step:
            remaining.append(pending)
            continue

        effects = _apply_effect(world, pending.effect, pending.participants)
        public_effects.extend(effects)
        provenance.append(
            CausalProvenance(
                rule_id=pending.rule_id,
                trigger=RuleTrigger.DELAYED,
                participants=dict(pending.participants),
                hidden_preconditions=("scheduled delayed consequence",),
                public_effects=effects,
            )
        )

    world.pending_effects = remaining
    return public_effects, provenance


def _evaluate_apply(world: WorldState, action: Action) -> tuple[list[str], list[CausalProvenance]]:
    actor_id = action.args.get("tool_id")
    target_id = action.args.get("target_id")
    if actor_id not in world.objects or target_id not in world.objects:
        return ["no_effect"], []

    actor = world.objects[actor_id]
    target = world.objects[target_id]
    public_effects: list[str] = []
    provenance: list[CausalProvenance] = []

    hazard_effects, hazard_provenance = _maybe_schedule_hazard(world, actor, target)
    public_effects.extend(hazard_effects)
    provenance.extend(hazard_provenance)

    if _is_affinity_blocked(actor, target):
        effects = [f"gate_resisted {target.obj_id}"]
        provenance.append(
            _provenance(
                "affinity_gate_modifier",
                RuleTrigger.APPLY,
                actor,
                target,
                effects,
                blocked_by="affinity mismatch",
            )
        )
        return public_effects + effects, provenance

    if _can_open_by_conductivity(actor, target):
        effects = _open_gate(target)
        provenance.append(
            _provenance("conductivity_gate_rule", RuleTrigger.APPLY, actor, target, effects)
        )
        return public_effects + effects, provenance

    if _can_open_by_magnetism(actor, target):
        effects = _open_gate(target)
        provenance.append(
            _provenance("magnetism_gate_rule", RuleTrigger.APPLY, actor, target, effects)
        )
        return public_effects + effects, provenance

    if _can_activate_key(actor, target):
        target.hidden.charge = True
        target.visible.state = ObjectState.ACTIVE
        target.visible.marker = "charged"
        effects = [f"activated {target.obj_id}"]
        provenance.append(
            _provenance("source_activation_rule", RuleTrigger.APPLY, actor, target, effects)
        )
        return public_effects + effects, provenance

    if _can_transform_block(actor, target):
        target.visible.state = ObjectState.DESTROYED
        target.visible.marker = "transformed"
        effects = [f"transformed {target.obj_id}"]
        provenance.append(
            _provenance("solubility_transform_rule", RuleTrigger.APPLY, actor, target, effects)
        )
        return public_effects + effects, provenance

    if actor.visible.obj_type == ObjectType.DETECTOR:
        target.visible.marker = _signal_marker(target)
        effects = [f"marked {target.obj_id} {target.visible.marker}"]
        provenance.append(
            _provenance("property_reveal_rule", RuleTrigger.APPLY, actor, target, effects)
        )
        return public_effects + effects, provenance

    return (public_effects or ["no_effect"]), provenance


def _evaluate_combine(
    world: WorldState,
    action: Action,
) -> tuple[list[str], list[CausalProvenance]]:
    id_a = action.args.get("obj_id_a")
    id_b = action.args.get("obj_id_b")
    if id_a not in world.objects or id_b not in world.objects:
        return ["no_effect"], []

    obj_a = world.objects[id_a]
    obj_b = world.objects[id_b]
    hazard_effects, hazard_provenance = _maybe_schedule_hazard(world, obj_a, obj_b)

    source, catalyst = _source_and_catalyst(obj_a, obj_b)
    if source is None or catalyst is None or source.visible.state != ObjectState.ACTIVE:
        return (hazard_effects or ["no_effect"]), hazard_provenance

    token_id = _produced_token_id(world, source, catalyst)
    if token_id not in world.objects:
        world.objects[token_id] = CrucibleObject(
            obj_id=token_id,
            visible=VisibleObjectState(
                obj_type=ObjectType.TOKEN,
                color=ObjectColor.YELLOW,
                shape=ObjectShape.SPHERE,
                texture=ObjectTexture.SMOOTH,
                size=ObjectSize.SMALL,
                marker="produced",
                pos=world.agent.pos,
                state=ObjectState.DEFAULT,
            ),
            hidden=HiddenObjectProps(
                conductivity=False,
                solubility=False,
                magnetism=False,
                charge=False,
                fragility=False,
                affinity=None,
            ),
        )
        _refresh_relations(world)

    effects = [f"produced {token_id}"]
    provenance = [
        _provenance("source_production_rule", RuleTrigger.COMBINE, source, catalyst, effects)
    ]
    return hazard_effects + effects, hazard_provenance + provenance


def _maybe_schedule_hazard(
    world: WorldState,
    actor: CrucibleObject,
    target: CrucibleObject,
) -> tuple[list[str], list[CausalProvenance]]:
    if actor.visible.obj_type != ObjectType.HAZARD and target.visible.obj_type != ObjectType.HAZARD:
        return [], []

    hazard = actor if actor.visible.obj_type == ObjectType.HAZARD else target
    damaged = target if actor.visible.obj_type == ObjectType.HAZARD else actor
    due_step = world.step + 2
    participants = {"hazard": hazard.obj_id, "target": damaged.obj_id}
    scheduled = [
        PendingEffect(
            rule_id="hazard_consequence_rule",
            due_step=due_step,
            effect=RuleEffect(kind=EffectKind.DRAIN_ENERGY, energy_delta=-10),
            participants=participants,
        ),
        PendingEffect(
            rule_id="hazard_consequence_rule",
            due_step=due_step,
            effect=RuleEffect(
                kind=EffectKind.DAMAGE_OBJECT,
                target_id=damaged.obj_id,
                target_state=ObjectState.DESTROYED,
            ),
            participants=participants,
        ),
    ]
    world.pending_effects.extend(scheduled)
    effects = [f"hazard_triggered {hazard.obj_id}"]
    return effects, [
        CausalProvenance(
            rule_id="hazard_consequence_rule",
            trigger=RuleTrigger.APPLY,
            participants=participants,
            hidden_preconditions=("hazard interaction occurred",),
            public_effects=effects,
            scheduled_effects=scheduled,
        )
    ]


def _apply_effect(
    world: WorldState,
    effect: RuleEffect,
    participants: dict[str, str],
) -> list[str]:
    if effect.kind == EffectKind.DRAIN_ENERGY:
        world.agent.energy += effect.energy_delta
        return [f"energy_changed {effect.energy_delta}"]

    if effect.kind == EffectKind.DAMAGE_OBJECT:
        target_id = effect.target_id or participants.get("target")
        if target_id in world.objects:
            target = world.objects[target_id]
            target.visible.state = effect.target_state or ObjectState.DESTROYED
            return [f"damaged {target_id}"]
    return ["no_effect"]


def _is_affinity_blocked(actor: CrucibleObject, target: CrucibleObject) -> bool:
    return (
        target.visible.obj_type == ObjectType.GATE
        and target.visible.state == ObjectState.CLOSED
        and actor.visible.obj_type in (ObjectType.TOOL, ObjectType.KEY)
        and actor.hidden.conductivity
        and target.hidden.affinity is not None
        and actor.hidden.affinity != target.hidden.affinity
    )


def _can_open_by_conductivity(actor: CrucibleObject, target: CrucibleObject) -> bool:
    return (
        target.visible.obj_type == ObjectType.GATE
        and target.visible.state == ObjectState.CLOSED
        and actor.visible.obj_type in (ObjectType.TOOL, ObjectType.KEY)
        and actor.hidden.conductivity
        and (target.hidden.affinity is None or actor.hidden.affinity == target.hidden.affinity)
    )


def _can_open_by_magnetism(actor: CrucibleObject, target: CrucibleObject) -> bool:
    return (
        target.visible.obj_type == ObjectType.GATE
        and target.visible.state == ObjectState.CLOSED
        and actor.visible.obj_type == ObjectType.KEY
        and actor.hidden.magnetism
        and actor.hidden.charge
    )


def _can_activate_key(actor: CrucibleObject, target: CrucibleObject) -> bool:
    return (
        actor.visible.obj_type == ObjectType.SOURCE
        and actor.visible.state == ObjectState.ACTIVE
        and target.visible.obj_type == ObjectType.KEY
        and target.hidden.magnetism
    )


def _can_transform_block(actor: CrucibleObject, target: CrucibleObject) -> bool:
    return (
        actor.visible.obj_type == ObjectType.SOURCE
        and actor.visible.state == ObjectState.ACTIVE
        and target.visible.obj_type == ObjectType.BLOCK
        and target.hidden.solubility
    )


def _open_gate(gate: CrucibleObject) -> list[str]:
    gate.visible.state = ObjectState.OPEN
    return [f"opened {gate.obj_id}"]


def _signal_marker(target: CrucibleObject) -> str:
    hidden = target.hidden
    signal = (
        hidden.conductivity
        or hidden.solubility
        or hidden.magnetism
        or hidden.charge
        or hidden.fragility
        or hidden.affinity is not None
    )
    return "signal-positive" if signal else "signal-negative"


def _source_and_catalyst(
    obj_a: CrucibleObject,
    obj_b: CrucibleObject,
) -> tuple[CrucibleObject | None, CrucibleObject | None]:
    if (
        obj_a.visible.obj_type == ObjectType.SOURCE
        and obj_b.visible.obj_type == ObjectType.CATALYST
    ):
        return obj_a, obj_b
    if (
        obj_b.visible.obj_type == ObjectType.SOURCE
        and obj_a.visible.obj_type == ObjectType.CATALYST
    ):
        return obj_b, obj_a
    return None, None


def _produced_token_id(
    world: WorldState,
    source: CrucibleObject,
    catalyst: CrucibleObject,
) -> str:
    return f"{world.seed}_token_{source.obj_id}_{catalyst.obj_id}"


def _provenance(
    rule_id: str,
    trigger: RuleTrigger,
    actor: CrucibleObject,
    target: CrucibleObject,
    effects: list[str],
    *,
    blocked_by: str | None = None,
) -> CausalProvenance:
    rule = _rule_by_id(rule_id)
    return CausalProvenance(
        rule_id=rule_id,
        trigger=trigger,
        participants={"actor": actor.obj_id, "target": target.obj_id},
        hidden_preconditions=rule.hidden_preconditions if rule else (),
        public_effects=effects,
        blocked_by=blocked_by,
    )


def _rule_by_id(rule_id: str) -> LatentRule | None:
    for rule in _rule_templates():
        if rule.rule_id == rule_id:
            return rule
    return None


def _refresh_relations(world: WorldState) -> None:
    world.relations = derive_relations(world.objects, world.agent.pos)
    for obj_id in world.agent.inventory:
        world.relations.append(Relation(kind=RelationKind.HELD, subject="agent", object_=obj_id))
