from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from crucible.grammar import TaskFamily, TaskSpec, build_world_from_spec
from crucible.objects import ObjectColor, ObjectType
from crucible.relations import Relation, RelationKind
from crucible.rules import attach_rule_set
from crucible.utils.seeding import make_python_rng
from crucible.utils.serialization import to_dict
from crucible.world import WorldState


class CounterfactualDeltaKind(str, Enum):
    LATENT_PROPERTY = "latent_property"
    VISIBLE_FEATURE = "visible_feature"
    RELATION = "relation"
    RULE_MODIFIER = "rule_modifier"


class PerturbationKind(str, Enum):
    """Targeted modifications to a TaskSpec that attack a specific surface shortcut.

    Each perturbation preserves hidden causal solvability while changing one
    visible feature an agent might rely on instead of probing hidden properties.
    """

    COLOR_AFFORDANCE_DECORRELATION = "color_affordance_decorrelation"
    OBJECT_ID_RESAMPLING = "object_id_resampling"
    LAYOUT_PERMUTATION = "layout_permutation"
    DISTRACTOR_INJECTION = "distractor_injection"
    TOOL_APPEARANCE_SWAP = "tool_appearance_swap"
    DELAYED_CONSEQUENCE = "delayed_consequence"
    FALSE_HINT_TEXT = "false_hint_text"
    RESOURCE_SCARCITY = "resource_scarcity"


@dataclass
class CounterfactualDelta:
    kind: CounterfactualDeltaKind
    object_id: str | None
    field_path: str
    before_value: Any
    after_value: Any
    public_visibility: bool


@dataclass
class CounterfactualPair:
    factual: WorldState
    counterfactual: WorldState
    delta: CounterfactualDelta
    factual_hash: str
    counterfactual_hash: str
    factual_public_hash: str
    counterfactual_public_hash: str


@dataclass
class PredictionQuery:
    query_id: str
    world_hash: str
    action: dict
    mode: str = "effect"


@dataclass
class PredictionScore:
    query_id: str
    predicted_effect: Any
    actual_effect: Any
    correct: bool


def stable_state_hash(world: WorldState, *, public: bool = False) -> str:
    """Return a deterministic hash for full evaluator state or public state."""
    payload = json.dumps(
        to_dict(world, public=public),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_counterfactual_pair(
    spec_or_world: TaskSpec | WorldState,
    delta_kind: CounterfactualDeltaKind | str | None = None,
) -> CounterfactualPair:
    """Clone a factual world and apply exactly one declared counterfactual delta."""
    kind = CounterfactualDeltaKind(delta_kind or CounterfactualDeltaKind.LATENT_PROPERTY)
    factual, family = _materialize_world(spec_or_world)
    if not factual.rules:
        attach_rule_set(factual)
    counterfactual = copy.deepcopy(factual)

    if isinstance(spec_or_world, TaskSpec):
        delta = _apply_task_delta(counterfactual, spec_or_world, family, kind)
    else:
        delta = _apply_generic_delta(counterfactual, kind)

    return CounterfactualPair(
        factual=factual,
        counterfactual=counterfactual,
        delta=delta,
        factual_hash=stable_state_hash(factual),
        counterfactual_hash=stable_state_hash(counterfactual),
        factual_public_hash=stable_state_hash(factual, public=True),
        counterfactual_public_hash=stable_state_hash(counterfactual, public=True),
    )


def score_prediction(
    query: PredictionQuery,
    predicted_effect: Any,
    actual_effect: Any,
) -> PredictionScore:
    return PredictionScore(
        query_id=query.query_id,
        predicted_effect=predicted_effect,
        actual_effect=actual_effect,
        correct=_canonical(predicted_effect) == _canonical(actual_effect),
    )


def changed_state_paths(
    world_a: WorldState,
    world_b: WorldState,
    *,
    public: bool = False,
) -> list[str]:
    """Return leaf paths that differ between two world serializations."""
    paths: list[str] = []
    _collect_diffs(to_dict(world_a, public=public), to_dict(world_b, public=public), "", paths)
    return paths


def _materialize_world(
    spec_or_world: TaskSpec | WorldState,
) -> tuple[WorldState, TaskFamily | None]:
    if isinstance(spec_or_world, TaskSpec):
        return build_world_from_spec(spec_or_world), spec_or_world.family
    return copy.deepcopy(spec_or_world), None


def _apply_task_delta(
    world: WorldState,
    spec: TaskSpec,
    family: TaskFamily | None,
    kind: CounterfactualDeltaKind,
) -> CounterfactualDelta:
    if kind == CounterfactualDeltaKind.VISIBLE_FEATURE:
        return _apply_visible_tool_delta(world, spec)
    if kind == CounterfactualDeltaKind.RELATION:
        return _apply_relation_delta(world)
    if kind == CounterfactualDeltaKind.RULE_MODIFIER:
        return _apply_rule_modifier_delta(world)

    if family == TaskFamily.AFFORDANCE:
        obj_id = _role_obj_id(spec, "correct_tool") or _first_type(world, ObjectType.TOOL)
        return _flip_hidden_bool(world, obj_id, "conductivity")
    if family == TaskFamily.CAUSAL_GATE:
        obj_id = _role_obj_id(spec, "key") or _first_type(world, ObjectType.KEY)
        return _flip_hidden_bool(world, obj_id, "magnetism")
    if family == TaskFamily.COUNTERFACTUAL:
        obj_id = spec.goal.classify_correct_obj_id or _first_type(world, ObjectType.BLOCK)
        return _flip_hidden_bool(world, obj_id, "solubility")
    if family == TaskFamily.TOOL_SUBSTITUTION:
        obj_id = _role_obj_id(spec, "correct_tool") or _first_type(world, ObjectType.TOOL)
        return _flip_hidden_bool(world, obj_id, "conductivity")
    if family == TaskFamily.CONTRADICTION:
        obj_id = _role_obj_id(spec, "gate") or _first_type(world, ObjectType.GATE)
        return _toggle_affinity(world, obj_id)

    return _apply_generic_delta(world, kind)


def _apply_generic_delta(
    world: WorldState,
    kind: CounterfactualDeltaKind,
) -> CounterfactualDelta:
    if kind == CounterfactualDeltaKind.VISIBLE_FEATURE:
        return _change_visible_color(world, next(iter(world.objects)))
    if kind == CounterfactualDeltaKind.RELATION:
        return _apply_relation_delta(world)
    if kind == CounterfactualDeltaKind.RULE_MODIFIER:
        return _apply_rule_modifier_delta(world)

    obj_id = next(iter(world.objects))
    return _flip_hidden_bool(world, obj_id, "conductivity")


def _flip_hidden_bool(world: WorldState, obj_id: str, field_name: str) -> CounterfactualDelta:
    obj = world.objects[obj_id]
    before = getattr(obj.hidden, field_name)
    after = not before
    setattr(obj.hidden, field_name, after)
    return CounterfactualDelta(
        kind=CounterfactualDeltaKind.LATENT_PROPERTY,
        object_id=obj_id,
        field_path=f"objects.{obj_id}.hidden.{field_name}",
        before_value=before,
        after_value=after,
        public_visibility=False,
    )


def _toggle_affinity(world: WorldState, obj_id: str) -> CounterfactualDelta:
    obj = world.objects[obj_id]
    before = obj.hidden.affinity
    after = None if before is not None else "organic"
    obj.hidden.affinity = after
    return CounterfactualDelta(
        kind=CounterfactualDeltaKind.LATENT_PROPERTY,
        object_id=obj_id,
        field_path=f"objects.{obj_id}.hidden.affinity",
        before_value=before,
        after_value=after,
        public_visibility=False,
    )


def _apply_visible_tool_delta(world: WorldState, spec: TaskSpec) -> CounterfactualDelta:
    obj_id = _role_obj_id(spec, "correct_tool") or _first_type(world, ObjectType.TOOL)
    return _change_visible_color(world, obj_id)


def _change_visible_color(world: WorldState, obj_id: str) -> CounterfactualDelta:
    obj = world.objects[obj_id]
    before = obj.visible.color
    after = ObjectColor.YELLOW if before != ObjectColor.YELLOW else ObjectColor.GREY
    obj.visible.color = after
    return CounterfactualDelta(
        kind=CounterfactualDeltaKind.VISIBLE_FEATURE,
        object_id=obj_id,
        field_path=f"objects.{obj_id}.visible.color",
        before_value=before.value,
        after_value=after.value,
        public_visibility=True,
    )


def _apply_relation_delta(world: WorldState) -> CounterfactualDelta:
    if world.relations:
        before = world.relations.pop(0)
        return CounterfactualDelta(
            kind=CounterfactualDeltaKind.RELATION,
            object_id=before.subject,
            field_path="relations[0]",
            before_value=to_dict(before, public=False),
            after_value=None,
            public_visibility=True,
        )

    obj_id = next(iter(world.objects))
    relation = Relation(kind=RelationKind.ADJACENT, subject="agent", object_=obj_id)
    world.relations.append(relation)
    return CounterfactualDelta(
        kind=CounterfactualDeltaKind.RELATION,
        object_id=obj_id,
        field_path="relations[append]",
        before_value=None,
        after_value=to_dict(relation, public=False),
        public_visibility=True,
    )


def _apply_rule_modifier_delta(world: WorldState) -> CounterfactualDelta:
    if not world.rules:
        attach_rule_set(world)
    rule = world.rules[0]
    before = rule.hidden_preconditions
    after = before + ("counterfactual_modifier",)
    rule.hidden_preconditions = after
    return CounterfactualDelta(
        kind=CounterfactualDeltaKind.RULE_MODIFIER,
        object_id=None,
        field_path=f"rules.{rule.rule_id}.hidden_preconditions",
        before_value=before,
        after_value=after,
        public_visibility=False,
    )


def _role_obj_id(spec: TaskSpec, role: str) -> str | None:
    for obj_spec in spec.object_specs:
        if obj_spec.role == role:
            return obj_spec.obj_id
    return None


def _first_type(world: WorldState, obj_type: ObjectType) -> str:
    for obj_id, obj in world.objects.items():
        if obj.visible.obj_type == obj_type:
            return obj_id
    return next(iter(world.objects))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _collect_diffs(value_a: Any, value_b: Any, path: str, paths: list[str]) -> None:
    if isinstance(value_a, dict) and isinstance(value_b, dict):
        for key in sorted(set(value_a) | set(value_b)):
            child = f"{path}.{key}" if path else str(key)
            if key not in value_a or key not in value_b:
                paths.append(child)
            else:
                _collect_diffs(value_a[key], value_b[key], child, paths)
        return

    if isinstance(value_a, (list, tuple)) and isinstance(value_b, (list, tuple)):
        max_len = max(len(value_a), len(value_b))
        for index in range(max_len):
            child = f"{path}[{index}]"
            if index >= len(value_a) or index >= len(value_b):
                paths.append(child)
            else:
                _collect_diffs(value_a[index], value_b[index], child, paths)
        return

    if value_a != value_b:
        paths.append(path)


# ---------------------------------------------------------------------------
# Perturbation suite
# ---------------------------------------------------------------------------


def apply_perturbation(
    spec: TaskSpec,
    kind: PerturbationKind | str,
    seed: int = 0,
) -> TaskSpec:
    """Return a copy of spec with exactly one perturbation applied.

    Each perturbation attacks a distinct surface shortcut while leaving the
    hidden causal solution intact.  The returned spec passes validate_task().
    """
    kind = PerturbationKind(kind)
    dispatch = {
        PerturbationKind.COLOR_AFFORDANCE_DECORRELATION: _perturb_color_decorrelation,
        PerturbationKind.OBJECT_ID_RESAMPLING: _perturb_object_id_resampling,
        PerturbationKind.LAYOUT_PERMUTATION: _perturb_layout_permutation,
        PerturbationKind.DISTRACTOR_INJECTION: _perturb_distractor_injection,
        PerturbationKind.TOOL_APPEARANCE_SWAP: _perturb_tool_appearance_swap,
        PerturbationKind.DELAYED_CONSEQUENCE: _perturb_delayed_consequence,
        PerturbationKind.FALSE_HINT_TEXT: _perturb_false_hint_text,
        PerturbationKind.RESOURCE_SCARCITY: _perturb_resource_scarcity,
    }
    return dispatch[kind](spec, seed)


# --- helpers ----------------------------------------------------------------


def _spec_copy(spec: TaskSpec) -> TaskSpec:
    """Deep copy a TaskSpec without importing dataclasses at call time."""
    return copy.deepcopy(spec)


def _find_role(spec: TaskSpec, role_prefix: str):
    """Return first ObjectSpec whose role starts with role_prefix, or None."""
    for o in spec.object_specs:
        if o.role.startswith(role_prefix):
            return o
    return None


def _perturb_color_decorrelation(spec: TaskSpec, seed: int) -> TaskSpec:
    """Randomise tool/key colours while preserving hidden conductivity assignments.

    Attacks colour-cue identification (e.g., 'pick the red one').
    """
    s = _spec_copy(spec)
    rng = make_python_rng(seed)
    colors = list(ObjectColor)
    for o in s.object_specs:
        if o.obj_type in (ObjectType.TOOL, ObjectType.KEY):
            o.color = rng.choice(colors)
    s.task_id = spec.task_id + "_pcd"
    return s


def _perturb_object_id_resampling(spec: TaskSpec, seed: int) -> TaskSpec:
    """Rename every object ID with a seed-derived suffix.

    Attacks memorised object-ID lookup tables (e.g., memorization agent).
    Updates goal, solution_certificate, and all ObjectSpec.obj_id fields.
    """
    s = _spec_copy(spec)
    suffix = f"_r{seed % 10_000}"
    mapping = {o.obj_id: o.obj_id + suffix for o in s.object_specs}

    for o in s.object_specs:
        o.obj_id = mapping[o.obj_id]

    if s.goal.target_obj_id and s.goal.target_obj_id in mapping:
        s.goal.target_obj_id = mapping[s.goal.target_obj_id]
    if s.goal.classify_correct_obj_id and s.goal.classify_correct_obj_id in mapping:
        s.goal.classify_correct_obj_id = mapping[s.goal.classify_correct_obj_id]

    for action in s.solution_certificate.action_sequence:
        args = action.get("args", {})
        for key in ("obj_id", "tool_id", "target_id", "obj_id_a", "obj_id_b"):
            if key in args and args[key] in mapping:
                args[key] = mapping[args[key]]

    s.task_id = spec.task_id + "_pir"
    return s


def _perturb_layout_permutation(spec: TaskSpec, seed: int) -> TaskSpec:
    """Change the position seed so objects appear at different grid positions.

    Attacks memorised spatial layouts.  build_world_from_spec reads
    metadata['position_seed'] and uses it instead of spec.seed for placement.
    """
    s = _spec_copy(spec)
    s.metadata = dict(s.metadata)
    s.metadata["position_seed"] = seed + 7_000_000
    s.task_id = spec.task_id + "_plp"
    return s


def _perturb_distractor_injection(spec: TaskSpec, seed: int) -> TaskSpec:
    """Add a non-causal object that looks like the correct tool/key.

    The injected distractor shares visible features with the correct tool but
    has conductivity=False.  Attacks 'pick the red one' heuristics.
    """
    from crucible.grammar import ObjectSpec

    s = _spec_copy(spec)
    template = _find_role(s, "correct_tool") or _find_role(s, "correct_key")
    if template is None:
        template = next(
            (o for o in s.object_specs if o.obj_type in (ObjectType.TOOL, ObjectType.KEY)),
            s.object_specs[0],
        )

    distractor = ObjectSpec(
        role="distractor",
        obj_id=f"{spec.task_id}_dist{seed % 1000}",
        obj_type=template.obj_type,
        color=template.color,
        shape=template.shape,
        texture=template.texture,
        size=template.size,
        marker=None,
        conductivity=False,
        solubility=False,
        magnetism=False,
        charge=False,
        fragility=False,
        affinity=None,
    )
    s.object_specs.append(distractor)
    s.task_id = spec.task_id + "_pdi"
    return s


def _perturb_tool_appearance_swap(spec: TaskSpec, seed: int) -> TaskSpec:
    """Swap (color, shape, texture) between correct_tool and first decoy.

    Hidden conductivity assignments are unchanged.  Attacks shape/colour-based
    identification without causal testing.
    """
    s = _spec_copy(spec)
    correct = _find_role(s, "correct_tool") or _find_role(s, "correct_key")
    decoy = _find_role(s, "decoy")
    if correct is None or decoy is None:
        return s
    correct.color, decoy.color = decoy.color, correct.color
    correct.shape, decoy.shape = decoy.shape, correct.shape
    correct.texture, decoy.texture = decoy.texture, correct.texture
    s.task_id = spec.task_id + "_ptas"
    return s


def _perturb_delayed_consequence(spec: TaskSpec, seed: int) -> TaskSpec:
    """Flag that causal effects are delayed by two steps.

    Sets metadata['pending_effect_steps'] = 2.  Full rule-engine enforcement
    of the delay is deferred; this flag allows agents to detect and adapt.
    Attacks agents that assume effects are immediate.
    """
    s = _spec_copy(spec)
    s.metadata = dict(s.metadata)
    s.metadata["pending_effect_steps"] = 2
    s.task_id = spec.task_id + "_pdc"
    return s


def _perturb_false_hint_text(spec: TaskSpec, seed: int) -> TaskSpec:
    """Add misleading 'conductive' marker to non-conductive objects.

    The correct tool/key gets marker=None; decoys get marker='conductive'.
    Hidden conductivity is unchanged.  Attacks agents that read visible markers
    instead of probing hidden properties through intervention.
    """
    s = _spec_copy(spec)
    for o in s.object_specs:
        if o.role.startswith("correct_tool") or o.role.startswith("correct_key"):
            o.marker = None
        elif o.obj_type in (ObjectType.TOOL, ObjectType.KEY):
            o.marker = "conductive"
    s.task_id = spec.task_id + "_pfh"
    return s


def _perturb_resource_scarcity(spec: TaskSpec, seed: int) -> TaskSpec:
    """Halve the energy budget (floor 20) to penalise wasteful exploration.

    The oracle solution always fits within the reduced budget (oracle ≤ 4 steps).
    Attacks random-walk agents that deplete energy before reaching the goal.
    """
    s = _spec_copy(spec)
    s.constraints.energy_budget = max(20, s.constraints.energy_budget // 2)
    s.task_id = spec.task_id + "_prs"
    return s
