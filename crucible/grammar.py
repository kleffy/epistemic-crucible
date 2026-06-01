from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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
from crucible.splits import SplitLabel, assign_split
from crucible.utils.seeding import make_rng
from crucible.world import AgentState, WorldState, derive_relations, sample_cell

# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------


class TaskFamily(str, Enum):
    AFFORDANCE = "affordance"
    CAUSAL_GATE = "causal_gate"
    COUNTERFACTUAL = "counterfactual"
    TOOL_SUBSTITUTION = "tool_substitution"
    CONTRADICTION = "contradiction"


class GoalKind(str, Enum):
    OPEN = "open"
    RETRIEVE = "retrieve"
    REACH = "reach"
    CLASSIFY = "classify"
    TRANSFORM = "transform"


@dataclass
class GoalSpec:
    kind: GoalKind
    target_obj_id: str | None = None
    target_type: str | None = None
    target_pos: tuple[int, int] | None = None
    classify_property: str | None = None
    classify_correct_obj_id: str | None = None


@dataclass
class ConstraintSpec:
    max_interventions: int
    max_steps: int
    energy_budget: int = 100


@dataclass
class ObjectSpec:
    """Full blueprint for one object including its hidden causal properties."""

    role: str
    obj_id: str
    obj_type: ObjectType
    color: ObjectColor
    shape: ObjectShape
    texture: ObjectTexture
    size: ObjectSize
    marker: str | None = None
    state: ObjectState = ObjectState.DEFAULT
    conductivity: bool = False
    solubility: bool = False
    magnetism: bool = False
    charge: bool = False
    fragility: bool = False
    affinity: str | None = None


@dataclass
class SolutionCertificate:
    description: str
    action_sequence: list[dict]
    oracle_rules_required: list[str]


@dataclass
class TaskSpec:
    task_id: str
    family: TaskFamily
    seed: int
    grid_size: int
    max_steps: int
    object_specs: list[ObjectSpec]
    goal: GoalSpec
    constraints: ConstraintSpec
    split: SplitLabel
    pressure_labels: list[str]
    solution_certificate: SolutionCertificate
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Family A: Affordance Discovery
# ---------------------------------------------------------------------------


def _generate_affordance(seed: int, split: SplitLabel) -> TaskSpec:
    task_id = f"aff_{seed}_{split.value}"
    gate_id = f"{task_id}_gate"
    tool_ids = [f"{task_id}_tool{i}" for i in range(3)]

    if split == SplitLabel.TRAIN:
        # RED reliably correlates with conductivity in training worlds.
        tool_colors = [ObjectColor.RED, ObjectColor.BLUE, ObjectColor.GREEN]
        correct_idx = 0
    else:
        # Decorrelate colour from conductivity in test worlds.
        rng = make_rng(seed + 2_000_000)
        correct_idx = int(rng.integers(3))
        color_pool = list(ObjectColor)
        rng.shuffle(color_pool)
        tool_colors = list(color_pool[:3])

    tool_shapes = [ObjectShape.ROD, ObjectShape.CUBE, ObjectShape.SPHERE]
    tool_textures = [ObjectTexture.SMOOTH, ObjectTexture.ROUGH, ObjectTexture.DOTTED]

    object_specs = [
        ObjectSpec(
            role="correct_tool" if i == correct_idx else f"decoy_tool{i}",
            obj_id=tool_ids[i],
            obj_type=ObjectType.TOOL,
            color=tool_colors[i],
            shape=tool_shapes[i],
            texture=tool_textures[i],
            size=ObjectSize.SMALL,
            conductivity=(i == correct_idx),
        )
        for i in range(3)
    ] + [
        ObjectSpec(
            role="gate",
            obj_id=gate_id,
            obj_type=ObjectType.GATE,
            color=ObjectColor.GREY,
            shape=ObjectShape.FLAT,
            texture=ObjectTexture.SMOOTH,
            size=ObjectSize.LARGE,
            state=ObjectState.CLOSED,
        ),
    ]

    correct_tool_id = tool_ids[correct_idx]
    return TaskSpec(
        task_id=task_id,
        family=TaskFamily.AFFORDANCE,
        seed=seed,
        grid_size=6,
        max_steps=30,
        object_specs=object_specs,
        goal=GoalSpec(kind=GoalKind.OPEN, target_obj_id=gate_id),
        constraints=ConstraintSpec(max_interventions=3, max_steps=30),
        split=split,
        pressure_labels=["affordance", "causal", "deception"],
        solution_certificate=SolutionCertificate(
            description="Pick up the conductive tool and apply it to the gate.",
            action_sequence=[
                {"kind": "pickup", "args": {"obj_id": correct_tool_id}},
                {"kind": "apply", "args": {"tool_id": correct_tool_id, "target_id": gate_id}},
            ],
            oracle_rules_required=["conductivity_gate_rule"],
        ),
        metadata={
            "train_correlation": "red=conductive"
            if split == SplitLabel.TRAIN
            else "decorrelated",
        },
    )


# ---------------------------------------------------------------------------
# Family B: Causal Gate Opening
# ---------------------------------------------------------------------------


def _generate_causal_gate(seed: int, split: SplitLabel) -> TaskSpec:
    task_id = f"cgate_{seed}_{split.value}"
    key_id = f"{task_id}_key"
    source_id = f"{task_id}_source"
    gate_id = f"{task_id}_gate"

    object_specs = [
        ObjectSpec(
            role="key",
            obj_id=key_id,
            obj_type=ObjectType.KEY,
            color=ObjectColor.RED,
            shape=ObjectShape.ROD,
            texture=ObjectTexture.SMOOTH,
            size=ObjectSize.SMALL,
            magnetism=True,
        ),
        ObjectSpec(
            role="source",
            obj_id=source_id,
            obj_type=ObjectType.SOURCE,
            color=ObjectColor.BLUE,
            shape=ObjectShape.FLAT,
            texture=ObjectTexture.SMOOTH,
            size=ObjectSize.MEDIUM,
            state=ObjectState.ACTIVE,
        ),
        ObjectSpec(
            role="gate",
            obj_id=gate_id,
            obj_type=ObjectType.GATE,
            color=ObjectColor.GREY,
            shape=ObjectShape.FLAT,
            texture=ObjectTexture.ROUGH,
            size=ObjectSize.LARGE,
            state=ObjectState.CLOSED,
        ),
    ]

    return TaskSpec(
        task_id=task_id,
        family=TaskFamily.CAUSAL_GATE,
        seed=seed,
        grid_size=6,
        max_steps=40,
        object_specs=object_specs,
        goal=GoalSpec(kind=GoalKind.OPEN, target_obj_id=gate_id),
        constraints=ConstraintSpec(max_interventions=4, max_steps=40),
        split=split,
        pressure_labels=["causal", "compositional"],
        solution_certificate=SolutionCertificate(
            description="Apply source to key to activate it, then apply key to gate.",
            action_sequence=[
                {"kind": "pickup", "args": {"obj_id": source_id}},
                {"kind": "apply", "args": {"tool_id": source_id, "target_id": key_id}},
                {"kind": "pickup", "args": {"obj_id": key_id}},
                {"kind": "apply", "args": {"tool_id": key_id, "target_id": gate_id}},
            ],
            oracle_rules_required=["magnetism_gate_rule", "source_activation_rule"],
        ),
        metadata={},
    )


# ---------------------------------------------------------------------------
# Family C: Counterfactual Classification
# ---------------------------------------------------------------------------


def _generate_counterfactual(seed: int, split: SplitLabel) -> TaskSpec:
    task_id = f"cf_{seed}_{split.value}"
    block0_id = f"{task_id}_block0"
    block1_id = f"{task_id}_block1"
    source_id = f"{task_id}_source"

    if split == SplitLabel.TRAIN:
        # block0 is soluble in training worlds
        b0_soluble, b1_soluble = True, False
        correct_id, apply_target = block0_id, block0_id
    else:
        # Counterfactual: property assignment is swapped in test worlds
        b0_soluble, b1_soluble = False, True
        correct_id, apply_target = block1_id, block1_id

    object_specs = [
        ObjectSpec(
            role="block_primary",
            obj_id=block0_id,
            obj_type=ObjectType.BLOCK,
            color=ObjectColor.RED,
            shape=ObjectShape.CUBE,
            texture=ObjectTexture.SMOOTH,
            size=ObjectSize.MEDIUM,
            solubility=b0_soluble,
            fragility=not b0_soluble,
        ),
        ObjectSpec(
            role="block_secondary",
            obj_id=block1_id,
            obj_type=ObjectType.BLOCK,
            color=ObjectColor.BLUE,
            shape=ObjectShape.CUBE,
            texture=ObjectTexture.ROUGH,
            size=ObjectSize.MEDIUM,
            solubility=b1_soluble,
            fragility=not b1_soluble,
        ),
        ObjectSpec(
            role="source",
            obj_id=source_id,
            obj_type=ObjectType.SOURCE,
            color=ObjectColor.GREY,
            shape=ObjectShape.FLAT,
            texture=ObjectTexture.SMOOTH,
            size=ObjectSize.SMALL,
            state=ObjectState.ACTIVE,
        ),
    ]

    return TaskSpec(
        task_id=task_id,
        family=TaskFamily.COUNTERFACTUAL,
        seed=seed,
        grid_size=6,
        max_steps=30,
        object_specs=object_specs,
        goal=GoalSpec(
            kind=GoalKind.CLASSIFY,
            classify_property="solubility",
            classify_correct_obj_id=correct_id,
        ),
        constraints=ConstraintSpec(max_interventions=2, max_steps=30),
        split=split,
        pressure_labels=["counterfactual", "causal", "scarcity"],
        solution_certificate=SolutionCertificate(
            description="Apply source to each block; the one that transforms is soluble.",
            action_sequence=[
                {"kind": "pickup", "args": {"obj_id": source_id}},
                {"kind": "apply", "args": {"tool_id": source_id, "target_id": apply_target}},
            ],
            oracle_rules_required=["solubility_transform_rule"],
        ),
        metadata={"correct_obj_id": correct_id},
    )


# ---------------------------------------------------------------------------
# Family D: Tool Substitution
# ---------------------------------------------------------------------------


def _generate_tool_substitution(seed: int, split: SplitLabel) -> TaskSpec:
    task_id = f"tsub_{seed}_{split.value}"
    gate_id = f"{task_id}_gate"
    decoy_id = f"{task_id}_decoy"

    if split == SplitLabel.TRAIN:
        correct_id = f"{task_id}_tool_std"
        correct_spec = ObjectSpec(
            role="correct_tool",
            obj_id=correct_id,
            obj_type=ObjectType.TOOL,
            color=ObjectColor.RED,
            shape=ObjectShape.ROD,
            texture=ObjectTexture.SMOOTH,
            size=ObjectSize.MEDIUM,
            conductivity=True,
        )
    else:
        # Novel appearance in test: same hidden affordance, different shape/colour
        correct_id = f"{task_id}_tool_novel"
        correct_spec = ObjectSpec(
            role="correct_tool",
            obj_id=correct_id,
            obj_type=ObjectType.TOOL,
            color=ObjectColor.YELLOW,
            shape=ObjectShape.FLAT,
            texture=ObjectTexture.DOTTED,
            size=ObjectSize.MEDIUM,
            conductivity=True,
        )

    object_specs = [
        correct_spec,
        ObjectSpec(
            role="decoy_tool",
            obj_id=decoy_id,
            obj_type=ObjectType.TOOL,
            color=ObjectColor.BLUE,
            shape=ObjectShape.SPHERE,
            texture=ObjectTexture.ROUGH,
            size=ObjectSize.SMALL,
            conductivity=False,
        ),
        ObjectSpec(
            role="gate",
            obj_id=gate_id,
            obj_type=ObjectType.GATE,
            color=ObjectColor.GREY,
            shape=ObjectShape.FLAT,
            texture=ObjectTexture.SMOOTH,
            size=ObjectSize.LARGE,
            state=ObjectState.CLOSED,
        ),
    ]

    return TaskSpec(
        task_id=task_id,
        family=TaskFamily.TOOL_SUBSTITUTION,
        seed=seed,
        grid_size=6,
        max_steps=30,
        object_specs=object_specs,
        goal=GoalSpec(kind=GoalKind.OPEN, target_obj_id=gate_id),
        constraints=ConstraintSpec(max_interventions=3, max_steps=30),
        split=split,
        pressure_labels=["affordance", "deception", "compositional"],
        solution_certificate=SolutionCertificate(
            description=(
                "Pick up the conductive tool (standard shape in train, "
                "novel shape in test) and apply it to the gate."
            ),
            action_sequence=[
                {"kind": "pickup", "args": {"obj_id": correct_id}},
                {"kind": "apply", "args": {"tool_id": correct_id, "target_id": gate_id}},
            ],
            oracle_rules_required=["conductivity_gate_rule"],
        ),
        metadata={"appearance": "standard" if split == SplitLabel.TRAIN else "novel"},
    )


# ---------------------------------------------------------------------------
# Family E: Rule Contradiction and Revision
# ---------------------------------------------------------------------------


def _generate_contradiction(seed: int, split: SplitLabel) -> TaskSpec:
    task_id = f"contra_{seed}_{split.value}"
    key_metal_id = f"{task_id}_key_metal"
    key_organic_id = f"{task_id}_key_organic"
    gate_id = f"{task_id}_gate"

    if split == SplitLabel.TRAIN:
        # No affinity restriction: any conductive key opens the gate
        gate_affinity = None
        correct_key_id = key_metal_id
        metal_role, organic_role = "correct_key", "alternative_key"
        description = "Any conductive key opens the gate."
        oracle_rules: list[str] = ["conductivity_gate_rule"]
    else:
        # Gate requires organic affinity — metal key fails despite conductivity
        gate_affinity = "organic"
        correct_key_id = key_organic_id
        metal_role, organic_role = "incorrect_key", "correct_key"
        description = (
            "Gate requires organic affinity; the metal key fails despite conductivity. "
            "Agent must detect the restriction and switch to key_organic."
        )
        oracle_rules = ["conductivity_gate_rule", "affinity_gate_modifier"]

    object_specs = [
        ObjectSpec(
            role=metal_role,
            obj_id=key_metal_id,
            obj_type=ObjectType.KEY,
            color=ObjectColor.RED,
            shape=ObjectShape.ROD,
            texture=ObjectTexture.SMOOTH,
            size=ObjectSize.SMALL,
            conductivity=True,
            affinity="metal",
        ),
        ObjectSpec(
            role=organic_role,
            obj_id=key_organic_id,
            obj_type=ObjectType.KEY,
            color=ObjectColor.BLUE,
            shape=ObjectShape.SPHERE,
            texture=ObjectTexture.ROUGH,
            size=ObjectSize.SMALL,
            conductivity=True,
            affinity="organic",
        ),
        ObjectSpec(
            role="gate",
            obj_id=gate_id,
            obj_type=ObjectType.GATE,
            color=ObjectColor.GREY,
            shape=ObjectShape.FLAT,
            texture=ObjectTexture.SMOOTH,
            size=ObjectSize.LARGE,
            state=ObjectState.CLOSED,
            affinity=gate_affinity,
        ),
    ]

    return TaskSpec(
        task_id=task_id,
        family=TaskFamily.CONTRADICTION,
        seed=seed,
        grid_size=6,
        max_steps=40,
        object_specs=object_specs,
        goal=GoalSpec(kind=GoalKind.OPEN, target_obj_id=gate_id),
        constraints=ConstraintSpec(max_interventions=4, max_steps=40),
        split=split,
        pressure_labels=["contradiction", "causal", "self_verification"],
        solution_certificate=SolutionCertificate(
            description=description,
            action_sequence=[
                {"kind": "pickup", "args": {"obj_id": correct_key_id}},
                {"kind": "apply", "args": {"tool_id": correct_key_id, "target_id": gate_id}},
            ],
            oracle_rules_required=oracle_rules,
        ),
        metadata={"gate_affinity": gate_affinity},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_GENERATORS = {
    TaskFamily.AFFORDANCE: _generate_affordance,
    TaskFamily.CAUSAL_GATE: _generate_causal_gate,
    TaskFamily.COUNTERFACTUAL: _generate_counterfactual,
    TaskFamily.TOOL_SUBSTITUTION: _generate_tool_substitution,
    TaskFamily.CONTRADICTION: _generate_contradiction,
}


def generate_task(
    family: TaskFamily | str, seed: int, split: SplitLabel | None = None
) -> TaskSpec:
    """Generate a deterministic TaskSpec. Split is auto-assigned from seed if None."""
    family = TaskFamily(family)
    if split is None:
        split = assign_split(seed)
    return _GENERATORS[family](seed, split)


def generate_perturbed_task(
    family: TaskFamily | str,
    seed: int,
    perturbation: str,
    split: SplitLabel | None = None,
) -> tuple[TaskSpec, TaskSpec]:
    """Return (clean_spec, perturbed_spec) for paired clean/perturbed episode analysis.

    The perturbed spec passes validate_task() and shares the same hidden causal
    solution as the clean spec; only a surface feature is changed.
    """
    from crucible.counterfactuals import PerturbationKind, apply_perturbation

    clean = generate_task(family, seed, split)
    perturbed = apply_perturbation(clean, PerturbationKind(perturbation), seed + 5_000_000)
    return clean, perturbed


def validate_task(spec: TaskSpec) -> list[str]:
    """Return a list of error strings. An empty list means the spec is valid."""
    errors: list[str] = []

    if spec.constraints.max_interventions <= 0:
        errors.append("max_interventions must be positive")
    if spec.constraints.max_steps <= 0:
        errors.append("max_steps must be positive")
    if spec.constraints.energy_budget <= 0:
        errors.append("energy_budget must be positive")
    if not spec.pressure_labels:
        errors.append("pressure_labels must be non-empty")
    if not spec.solution_certificate.action_sequence:
        errors.append("solution_certificate.action_sequence must be non-empty")
    if spec.split not in (SplitLabel.TRAIN, SplitLabel.TEST):
        errors.append(f"split must be TRAIN or TEST, got {spec.split!r}")

    obj_id_set = {s.obj_id for s in spec.object_specs}

    if spec.goal.target_obj_id and spec.goal.target_obj_id not in obj_id_set:
        errors.append(f"goal.target_obj_id {spec.goal.target_obj_id!r} not in object_specs")
    if spec.goal.classify_correct_obj_id and spec.goal.classify_correct_obj_id not in obj_id_set:
        errors.append(
            f"goal.classify_correct_obj_id {spec.goal.classify_correct_obj_id!r} "
            "not in object_specs"
        )

    for action_dict in spec.solution_certificate.action_sequence:
        args = action_dict.get("args", {})
        for key in ("obj_id", "tool_id", "target_id", "obj_id_a", "obj_id_b"):
            if key in args and args[key] not in obj_id_set:
                errors.append(
                    f"solution action references unknown id {args[key]!r} via key {key!r}"
                )

    return errors


def build_world_from_spec(spec: TaskSpec, compact: bool = False) -> WorldState:
    """Materialise a WorldState from a TaskSpec. Object positions are random but deterministic.

    With ``compact=True`` the agent starts at the grid centre and objects are
    placed in the nearest cells (assignment randomised per seed). This isolates
    the causal decision (which object to act on) from long-horizon navigation —
    a "decision-focused" variant. The task's objects and hidden properties are
    unchanged; only positions differ.
    """
    position_seed = spec.metadata.get("position_seed", spec.seed)
    rng = make_rng(position_seed)
    grid_size = spec.grid_size
    all_cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    occupied: set[tuple[int, int]] = set()

    compact_positions: list[tuple[int, int]] = []
    if compact:
        center = (grid_size // 2, grid_size // 2)
        agent_pos = center
        occupied.add(agent_pos)
        ring = sorted(
            (c for c in all_cells if c != center),
            key=lambda c: abs(c[0] - center[0]) + abs(c[1] - center[1]),
        )
        compact_positions = ring[: len(spec.object_specs)]
        rng.shuffle(compact_positions)  # randomise which object goes where (still near)
    else:
        agent_pos = sample_cell(rng, all_cells, occupied)
        occupied.add(agent_pos)

    objects: dict[str, CrucibleObject] = {}
    for os in spec.object_specs:
        if compact:
            pos = compact_positions.pop()
        else:
            pos = sample_cell(rng, all_cells, occupied)
        occupied.add(pos)
        visible = VisibleObjectState(
            obj_type=os.obj_type,
            color=os.color,
            shape=os.shape,
            texture=os.texture,
            size=os.size,
            marker=os.marker,
            pos=pos,
            state=os.state,
        )
        hidden = HiddenObjectProps(
            conductivity=os.conductivity,
            solubility=os.solubility,
            magnetism=os.magnetism,
            charge=os.charge,
            fragility=os.fragility,
            affinity=os.affinity,
        )
        objects[os.obj_id] = CrucibleObject(obj_id=os.obj_id, visible=visible, hidden=hidden)

    relations = derive_relations(objects, agent_pos)
    world = WorldState(
        seed=spec.seed,
        grid_size=spec.grid_size,
        step=0,
        max_steps=spec.max_steps,
        objects=objects,
        relations=relations,
        agent=AgentState(pos=agent_pos),
    )
    from crucible.rules import attach_rule_set

    attach_rule_set(world)
    return world


def check_goal(goal: GoalSpec, world: WorldState) -> bool:
    """Machine-checkable goal predicate evaluated against visible world state."""
    if goal.kind == GoalKind.OPEN:
        obj = world.objects.get(goal.target_obj_id or "")
        return obj is not None and obj.visible.state == ObjectState.OPEN

    if goal.kind == GoalKind.RETRIEVE:
        if not goal.target_type:
            return False
        target = ObjectType(goal.target_type)
        return any(
            world.objects[oid].visible.obj_type == target
            for oid in world.agent.inventory
            if oid in world.objects
        )

    if goal.kind == GoalKind.REACH:
        return world.agent.pos == goal.target_pos

    if goal.kind == GoalKind.TRANSFORM:
        obj = world.objects.get(goal.target_obj_id or "")
        return obj is not None and obj.visible.state.value == goal.target_state

    # CLASSIFY: satisfied only via trace inspection (Phase 6); not checkable from
    # visible state alone.
    return False
