"""Tests for action legality, observation boundary, object serialisation, and task grammar."""

from crucible.actions import Action, ActionKind, Direction
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
from crucible.observations import observe
from crucible.utils.serialization import to_dict
from crucible.world import AgentState, WorldState, generate_world


def _make_world(agent_pos=(0, 0), obj_pos=(0, 1), obj_in_inventory=False):
    """Minimal 6x6 world with one object for action legality tests."""
    obj = CrucibleObject(
        obj_id="test_000",
        visible=VisibleObjectState(
            obj_type=ObjectType.KEY,
            color=ObjectColor.RED,
            shape=ObjectShape.ROD,
            texture=ObjectTexture.SMOOTH,
            size=ObjectSize.SMALL,
            marker=None,
            pos=None if obj_in_inventory else obj_pos,
            state=ObjectState.DEFAULT,
        ),
        hidden=HiddenObjectProps(
            conductivity=True,
            solubility=False,
            magnetism=True,
            charge=False,
            fragility=False,
            affinity="metal",
        ),
    )
    inventory = ["test_000"] if obj_in_inventory else []
    return WorldState(
        seed=0,
        grid_size=6,
        step=0,
        max_steps=50,
        objects={"test_000": obj},
        relations=[],
        agent=AgentState(pos=agent_pos, inventory=inventory),
    )


# --- MOVE ---

def test_legal_move_in_bounds():
    world = _make_world(agent_pos=(3, 3))
    action = Action(ActionKind.MOVE, {"direction": Direction.SOUTH})
    from crucible.actions import is_legal
    assert is_legal(action, world) is True


def test_illegal_move_out_of_bounds():
    world = _make_world(agent_pos=(0, 0))
    action = Action(ActionKind.MOVE, {"direction": Direction.NORTH})
    from crucible.actions import is_legal
    assert is_legal(action, world) is False


# --- PICKUP ---

def test_pickup_at_position():
    world = _make_world(agent_pos=(0, 1), obj_pos=(0, 1))
    action = Action(ActionKind.PICKUP, {"obj_id": "test_000"})
    from crucible.actions import is_legal
    assert is_legal(action, world) is True


def test_pickup_wrong_position():
    world = _make_world(agent_pos=(0, 0), obj_pos=(3, 3))
    action = Action(ActionKind.PICKUP, {"obj_id": "test_000"})
    from crucible.actions import is_legal
    assert is_legal(action, world) is False


def test_pickup_already_held():
    world = _make_world(agent_pos=(0, 0), obj_in_inventory=True)
    action = Action(ActionKind.PICKUP, {"obj_id": "test_000"})
    from crucible.actions import is_legal
    assert is_legal(action, world) is False


# --- DROP ---

def test_drop_from_inventory():
    world = _make_world(agent_pos=(0, 0), obj_in_inventory=True)
    action = Action(ActionKind.DROP, {"obj_id": "test_000"})
    from crucible.actions import is_legal
    assert is_legal(action, world) is True


def test_drop_not_in_inventory():
    world = _make_world(agent_pos=(0, 0), obj_pos=(0, 1))
    action = Action(ActionKind.DROP, {"obj_id": "test_000"})
    from crucible.actions import is_legal
    assert is_legal(action, world) is False


# --- OBSERVATION BOUNDARY ---

_HIDDEN_FIELD_NAMES = {
    "hidden", "conductivity", "solubility", "magnetism", "charge", "fragility", "affinity"
}


def test_observation_excludes_hidden():
    world = generate_world(seed=42)
    obs = observe(world)
    obs_str = str(obs)
    for field_name in _HIDDEN_FIELD_NAMES:
        assert field_name not in obs_str, f"Hidden field '{field_name}' leaked into observation"


# --- SERIALISATION ---

def test_object_serialization_roundtrip():
    obj = CrucibleObject(
        obj_id="rt_000",
        visible=VisibleObjectState(
            obj_type=ObjectType.TOOL,
            color=ObjectColor.BLUE,
            shape=ObjectShape.CYLINDER,
            texture=ObjectTexture.ROUGH,
            size=ObjectSize.MEDIUM,
            marker="X",
            pos=(2, 3),
            state=ObjectState.ACTIVE,
        ),
        hidden=HiddenObjectProps(
            conductivity=True,
            solubility=True,
            magnetism=False,
            charge=True,
            fragility=False,
            affinity="organic",
        ),
    )
    full_dict = to_dict(obj, public=False)
    # hidden must be present in full serialisation
    assert "hidden" in full_dict
    assert full_dict["hidden"]["conductivity"] is True

    # public serialisation must not contain hidden
    public_dict = to_dict(obj, public=True)
    assert "hidden" not in public_dict
    assert "conductivity" not in str(public_dict)

    # full round-trip preserves obj_id and visible fields
    assert full_dict["obj_id"] == "rt_000"
    assert full_dict["visible"]["color"] == "blue"
    assert full_dict["visible"]["pos"] == (2, 3)


# --- TASK GRAMMAR ---

from crucible.grammar import (  # noqa: E402
    GoalKind,
    GoalSpec,
    TaskFamily,
    build_world_from_spec,
    check_goal,
    generate_task,
    validate_task,
)


def test_grammar_all_families_produce_valid_tasks():
    for family in TaskFamily:
        spec = generate_task(family, seed=42)
        errors = validate_task(spec)
        assert errors == [], f"{family}: {errors}"


def test_grammar_100_seeds_valid():
    for seed in range(100):
        spec = generate_task(TaskFamily.AFFORDANCE, seed=seed)
        assert validate_task(spec) == [], f"seed={seed}"


def test_goal_reachable_certificate_nonempty():
    for family in TaskFamily:
        spec = generate_task(family, seed=7)
        assert len(spec.solution_certificate.action_sequence) > 0, family


def test_constraints_non_negative():
    for family in TaskFamily:
        spec = generate_task(family, seed=1)
        assert spec.constraints.max_interventions > 0
        assert spec.constraints.max_steps > 0
        assert spec.constraints.energy_budget > 0


def test_pressure_labels_present():
    for family in TaskFamily:
        spec = generate_task(family, seed=2)
        assert len(spec.pressure_labels) > 0, family


def test_split_label_present():
    from crucible.splits import SplitLabel

    for family in TaskFamily:
        spec = generate_task(family, seed=3)
        assert spec.split in (SplitLabel.TRAIN, SplitLabel.TEST), family


def test_check_goal_open():
    spec = generate_task(TaskFamily.CAUSAL_GATE, seed=10)
    world = build_world_from_spec(spec)
    gate_id = spec.goal.target_obj_id
    assert gate_id is not None
    # Initially closed — goal not satisfied
    assert check_goal(spec.goal, world) is False
    # Force the gate open
    world.objects[gate_id].visible.state = ObjectState.OPEN
    assert check_goal(spec.goal, world) is True


def test_check_goal_retrieve():
    spec = generate_task(TaskFamily.AFFORDANCE, seed=11)
    world = build_world_from_spec(spec)
    retrieve_goal = GoalSpec(kind=GoalKind.RETRIEVE, target_type="tool")
    assert check_goal(retrieve_goal, world) is False
    # Put a TOOL in inventory
    tool_id = next(
        oid for oid, obj in world.objects.items() if obj.visible.obj_type.value == "tool"
    )
    world.agent.inventory.append(tool_id)
    assert check_goal(retrieve_goal, world) is True


def test_build_world_object_count_matches_spec():
    for family in TaskFamily:
        spec = generate_task(family, seed=5)
        world = build_world_from_spec(spec)
        assert len(world.objects) == len(spec.object_specs), family


def test_task_metadata_field_exists():
    for family in TaskFamily:
        spec = generate_task(family, seed=6)
        assert isinstance(spec.metadata, dict)


# ---------------------------------------------------------------------------
# Perturbation suite
# ---------------------------------------------------------------------------

from crucible.counterfactuals import PerturbationKind, apply_perturbation  # noqa: E402
from crucible.grammar import generate_perturbed_task  # noqa: E402
from crucible.splits import SplitLabel  # noqa: E402, F811


def test_all_perturbations_produce_valid_tasks():
    spec = generate_task(TaskFamily.AFFORDANCE, seed=10, split=SplitLabel.TRAIN)
    for kind in PerturbationKind:
        perturbed = apply_perturbation(spec, kind, seed=99)
        errors = validate_task(perturbed)
        assert errors == [], f"{kind}: {errors}"


def test_color_decorrelation_preserves_conductivity():
    spec = generate_task(TaskFamily.AFFORDANCE, seed=10, split=SplitLabel.TRAIN)
    perturbed = apply_perturbation(spec, PerturbationKind.COLOR_AFFORDANCE_DECORRELATION, seed=1)
    orig_cond = {o.obj_id: o.conductivity for o in spec.object_specs}
    pert_cond = {o.obj_id: o.conductivity for o in perturbed.object_specs}
    assert orig_cond == pert_cond
    # At least one tool colour must have changed.
    orig_cols = {o.obj_id: o.color for o in spec.object_specs}
    pert_cols = {o.obj_id: o.color for o in perturbed.object_specs}
    assert orig_cols != pert_cols


def test_object_id_resampling_updates_all_references():
    spec = generate_task(TaskFamily.AFFORDANCE, seed=11, split=SplitLabel.TRAIN)
    original_ids = {o.obj_id for o in spec.object_specs}
    perturbed = apply_perturbation(spec, PerturbationKind.OBJECT_ID_RESAMPLING, seed=2)
    new_ids = {o.obj_id for o in perturbed.object_specs}
    assert original_ids.isdisjoint(new_ids)
    assert validate_task(perturbed) == []


def test_distractor_injection_adds_one_object():
    spec = generate_task(TaskFamily.AFFORDANCE, seed=12, split=SplitLabel.TRAIN)
    perturbed = apply_perturbation(spec, PerturbationKind.DISTRACTOR_INJECTION, seed=3)
    assert len(perturbed.object_specs) == len(spec.object_specs) + 1
    distractor = next(o for o in perturbed.object_specs if o.role == "distractor")
    assert distractor.conductivity is False


def test_tool_appearance_swap_preserves_hidden_properties():
    spec = generate_task(TaskFamily.AFFORDANCE, seed=13, split=SplitLabel.TRAIN)
    perturbed = apply_perturbation(spec, PerturbationKind.TOOL_APPEARANCE_SWAP, seed=4)
    orig_cond = {o.obj_id: o.conductivity for o in spec.object_specs}
    pert_cond = {o.obj_id: o.conductivity for o in perturbed.object_specs}
    assert orig_cond == pert_cond


def test_false_hint_text_does_not_alter_ground_truth():
    spec = generate_task(TaskFamily.AFFORDANCE, seed=14, split=SplitLabel.TRAIN)
    perturbed = apply_perturbation(spec, PerturbationKind.FALSE_HINT_TEXT, seed=5)
    orig_cond = {o.obj_id: o.conductivity for o in spec.object_specs}
    pert_cond = {o.obj_id: o.conductivity for o in perturbed.object_specs}
    assert orig_cond == pert_cond
    assert validate_task(perturbed) == []


def test_resource_scarcity_reduces_budget():
    spec = generate_task(TaskFamily.AFFORDANCE, seed=15, split=SplitLabel.TRAIN)
    perturbed = apply_perturbation(spec, PerturbationKind.RESOURCE_SCARCITY, seed=6)
    assert perturbed.constraints.energy_budget < spec.constraints.energy_budget
    assert perturbed.constraints.energy_budget >= 20


def test_layout_permutation_changes_position_seed():
    spec = generate_task(TaskFamily.AFFORDANCE, seed=16, split=SplitLabel.TRAIN)
    perturbed = apply_perturbation(spec, PerturbationKind.LAYOUT_PERMUTATION, seed=7)
    assert perturbed.metadata.get("position_seed") is not None
    assert perturbed.metadata["position_seed"] != spec.seed


def test_generate_perturbed_task_returns_differing_pair():
    clean, perturbed = generate_perturbed_task(
        TaskFamily.AFFORDANCE,
        seed=5,
        perturbation=PerturbationKind.COLOR_AFFORDANCE_DECORRELATION.value,
    )
    assert validate_task(clean) == []
    assert validate_task(perturbed) == []
    clean_cols = [o.color for o in clean.object_specs]
    pert_cols = [o.color for o in perturbed.object_specs]
    assert clean_cols != pert_cols
