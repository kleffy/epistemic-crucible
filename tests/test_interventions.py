"""Hidden intervention transition tests."""

from crucible.actions import Action, ActionKind
from crucible.counterfactuals import stable_state_hash
from crucible.env import CrucibleEnv
from crucible.grammar import (
    GoalKind,
    TaskFamily,
    build_world_from_spec,
    check_goal,
    generate_task,
)
from crucible.interventions import execute_intervention, execute_oracle_certificate
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
from crucible.rules import attach_rule_set, resolve_pending_effects
from crucible.splits import SplitLabel
from crucible.world import AgentState, WorldState, derive_relations


def _object(
    obj_id: str,
    obj_type: ObjectType,
    *,
    pos: tuple[int, int] | None = (0, 0),
    state: ObjectState = ObjectState.DEFAULT,
    conductivity: bool = False,
    solubility: bool = False,
    magnetism: bool = False,
    charge: bool = False,
    fragility: bool = False,
    affinity: str | None = None,
) -> CrucibleObject:
    return CrucibleObject(
        obj_id=obj_id,
        visible=VisibleObjectState(
            obj_type=obj_type,
            color=ObjectColor.GREY,
            shape=ObjectShape.CUBE,
            texture=ObjectTexture.SMOOTH,
            size=ObjectSize.SMALL,
            marker=None,
            pos=pos,
            state=state,
        ),
        hidden=HiddenObjectProps(
            conductivity=conductivity,
            solubility=solubility,
            magnetism=magnetism,
            charge=charge,
            fragility=fragility,
            affinity=affinity,
        ),
    )


def _world(objects: list[CrucibleObject], *, inventory: list[str] | None = None) -> WorldState:
    inventory = inventory or []
    object_map = {obj.obj_id: obj for obj in objects}
    for obj_id in inventory:
        object_map[obj_id].visible.pos = None
    world = WorldState(
        seed=123,
        grid_size=6,
        step=0,
        max_steps=20,
        objects=object_map,
        relations=derive_relations(object_map, (0, 0)),
        agent=AgentState(pos=(0, 0), inventory=list(inventory)),
    )
    attach_rule_set(world)
    return world


def _rule_ids(result) -> list[str]:
    return [prov.rule_id for prov in result.causal_provenance]


def test_conductive_tool_opens_gate_and_decoy_does_not():
    conductive_world = _world(
        [
            _object("tool", ObjectType.TOOL, conductivity=True),
            _object("gate", ObjectType.GATE, pos=(0, 1), state=ObjectState.CLOSED),
        ],
        inventory=["tool"],
    )

    result = execute_intervention(
        conductive_world,
        Action(ActionKind.APPLY, {"tool_id": "tool", "target_id": "gate"}),
    )

    assert conductive_world.objects["gate"].visible.state == ObjectState.OPEN
    assert "opened gate" in result.public_effects
    assert _rule_ids(result) == ["conductivity_gate_rule"]

    decoy_world = _world(
        [
            _object("decoy", ObjectType.TOOL, conductivity=False),
            _object("gate", ObjectType.GATE, pos=(0, 1), state=ObjectState.CLOSED),
        ],
        inventory=["decoy"],
    )

    result = execute_intervention(
        decoy_world,
        Action(ActionKind.APPLY, {"tool_id": "decoy", "target_id": "gate"}),
    )

    assert decoy_world.objects["gate"].visible.state == ObjectState.CLOSED
    assert result.public_effects == ["no_effect"]


def test_source_activates_magnetic_key_then_key_opens_gate():
    world = _world(
        [
            _object("source", ObjectType.SOURCE, state=ObjectState.ACTIVE),
            _object("key", ObjectType.KEY, pos=(0, 1), magnetism=True),
            _object("gate", ObjectType.GATE, pos=(0, 1), state=ObjectState.CLOSED),
        ],
        inventory=["source", "key"],
    )

    activation = execute_intervention(
        world,
        Action(ActionKind.APPLY, {"tool_id": "source", "target_id": "key"}),
    )
    opening = execute_intervention(
        world,
        Action(ActionKind.APPLY, {"tool_id": "key", "target_id": "gate"}),
    )

    assert world.objects["key"].hidden.charge is True
    assert world.objects["key"].visible.state == ObjectState.ACTIVE
    assert world.objects["gate"].visible.state == ObjectState.OPEN
    assert _rule_ids(activation) == ["source_activation_rule"]
    assert _rule_ids(opening) == ["magnetism_gate_rule"]


def test_soluble_block_transforms_and_non_soluble_block_does_not():
    soluble_world = _world(
        [
            _object("source", ObjectType.SOURCE, state=ObjectState.ACTIVE),
            _object("block", ObjectType.BLOCK, pos=(0, 1), solubility=True),
        ],
        inventory=["source"],
    )

    result = execute_intervention(
        soluble_world,
        Action(ActionKind.APPLY, {"tool_id": "source", "target_id": "block"}),
    )

    assert soluble_world.objects["block"].visible.state == ObjectState.DESTROYED
    assert "transformed block" in result.public_effects
    assert _rule_ids(result) == ["solubility_transform_rule"]

    non_soluble_world = _world(
        [
            _object("source", ObjectType.SOURCE, state=ObjectState.ACTIVE),
            _object("block", ObjectType.BLOCK, pos=(0, 1), solubility=False),
        ],
        inventory=["source"],
    )

    result = execute_intervention(
        non_soluble_world,
        Action(ActionKind.APPLY, {"tool_id": "source", "target_id": "block"}),
    )

    assert non_soluble_world.objects["block"].visible.state == ObjectState.DEFAULT
    assert result.public_effects == ["no_effect"]


def test_detector_marks_public_signal_without_hidden_field_names():
    world = _world(
        [
            _object("detector", ObjectType.DETECTOR),
            _object("target", ObjectType.TOOL, pos=(0, 1), conductivity=True),
        ],
        inventory=["detector"],
    )

    result = execute_intervention(
        world,
        Action(ActionKind.APPLY, {"tool_id": "detector", "target_id": "target"}),
    )

    assert world.objects["target"].visible.marker == "signal-positive"
    assert "marked target signal-positive" in result.public_effects
    assert "conductivity" not in str(result.public_effects)
    assert "conductivity" not in str(observe(world))


def test_source_and_catalyst_produce_deterministic_token():
    world = _world(
        [
            _object("source", ObjectType.SOURCE, state=ObjectState.ACTIVE),
            _object("catalyst", ObjectType.CATALYST, pos=(0, 1)),
        ],
        inventory=["source", "catalyst"],
    )

    result = execute_intervention(
        world,
        Action(ActionKind.COMBINE, {"obj_id_a": "source", "obj_id_b": "catalyst"}),
    )

    produced = [obj for obj in world.objects.values() if obj.visible.obj_type == ObjectType.TOKEN]
    assert len(produced) == 1
    assert produced[0].obj_id == "123_token_source_catalyst"
    assert result.public_effects == ["produced 123_token_source_catalyst"]
    assert _rule_ids(result) == ["source_production_rule"]


def test_hazard_delayed_consequence_fires_after_one_intervening_step():
    world = _world(
        [
            _object("tool", ObjectType.TOOL),
            _object("hazard", ObjectType.HAZARD, pos=(0, 1)),
        ],
        inventory=["tool"],
    )

    result = execute_intervention(
        world,
        Action(ActionKind.APPLY, {"tool_id": "tool", "target_id": "hazard"}),
    )

    assert result.public_effects == ["hazard_triggered hazard"]
    assert len(world.pending_effects) == 2
    world.step += 1
    early_effects, _ = resolve_pending_effects(world)
    assert early_effects == []
    assert world.agent.energy == 100

    world.step += 1
    due_effects, due_provenance = resolve_pending_effects(world)

    assert "energy_changed -10" in due_effects
    assert "damaged tool" in due_effects
    assert world.agent.energy == 90
    assert world.objects["tool"].visible.state == ObjectState.DESTROYED
    assert [prov.rule_id for prov in due_provenance] == [
        "hazard_consequence_rule",
        "hazard_consequence_rule",
    ]


def test_oracle_certificate_solves_state_checkable_task_families():
    state_checkable_families = [
        TaskFamily.AFFORDANCE,
        TaskFamily.CAUSAL_GATE,
        TaskFamily.TOOL_SUBSTITUTION,
        TaskFamily.CONTRADICTION,
    ]

    for family in state_checkable_families:
        for split in (SplitLabel.TRAIN, SplitLabel.TEST):
            spec = generate_task(family, seed=7, split=split)
            assert spec.goal.kind != GoalKind.CLASSIFY
            world = build_world_from_spec(spec)

            trace = execute_oracle_certificate(world, spec.solution_certificate)

            assert all(record["legal"] for record in trace), (family, split, trace)
            assert check_goal(spec.goal, world), (family, split, trace)


def test_swap_exchanges_object_positions_and_records_hashes():
    world = _world(
        [
            _object("left", ObjectType.TOOL, pos=(0, 0)),
            _object("right", ObjectType.BLOCK, pos=(0, 1)),
        ]
    )

    result = execute_intervention(
        world,
        Action(ActionKind.SWAP, {"obj_id_a": "left", "obj_id_b": "right"}),
    )

    assert result.valid is True
    assert result.public_effects == ["swapped left right"]
    assert result.state_hash_before != result.state_hash_after
    assert world.objects["left"].visible.pos == (0, 1)
    assert world.objects["right"].visible.pos == (0, 0)


def test_isolate_moves_accessible_object_to_deterministic_test_cell():
    world = _world([_object("tool", ObjectType.TOOL)], inventory=["tool"])

    result = execute_intervention(
        world,
        Action(ActionKind.ISOLATE, {"obj_id": "tool"}),
    )

    assert result.valid is True
    assert result.public_effects == ["isolated tool at (5, 5)"]
    assert "tool" not in world.agent.inventory
    assert world.objects["tool"].visible.pos == (5, 5)


def test_env_trace_records_state_hashes_for_intervention_action():
    env = CrucibleEnv(seed=31)
    env.reset()
    _, _, _, info = env.step(Action(ActionKind.PREDICT, {"query_id": "q1"}))
    trace = env.get_trace()

    assert info["legal"] is True
    assert trace[0]["state_hash_before"]
    assert trace[0]["state_hash_after"]
    assert trace[0]["public_state_hash_before"]
    assert trace[0]["public_state_hash_after"]


def test_invalid_interventions_preserve_state_hash_and_pending_effects():
    actions = [
        Action(ActionKind.APPLY, {"tool_id": "missing", "target_id": "gate"}),
        Action(ActionKind.COMBINE, {"obj_id_a": "tool", "obj_id_b": "tool"}),
        Action(ActionKind.SWAP, {"obj_id_a": "tool", "obj_id_b": "missing"}),
        Action(ActionKind.ISOLATE, {"obj_id": "missing"}),
    ]

    for action in actions:
        world = _world([_object("tool", ObjectType.TOOL)])
        before = stable_state_hash(world)

        result = execute_intervention(world, action)

        assert result.valid is False
        assert result.error == "illegal_action"
        assert result.state_hash_before == before
        assert result.state_hash_after == before
        assert stable_state_hash(world) == before
        assert world.pending_effects == []


def test_predict_records_query_without_mutating_world():
    world = _world([_object("tool", ObjectType.TOOL)])

    result = execute_intervention(
        world,
        Action(
            ActionKind.PREDICT,
            {"query_id": "q-predict", "predicted_effect": ["no_effect"]},
        ),
    )

    assert result.valid is True
    assert result.public_effects == ["prediction_recorded q-predict"]
    assert result.state_hash_before == result.state_hash_after
    assert result.public_state_hash_before == result.public_state_hash_after
