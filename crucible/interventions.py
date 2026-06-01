from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crucible.actions import Action, ActionKind, Direction, is_legal
from crucible.counterfactuals import stable_state_hash
from crucible.relations import Relation, RelationKind
from crucible.rules import (
    CausalProvenance,
    PendingEffect,
    evaluate_action_rules,
    resolve_pending_effects,
)
from crucible.world import WorldState, derive_relations


@dataclass
class InterventionResult:
    public_effects: list[str]
    causal_provenance: list[CausalProvenance]
    scheduled_effects: list[PendingEffect] = field(default_factory=list)
    state_hash_before: str = ""
    state_hash_after: str = ""
    public_state_hash_before: str = ""
    public_state_hash_after: str = ""
    valid: bool = True
    error: str | None = None


def execute_intervention(world: WorldState, action: Action) -> InterventionResult:
    """Execute an intervention action and return public plus evaluator-only data."""
    state_hash_before = stable_state_hash(world)
    public_state_hash_before = stable_state_hash(world, public=True)
    if not is_legal(action, world):
        return InterventionResult(
            public_effects=["invalid_intervention"],
            causal_provenance=[],
            state_hash_before=state_hash_before,
            state_hash_after=stable_state_hash(world),
            public_state_hash_before=public_state_hash_before,
            public_state_hash_after=stable_state_hash(world, public=True),
            valid=False,
            error="illegal_action",
        )

    before = len(world.pending_effects)
    if action.kind in (ActionKind.APPLY, ActionKind.COMBINE):
        public_effects, provenance = evaluate_action_rules(world, action)
    elif action.kind == ActionKind.SWAP:
        public_effects, provenance = _execute_swap(world, action), []
    elif action.kind == ActionKind.ISOLATE:
        public_effects, provenance = _execute_isolate(world, action), []
    elif action.kind == ActionKind.PREDICT:
        public_effects, provenance = _execute_predict(action), []
    else:
        public_effects, provenance = ["no_effect"], []

    return InterventionResult(
        public_effects=public_effects,
        causal_provenance=provenance,
        scheduled_effects=list(world.pending_effects[before:]),
        state_hash_before=state_hash_before,
        state_hash_after=stable_state_hash(world),
        public_state_hash_before=public_state_hash_before,
        public_state_hash_after=stable_state_hash(world, public=True),
    )


def execute_oracle_certificate(world: WorldState, certificate: Any) -> list[dict]:
    """Execute a solution certificate with deterministic hidden-state access.

    The helper moves the oracle agent deterministically to make certificate actions
    legal, then applies the same transition engine used by the environment.
    """
    trace: list[dict] = []

    for action_dict in certificate.action_sequence:
        action = Action(ActionKind(action_dict["kind"]), dict(action_dict.get("args", {})))
        _prepare_oracle_action(world, action)
        legal = is_legal(action, world)
        effects: list[str]
        provenance: list[CausalProvenance]

        if not legal:
            effects = ["oracle_illegal_action"]
            provenance = []
        elif action.kind in (
            ActionKind.APPLY,
            ActionKind.COMBINE,
            ActionKind.SWAP,
            ActionKind.ISOLATE,
            ActionKind.PREDICT,
        ):
            result = execute_intervention(world, action)
            effects = result.public_effects
            provenance = result.causal_provenance
        else:
            effects = _execute_primitive(world, action)
            provenance = []

        world.step += 1
        delayed_effects, delayed_provenance = resolve_pending_effects(world)
        effects.extend(delayed_effects)
        provenance.extend(delayed_provenance)

        trace.append(
            {
                "action": action,
                "legal": legal,
                "effects": effects,
                "causal_provenance": provenance,
            }
        )

    return trace


def _execute_swap(world: WorldState, action: Action) -> list[str]:
    id_a = action.args["obj_id_a"]
    id_b = action.args["obj_id_b"]
    obj_a = world.objects[id_a]
    obj_b = world.objects[id_b]
    a_held = id_a in world.agent.inventory
    b_held = id_b in world.agent.inventory

    if a_held and b_held:
        idx_a = world.agent.inventory.index(id_a)
        idx_b = world.agent.inventory.index(id_b)
        world.agent.inventory[idx_a], world.agent.inventory[idx_b] = (
            world.agent.inventory[idx_b],
            world.agent.inventory[idx_a],
        )
    elif a_held:
        _replace_inventory(world, id_a, id_b)
        obj_a.visible.pos = obj_b.visible.pos
        obj_b.visible.pos = None
    elif b_held:
        _replace_inventory(world, id_b, id_a)
        obj_b.visible.pos = obj_a.visible.pos
        obj_a.visible.pos = None
    else:
        obj_a.visible.pos, obj_b.visible.pos = obj_b.visible.pos, obj_a.visible.pos

    _refresh_relations(world)
    return [f"swapped {id_a} {id_b}"]


def _execute_isolate(world: WorldState, action: Action) -> list[str]:
    obj_id = action.args["obj_id"]
    obj = world.objects[obj_id]
    if obj_id in world.agent.inventory:
        world.agent.inventory.remove(obj_id)
    cell = _isolation_cell(world)
    obj.visible.pos = cell
    _refresh_relations(world)
    return [f"isolated {obj_id} at {cell}"]


def _execute_predict(action: Action) -> list[str]:
    query_id = action.args.get("query_id")
    if query_id is None:
        return ["prediction_recorded"]
    return [f"prediction_recorded {query_id}"]


def _prepare_oracle_action(world: WorldState, action: Action) -> None:
    if action.kind == ActionKind.PICKUP:
        obj_id = action.args.get("obj_id")
        if obj_id in world.objects and world.objects[obj_id].visible.pos is not None:
            _move_agent_to(world, world.objects[obj_id].visible.pos)

    if action.kind == ActionKind.APPLY:
        tool_id = action.args.get("tool_id")
        target_id = action.args.get("target_id")
        if tool_id in world.objects and tool_id not in world.agent.inventory:
            tool_pos = world.objects[tool_id].visible.pos
            if tool_pos is not None:
                _move_agent_to(world, tool_pos)
        if target_id in world.objects:
            target_pos = world.objects[target_id].visible.pos
            if target_pos is not None:
                _move_agent_to(world, target_pos)

    if action.kind == ActionKind.COMBINE:
        for arg_name in ("obj_id_a", "obj_id_b"):
            obj_id = action.args.get(arg_name)
            if obj_id in world.objects and obj_id not in world.agent.inventory:
                obj_pos = world.objects[obj_id].visible.pos
                if obj_pos is not None:
                    _move_agent_to(world, obj_pos)

    if action.kind in (ActionKind.SWAP, ActionKind.ISOLATE):
        obj_id = action.args.get("obj_id") or action.args.get("obj_id_a")
        if obj_id in world.objects and obj_id not in world.agent.inventory:
            obj_pos = world.objects[obj_id].visible.pos
            if obj_pos is not None:
                _move_agent_to(world, obj_pos)


def _execute_primitive(world: WorldState, action: Action) -> list[str]:
    if action.kind == ActionKind.WAIT:
        return ["waited"]

    if action.kind == ActionKind.PICKUP:
        obj_id = action.args["obj_id"]
        obj = world.objects[obj_id]
        world.agent.inventory.append(obj_id)
        obj.visible.pos = None
        _refresh_relations(world)
        return [f"picked_up {obj_id}"]

    if action.kind == ActionKind.DROP:
        obj_id = action.args["obj_id"]
        obj = world.objects[obj_id]
        world.agent.inventory.remove(obj_id)
        obj.visible.pos = world.agent.pos
        _refresh_relations(world)
        return [f"dropped {obj_id} at {world.agent.pos}"]

    if action.kind == ActionKind.MOVE:
        direction = Direction(action.args["direction"])
        row, col = world.agent.pos
        if direction == Direction.NORTH:
            world.agent.pos = (row - 1, col)
        elif direction == Direction.SOUTH:
            world.agent.pos = (row + 1, col)
        elif direction == Direction.EAST:
            world.agent.pos = (row, col + 1)
        elif direction == Direction.WEST:
            world.agent.pos = (row, col - 1)
        _refresh_relations(world)
        return [f"moved_to {world.agent.pos}"]

    if action.kind == ActionKind.INSPECT:
        obj_id = action.args["obj_id"]
        return [f"inspect:{obj_id}"]

    return ["no_effect"]


def _move_agent_to(world: WorldState, target: tuple[int, int]) -> None:
    row, col = world.agent.pos
    target_row, target_col = target
    while row != target_row:
        row += 1 if target_row > row else -1
    while col != target_col:
        col += 1 if target_col > col else -1
    world.agent.pos = (row, col)
    _refresh_relations(world)


def _replace_inventory(world: WorldState, old_obj_id: str, new_obj_id: str) -> None:
    idx = world.agent.inventory.index(old_obj_id)
    world.agent.inventory[idx] = new_obj_id


def _isolation_cell(world: WorldState) -> tuple[int, int]:
    occupied = {
        obj.visible.pos
        for obj in world.objects.values()
        if obj.visible.pos is not None
    }
    occupied.add(world.agent.pos)
    preferred = (world.grid_size - 1, world.grid_size - 1)
    if preferred not in occupied:
        return preferred
    for row in range(world.grid_size):
        for col in range(world.grid_size):
            cell = (row, col)
            if cell not in occupied:
                return cell
    raise ValueError("No free isolation cell available.")


def _refresh_relations(world: WorldState) -> None:
    world.relations = derive_relations(world.objects, world.agent.pos)
    for obj_id in world.agent.inventory:
        world.relations.append(Relation(kind=RelationKind.HELD, subject="agent", object_=obj_id))
