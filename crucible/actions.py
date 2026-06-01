from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crucible.world import WorldState

_INVENTORY_CAP = 3


class ActionKind(str, Enum):
    MOVE = "move"
    PICKUP = "pickup"
    DROP = "drop"
    INSPECT = "inspect"
    APPLY = "apply"
    COMBINE = "combine"
    SWAP = "swap"
    ISOLATE = "isolate"
    PREDICT = "predict"
    WAIT = "wait"


class Direction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


DIRECTION_DELTA: dict[Direction, tuple[int, int]] = {
    Direction.NORTH: (-1, 0),
    Direction.SOUTH: (1, 0),
    Direction.EAST: (0, 1),
    Direction.WEST: (0, -1),
}


@dataclass
class Action:
    kind: ActionKind
    args: dict = field(default_factory=dict)


def is_legal(action: Action, world: WorldState) -> bool:
    """Return True iff the action is valid given the current visible world state."""
    kind = action.kind
    args = action.args
    agent = world.agent

    if kind == ActionKind.WAIT:
        return True

    if kind == ActionKind.MOVE:
        direction = args.get("direction")
        if direction is None:
            return False
        dr, dc = DIRECTION_DELTA[Direction(direction)]
        r, c = agent.pos
        nr, nc = r + dr, c + dc
        return 0 <= nr < world.grid_size and 0 <= nc < world.grid_size

    if kind == ActionKind.PICKUP:
        obj_id = args.get("obj_id")
        if obj_id is None or obj_id not in world.objects:
            return False
        if obj_id in agent.inventory:
            return False
        if len(agent.inventory) >= _INVENTORY_CAP:
            return False
        obj = world.objects[obj_id]
        return obj.visible.pos == agent.pos

    if kind == ActionKind.DROP:
        obj_id = args.get("obj_id")
        return obj_id in agent.inventory if obj_id is not None else False

    if kind == ActionKind.INSPECT:
        obj_id = args.get("obj_id")
        return obj_id in world.objects if obj_id is not None else False

    if kind == ActionKind.APPLY:
        tool_id = args.get("tool_id")
        target_id = args.get("target_id")
        if tool_id is None or target_id is None:
            return False
        if tool_id not in world.objects or target_id not in world.objects:
            return False
        tool_at_agent = (
            tool_id in agent.inventory or world.objects[tool_id].visible.pos == agent.pos
        )
        target_pos = world.objects[target_id].visible.pos
        target_reachable = (
            target_id in agent.inventory
            or target_pos == agent.pos
            or _adjacent(agent.pos, target_pos)
        )
        return tool_at_agent and target_reachable

    if kind == ActionKind.COMBINE:
        id_a = args.get("obj_id_a")
        id_b = args.get("obj_id_b")
        if id_a is None or id_b is None or id_a == id_b:
            return False
        if id_a not in world.objects or id_b not in world.objects:
            return False
        accessible = {*agent.inventory}
        for oid in (id_a, id_b):
            obj = world.objects[oid]
            if obj.visible.pos == agent.pos:
                accessible.add(oid)
        return id_a in accessible and id_b in accessible

    if kind == ActionKind.SWAP:
        id_a = args.get("obj_id_a")
        id_b = args.get("obj_id_b")
        if id_a is None or id_b is None or id_a == id_b:
            return False
        if id_a not in world.objects or id_b not in world.objects:
            return False
        return _accessible(id_a, world) and _accessible(id_b, world)

    if kind == ActionKind.ISOLATE:
        obj_id = args.get("obj_id")
        if obj_id is None or obj_id not in world.objects:
            return False
        return _accessible(obj_id, world) and _has_free_cell(world)

    if kind == ActionKind.PREDICT:
        return True

    return False


def _accessible(obj_id: str, world: WorldState) -> bool:
    if obj_id in world.agent.inventory:
        return True
    obj_pos = world.objects[obj_id].visible.pos
    return obj_pos == world.agent.pos or _adjacent(world.agent.pos, obj_pos)


def _has_free_cell(world: WorldState) -> bool:
    occupied = {
        obj.visible.pos
        for obj in world.objects.values()
        if obj.visible.pos is not None
    }
    occupied.add(world.agent.pos)
    return any(
        (row, col) not in occupied
        for row in range(world.grid_size)
        for col in range(world.grid_size)
    )


def _adjacent(pos_a: tuple[int, int], pos_b: tuple[int, int] | None) -> bool:
    if pos_b is None:
        return False
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1]) == 1
