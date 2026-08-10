from __future__ import annotations

from crucible.actions import Action, ActionKind, Direction
from crucible.agents.base import Agent, _manhattan
from crucible.utils.logging import get_logger

_log = get_logger(__name__)

# Object types that can directly interact with gates.
_ACTOR_TYPES = {"tool", "key"}
# Object types worth picking up.
_PICKUP_TYPES = {"tool", "key", "source", "detector", "catalyst"}

_DIRECTIONS = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
_DELTA = {
    Direction.NORTH: (-1, 0),
    Direction.SOUTH: (1, 0),
    Direction.EAST: (0, 1),
    Direction.WEST: (0, -1),
}


class HeuristicAgent(Agent):
    """Visible-only rule-of-thumb planner.

    Uses only public observation fields — no latent property access, no rule-engine imports.

    Priority chain:
    1. Held actor + closed gate reachable → APPLY actor to gate.
    2. Object worth picking up at current position → PICKUP.
    3. SOURCE held + key/block adjacent → APPLY source to that object.
    4. MOVE toward nearest unvisited interesting object.
    5. WAIT.
    """

    name = "heuristic"

    def __init__(self, grid_size: int = 6) -> None:
        self._grid_size = grid_size
        self._visited: set[tuple[int, int]] = set()
        self._last_action: Action | None = None

    def reset(self) -> None:
        self._visited = set()
        self._last_action = None

    def act(self, obs: dict) -> Action:
        agent_pos: tuple[int, int] = tuple(obs["agent"]["pos"])  # type: ignore[arg-type]
        inventory: list[str] = obs["agent"]["inventory"]
        objects: dict = obs["objects"]
        self._visited.add(agent_pos)

        # 1. If holding an actor (tool/key) and a closed gate is accessible → APPLY.
        held_actors = [oid for oid in inventory if objects.get(oid, {}).get("type") in _ACTOR_TYPES]
        closed_gates = [
            oid
            for oid, o in objects.items()
            if o.get("type") == "gate" and o.get("state") == "closed"
        ]
        if held_actors and closed_gates:
            gate_id = closed_gates[0]
            gate_pos = objects[gate_id].get("pos")
            if gate_pos is not None:
                dist = _manhattan(tuple(gate_pos), agent_pos)  # type: ignore[arg-type]
                if dist <= 1:
                    action = Action(
                        kind=ActionKind.APPLY,
                        args={"tool_id": held_actors[0], "target_id": gate_id},
                    )
                    _log.debug("heuristic: apply actor to gate")
                    return action
                # Move toward gate instead.
                move = self._move_toward(agent_pos, tuple(gate_pos))  # type: ignore[arg-type]
                if move is not None:
                    return move

        # 2. Interesting object at current position not in inventory → PICKUP.
        at_pos = [
            oid
            for oid, o in objects.items()
            if o.get("pos") is not None
            and tuple(o["pos"]) == agent_pos  # type: ignore[arg-type]
            and oid not in inventory
            and o.get("type") in _PICKUP_TYPES
        ]
        if at_pos:
            _log.debug("heuristic: pickup %s", at_pos[0])
            return Action(kind=ActionKind.PICKUP, args={"obj_id": at_pos[0]})

        # 3. SOURCE held + key/block adjacent → APPLY source to it.
        held_sources = [oid for oid in inventory if objects.get(oid, {}).get("type") == "source"]
        if held_sources:
            source_id = held_sources[0]
            for oid, o in objects.items():
                if o.get("type") in {"key", "block"} and o.get("pos") is not None:
                    if _manhattan(tuple(o["pos"]), agent_pos) <= 1:  # type: ignore[arg-type]
                        _log.debug("heuristic: apply source to %s", oid)
                        return Action(
                            kind=ActionKind.APPLY,
                            args={"tool_id": source_id, "target_id": oid},
                        )

        # 4. MOVE toward nearest unvisited interesting object.
        targets = [
            (oid, tuple(o["pos"]))
            for oid, o in objects.items()
            if o.get("pos") is not None
            and o.get("type") in _PICKUP_TYPES | {"gate"}
            and tuple(o["pos"]) not in self._visited  # type: ignore[arg-type]
            and oid not in inventory
        ]
        if targets:
            targets.sort(key=lambda x: _manhattan(x[1], agent_pos))  # type: ignore[arg-type]
            _, target_pos = targets[0]
            move = self._move_toward(agent_pos, target_pos)  # type: ignore[arg-type]
            if move is not None:
                _log.debug("heuristic: move toward %s", target_pos)
                return move

        return Action(kind=ActionKind.WAIT, args={})

    def _move_toward(self, agent_pos: tuple[int, int], target: tuple[int, int]) -> Action | None:
        dr = target[0] - agent_pos[0]
        dc = target[1] - agent_pos[1]
        if dr < 0:
            return Action(kind=ActionKind.MOVE, args={"direction": Direction.NORTH})
        if dr > 0:
            return Action(kind=ActionKind.MOVE, args={"direction": Direction.SOUTH})
        if dc > 0:
            return Action(kind=ActionKind.MOVE, args={"direction": Direction.EAST})
        if dc < 0:
            return Action(kind=ActionKind.MOVE, args={"direction": Direction.WEST})
        return None
