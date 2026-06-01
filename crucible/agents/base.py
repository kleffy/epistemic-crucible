from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations

from crucible.actions import Action, ActionKind, Direction

# All four cardinal directions in a stable order.
_DIRECTIONS = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]

# ActionKinds that count as interventions for budget tracking.
INTERVENTION_KINDS = {ActionKind.APPLY, ActionKind.COMBINE, ActionKind.INSPECT}


class Agent(ABC):
    """Shared interface for all Epistemic Crucible agents.

    Agents receive only public observations (never WorldState or hidden fields).
    The environment enforces legality; agents may propose illegal actions and
    the env will return legal=False without mutating state.
    """

    name: str = "agent"

    @abstractmethod
    def reset(self) -> None:
        """Called at the start of each episode before the first act()."""

    @abstractmethod
    def act(self, obs: dict) -> Action:
        """Select an action given a public observation."""

    def observe_result(
        self, obs: dict, reward: float, done: bool, info: dict
    ) -> None:
        """Hook called after env.step() returns. Default is no-op.

        Learning agents override this to update their policy or model.
        `info` contains: legal, effects, public_state_hash_before/after.
        """


def enumerate_candidate_actions(obs: dict, grid_size: int = 6) -> list[Action]:
    """Return all plausible actions derivable from public observation.

    The environment enforces actual legality; this function produces the
    superset of candidates an agent might want to consider. Only uses
    visible observation fields — no hidden properties accessed.
    """
    candidates: list[Action] = [Action(kind=ActionKind.WAIT, args={})]

    agent_pos: tuple[int, int] = tuple(obs["agent"]["pos"])  # type: ignore[arg-type]
    inventory: list[str] = obs["agent"]["inventory"]
    objects: dict = obs["objects"]

    # MOVE — all four directions (env rejects out-of-bounds)
    for d in _DIRECTIONS:
        candidates.append(Action(kind=ActionKind.MOVE, args={"direction": d}))

    # Classify objects by accessibility
    at_pos = [
        oid
        for oid, o in objects.items()
        if o["pos"] is not None and tuple(o["pos"]) == agent_pos and oid not in inventory
    ]
    adjacent_pos = [
        oid
        for oid, o in objects.items()
        if o["pos"] is not None
        and _manhattan(tuple(o["pos"]), agent_pos) == 1  # type: ignore[arg-type]
        and oid not in inventory
    ]
    accessible = set(at_pos + adjacent_pos + inventory)

    # PICKUP — objects at agent position not already held
    for oid in at_pos:
        candidates.append(Action(kind=ActionKind.PICKUP, args={"obj_id": oid}))

    # DROP — objects in inventory
    for oid in inventory:
        candidates.append(Action(kind=ActionKind.DROP, args={"obj_id": oid}))

    # INSPECT — any visible object
    for oid in objects:
        candidates.append(Action(kind=ActionKind.INSPECT, args={"obj_id": oid}))

    # APPLY — (tool from inventory or at pos) × (accessible target)
    actor_pool = set(inventory + at_pos)
    for tool_id in actor_pool:
        for target_id in accessible:
            if tool_id != target_id:
                candidates.append(
                    Action(
                        kind=ActionKind.APPLY,
                        args={"tool_id": tool_id, "target_id": target_id},
                    )
                )

    # COMBINE — any 2-combination of accessible objects
    for a_id, b_id in combinations(sorted(accessible), 2):
        candidates.append(
            Action(kind=ActionKind.COMBINE, args={"obj_id_a": a_id, "obj_id_b": b_id})
        )

    return candidates


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
