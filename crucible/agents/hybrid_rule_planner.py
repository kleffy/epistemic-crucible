from __future__ import annotations

from dataclasses import dataclass, field

from crucible.actions import Action, ActionKind, Direction
from crucible.agents.base import Agent, _manhattan
from crucible.utils.logging import get_logger
from crucible.utils.seeding import make_python_rng

_log = get_logger(__name__)

# Effect strings produced by rules (Phase 3) that reveal hidden properties.
_GATE_OPEN_EFFECTS = {"opened gate", "gate opened"}
_NO_EFFECT_EFFECTS = {"no_effect", "no effect"}
_SIGNAL_POS = "signal-positive"
_SIGNAL_NEG = "signal-negative"
_CHARGED_MARKER = "charged"
_TRANSFORMED_MARKER = "transformed"


@dataclass
class PropertyBelief:
    """Confidence that an object has a given latent property (0.0–1.0)."""

    obj_id: str
    prop: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


class HybridRulePlannerAgent(Agent):
    """Belief-tracking planner that combines heuristic rules with inferred properties.

    Maintains a confidence table over latent object properties and uses it to
    prioritise APPLY actions toward objects most likely to produce effects.
    Belief updates are based solely on public effect strings and visible markers —
    no hidden state is accessed.
    """

    name = "hybrid_rule_planner"

    def __init__(self, seed: int = 0, grid_size: int = 6) -> None:
        self._seed = seed
        self._grid_size = grid_size
        self._rng = make_python_rng(seed)
        # obj_id → {prop: confidence}
        self._beliefs: dict[str, dict[str, float]] = {}
        self._last_action: Action | None = None
        self._visited: set[tuple[int, int]] = set()

    def reset(self) -> None:
        self._beliefs = {}
        self._last_action = None
        self._visited = set()

    def _belief(self, obj_id: str, prop: str) -> float:
        return self._beliefs.get(obj_id, {}).get(prop, 0.5)

    def _set_belief(self, obj_id: str, prop: str, value: float, evidence: str) -> None:
        self._beliefs.setdefault(obj_id, {})[prop] = value
        _log.debug("belief: %s.%s=%.2f (%s)", obj_id, prop, value, evidence)

    def _update_beliefs(self, action: Action, effects: list[str], obs: dict) -> None:
        """Update property beliefs based on observed effects and visible markers."""
        if action.kind == ActionKind.APPLY:
            tool_id = action.args.get("tool_id", "")
            target_id = action.args.get("target_id", "")
            effects_set = {e.lower() for e in effects}

            # Conductive tool opened gate → conductivity confirmed.
            if effects_set & _GATE_OPEN_EFFECTS:
                self._set_belief(tool_id, "conductivity", 1.0, "opened gate")
            # No effect on gate → likely not conductive.
            elif not effects and target_id and _is_gate(obs, target_id):
                old = self._belief(tool_id, "conductivity")
                self._set_belief(tool_id, "conductivity", old * 0.3, "no gate effect")

        # Scan visible markers for property signals.
        for oid, obj in obs.get("objects", {}).items():
            marker = obj.get("marker")
            if marker == _SIGNAL_POS:
                self._set_belief(oid, "conductivity", 0.9, "signal-positive")
            elif marker == _SIGNAL_NEG:
                self._set_belief(oid, "conductivity", 0.1, "signal-negative")
            elif marker == _CHARGED_MARKER:
                self._set_belief(oid, "magnetism", 1.0, "charged marker")
            elif marker == _TRANSFORMED_MARKER:
                self._set_belief(oid, "solubility", 1.0, "transformed marker")

    def observe_result(self, obs: dict, reward: float, done: bool, info: dict) -> None:
        if self._last_action is not None:
            self._update_beliefs(self._last_action, info.get("effects", []), obs)

    def act(self, obs: dict) -> Action:
        agent_pos: tuple[int, int] = tuple(obs["agent"]["pos"])  # type: ignore[arg-type]
        inventory: list[str] = obs["agent"]["inventory"]
        objects: dict = obs["objects"]
        self._visited.add(agent_pos)

        # 1. If holding a high-belief actor and a closed gate is accessible → APPLY.
        held_actors = [
            oid for oid in inventory if objects.get(oid, {}).get("type") in {"tool", "key"}
        ]
        held_actors.sort(key=lambda oid: self._belief(oid, "conductivity"), reverse=True)
        closed_gates = [
            oid
            for oid, o in objects.items()
            if o.get("type") == "gate" and o.get("state") == "closed"
        ]

        if held_actors and closed_gates:
            gate_id = closed_gates[0]
            gate_pos = objects[gate_id].get("pos")
            if gate_pos is not None and _manhattan(tuple(gate_pos), agent_pos) <= 1:  # type: ignore[arg-type]
                best_actor = held_actors[0]
                action = Action(
                    kind=ActionKind.APPLY,
                    args={"tool_id": best_actor, "target_id": gate_id},
                )
                self._last_action = action
                return action

        # 2. Move toward closed gate if holding an actor.
        if held_actors and closed_gates:
            gate_id = closed_gates[0]
            gate_pos = objects[gate_id].get("pos")
            if gate_pos is not None:
                move = self._move_toward(agent_pos, tuple(gate_pos))  # type: ignore[arg-type]
                if move is not None:
                    self._last_action = move
                    return move

        # 3. Pick up highest-belief tool/key at current position.
        at_pos_actors = [
            oid
            for oid, o in objects.items()
            if o.get("pos") is not None
            and tuple(o["pos"]) == agent_pos  # type: ignore[arg-type]
            and oid not in inventory
            and o.get("type") in {"tool", "key", "source"}
        ]
        if at_pos_actors:
            at_pos_actors.sort(key=lambda oid: self._belief(oid, "conductivity"), reverse=True)
            action = Action(kind=ActionKind.PICKUP, args={"obj_id": at_pos_actors[0]})
            self._last_action = action
            return action

        # 4. Move toward highest-belief unseen actor.
        actor_targets = [
            (oid, tuple(o["pos"]))
            for oid, o in objects.items()
            if o.get("type") in {"tool", "key", "source"}
            and o.get("pos") is not None
            and oid not in inventory
            and tuple(o["pos"]) not in self._visited  # type: ignore[arg-type]
        ]
        if actor_targets:
            actor_targets.sort(
                key=lambda x: (-self._belief(x[0], "conductivity"), _manhattan(x[1], agent_pos))
            )
            _, target_pos = actor_targets[0]
            move = self._move_toward(agent_pos, target_pos)  # type: ignore[arg-type]
            if move is not None:
                self._last_action = move
                return move

        action = Action(kind=ActionKind.WAIT, args={})
        self._last_action = action
        return action

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


def _is_gate(obs: dict, obj_id: str) -> bool:
    return obs.get("objects", {}).get(obj_id, {}).get("type") == "gate"
