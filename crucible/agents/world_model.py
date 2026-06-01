from __future__ import annotations

from crucible.actions import Action
from crucible.agents.base import Agent, enumerate_candidate_actions
from crucible.utils.logging import get_logger
from crucible.utils.seeding import make_python_rng

_log = get_logger(__name__)


def _action_key(action: Action) -> tuple[str, tuple]:
    """Stable hashable representation of an action."""
    args_key = tuple(sorted((k, str(v)) for k, v in action.args.items()))
    return (action.kind.value, args_key)


class WorldModelAgent(Agent):
    """Transition-model-based agent that plans from public observations.

    Builds a table: (state_hash, action_key) → (next_state_hash, frozenset(effects)).
    Planning heuristic: prefer actions that have previously produced non-empty effects.
    Falls back to random when no useful prior experience exists.
    """

    name = "world_model"

    def __init__(self, seed: int = 0, grid_size: int = 6) -> None:
        self._seed = seed
        self._grid_size = grid_size
        self._rng = make_python_rng(seed)
        # (state_hash, action_key) → (next_state_hash, frozenset[str])
        self._transitions: dict[tuple, tuple[str, frozenset[str]]] = {}
        # Effect frequency: action_key → count of non-empty effect steps
        self._effect_counts: dict[tuple, int] = {}
        self._prev_state: str | None = None
        self._prev_action_key: tuple | None = None

    def reset(self) -> None:
        self._prev_state = None
        self._prev_action_key = None

    def act(self, obs: dict) -> Action:
        candidates = enumerate_candidate_actions(obs, self._grid_size)
        state_key = obs.get("_public_hash", "")

        # Find candidates that have previously produced effects in any state.
        scored: list[tuple[float, int, Action]] = []
        for i, action in enumerate(candidates):
            ak = _action_key(action)
            score = self._effect_counts.get(ak, 0)
            # Prefer actions with known transitions from this exact state.
            transition_key = (state_key, ak)
            if transition_key in self._transitions:
                _, effects = self._transitions[transition_key]
                score += 10 if effects else 0
            scored.append((score, i, action))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score = scored[0][0] if scored else 0

        if best_score > 0:
            # Pick highest-scored (with random tie-breaking among equals).
            best = [t for t in scored if t[0] == best_score]
            _, _, action = self._rng.choice(best)
        else:
            action = self._rng.choice(candidates)

        self._prev_state = state_key
        self._prev_action_key = _action_key(action)
        _log.debug("world_model: %s (score=%d)", action.kind.value, best_score)
        return action

    def observe_result(self, obs: dict, reward: float, done: bool, info: dict) -> None:
        if self._prev_state is None or self._prev_action_key is None:
            return

        next_hash = info.get("public_state_hash_after", "")
        effects = frozenset(info.get("effects", []))
        tk = (self._prev_state, self._prev_action_key)
        self._transitions[tk] = (next_hash, effects)

        if effects:
            self._effect_counts[self._prev_action_key] = (
                self._effect_counts.get(self._prev_action_key, 0) + 1
            )
