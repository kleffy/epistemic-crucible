from __future__ import annotations

from crucible.actions import Action, ActionKind
from crucible.agents.base import Agent
from crucible.utils.logging import get_logger

_log = get_logger(__name__)

# Lookup key type: (family_str, seed_int, split_str)
_MemKey = tuple[str, int, str]


class MemorizationAgent(Agent):
    """Store-and-replay agent that memorizes solutions from prior rollouts.

    The runner (or caller) populates the memory via store() before evaluation.
    During act(), the agent replays the stored action sequence for the current
    task key. Falls back to WAIT when the task is unknown.

    Key convention: (family.value, seed, split.value)
    """

    name = "memorization"

    def __init__(self) -> None:
        self._memory: dict[_MemKey, list[Action]] = {}
        self._current_key: _MemKey | None = None
        self._replay_idx: int = 0

    def store(self, key: _MemKey, action_sequence: list[Action]) -> None:
        """Store an action sequence for retrieval during episodes with matching key."""
        self._memory[key] = list(action_sequence)
        _log.debug("memorization: stored %d actions for key %s", len(action_sequence), key)

    def set_episode_key(self, key: _MemKey) -> None:
        """Call before reset() / before the episode begins to set the lookup key."""
        self._current_key = key

    def reset(self) -> None:
        self._replay_idx = 0

    def act(self, obs: dict) -> Action:
        if self._current_key is None or self._current_key not in self._memory:
            _log.debug("memorization: no memory for key %s, returning WAIT", self._current_key)
            return Action(kind=ActionKind.WAIT, args={})

        sequence = self._memory[self._current_key]
        if self._replay_idx >= len(sequence):
            _log.debug("memorization: sequence exhausted, returning WAIT")
            return Action(kind=ActionKind.WAIT, args={})

        action = sequence[self._replay_idx]
        self._replay_idx += 1
        _log.debug("memorization: replaying step %d: %s", self._replay_idx, action.kind.value)
        return action
