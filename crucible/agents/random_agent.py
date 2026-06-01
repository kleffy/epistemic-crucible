from __future__ import annotations

from crucible.actions import Action
from crucible.agents.base import Agent, enumerate_candidate_actions
from crucible.utils.logging import get_logger
from crucible.utils.seeding import make_python_rng

_log = get_logger(__name__)


class RandomAgent(Agent):
    """Uniform-random policy over plausible candidate actions.

    Candidate actions are derived solely from the public observation;
    the environment enforces legality. Deterministic per seed.
    """

    name = "random"

    def __init__(self, seed: int = 0, grid_size: int = 6) -> None:
        self._seed = seed
        self._grid_size = grid_size
        self._rng = make_python_rng(seed)

    def reset(self) -> None:
        self._rng = make_python_rng(self._seed)

    def act(self, obs: dict) -> Action:
        candidates = enumerate_candidate_actions(obs, self._grid_size)
        action = self._rng.choice(candidates)
        _log.debug("random_agent selected %s", action.kind.value)
        return action
