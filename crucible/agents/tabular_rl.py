from __future__ import annotations

from crucible.actions import Action
from crucible.agents.base import Agent, enumerate_candidate_actions
from crucible.utils.logging import get_logger
from crucible.utils.seeding import make_python_rng

_log = get_logger(__name__)


class TabularRLAgent(Agent):
    """Epsilon-greedy Q-learning agent with public-state-hash as state key.

    Q-table is a plain dict: state_key → list[float] (one value per candidate
    action index). State keys are `public_state_hash_before` from info.
    The table persists across episodes; each episode is treated as a new
    experience trajectory.

    With reward=0.0 (Phase 1-4) the table stays at its initial values,
    making this equivalent to random. It becomes meaningful once reward
    signals are introduced in later phases.
    """

    name = "tabular_rl"

    def __init__(
        self,
        seed: int = 0,
        grid_size: int = 6,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
    ) -> None:
        self._seed = seed
        self._grid_size = grid_size
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._rng = make_python_rng(seed)
        # Q-table: state_key → {action_idx: float}
        self._q: dict[str, dict[int, float]] = {}
        # Per-step tracking for TD update
        self._prev_state: str | None = None
        self._prev_action_idx: int | None = None
        self._prev_candidates: list[Action] = []

    def reset(self) -> None:
        self._prev_state = None
        self._prev_action_idx = None
        self._prev_candidates = []

    def act(self, obs: dict) -> Action:
        candidates = enumerate_candidate_actions(obs, self._grid_size)
        # Use public_state_hash injected into obs by the runner; fallback to str(obs).
        state_key = obs.get("_public_hash", str(sorted(obs.get("agent", {}).items())))

        self._prev_candidates = candidates
        self._prev_state = state_key

        if self._rng.random() < self._epsilon or state_key not in self._q:
            # Explore: pick uniformly
            idx = self._rng.randint(0, len(candidates) - 1)
        else:
            q_row = self._q[state_key]
            idx = max(range(len(candidates)), key=lambda i: q_row.get(i, 0.0))

        self._prev_action_idx = idx
        action = candidates[idx]
        _log.debug("tabular_rl: %s (ε-greedy idx=%d)", action.kind.value, idx)
        return action

    def observe_result(self, obs: dict, reward: float, done: bool, info: dict) -> None:
        if self._prev_state is None or self._prev_action_idx is None:
            return

        next_state_key = info.get("public_state_hash_after", "")
        if next_state_key not in self._q:
            self._q[next_state_key] = {}

        max_next_q = max(self._q[next_state_key].values(), default=0.0)

        prev_q = self._q.setdefault(self._prev_state, {})
        current = prev_q.get(self._prev_action_idx, 0.0)
        td_target = reward + (0.0 if done else self._gamma * max_next_q)
        prev_q[self._prev_action_idx] = current + self._alpha * (td_target - current)
