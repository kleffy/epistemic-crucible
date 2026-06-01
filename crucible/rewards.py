"""Task reward signal for RL training.

The base environment is reward-free by design (``CrucibleEnv.step`` returns 0.0
and only terminates on timeout). That keeps the diagnostic traces and the
symbolic baselines untouched. Reinforcement-learning agents, however, need a
scalar signal, so this module defines the task reward as *environment logic*
(hence it lives in ``crucible/`` per the architecture boundary).

The default reward is purely sparse: +``goal_reward`` on the first step the task
goal predicate becomes true, and 0 otherwise. Optional shaping penalties are
provided for harder families but are off by default, so enabling rewards never
changes the meaning of the benchmark — only the learning signal handed to a
trainer that explicitly opts in.
"""

from __future__ import annotations

from dataclasses import dataclass

from crucible.grammar import GoalSpec, check_goal
from crucible.objects import ObjectType
from crucible.world import WorldState


@dataclass
class RewardConfig:
    """Reward shaping parameters. Defaults give a pure sparse goal reward."""

    goal_reward: float = 1.0
    step_penalty: float = 0.0  # subtracted every step (>= 0)
    illegal_penalty: float = 0.0  # subtracted when an action is illegal (>= 0)
    # Potential-based navigation shaping coefficient. Uses only visible object
    # positions (guide toward any tool, then the goal object), so it aids
    # exploration WITHOUT revealing which tool is causal — the tool-selection
    # decision that distinguishes shortcut from causal learning stays unshaped.
    navigation_shaping: float = 0.0
    shaping_gamma: float = 0.99


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def navigation_potential(world: WorldState, goal: GoalSpec, grid_size: int) -> float:
    """Public navigation potential for shaping (never reads hidden properties).

    Before a tool is held: negative normalized distance to the nearest free
    tool. After a tool is held: negative normalized distance to goal-adjacency.
    Higher (closer to zero) is better. Returns 0 when there is nothing to guide
    toward, so non-navigation families are unaffected.
    """
    agent_pos = world.agent.pos
    inv = world.agent.inventory
    holds_tool = any(
        i in world.objects and world.objects[i].visible.obj_type == ObjectType.TOOL
        for i in inv
    )
    if not holds_tool:
        tool_positions = [
            o.visible.pos
            for o in world.objects.values()
            if o.visible.obj_type == ObjectType.TOOL and o.visible.pos is not None
        ]
        if not tool_positions:
            return 0.0
        d = min(_manhattan(agent_pos, p) for p in tool_positions)
    else:
        gate = world.objects.get(goal.target_obj_id or "")
        if gate is None or gate.visible.pos is None:
            return 0.0
        d = max(0, _manhattan(agent_pos, gate.visible.pos) - 1)  # apply needs adjacency
    return -float(d) / max(grid_size, 1)


def goal_reached(goal: GoalSpec, world: WorldState) -> bool:
    """True iff the task goal predicate holds in the current visible world state.

    Thin wrapper over :func:`crucible.grammar.check_goal` so RL code does not
    reach past the public goal predicate. Note that ``CLASSIFY`` goals
    (counterfactual family) are not checkable from visible state and so never
    yield reward — the neural baseline learns the OPEN-goal families.
    """
    return check_goal(goal, world)


def compute_step_reward(
    goal: GoalSpec,
    world: WorldState,
    info: dict,
    config: RewardConfig,
    *,
    already_solved: bool,
) -> tuple[float, bool]:
    """Return ``(reward, solved_now)`` for the step that just completed.

    ``reward`` is ``goal_reward`` on the first step the goal is satisfied, plus
    optional shaping penalties. ``already_solved`` guards against awarding the
    goal reward more than once within an episode.
    """
    reward = -config.step_penalty
    if not info.get("legal", True):
        reward -= config.illegal_penalty
    solved_now = False
    if not already_solved and goal_reached(goal, world):
        reward += config.goal_reward
        solved_now = True
    return reward, solved_now
