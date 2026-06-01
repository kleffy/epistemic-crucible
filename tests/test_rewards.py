"""Tests for the opt-in RL reward layer (crucible/rewards.py + env flag).

The task solution certificates are logical proofs without navigation steps (a
known "oracle gap"), so replaying them does not open the gate. These tests
therefore drive the world to the goal state directly to exercise the reward
semantics, which is exactly what the env checks after each step.
"""

from __future__ import annotations

from crucible.actions import Action, ActionKind
from crucible.env import CrucibleEnv
from crucible.grammar import TaskFamily, build_world_from_spec, generate_task
from crucible.objects import ObjectState
from crucible.rewards import RewardConfig, compute_step_reward, goal_reached
from crucible.splits import SplitLabel


def _affordance_spec(seed: int = 3):
    return generate_task(TaskFamily.AFFORDANCE, seed=seed, split=SplitLabel.TRAIN)


def _open_gate(world, goal) -> None:
    world.objects[goal.target_obj_id].visible.state = ObjectState.OPEN


def test_goal_reached_and_reward_semantics():
    """goal_reached tracks the gate state; reward fires once, then never again."""
    spec = _affordance_spec()
    world = build_world_from_spec(spec)
    cfg = RewardConfig()

    assert goal_reached(spec.goal, world) is False
    r, solved = compute_step_reward(spec.goal, world, {"legal": True}, cfg, already_solved=False)
    assert r == 0.0 and solved is False

    _open_gate(world, spec.goal)
    assert goal_reached(spec.goal, world) is True
    r, solved = compute_step_reward(spec.goal, world, {"legal": True}, cfg, already_solved=False)
    assert r == 1.0 and solved is True
    # Once solved, the goal reward is not paid again.
    r, solved = compute_step_reward(spec.goal, world, {"legal": True}, cfg, already_solved=True)
    assert r == 0.0 and solved is False


def test_env_emits_reward_and_terminates_on_goal():
    """With the flag on, reaching the goal yields reward 1.0 and done=True once."""
    spec = _affordance_spec()
    env = CrucibleEnv(seed=spec.seed, config={"task_spec": spec, "terminate_on_goal": True})
    env.reset()

    # Simulate that prior actions opened the gate, then take one more step.
    _open_gate(env.world, spec.goal)
    _, reward, done, _ = env.step(Action(kind=ActionKind.WAIT, args={}))

    assert reward == 1.0
    assert done is True
    assert env._solved is True


def test_flag_off_is_reward_free_and_timeout_only():
    """Default env (flag off) returns reward 0.0 even when the goal is satisfied."""
    spec = _affordance_spec()
    env = CrucibleEnv(seed=spec.seed, config={"task_spec": spec})  # no terminate_on_goal
    env.reset()
    _open_gate(env.world, spec.goal)
    _, reward, done, _ = env.step(Action(kind=ActionKind.WAIT, args={}))

    assert reward == 0.0, "reward must stay 0.0 when the reward layer is disabled"
    assert done is False, "no early termination when reward layer is disabled"


def test_no_reward_for_non_solving_trajectory():
    """A trajectory of WAITs never reaches the goal and earns zero reward."""
    spec = _affordance_spec()
    env = CrucibleEnv(seed=spec.seed, config={"task_spec": spec, "terminate_on_goal": True})
    env.reset()
    rewards, dones = [], []
    for _ in range(spec.max_steps):
        _, reward, done, _ = env.step(Action(kind=ActionKind.WAIT, args={}))
        rewards.append(reward)
        dones.append(done)
        if done:
            break
    assert sum(rewards) == 0.0
    assert dones[-1] is True, "WAIT trajectory ends only via timeout"


def test_shaping_penalties_apply_when_configured():
    """Step and illegal penalties are subtracted when a RewardConfig opts in."""
    spec = _affordance_spec()
    cfg = RewardConfig(goal_reward=1.0, step_penalty=0.01, illegal_penalty=0.5)
    env = CrucibleEnv(
        seed=spec.seed,
        config={"task_spec": spec, "terminate_on_goal": True, "reward_config": cfg},
    )
    env.reset()
    # PICKUP of a nonexistent object is illegal and does not advance the world.
    _, reward, _, info = env.step(Action(kind=ActionKind.PICKUP, args={"obj_id": "nope"}))

    assert info["legal"] is False
    assert reward == -0.51, f"expected -0.51 (step+illegal), got {reward}"
