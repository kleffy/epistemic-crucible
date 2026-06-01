from __future__ import annotations

import json

from crucible.actions import DIRECTION_DELTA, Action, ActionKind, Direction, is_legal
from crucible.counterfactuals import stable_state_hash
from crucible.grammar import GoalSpec, TaskSpec, build_world_from_spec, generate_task
from crucible.interventions import execute_intervention
from crucible.observations import observe
from crucible.relations import Relation, RelationKind
from crucible.rewards import RewardConfig, compute_step_reward, navigation_potential
from crucible.rules import CausalProvenance, resolve_pending_effects
from crucible.splits import SplitLabel
from crucible.utils.logging import get_logger
from crucible.utils.serialization import to_dict
from crucible.world import WorldState, generate_world

_log = get_logger(__name__)


class CrucibleEnv:
    """Minimal symbolic environment for Crucible-Micro.

    step() returns (observation, reward, done, info) following Gymnasium conventions.
    Reward is always 0 in Phase 1; task-specific rewards are added in later phases.
    """

    def __init__(self, seed: int, config: dict | None = None) -> None:
        self._seed = seed
        cfg = config or {}
        self._grid_size = int(cfg.get("grid_size", 6))
        self._num_objects = int(cfg.get("num_objects", 4))
        self._max_steps = int(cfg.get("max_steps", 50))
        self._task_spec = cfg.get("task_spec")
        self._task_family = cfg.get("task_family")
        self._split = cfg.get("split")
        # Opt-in RL reward layer. When disabled (default) the env stays
        # reward-free and only terminates on timeout — byte-identical to before.
        self._terminate_on_goal = bool(cfg.get("terminate_on_goal", False))
        self._reward_config = cfg.get("reward_config") or RewardConfig()
        # Decision-focused variant: cluster objects around the agent so the task
        # isolates the causal decision from long-horizon navigation.
        self._compact_layout = bool(cfg.get("compact_layout", False))
        self._active_goal: GoalSpec | None = None
        self._solved = False
        self._world: WorldState | None = None
        self._trace: list[dict] = []

    def reset(self) -> dict:
        if isinstance(self._task_spec, TaskSpec):
            self._world = build_world_from_spec(self._task_spec, compact=self._compact_layout)
            self._active_goal = self._task_spec.goal
        elif self._task_family is not None:
            split = SplitLabel(self._split) if self._split is not None else None
            spec = generate_task(self._task_family, seed=self._seed, split=split)
            self._world = build_world_from_spec(spec, compact=self._compact_layout)
            self._active_goal = spec.goal
        else:
            self._world = generate_world(
                seed=self._seed,
                grid_size=self._grid_size,
                num_objects=self._num_objects,
                max_steps=self._max_steps,
            )
            self._active_goal = None
        self._solved = False
        self._prev_potential = 0.0
        if (
            self._terminate_on_goal
            and self._active_goal is not None
            and self._reward_config.navigation_shaping
        ):
            self._prev_potential = navigation_potential(
                self._world, self._active_goal, self._grid_size
            )
        self._trace = []
        obs = observe(self._world)
        _log.debug(
            "reset seed=%d grid=%d objects=%d", self._seed, self._grid_size, self._num_objects
        )
        return obs

    def step(self, action: Action) -> tuple[dict, float, bool, dict]:
        if self._world is None:
            raise RuntimeError("Call reset() before step().")

        obs_before = observe(self._world)
        state_hash_before = stable_state_hash(self._world)
        public_state_hash_before = stable_state_hash(self._world, public=True)
        legal = is_legal(action, self._world)

        causal_provenance: list[CausalProvenance] = []
        if legal:
            effects, causal_provenance = self._apply_action(action)
        else:
            effects = ["illegal_action"]
            _log.debug("illegal action %s at step %d", action, self._world.step)

        if legal:
            self._world.step += 1
            delayed_effects, delayed_provenance = resolve_pending_effects(self._world)
            effects.extend(delayed_effects)
            causal_provenance.extend(delayed_provenance)
        done = self._world.step >= self._world.max_steps

        # Reward layer (opt-in). Default path: reward stays 0.0 and termination
        # is timeout-only, identical to the reward-free benchmark.
        reward = 0.0
        if self._terminate_on_goal and self._active_goal is not None:
            reward, solved_now = compute_step_reward(
                self._active_goal,
                self._world,
                {"legal": legal},
                self._reward_config,
                already_solved=self._solved,
            )
            if solved_now:
                self._solved = True
                done = True
            shaping = self._reward_config.navigation_shaping
            if shaping:
                new_potential = navigation_potential(
                    self._world, self._active_goal, self._grid_size
                )
                reward += shaping * (
                    self._reward_config.shaping_gamma * new_potential - self._prev_potential
                )
                self._prev_potential = new_potential

        obs_after = observe(self._world)
        state_hash_after = stable_state_hash(self._world)
        public_state_hash_after = stable_state_hash(self._world, public=True)

        self._trace.append(
            {
                "step": self._world.step,
                "action": to_dict(action),
                "obs_before": obs_before,
                "obs_after": obs_after,
                "legal": legal,
                "effects": effects,
                "causal_provenance": to_dict(causal_provenance, public=False),
                "state_hash_before": state_hash_before,
                "state_hash_after": state_hash_after,
                "public_state_hash_before": public_state_hash_before,
                "public_state_hash_after": public_state_hash_after,
            }
        )

        return obs_after, reward, done, {
            "legal": legal,
            "effects": effects,
            "solved": self._solved,
            "state_hash_before": state_hash_before,
            "state_hash_after": state_hash_after,
            "public_state_hash_before": public_state_hash_before,
            "public_state_hash_after": public_state_hash_after,
        }

    def get_trace(self) -> list[dict]:
        return list(self._trace)

    def dump_trace_jsonl(self) -> str:
        return "\n".join(json.dumps(record) for record in self._trace)

    @property
    def world(self) -> WorldState:
        if self._world is None:
            raise RuntimeError("Call reset() before accessing world.")
        return self._world

    def _apply_action(self, action: Action) -> tuple[list[str], list[CausalProvenance]]:
        world = self._world
        assert world is not None
        agent = world.agent
        args = action.args

        if action.kind == ActionKind.WAIT:
            return ["waited"], []

        if action.kind == ActionKind.MOVE:
            dr, dc = DIRECTION_DELTA[Direction(args["direction"])]
            r, c = agent.pos
            agent.pos = (r + dr, c + dc)
            _refresh_agent_adjacency(world)
            return [f"moved_to {agent.pos}"], []

        if action.kind == ActionKind.PICKUP:
            obj_id = args["obj_id"]
            obj = world.objects[obj_id]
            agent.inventory.append(obj_id)
            obj.visible.pos = None
            world.relations = [
                r for r in world.relations
                if not (r.subject == obj_id or r.object_ == obj_id)
            ]
            world.relations.append(
                Relation(kind=RelationKind.HELD, subject="agent", object_=obj_id)
            )
            return [f"picked_up {obj_id}"], []

        if action.kind == ActionKind.DROP:
            obj_id = args["obj_id"]
            obj = world.objects[obj_id]
            agent.inventory.remove(obj_id)
            obj.visible.pos = agent.pos
            world.relations = [
                r for r in world.relations
                if not (r.kind == RelationKind.HELD and r.object_ == obj_id)
            ]
            _refresh_object_adjacency(world, obj_id)
            return [f"dropped {obj_id} at {agent.pos}"], []

        if action.kind == ActionKind.INSPECT:
            obj_id = args["obj_id"]
            obj = world.objects[obj_id]
            v = obj.visible
            desc = (
                f"{v.obj_type.value} color={v.color.value} shape={v.shape.value} "
                f"texture={v.texture.value} size={v.size.value} state={v.state.value}"
            )
            return [f"inspect:{obj_id}:{desc}"], []

        if action.kind in (
            ActionKind.APPLY,
            ActionKind.COMBINE,
            ActionKind.SWAP,
            ActionKind.ISOLATE,
            ActionKind.PREDICT,
        ):
            result = execute_intervention(world, action)
            return result.public_effects, result.causal_provenance

        return [], []


def _refresh_agent_adjacency(world: WorldState) -> None:
    world.relations = [
        r for r in world.relations if r.subject != "agent" and r.object_ != "agent"
    ]
    agent_pos = world.agent.pos
    for obj_id, obj in world.objects.items():
        if obj.visible.pos is None:
            continue
        if abs(obj.visible.pos[0] - agent_pos[0]) + abs(obj.visible.pos[1] - agent_pos[1]) == 1:
            world.relations.append(
                Relation(kind=RelationKind.ADJACENT, subject="agent", object_=obj_id)
            )
            world.relations.append(
                Relation(kind=RelationKind.ADJACENT, subject=obj_id, object_="agent")
            )


def _refresh_object_adjacency(world: WorldState, obj_id: str) -> None:
    obj = world.objects[obj_id]
    if obj.visible.pos is None:
        return
    for other_id, other in world.objects.items():
        if other_id == obj_id or other.visible.pos is None:
            continue
        dist = (
            abs(obj.visible.pos[0] - other.visible.pos[0])
            + abs(obj.visible.pos[1] - other.visible.pos[1])
        )
        if dist == 1:
            world.relations.append(
                Relation(kind=RelationKind.ADJACENT, subject=obj_id, object_=other_id)
            )
            world.relations.append(
                Relation(kind=RelationKind.ADJACENT, subject=other_id, object_=obj_id)
            )
    agent_pos = world.agent.pos
    if (
        abs(obj.visible.pos[0] - agent_pos[0])
        + abs(obj.visible.pos[1] - agent_pos[1])
        == 1
    ):
        world.relations.append(
            Relation(kind=RelationKind.ADJACENT, subject="agent", object_=obj_id)
        )
        world.relations.append(
            Relation(kind=RelationKind.ADJACENT, subject=obj_id, object_="agent")
        )
