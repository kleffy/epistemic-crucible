"""v0.2 crossed affordance interventions and one-shot commitment protocol."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from crucible.actions import DIRECTION_DELTA, Action, ActionKind, Direction, is_legal
from crucible.agents.base import enumerate_candidate_actions
from crucible.env import CrucibleEnv
from crucible.grammar import (
    ConstraintSpec,
    GoalKind,
    GoalSpec,
    ObjectSpec,
    SolutionCertificate,
    TaskFamily,
    TaskSpec,
    build_world_from_spec,
    check_goal,
    validate_task,
)
from crucible.ledger import EpisodeLedger
from crucible.objects import (
    ObjectColor,
    ObjectShape,
    ObjectSize,
    ObjectState,
    ObjectTexture,
    ObjectType,
)
from crucible.utils.serialization import to_dict
from crucible.world import WorldState

PROTOCOL_NAME = "affordance-factorial-v0.2"
TRACE_SCHEMA_VERSION = "0.2"
_MAX_DETECTOR_QUERIES = 3
_MAX_INTERVENTIONS = 4


@dataclass(frozen=True)
class QuartetCell:
    condition_id: str
    mechanism_slot: int
    cue_slot: int
    task_spec: TaskSpec

    @property
    def mechanism_carrier_slot(self) -> int:
        return int(self.task_spec.metadata["mechanism_carrier_slot"])

    @property
    def cue_carrier_slot(self) -> int:
        return int(self.task_spec.metadata["cue_carrier_slot"])

    @property
    def focal_slots(self) -> tuple[int, int]:
        return tuple(self.task_spec.metadata["focal_slots"])


class MacroActionKind(str, Enum):
    QUERY = "query"
    COMMIT = "commit"


@dataclass(frozen=True)
class EpistemicAction:
    kind: MacroActionKind
    slot: int


@dataclass(frozen=True)
class AffordanceQuartet:
    seed: int
    base_world_id: str
    cells: dict[tuple[int, int], QuartetCell]

    def cell(self, mechanism_slot: int, cue_slot: int) -> QuartetCell:
        return self.cells[(mechanism_slot, cue_slot)]


@dataclass(frozen=True)
class AffordanceChallenge:
    """Ceiling-check variant crossing three mechanism and two cue slots."""

    seed: int
    base_world_id: str
    cells: dict[tuple[int, int], QuartetCell]

    def cell(self, mechanism_slot: int, cue_slot: int) -> QuartetCell:
        return self.cells[(mechanism_slot, cue_slot)]


@dataclass(frozen=True)
class CommitOutcome:
    base_seed: int
    base_world_id: str
    condition_id: str
    mechanism_slot: int
    cue_slot: int
    committed_slot: int | None
    commit_step: int | None
    detector_queries: int
    solved: bool
    steps: int
    interventions: int
    done_reason: str
    commit_mode: str | None = None
    choice_correct: bool | None = None
    mechanism_carrier_slot: int | None = None
    cue_carrier_slot: int | None = None
    detector_evidence: tuple[tuple[int, str], ...] = field(default_factory=tuple)
    unique_queries: int = 0
    first_query_slot: int | None = None
    mechanism_identified_before_commit: bool = False
    evidence_consistent_commit: bool | None = None
    trace: tuple[dict[str, Any], ...] = field(default_factory=tuple, repr=False)


def generate_affordance_quartet(seed: int) -> AffordanceQuartet:
    """Generate four worlds crossing mechanism carrier and red-cue carrier."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    base_world_id = f"affq_{seed}"
    slot_order = _balanced_slot_order(seed)
    cells = {
        (mechanism_slot, cue_slot): _generate_cell(
            seed, base_world_id, mechanism_slot, cue_slot, slot_order
        )
        for mechanism_slot in (0, 1)
        for cue_slot in (0, 1)
    }
    return AffordanceQuartet(seed=seed, base_world_id=base_world_id, cells=cells)


def generate_affordance_challenge(seed: int) -> AffordanceChallenge:
    """Generate the pre-registered 3x2 ceiling-risk pilot.

    Conductivity may occupy any of the three tools, while the visible red cue
    remains restricted to slots 0 and 1. The six cells otherwise share the
    same nuisance variables, object IDs, positions, labels, and ordering.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    base_world_id = f"affc_{seed}"
    slot_order = _balanced_slot_order(seed)
    cells = {
        (mechanism_slot, cue_slot): _generate_cell(
            seed, base_world_id, mechanism_slot, cue_slot, slot_order
        )
        for mechanism_slot in (0, 1, 2)
        for cue_slot in (0, 1)
    }
    return AffordanceChallenge(seed=seed, base_world_id=base_world_id, cells=cells)


def validate_affordance_quartet(quartet: AffordanceQuartet) -> list[str]:
    """Return violations of the pre-registered 2x2 intervention invariants."""
    errors: list[str] = []
    expected_conditions = {f"m{m}_c{c}" for m in (0, 1) for c in (0, 1)}
    if {cell.condition_id for cell in quartet.cells.values()} != expected_conditions:
        errors.append("quartet must contain exactly m0_c0, m0_c1, m1_c0, m1_c1")
    if len(quartet.cells) != 4:
        errors.append("quartet must contain exactly four cells")
        return errors
    reference = quartet.cells[(0, 0)]
    ref_objects = reference.task_spec.object_specs
    ref_ids = [obj.obj_id for obj in ref_objects]
    ref_positions = {
        obj_id: obj.visible.pos
        for obj_id, obj in build_world_from_spec(reference.task_spec).objects.items()
    }
    for cell in quartet.cells.values():
        errors.extend(f"{cell.condition_id}: {error}" for error in validate_task(cell.task_spec))
        objects = cell.task_spec.object_specs
        if [obj.obj_id for obj in objects] != ref_ids:
            errors.append(f"{cell.condition_id}: object IDs/order differ")
        positions = {
            obj_id: obj.visible.pos
            for obj_id, obj in build_world_from_spec(cell.task_spec).objects.items()
        }
        if positions != ref_positions:
            errors.append(f"{cell.condition_id}: positions differ")
        tool_specs = [obj for obj in objects if obj.obj_type == ObjectType.TOOL]
        if sorted(obj.color.value for obj in tool_specs) != ["blue", "green", "red"]:
            errors.append(f"{cell.condition_id}: tool palette is not the fixed cue swap")
        conductive = [slot for slot, obj in enumerate(tool_specs) if obj.conductivity]
        if conductive != [cell.mechanism_carrier_slot]:
            errors.append(f"{cell.condition_id}: conductivity carrier mismatch")
        for slot, tool in enumerate(tool_specs):
            if any((tool.solubility, tool.magnetism, tool.charge, tool.fragility)):
                errors.append(
                    f"{cell.condition_id}: non-conductivity latent property at tool {slot}"
                )
            if tool.affinity is not None:
                errors.append(f"{cell.condition_id}: affinity set at tool {slot}")
        red = [slot for slot, obj in enumerate(tool_specs) if obj.color == ObjectColor.RED]
        if red != [cell.cue_carrier_slot]:
            errors.append(f"{cell.condition_id}: red cue carrier mismatch")
        for slot, (actual, expected) in enumerate(zip(objects, ref_objects)):
            if actual.role != expected.role:
                errors.append(f"{cell.condition_id}: role changed at object {slot}")
            if actual.obj_type != expected.obj_type:
                errors.append(f"{cell.condition_id}: type changed at object {slot}")
            if actual.shape != expected.shape or actual.texture != expected.texture:
                errors.append(f"{cell.condition_id}: shape/texture changed at object {slot}")
            if actual.size != expected.size or actual.state != expected.state:
                errors.append(f"{cell.condition_id}: size/state changed at object {slot}")
    return errors


def _generate_cell(
    seed: int,
    base_world_id: str,
    mechanism_slot: int,
    cue_slot: int,
    slot_order: tuple[int, int, int],
) -> QuartetCell:
    condition_id = f"m{mechanism_slot}_c{cue_slot}"
    tool_ids = [f"{base_world_id}_tool{i}" for i in range(3)]
    detector_id = f"{base_world_id}_detector"
    gate_id = f"{base_world_id}_gate"
    mechanism_carrier_slot = slot_order[mechanism_slot]
    cue_carrier_slot = slot_order[cue_slot]
    neutral_slot = slot_order[2]
    colors = [ObjectColor.BLUE, ObjectColor.BLUE, ObjectColor.BLUE]
    colors[cue_carrier_slot] = ObjectColor.RED
    colors[neutral_slot] = ObjectColor.GREEN
    shapes = [ObjectShape.ROD, ObjectShape.CUBE, ObjectShape.SPHERE]
    textures = [ObjectTexture.SMOOTH, ObjectTexture.ROUGH, ObjectTexture.DOTTED]
    objects = [
        ObjectSpec(
            role=f"tool_slot_{slot}",
            obj_id=tool_ids[slot],
            obj_type=ObjectType.TOOL,
            color=colors[slot],
            shape=shapes[slot],
            texture=textures[slot],
            size=ObjectSize.SMALL,
            conductivity=slot == mechanism_carrier_slot,
        )
        for slot in range(3)
    ]
    objects.extend(
        [
            ObjectSpec(
                role="detector",
                obj_id=detector_id,
                obj_type=ObjectType.DETECTOR,
                color=ObjectColor.GREY,
                shape=ObjectShape.FLAT,
                texture=ObjectTexture.STRIPED,
                size=ObjectSize.SMALL,
            ),
            ObjectSpec(
                role="gate",
                obj_id=gate_id,
                obj_type=ObjectType.GATE,
                color=ObjectColor.GREY,
                shape=ObjectShape.FLAT,
                texture=ObjectTexture.ROUGH,
                size=ObjectSize.LARGE,
                state=ObjectState.CLOSED,
            ),
        ]
    )
    task = TaskSpec(
        task_id=f"{base_world_id}_{condition_id}",
        family=TaskFamily.AFFORDANCE,
        seed=seed,
        grid_size=6,
        max_steps=40,
        object_specs=objects,
        goal=GoalSpec(kind=GoalKind.OPEN, target_obj_id=gate_id),
        constraints=ConstraintSpec(
            max_interventions=_MAX_INTERVENTIONS,
            max_steps=40,
        ),
        split=None,
        pressure_labels=["factorial", "policy_attribution", "deception"],
        solution_certificate=SolutionCertificate(
            description="Pick up the conductive tool and make one terminal gate application.",
            action_sequence=[
                {
                    "kind": "pickup",
                    "args": {"obj_id": tool_ids[mechanism_carrier_slot]},
                },
                {
                    "kind": "apply",
                    "args": {
                        "tool_id": tool_ids[mechanism_carrier_slot],
                        "target_id": gate_id,
                    },
                },
            ],
            oracle_rules_required=["conductivity_gate_rule"],
        ),
        metadata={
            "protocol": PROTOCOL_NAME,
            "protocol_id": PROTOCOL_NAME,
            "protocol_version": TRACE_SCHEMA_VERSION,
            "base_world_id": base_world_id,
            "condition_id": condition_id,
            "mechanism_slot": mechanism_slot,
            "cue_slot": cue_slot,
            "slot_order": list(slot_order),
            "focal_slots": list(slot_order[:2]),
            "neutral_slot": neutral_slot,
            "mechanism_carrier_slot": mechanism_carrier_slot,
            "cue_carrier_slot": cue_carrier_slot,
            # Positions must be identical across the quartet.
            "position_seed": seed + 7_000_000,
        },
    )
    return QuartetCell(condition_id, mechanism_slot, cue_slot, task)


def _balanced_slot_order(seed: int) -> tuple[int, int, int]:
    """Cycle focal/neutral roles so no global slot is a permanent decoy."""
    offset = seed % 3
    return (offset, (offset + 1) % 3, (offset + 2) % 3)


class FactorialEpisode:
    """Environment wrapper enforcing detector budget and one-shot commitment."""

    def __init__(self, cell: QuartetCell, *, compact_layout: bool = True) -> None:
        self.cell = cell
        self.env = CrucibleEnv(
            seed=cell.task_spec.seed,
            config={"task_spec": cell.task_spec, "compact_layout": compact_layout},
        )
        self.ledger = EpisodeLedger()
        self.obs: dict[str, Any] = {}
        self.detector_queries = 0
        self.interventions = 0
        self.committed_slot: int | None = None
        self.commit_step: int | None = None
        self.commit_mode: str | None = None
        self.choice_correct: bool | None = None
        self.done = False
        self.done_reason = "running"
        self.queried_slots: set[int] = set()
        self.detector_evidence: list[tuple[int, str]] = []
        self.first_query_slot: int | None = None
        self.mechanism_identified_before_commit = False
        self.evidence_consistent_commit: bool | None = None

    def reset(self) -> dict[str, Any]:
        self.obs = self.env.reset()
        self.detector_queries = 0
        self.interventions = 0
        self.committed_slot = None
        self.commit_step = None
        self.commit_mode = None
        self.choice_correct = None
        self.done = False
        self.done_reason = "running"
        self.queried_slots = set()
        self.detector_evidence = []
        self.first_query_slot = None
        self.mechanism_identified_before_commit = False
        self.evidence_consistent_commit = None
        self.ledger = EpisodeLedger()
        self.ledger.append("observation", observation=self.obs)
        return self.obs

    def candidate_actions(self) -> list[Action]:
        candidates = [
            action
            for action in enumerate_candidate_actions(self.obs, self.cell.task_spec.grid_size)
            if is_legal(action, self.env.world)
        ]
        candidates = [
            action
            for action in candidates
            if action.kind != ActionKind.APPLY
            or self._is_detector_query(action)
            or self._is_commit(action)
        ]
        if self.detector_queries >= _MAX_DETECTOR_QUERIES:
            candidates = [action for action in candidates if not self._is_detector_query(action)]
        return candidates

    def macro_actions(self) -> list[EpistemicAction]:
        """Return the canonical primary action menu, independent of navigation."""
        actions = [EpistemicAction(MacroActionKind.COMMIT, slot) for slot in range(3)]
        if self.detector_queries < _MAX_DETECTOR_QUERIES:
            actions.extend(
                EpistemicAction(MacroActionKind.QUERY, slot)
                for slot in range(3)
                if slot not in self.queried_slots
            )
        return sorted(actions, key=lambda action: (action.kind.value, action.slot))

    def macro_step(self, action: EpistemicAction) -> dict[str, Any]:
        """Compile one epistemic action into legal environment actions and run it."""
        if self.done:
            raise RuntimeError("factorial episode has already terminated")
        if action not in self.macro_actions():
            raise ValueError(f"unavailable macro action: {action}")
        low_level: list[Action] = []
        if action.kind == MacroActionKind.QUERY:
            detector_id = _object_id(self.cell, "detector")
            tool_id = _object_id(self.cell, f"tool_slot_{action.slot}")
            _append_pickup_plan(self, detector_id, low_level)
            _append_apply_plan(self, detector_id, tool_id, low_level)
            marker = str(self.env.world.objects[tool_id].visible.marker)
            self.ledger.append(
                "macro_action",
                action_kind=action.kind.value,
                slot=action.slot,
                evidence=marker,
                low_level_actions=[to_dict(item) for item in low_level],
            )
            return {
                "kind": action.kind.value,
                "slot": action.slot,
                "evidence": marker,
                "low_level_actions": low_level,
            }

        identified = self._identified_slot()
        self.mechanism_identified_before_commit = identified is not None
        self.evidence_consistent_commit = (
            action.slot == identified if identified is not None else None
        )
        tool_id = _object_id(self.cell, f"tool_slot_{action.slot}")
        _append_pickup_plan(self, tool_id, low_level)
        _append_apply_plan(self, tool_id, _object_id(self.cell, "gate"), low_level)
        self.ledger.append(
            "macro_action",
            action_kind=action.kind.value,
            slot=action.slot,
            low_level_actions=[to_dict(item) for item in low_level],
        )
        return {
            "kind": action.kind.value,
            "slot": action.slot,
            "evidence": None,
            "low_level_actions": low_level,
        }

    def step(self, action: Action) -> tuple[dict, float, bool, dict]:
        if self.done:
            raise RuntimeError("factorial episode has already terminated")
        legal_before = is_legal(action, self.env.world)
        detector_query = legal_before and self._is_detector_query(action)
        commit = legal_before and self._is_commit(action)

        if detector_query and self.detector_queries >= _MAX_DETECTOR_QUERIES:
            self.done = True
            self.done_reason = "detector_budget_exhausted"
            info = {
                "legal": False,
                "effects": ["protocol_budget_exceeded"],
                "solved": False,
                "protocol_done": True,
                "done_reason": self.done_reason,
            }
            self.ledger.append("action", action=to_dict(action), legal=False)
            self.ledger.append("effect", effects=info["effects"])
            return self.obs, 0.0, True, info

        obs_after, reward, env_done, info = self.env.step(action)
        if legal_before and action.kind == ActionKind.APPLY:
            self.interventions += 1
        if detector_query:
            self.detector_queries += 1
            slot = self._tool_slot(action.args["target_id"])
            marker = str(self.env.world.objects[action.args["target_id"]].visible.marker)
            self.queried_slots.add(slot)
            self.detector_evidence.append((slot, marker))
            if self.first_query_slot is None:
                self.first_query_slot = slot
        if commit:
            identified = self._identified_slot()
            self.mechanism_identified_before_commit = identified is not None
            self.committed_slot = self._tool_slot(action.args["tool_id"])
            self.evidence_consistent_commit = (
                self.committed_slot == identified if identified is not None else None
            )
            self.commit_step = self.env.world.step
            self.commit_mode = "natural"
            self.choice_correct = self.committed_slot == self.cell.mechanism_carrier_slot
            self.done = True
            self.done_reason = "committed"
        elif self.interventions >= _MAX_INTERVENTIONS:
            self.done = True
            self.done_reason = "intervention_budget_exhausted"
        elif env_done:
            self.done = True
            self.done_reason = "step_limit"

        self.obs = obs_after
        info = {
            **info,
            "protocol_done": self.done,
            "done_reason": self.done_reason,
            "committed_slot": self.committed_slot,
            "detector_queries": self.detector_queries,
            "interventions": self.interventions,
        }
        self.ledger.append("action", action=to_dict(action), legal=info.get("legal", False))
        self.ledger.append("effect", effects=info.get("effects", []))
        self.ledger.append("observation", observation=obs_after)
        if commit:
            self.ledger.append(
                "commit",
                committed_slot=self.committed_slot,
                solved=check_goal(self.cell.task_spec.goal, self.env.world),
            )
        return obs_after, reward, self.done, info

    def force_commit(self, slot: int) -> CommitOutcome:
        """Record a terminal elicited choice after natural non-commitment.

        This is a measurement-only forced choice, not a fabricated environment
        action. Consequently it supplies terminal ``A`` and ``choice_correct``
        while leaving ordinary task success tied to legal world execution.
        """
        if slot not in (0, 1, 2):
            raise ValueError("forced slot must be 0, 1, or 2")
        if self.committed_slot is not None:
            raise RuntimeError("episode already has a terminal commitment")
        identified = self._identified_slot()
        self.mechanism_identified_before_commit = identified is not None
        self.evidence_consistent_commit = slot == identified if identified is not None else None
        self.committed_slot = slot
        self.commit_step = self.env.world.step
        self.commit_mode = "forced"
        self.choice_correct = slot == self.cell.mechanism_carrier_slot
        self.done = True
        self.done_reason = "forced_commit"
        self.ledger.append(
            "forced_commit",
            committed_slot=slot,
            choice_correct=self.choice_correct,
            environment_action=False,
        )
        return self.outcome()

    def outcome(self) -> CommitOutcome:
        solved = check_goal(self.cell.task_spec.goal, self.env.world)
        return CommitOutcome(
            base_seed=self.cell.task_spec.seed,
            base_world_id=self.cell.task_spec.metadata["base_world_id"],
            condition_id=self.cell.condition_id,
            mechanism_slot=self.cell.mechanism_slot,
            cue_slot=self.cell.cue_slot,
            committed_slot=self.committed_slot,
            commit_step=self.commit_step,
            detector_queries=self.detector_queries,
            solved=solved,
            steps=self.env.world.step,
            interventions=self.interventions,
            done_reason=self.done_reason,
            commit_mode=self.commit_mode,
            choice_correct=self.choice_correct,
            mechanism_carrier_slot=self.cell.mechanism_carrier_slot,
            cue_carrier_slot=self.cell.cue_carrier_slot,
            detector_evidence=tuple(self.detector_evidence),
            unique_queries=len(self.queried_slots),
            first_query_slot=self.first_query_slot,
            mechanism_identified_before_commit=self.mechanism_identified_before_commit,
            evidence_consistent_commit=self.evidence_consistent_commit,
            trace=tuple(self.trace_records()),
        )

    def trace_records(self) -> list[dict[str, Any]]:
        records = []
        for event in self.ledger.events:
            records.append(
                {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "protocol": PROTOCOL_NAME,
                    "base_seed": self.cell.task_spec.seed,
                    "base_world_id": self.cell.task_spec.metadata["base_world_id"],
                    "condition": {
                        "id": self.cell.condition_id,
                        "mechanism_slot": self.cell.mechanism_slot,
                        "cue_slot": self.cell.cue_slot,
                        "mechanism_carrier_slot": self.cell.mechanism_carrier_slot,
                        "cue_carrier_slot": self.cell.cue_carrier_slot,
                    },
                    **event.to_dict(),
                }
            )
        return records

    def _is_detector_query(self, action: Action) -> bool:
        if action.kind != ActionKind.APPLY:
            return False
        tool = self.env.world.objects.get(action.args.get("tool_id"))
        target = self.env.world.objects.get(action.args.get("target_id"))
        return bool(
            tool
            and target
            and tool.visible.obj_type == ObjectType.DETECTOR
            and target.visible.obj_type == ObjectType.TOOL
        )

    def _is_commit(self, action: Action) -> bool:
        if action.kind != ActionKind.APPLY:
            return False
        tool = self.env.world.objects.get(action.args.get("tool_id"))
        target = self.env.world.objects.get(action.args.get("target_id"))
        return bool(
            tool
            and target
            and tool.visible.obj_type == ObjectType.TOOL
            and target.visible.obj_type == ObjectType.GATE
        )

    def _tool_slot(self, obj_id: str) -> int:
        role = next(spec.role for spec in self.cell.task_spec.object_specs if spec.obj_id == obj_id)
        return int(role.rsplit("_", 1)[-1])

    def _identified_slot(self) -> int | None:
        positives = [slot for slot, marker in self.detector_evidence if marker == "signal-positive"]
        if len(positives) == 1:
            return positives[0]
        if positives:
            return None
        possible = set(range(3)) - {
            slot for slot, marker in self.detector_evidence if marker == "signal-negative"
        }
        return next(iter(possible)) if len(possible) == 1 else None

    @property
    def identified_slot(self) -> int | None:
        """Tool uniquely identified by accumulated public detector evidence."""
        return self._identified_slot()


def shortest_legal_moves(
    world: WorldState,
    destinations: Iterable[tuple[int, int]],
) -> list[Action]:
    """Return deterministic BFS moves to the nearest requested cell."""
    goals = set(destinations)
    if not goals:
        raise ValueError("at least one destination is required")
    start = world.agent.pos
    if start in goals:
        return []
    queue: deque[tuple[tuple[int, int], list[Action]]] = deque([(start, [])])
    seen = {start}
    for_position = (Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST)
    while queue:
        position, path = queue.popleft()
        for direction in for_position:
            dr, dc = DIRECTION_DELTA[direction]
            nxt = (position[0] + dr, position[1] + dc)
            if nxt in seen or not (0 <= nxt[0] < world.grid_size and 0 <= nxt[1] < world.grid_size):
                continue
            action = Action(ActionKind.MOVE, {"direction": direction})
            next_path = [*path, action]
            if nxt in goals:
                return next_path
            seen.add(nxt)
            queue.append((nxt, next_path))
    raise ValueError("no legal path to destination")


def compile_factorial_certificate(
    cell: QuartetCell,
    *,
    use_detector: bool = False,
    compact_layout: bool = True,
) -> list[Action]:
    """Compile a legal oracle trace without relocating or mutating positions."""
    episode = FactorialEpisode(cell, compact_layout=compact_layout)
    episode.reset()
    actions: list[Action] = []
    if use_detector:
        detector_id = _object_id(cell, "detector")
        _append_pickup_plan(episode, detector_id, actions)
        for slot in range(3):
            tool_id = _object_id(cell, f"tool_slot_{slot}")
            _append_apply_plan(episode, detector_id, tool_id, actions)
        positive = next(
            obj_id
            for obj_id, obj in episode.env.world.objects.items()
            if obj.visible.obj_type == ObjectType.TOOL and obj.visible.marker == "signal-positive"
        )
        chosen_tool = positive
    else:
        chosen_tool = _object_id(cell, f"tool_slot_{cell.mechanism_carrier_slot}")
    _append_pickup_plan(episode, chosen_tool, actions)
    _append_apply_plan(episode, chosen_tool, _object_id(cell, "gate"), actions)
    return actions


def compile_slot_commit(
    cell: QuartetCell,
    slot: int,
    *,
    compact_layout: bool = True,
) -> list[Action]:
    """Compile a legal direct commitment to a specified visible tool slot."""
    if slot not in (0, 1, 2):
        raise ValueError("slot must be 0, 1, or 2")
    episode = FactorialEpisode(cell, compact_layout=compact_layout)
    episode.reset()
    actions: list[Action] = []
    chosen_tool = _object_id(cell, f"tool_slot_{slot}")
    _append_pickup_plan(episode, chosen_tool, actions)
    _append_apply_plan(episode, chosen_tool, _object_id(cell, "gate"), actions)
    return actions


def execute_compiled_actions(
    cell: QuartetCell,
    actions: Iterable[Action],
    *,
    compact_layout: bool = True,
) -> CommitOutcome:
    episode = FactorialEpisode(cell, compact_layout=compact_layout)
    episode.reset()
    for action in actions:
        if not is_legal(action, episode.env.world):
            raise ValueError(f"compiled illegal action at step {episode.env.world.step}: {action}")
        _, _, done, _ = episode.step(action)
        if done:
            break
    return episode.outcome()


def run_scripted_control(
    cell: QuartetCell,
    control: str,
    *,
    compact_layout: bool = True,
) -> CommitOutcome:
    """Execute one of the pre-registered integrity-control policies."""
    if control == "abstain":
        episode = FactorialEpisode(cell, compact_layout=compact_layout)
        episode.reset()
        while not episode.done:
            episode.step(Action(ActionKind.WAIT))
        return episode.outcome()
    if control == "mechanism_oracle":
        slot = cell.mechanism_carrier_slot
        return _run_direct_commit(cell, slot, compact_layout)
    if control == "cue_follower":
        slot = cell.cue_carrier_slot
        return _run_direct_commit(cell, slot, compact_layout)
    if control == "anti_cue":
        other_focal = next(slot for slot in cell.focal_slots if slot != cell.cue_carrier_slot)
        return _run_direct_commit(cell, other_focal, compact_layout)
    if control == "fixed_slot":
        return _run_direct_commit(cell, 0, compact_layout)
    if control == "random_committer":
        # Per-cell deterministic randomness makes the control reproducible
        # without coupling its draws to iteration order.
        control_seed = (
            cell.task_spec.seed * 10_007 + cell.mechanism_slot * 101 + cell.cue_slot * 17 + 31
        )
        slot = random.Random(control_seed).randrange(3)
        return _run_direct_commit(cell, slot, compact_layout)
    if control == "detector_policy":
        actions = compile_factorial_certificate(
            cell, use_detector=True, compact_layout=compact_layout
        )
        return execute_compiled_actions(cell, actions, compact_layout=compact_layout)
    raise ValueError(f"unknown scripted control {control!r}")


def run_policy(
    cell: QuartetCell,
    policy: Callable[[FactorialEpisode], Action],
    *,
    compact_layout: bool = True,
) -> CommitOutcome:
    episode = FactorialEpisode(cell, compact_layout=compact_layout)
    episode.reset()
    while not episode.done:
        episode.step(policy(episode))
    return episode.outcome()


def _run_direct_commit(cell: QuartetCell, slot: int, compact_layout: bool) -> CommitOutcome:
    chosen_tool = _object_id(cell, f"tool_slot_{slot}")
    gate = _object_id(cell, "gate")
    episode = FactorialEpisode(cell, compact_layout=compact_layout)
    episode.reset()
    actions: list[Action] = []
    _append_pickup_plan(episode, chosen_tool, actions)
    _append_apply_plan(episode, chosen_tool, gate, actions)
    return episode.outcome()


def _append_pickup_plan(episode: FactorialEpisode, obj_id: str, actions: list[Action]) -> None:
    obj_pos = episode.env.world.objects[obj_id].visible.pos
    if obj_pos is None:
        return
    _execute_and_append(episode, shortest_legal_moves(episode.env.world, [obj_pos]), actions)
    _execute_and_append(episode, [Action(ActionKind.PICKUP, {"obj_id": obj_id})], actions)


def _append_apply_plan(
    episode: FactorialEpisode,
    tool_id: str,
    target_id: str,
    actions: list[Action],
) -> None:
    target = episode.env.world.objects[target_id]
    if target_id in episode.env.world.agent.inventory:
        destinations = [episode.env.world.agent.pos]
    else:
        assert target.visible.pos is not None
        row, col = target.visible.pos
        destinations = [
            pos
            for pos in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1), (row, col))
            if 0 <= pos[0] < episode.env.world.grid_size
            and 0 <= pos[1] < episode.env.world.grid_size
        ]
    _execute_and_append(episode, shortest_legal_moves(episode.env.world, destinations), actions)
    _execute_and_append(
        episode,
        [Action(ActionKind.APPLY, {"tool_id": tool_id, "target_id": target_id})],
        actions,
    )


def _execute_and_append(
    episode: FactorialEpisode,
    new_actions: Iterable[Action],
    actions: list[Action],
) -> None:
    for action in new_actions:
        if not is_legal(action, episode.env.world):
            raise ValueError(f"planner emitted illegal action: {action}")
        actions.append(action)
        episode.step(action)


def _object_id(cell: QuartetCell, role: str) -> str:
    return next(spec.obj_id for spec in cell.task_spec.object_specs if spec.role == role)
