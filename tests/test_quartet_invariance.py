"""Acceptance tests for the v0.2 affordance quartet.

These fail until the v0.2 generator lands. Required API:

    crucible.grammar.generate_affordance_quartet(seed) -> AffordanceQuartet
    AffordanceQuartet.cells: dict[tuple[int, int], QuartetCell]   # (mechanism_slot, cue_slot)
    QuartetCell.task_spec: TaskSpec
    QuartetCell.mechanism_slot: int
    QuartetCell.cue_slot: int
    QuartetCell.condition_id: str
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

from crucible.agents.base import enumerate_candidate_actions
from crucible.agents.prompting import build_label_map, build_user_message, describe_goal
from crucible.grammar import build_world_from_spec, generate_affordance_quartet
from crucible.objects import ObjectColor, ObjectType
from crucible.observations import observe

SEEDS = range(64)
CELLS = [(0, 0), (0, 1), (1, 0), (1, 1)]
MECHANISM_PAIRS = [((0, 0), (1, 0)), ((0, 1), (1, 1))]
CUE_PAIRS = [((0, 0), (0, 1)), ((1, 0), (1, 1))]

# Fields permitted to differ across the mechanism axis. Everything else is
# either visible to the agent or must be held fixed by construction.
MECHANISM_AXIS_ALLOWED = frozenset(
    {"conductivity", "role", "task_id", "condition_id", "metadata", "solution_certificate"}
)


def _obs_for(spec):
    return observe(build_world_from_spec(spec, compact=True))


def _render(spec):
    obs = _obs_for(spec)
    candidates = enumerate_candidate_actions(obs, spec.grid_size)
    msg, candidates, label_map = build_user_message(
        obs, describe_goal(spec.goal), spec.grid_size, candidates
    )
    return msg, candidates, label_map


def _mask_colour(spec):
    masked = copy.deepcopy(spec)
    for obj in masked.object_specs:
        obj.color = ObjectColor.GREY
    return masked


def _differing_roots(spec_a, spec_b) -> set[str]:
    a = dataclasses.asdict(spec_a)
    b = dataclasses.asdict(spec_b)
    roots = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    if a["object_specs"] != b["object_specs"]:
        roots.discard("object_specs")
        for obj_a, obj_b in zip(a["object_specs"], b["object_specs"], strict=True):
            roots |= {k for k in obj_a if obj_a[k] != obj_b[k]}
    return roots


def _tools(spec):
    return [o for o in spec.object_specs if o.obj_type is ObjectType.TOOL]


# --- mechanism axis: the two cells differ only in a hidden field -------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cell_a,cell_b", MECHANISM_PAIRS)
def test_mechanism_axis_prompt_byte_identical(seed, cell_a, cell_b):
    quartet = generate_affordance_quartet(seed)
    msg_a, _, _ = _render(quartet.cells[cell_a].task_spec)
    msg_b, _, _ = _render(quartet.cells[cell_b].task_spec)
    assert msg_a == msg_b


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cell_a,cell_b", MECHANISM_PAIRS)
def test_mechanism_axis_only_hidden_fields_differ(seed, cell_a, cell_b):
    quartet = generate_affordance_quartet(seed)
    diff = _differing_roots(quartet.cells[cell_a].task_spec, quartet.cells[cell_b].task_spec)
    assert diff <= MECHANISM_AXIS_ALLOWED, sorted(diff - MECHANISM_AXIS_ALLOWED)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cell_a,cell_b", MECHANISM_PAIRS)
def test_mechanism_axis_conductivity_actually_moves(seed, cell_a, cell_b):
    quartet = generate_affordance_quartet(seed)
    cond_a = [o.conductivity for o in _tools(quartet.cells[cell_a].task_spec)]
    cond_b = [o.conductivity for o in _tools(quartet.cells[cell_b].task_spec)]
    assert cond_a != cond_b


# --- cue axis: the two cells differ only in colour ---------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cell_a,cell_b", CUE_PAIRS)
def test_cue_axis_identical_under_colour_mask(seed, cell_a, cell_b):
    quartet = generate_affordance_quartet(seed)
    msg_a, _, _ = _render(_mask_colour(quartet.cells[cell_a].task_spec))
    msg_b, _, _ = _render(_mask_colour(quartet.cells[cell_b].task_spec))
    assert msg_a == msg_b


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cell_a,cell_b", CUE_PAIRS)
def test_cue_axis_colour_actually_moves(seed, cell_a, cell_b):
    quartet = generate_affordance_quartet(seed)
    msg_a, _, _ = _render(quartet.cells[cell_a].task_spec)
    msg_b, _, _ = _render(quartet.cells[cell_b].task_spec)
    assert msg_a != msg_b


# --- invariants shared by all four cells -------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_palette_is_exactly_red_blue_green(seed):
    for cell in generate_affordance_quartet(seed).cells.values():
        colours = sorted(o.color.value for o in _tools(cell.task_spec))
        assert colours == ["blue", "green", "red"]


@pytest.mark.parametrize("seed", SEEDS)
def test_object_ids_stable_across_cells(seed):
    quartet = generate_affordance_quartet(seed)
    id_sets = {
        cell: tuple(o.obj_id for o in quartet.cells[cell].task_spec.object_specs) for cell in CELLS
    }
    assert len(set(id_sets.values())) == 1


@pytest.mark.parametrize("seed", SEEDS)
def test_ids_do_not_encode_condition(seed):
    quartet = generate_affordance_quartet(seed)
    for cell in quartet.cells.values():
        for obj in cell.task_spec.object_specs:
            assert cell.condition_id not in obj.obj_id


@pytest.mark.parametrize("seed", SEEDS)
def test_label_map_identical_across_cells(seed):
    quartet = generate_affordance_quartet(seed)
    maps = [build_label_map(_obs_for(quartet.cells[c].task_spec)) for c in CELLS]
    assert all(m == maps[0] for m in maps[1:])


@pytest.mark.parametrize("seed", SEEDS)
def test_candidate_order_identical_across_cells(seed):
    quartet = generate_affordance_quartet(seed)
    orders = [_render(quartet.cells[c].task_spec)[1] for c in CELLS]
    assert all(o == orders[0] for o in orders[1:])


@pytest.mark.parametrize("seed", SEEDS)
def test_candidate_enumeration_is_canonically_ordered(seed):
    """Guards the unordered-set iteration in enumerate_candidate_actions."""
    for cell in generate_affordance_quartet(seed).cells.values():
        candidates = _render(cell.task_spec)[1]
        keys = [(c.kind.value, sorted(map(str, c.args.items()))) for c in candidates]
        assert keys == sorted(keys)
