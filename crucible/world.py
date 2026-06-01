from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from crucible.objects import (
    AFFINITY_VALUES,
    CrucibleObject,
    HiddenObjectProps,
    ObjectColor,
    ObjectShape,
    ObjectSize,
    ObjectState,
    ObjectTexture,
    ObjectType,
    VisibleObjectState,
)
from crucible.relations import Relation, RelationKind
from crucible.utils.seeding import make_rng


@dataclass
class AgentState:
    pos: tuple[int, int]
    inventory: list[str] = field(default_factory=list)
    energy: int = 100


@dataclass
class WorldState:
    seed: int
    grid_size: int
    step: int
    max_steps: int
    objects: dict[str, CrucibleObject]
    relations: list[Relation]
    agent: AgentState
    rules: list = field(default_factory=list)  # populated in Phase 3
    pending_effects: list = field(default_factory=list)  # populated in Phase 3


def generate_world(
    seed: int,
    grid_size: int = 6,
    num_objects: int = 4,
    max_steps: int = 50,
) -> WorldState:
    """Return a deterministic WorldState for the given seed and parameters."""
    rng = make_rng(seed)

    all_cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    occupied: set[tuple[int, int]] = set()

    agent_pos = sample_cell(rng, all_cells, occupied)
    occupied.add(agent_pos)

    objects: dict[str, CrucibleObject] = {}
    obj_types = list(ObjectType)
    obj_colors = list(ObjectColor)
    obj_shapes = list(ObjectShape)
    obj_textures = list(ObjectTexture)
    obj_sizes = list(ObjectSize)
    obj_states = [ObjectState.DEFAULT, ObjectState.CLOSED, ObjectState.ACTIVE]
    affinity_choices = list(AFFINITY_VALUES)

    for i in range(num_objects):
        pos = sample_cell(rng, all_cells, occupied)
        occupied.add(pos)

        obj_id = f"{seed}_{i:03d}"
        visible = VisibleObjectState(
            obj_type=obj_types[int(rng.integers(len(obj_types)))],
            color=obj_colors[int(rng.integers(len(obj_colors)))],
            shape=obj_shapes[int(rng.integers(len(obj_shapes)))],
            texture=obj_textures[int(rng.integers(len(obj_textures)))],
            size=obj_sizes[int(rng.integers(len(obj_sizes)))],
            marker=None,
            pos=pos,
            state=obj_states[int(rng.integers(len(obj_states)))],
        )
        hidden = HiddenObjectProps(
            conductivity=bool(rng.integers(2)),
            solubility=bool(rng.integers(2)),
            magnetism=bool(rng.integers(2)),
            charge=bool(rng.integers(2)),
            fragility=bool(rng.integers(2)),
            affinity=affinity_choices[int(rng.integers(len(affinity_choices)))],
        )
        objects[obj_id] = CrucibleObject(obj_id=obj_id, visible=visible, hidden=hidden)

    relations = derive_relations(objects, agent_pos)

    world = WorldState(
        seed=seed,
        grid_size=grid_size,
        step=0,
        max_steps=max_steps,
        objects=objects,
        relations=relations,
        agent=AgentState(pos=agent_pos),
    )
    from crucible.rules import attach_rule_set

    attach_rule_set(world)
    return world


def sample_cell(
    rng: np.random.Generator,
    all_cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> tuple[int, int]:
    free = [c for c in all_cells if c not in occupied]
    idx = int(rng.integers(len(free)))
    return free[idx]


def derive_relations(
    objects: dict[str, CrucibleObject],
    agent_pos: tuple[int, int],
) -> list[Relation]:
    relations: list[Relation] = []
    obj_list = list(objects.values())
    for i, a in enumerate(obj_list):
        for b in obj_list[i + 1 :]:
            if _manhattan(a.visible.pos, b.visible.pos) == 1:
                relations.append(
                    Relation(kind=RelationKind.ADJACENT, subject=a.obj_id, object_=b.obj_id)
                )
                relations.append(
                    Relation(kind=RelationKind.ADJACENT, subject=b.obj_id, object_=a.obj_id)
                )
        if _manhattan(a.visible.pos, agent_pos) == 1:
            relations.append(
                Relation(kind=RelationKind.ADJACENT, subject="agent", object_=a.obj_id)
            )
            relations.append(
                Relation(kind=RelationKind.ADJACENT, subject=a.obj_id, object_="agent")
            )
    return relations


def _manhattan(a: tuple[int, int] | None, b: tuple[int, int] | None) -> int:
    if a is None or b is None:
        return 999
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
