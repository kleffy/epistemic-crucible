from __future__ import annotations

from crucible.world import WorldState


def observe(world: WorldState) -> dict:
    """Return the agent-facing observation — hidden fields are never included."""
    return {
        "step": world.step,
        "max_steps": world.max_steps,
        "agent": {
            "pos": world.agent.pos,
            "inventory": list(world.agent.inventory),
            "energy": world.agent.energy,
        },
        "objects": {
            obj_id: {
                "obj_id": obj.obj_id,
                "type": obj.visible.obj_type.value,
                "color": obj.visible.color.value,
                "shape": obj.visible.shape.value,
                "texture": obj.visible.texture.value,
                "size": obj.visible.size.value,
                "marker": obj.visible.marker,
                "pos": obj.visible.pos,
                "state": obj.visible.state.value,
            }
            for obj_id, obj in world.objects.items()
        },
        "relations": [
            {
                "kind": r.kind.value,
                "subject": r.subject,
                "object": r.object_,
                "metadata": r.metadata,
            }
            for r in world.relations
        ],
    }
