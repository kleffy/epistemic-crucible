"""Publication renderer for crossed affordance-factorial worlds."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crucible.grammar import build_world_from_spec
from crucible.objects import ObjectType

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from crucible.factorial import QuartetCell


_COLORS = {
    "red": "#D1495B",
    "blue": "#3979B9",
    "green": "#3A9D5D",
    "grey": "#777777",
}


def render_factorial_world(
    cell: QuartetCell,
    ax: Axes | None = None,
    reveal_mechanism: bool = False,
) -> Figure:
    """Render one factorial cell without exposing its hidden mechanism by default.

    Visible properties and positions are taken from the public world state.  The
    conductivity field is read only when ``reveal_mechanism=True``, making the
    default rendering safe for prompts, anonymous artifacts, and paper figures.
    Stable slot labels are used instead of seed-bearing object identifiers.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, RegularPolygon
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install with: pip install epistemic-crucible[notebooks]"
        ) from exc

    if ax is None:
        figure, ax = plt.subplots(figsize=(6.4, 5.4))
    else:
        figure = ax.figure

    world = build_world_from_spec(cell.task_spec)
    size = cell.task_spec.grid_size
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.grid(color="#D9DEE5", linewidth=0.8, zorder=0)
    ax.set_facecolor("#FAFBFC")

    for spec in cell.task_spec.object_specs:
        public = world.objects[spec.obj_id].visible
        if public.pos is None:
            continue
        row, col = public.pos
        color = _COLORS.get(public.color.value, "#BBBBBB")

        if spec.obj_type == ObjectType.TOOL:
            slot = int(spec.role.rsplit("_", maxsplit=1)[-1])
            edge = "#20252B"
            width = 1.2
            if reveal_mechanism and spec.conductivity:
                edge = "#F2B134"
                width = 3.2
            patch = RegularPolygon(
                (col, row),
                numVertices=6,
                radius=0.31,
                orientation=0.0,
                facecolor=color,
                edgecolor=edge,
                linewidth=width,
                zorder=3,
            )
            ax.add_patch(patch)
            ax.text(
                col,
                row + 0.48,
                f"tool {slot}",
                ha="center",
                va="top",
                fontsize=8,
                color="#20252B",
                zorder=4,
            )
            if reveal_mechanism and spec.conductivity:
                ax.text(
                    col,
                    row - 0.47,
                    "conductive",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    weight="bold",
                    color="#8A5A00",
                    zorder=4,
                )
        elif spec.obj_type == ObjectType.DETECTOR:
            patch = FancyBboxPatch(
                (col - 0.34, row - 0.22),
                0.68,
                0.44,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor="#F3F5F7",
                edgecolor="#30363D",
                linewidth=1.4,
                zorder=3,
            )
            ax.add_patch(patch)
            ax.text(col, row, "DETECT", ha="center", va="center", fontsize=6.5, zorder=4)
            label_y = row - 0.47 if row == size - 1 else row + 0.47
            label_va = "bottom" if row == size - 1 else "top"
            ax.text(col, label_y, "detector", ha="center", va=label_va, fontsize=8)
        elif spec.obj_type == ObjectType.GATE:
            ax.plot(
                [col - 0.34, col - 0.34, col + 0.34, col + 0.34],
                [row + 0.27, row - 0.27, row - 0.27, row + 0.27],
                color="#30363D",
                linewidth=2.4,
                zorder=3,
            )
            for offset in (-0.2, 0.0, 0.2):
                ax.plot(
                    [col + offset, col + offset],
                    [row - 0.25, row + 0.25],
                    color="#777777",
                    linewidth=1.1,
                    zorder=3,
                )
            ax.text(col, row + 0.47, "gate", ha="center", va="top", fontsize=8)

    agent_row, agent_col = world.agent.pos
    ax.scatter(
        [agent_col],
        [agent_row],
        marker="*",
        s=220,
        facecolor="#20252B",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    ax.text(agent_col, agent_row + 0.47, "agent", ha="center", va="top", fontsize=8)
    return figure


__all__ = ["render_factorial_world"]
