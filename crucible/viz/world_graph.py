"""Grid world renderer — plots agent position, objects, and inventory sidebar."""

from crucible.utils.logging import get_logger

_log = get_logger(__name__)

_COLOUR_MAP = {
    "red": "red",
    "blue": "steelblue",
    "green": "forestgreen",
    "yellow": "gold",
    "grey": "dimgray",
}

_SHAPE_MARKER = {
    "cube": "s",
    "sphere": "o",
    "cylinder": "D",
    "rod": "|",
    "flat": "_",
}


def plot_world(world, *, ax=None, title=None, save_to=None):
    """Render WorldState as a grid with coloured object markers and agent star.

    Parameters
    ----------
    world : WorldState
    ax : matplotlib.axes.Axes, optional
    title : str, optional
    save_to : path-like, optional  — if given, save PNG to this path

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install with: pip install epistemic-crucible[notebooks]"
        ) from exc

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    n = world.grid_size

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n))
    ax.set_yticklabels(range(n - 1, -1, -1))  # row 0 at top
    ax.grid(True, color="lightgrey", linewidth=0.8, zorder=0)
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    inventory_labels = []

    for obj_id, obj in world.objects.items():
        if obj.visible.pos is None:
            inventory_labels.append(f"{obj_id[-6:]} ({obj.visible.obj_type.value})")
            continue
        row, col = obj.visible.pos
        y = n - 1 - row  # invert so row 0 is at top
        color = _COLOUR_MAP.get(obj.visible.color.value, "white")
        marker = _SHAPE_MARKER.get(obj.visible.shape.value, "o")
        ax.plot(
            col,
            y,
            marker,
            color=color,
            markersize=16,
            markeredgecolor="black",
            markeredgewidth=0.6,
            zorder=3,
        )
        ax.text(
            col,
            y - 0.38,
            obj.visible.obj_type.value[:4],
            ha="center",
            va="top",
            fontsize=6,
            zorder=4,
        )

    # Agent
    ar, ac = world.agent.pos
    ay = n - 1 - ar
    ax.plot(ac, ay, "*", color="black", markersize=20, zorder=5, label="agent")

    # Inventory sidebar
    inv_text = "Inventory:\n" + ("\n".join(inventory_labels) if inventory_labels else "(empty)")
    fig.text(
        0.82,
        0.5,
        inv_text,
        fontsize=8,
        va="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
    )

    ax.set_title(title or f"World (seed={world.seed}, step={world.step})")

    if save_to is not None:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
        _log.debug("Saved world plot to %s", save_to)

    return fig
