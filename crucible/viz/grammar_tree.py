"""Task grammar tree renderer — plots a TaskSpec as a shallow hierarchy."""

from crucible.utils.logging import get_logger

_log = get_logger(__name__)


def plot_task_tree(spec, *, ax=None, title=None, save_to=None):
    """Render TaskSpec as a hierarchical tree using matplotlib text nodes.

    Layout (normalised coordinates):
        Root (family / seed / split)
          ├── Goal node
          ├── Constraints node
          └── Object nodes (one per ObjectSpec)

    No networkx dependency — positions are computed manually.

    Parameters
    ----------
    spec : TaskSpec
    ax : matplotlib.axes.Axes, optional
    title : str, optional
    save_to : path-like, optional

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
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _node_kw = dict(ha="center", va="center", fontsize=8, wrap=True,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"))

    # Root node
    root_x, root_y = 0.5, 0.88
    root_label = f"{spec.family.value}\nseed={spec.seed}  ({spec.split.value})"
    ax.text(root_x, root_y, root_label, **_node_kw)

    # Goal node
    goal_x, goal_y = 0.18, 0.62
    target = spec.goal.target_obj_id or spec.goal.classify_property or ""
    goal_label = f"Goal: {spec.goal.kind.value}\n{target}"
    ax.text(goal_x, goal_y, goal_label, **_node_kw)
    _edge(ax, root_x, root_y, goal_x, goal_y)

    # Constraints node
    con_x, con_y = 0.82, 0.62
    con_label = (
        f"Constraints\nsteps≤{spec.constraints.max_steps}"
        f"\nenergy={spec.constraints.energy_budget}"
    )
    ax.text(con_x, con_y, con_label, **_node_kw)
    _edge(ax, root_x, root_y, con_x, con_y)

    # Object nodes
    n_objs = len(spec.object_specs)
    if n_objs > 0:
        xs = [0.05 + i * (0.90 / max(n_objs - 1, 1)) for i in range(n_objs)]
        for i, obj in enumerate(spec.object_specs):
            ox, oy = xs[i], 0.30
            obj_label = f"{obj.role}\n{obj.obj_type.value} / {obj.color.value}"
            ax.text(ox, oy, obj_label, **_node_kw)
            _edge(ax, root_x, root_y, ox, oy)

    ax.set_title(title or f"Task: {spec.task_id}", pad=4)

    if save_to is not None:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
        _log.debug("Saved grammar tree to %s", save_to)

    return fig


def _edge(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-", color="gray", lw=1.0),
        annotation_clip=False,
    )
