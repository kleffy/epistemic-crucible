"""Intervention trace visualisations — episode × step heatmap."""

from crucible.utils.logging import get_logger

_log = get_logger(__name__)

_INTERVENTION_KINDS = frozenset({"apply", "combine", "inspect"})


def plot_intervention_trace(
    steps,
    *,
    agent=None,
    family=None,
    max_episodes=5,
    max_steps=30,
    ax=None,
    save_to=None,
):
    """Render a heatmap of intervention outcomes across episodes × steps.

    Cell values:
        0 — non-intervention action
        1 — intervention with no causal effect
        2 — intervention that produced at least one effect

    Parameters
    ----------
    steps : list[dict]
        Step records from a JSONL trace.
    agent : str, optional
        Filter to a single agent.
    family : str, optional
        Filter to a single task family.
    max_episodes : int
        Maximum number of episode rows to show.
    max_steps : int
        Maximum number of step columns to show.
    ax : matplotlib.axes.Axes, optional
    save_to : path-like, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install with: pip install epistemic-crucible[notebooks]"
        ) from exc

    from crucible.metrics import filter_records

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "savefig.bbox": "tight",
    })
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(8, 3.8))
    else:
        fig = ax.figure

    filtered = filter_records(steps, agent=agent, family=family)

    if not filtered:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Intervention Trace (no data)")
        if save_to is not None:
            fig.savefig(save_to, dpi=300, bbox_inches="tight")
        return fig

    # Collect episodes in order of first appearance, up to max_episodes
    seen_eps: dict[tuple, int] = {}
    for r in filtered:
        key = (r.get("episode"), r.get("seed"), r.get("split"), r.get("agent"))
        if key not in seen_eps:
            if len(seen_eps) >= max_episodes:
                break
            seen_eps[key] = len(seen_eps)

    n_eps = len(seen_eps)
    matrix = np.zeros((n_eps, max_steps), dtype=float)

    for r in filtered:
        key = (r.get("episode"), r.get("seed"), r.get("split"), r.get("agent"))
        if key not in seen_eps:
            continue
        step_i = r.get("step", 0)
        if step_i >= max_steps:
            continue
        ep_i = seen_eps[key]
        action_kind = r.get("action", {}).get("kind", "")
        if action_kind in _INTERVENTION_KINDS:
            matrix[ep_i, step_i] = 2.0 if r.get("effects") else 1.0

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto", origin="upper",
                   interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["none", "no effect", "effect"], fontsize=12)

    ep_keys = sorted(seen_eps, key=lambda k: seen_eps[k])
    ax.set_yticks(range(n_eps))
    ax.set_yticklabels([f"ep{k[0]}" for k in ep_keys])
    ax.set_xlabel("Step")
    ax.set_ylabel("Episode")

    agent_str = agent or "all agents"
    family_str = f" / {family}" if family else ""
    ax.set_title(f"Intervention Trace — {agent_str}{family_str}")

    if save_to is not None:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
        _log.debug("Saved intervention trace to %s", save_to)

    return fig
