"""Recombination, shortcut-exposure, and failure-mode visualisations."""

from crucible.utils.logging import get_logger

_log = get_logger(__name__)

_FAILURE_MODES = ["timeout", "no_interaction", "high_illegal", "no_effects", "energy_depleted"]
_FAILURE_COLOURS = {
    "timeout": "steelblue",
    "no_interaction": "tomato",
    "high_illegal": "orange",
    "no_effects": "purple",
    "energy_depleted": "brown",
}

# Publication-standard font sizing. These figures are saved at 300 dpi and then
# scaled to the column width in the paper, so the native fonts are set large enough
# to stay legible after that downscaling.
_PUB_RC = {
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "savefig.bbox": "tight",
}


def plot_shortcut_exposure(outcomes, *, families=None, agents=None, save_to=None):
    """Side-by-side bar chart: train TSR vs test TSR per agent, one subplot per family.

    High gap (train TSR >> test TSR) signals shortcut reliance.

    Parameters
    ----------
    outcomes : list[dict]
    families : list[str], optional — subset of families to show
    agents : list[str], optional — subset of agents to show
    save_to : path-like, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required. Install with: pip install epistemic-crucible[notebooks]"
        ) from exc

    import math

    from crucible.metrics import filter_records, task_success_rate

    plt.rcParams.update(_PUB_RC)
    all_families = families or sorted({r["family"] for r in outcomes if "family" in r})
    all_agents = agents or sorted({r["agent"] for r in outcomes if "agent" in r})

    # Lay the per-family panels out in a grid (at most two columns) so they are not
    # squeezed into one wide row that shrinks the text to nothing.
    n_fam = max(len(all_families), 1)
    ncols = min(2, n_fam)
    nrows = math.ceil(n_fam / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.7 * nrows), squeeze=False)
    flat = list(axes.flat)

    for fi, fam in enumerate(all_families):
        ax = flat[fi]
        x = list(range(len(all_agents)))
        train_tsrs, test_tsrs = [], []
        for ag in all_agents:
            t = filter_records(outcomes, family=fam, agent=ag, split="train")
            v = filter_records(outcomes, family=fam, agent=ag, split="test")
            train_tsrs.append(task_success_rate(t).value)
            test_tsrs.append(task_success_rate(v).value)

        w = 0.38
        ax.bar([xi - w / 2 for xi in x], train_tsrs, width=w, label="train", color="steelblue")
        ax.bar([xi + w / 2 for xi in x], test_tsrs, width=w, label="test", color="tomato")
        ax.set_xticks(x)
        ax.set_xticklabels(all_agents, rotation=35, ha="right")
        ax.set_ylim(0, 1.12)
        if fi % ncols == 0:
            ax.set_ylabel("TSR")
        ax.set_title(fam)

    if all_families:
        flat[0].legend(loc="upper right")
    else:
        flat[0].text(
            0.5, 0.5, "No data", ha="center", va="center", transform=flat[0].transAxes, color="gray"
        )
    for j in range(n_fam, nrows * ncols):  # hide unused grid cells
        flat[j].axis("off")

    fig.suptitle("Shortcut Exposure: Train vs Test TSR")
    fig.tight_layout()

    if save_to is not None:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
        _log.debug("Saved shortcut exposure plot to %s", save_to)

    return fig


def plot_recombination_heatmap(outcomes, *, families=None, agents=None, save_to=None):
    """TSR heatmap: rows=agents, cols=families, two subplots (train | test).

    Parameters
    ----------
    outcomes : list[dict]
    families : list[str], optional
    agents : list[str], optional
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
            "matplotlib is required. Install with: pip install epistemic-crucible[notebooks]"
        ) from exc

    from crucible.metrics import filter_records, task_success_rate

    plt.rcParams.update(_PUB_RC)
    all_agents = agents or sorted({r["agent"] for r in outcomes if "agent" in r})
    all_families = families or sorted({r["family"] for r in outcomes if "family" in r})

    # Stack the train and test panels vertically so the figure stays narrow and the
    # per-cell annotations stay legible after the figure is scaled to column width.
    width = max(6.5, len(all_families) * 1.5 + 2.5)
    height = max(7.0, len(all_agents) * 0.9 + 2.0)
    fig, (ax_train, ax_test) = plt.subplots(2, 1, figsize=(width, height))

    for ax, split in [(ax_train, "train"), (ax_test, "test")]:
        if not all_agents or not all_families:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray"
            )
            ax.set_title(split.capitalize())
            continue

        matrix = np.full((len(all_agents), len(all_families)), float("nan"))
        for ai, ag in enumerate(all_agents):
            for fi, fam in enumerate(all_families):
                filtered = filter_records(outcomes, agent=ag, family=fam, split=split)
                if filtered:
                    matrix[ai, fi] = task_success_rate(filtered).value

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(all_families)))
        ax.set_xticklabels(all_families, rotation=20, ha="right")
        ax.set_yticks(range(len(all_agents)))
        ax.set_yticklabels(all_agents)
        ax.set_title(split.capitalize())

        for ai in range(len(all_agents)):
            for fi in range(len(all_families)):
                v = matrix[ai, fi]
                if not np.isnan(v):
                    color = "white" if v < 0.5 else "black"
                    ax.text(fi, ai, f"{v:.2f}", ha="center", va="center", fontsize=12, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Train/Test Recombination — TSR Heatmap")
    fig.tight_layout()

    if save_to is not None:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
        _log.debug("Saved recombination heatmap to %s", save_to)

    return fig


def plot_failure_map(steps, outcomes, *, agents=None, families=None, save_to=None):
    """Stacked horizontal bar chart: failure mode counts per agent.

    Parameters
    ----------
    steps : list[dict]
    outcomes : list[dict]
    agents : list[str], optional
    families : list[str], optional
    save_to : path-like, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required. Install with: pip install epistemic-crucible[notebooks]"
        ) from exc

    from crucible.metrics import failure_diversity, filter_records

    plt.rcParams.update(_PUB_RC)
    all_agents = agents or sorted({r["agent"] for r in outcomes if "agent" in r})

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(all_agents) * 0.7 + 2.0)))

    if not all_agents:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("Failure Mode Distribution")
        if save_to is not None:
            fig.savefig(save_to, dpi=300, bbox_inches="tight")
        return fig

    # Collect per-agent failure mode counts
    mode_data: dict[str, dict[str, int]] = {}
    for ag in all_agents:
        ag_steps = filter_records(steps, agent=ag)
        ag_outcomes = filter_records(outcomes, agent=ag)
        if families:
            ag_steps = filter_records(ag_steps, family=None)  # already filtered above
            ag_outcomes = [o for o in ag_outcomes if o.get("family") in families]
        fd = failure_diversity(ag_steps, ag_outcomes)
        mode_data[ag] = {k: v for k, v in fd.value.items() if k != "distinct_modes"}

    bottoms = [0.0] * len(all_agents)
    for mode in _FAILURE_MODES:
        values = [mode_data[ag].get(mode, 0) for ag in all_agents]
        color = _FAILURE_COLOURS.get(mode, "gray")
        bars = ax.barh(all_agents, values, left=bottoms, color=color, label=mode)
        for bar, v, b in zip(bars, values, bottoms):
            if v > 0:
                ax.text(
                    b + v / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white",
                    fontweight="bold",
                )
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_xlabel("Episode count")
    ax.set_title("Failure Mode Distribution per Agent")
    # Legend below the axes so it never overlaps the bars (the longest bar reaches
    # the right edge, where an in-axes legend used to sit).
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(_FAILURE_MODES), framealpha=0.9
    )
    ax.invert_yaxis()

    if save_to is not None:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
        _log.debug("Saved failure map to %s", save_to)

    return fig
