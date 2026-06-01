"""Tests for crucible/viz — visualization functions run without crashes and return Figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from crucible.grammar import TaskFamily, build_world_from_spec, generate_task
from crucible.splits import SplitLabel
from crucible.viz.grammar_tree import plot_task_tree
from crucible.viz.heatmaps import (
    plot_failure_map,
    plot_recombination_heatmap,
    plot_shortcut_exposure,
)
from crucible.viz.traces import plot_intervention_trace
from crucible.viz.world_graph import plot_world

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_outcomes(
    n: int = 5,
    goal: bool = True,
    family: str = "affordance",
    agent: str = "random",
    split: str = "train",
) -> list[dict]:
    return [
        {
            "kind": "outcome",
            "episode": i,
            "seed": i,
            "family": family,
            "split": split,
            "agent": agent,
            "goal_achieved": goal,
            "steps": 10,
            "interventions": 2,
            "energy_remaining": 80,
            "illegal_rate": 0.0,
            "unique_effects": ["opened gate"] if goal else [],
        }
        for i in range(n)
    ]


def _make_steps(
    n: int = 10,
    family: str = "affordance",
    agent: str = "random",
    split: str = "train",
) -> list[dict]:
    kinds = ["apply", "move"]
    return [
        {
            "kind": "step",
            "episode": i // 5,
            "seed": i // 5,
            "family": family,
            "split": split,
            "agent": agent,
            "step": i % 5,
            "action": {"kind": kinds[i % 2], "args": {}},
            "effects": ["opened gate"] if i % 3 == 0 else [],
            "legal": True,
            "energy": 90,
            "done": False,
        }
        for i in range(n)
    ]


def _minimal_trace(tmp_path: Path) -> Path:
    """Write a minimal valid JSONL trace to tmp_path and return its path."""
    step = {
        "kind": "step", "episode": 0, "seed": 0, "family": "affordance",
        "split": "train", "agent": "random", "step": 0,
        "action": {"kind": "move", "args": {}}, "effects": [],
        "legal": True, "energy": 100, "done": False,
    }
    outcome = {
        "kind": "outcome", "episode": 0, "seed": 0, "family": "affordance",
        "split": "train", "agent": "random", "goal_achieved": False,
        "steps": 1, "interventions": 0, "energy_remaining": 100,
        "illegal_rate": 0.0, "unique_effects": [],
    }
    p = tmp_path / "trace.jsonl"
    p.write_text(json.dumps(step) + "\n" + json.dumps(outcome) + "\n")
    return p


# ---------------------------------------------------------------------------
# world_graph
# ---------------------------------------------------------------------------


def test_plot_world_returns_figure():
    spec = generate_task(TaskFamily.AFFORDANCE, seed=0, split=SplitLabel.TRAIN)
    world = build_world_from_spec(spec)
    fig = plot_world(world)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_world_all_families():
    for family in TaskFamily:
        spec = generate_task(family, seed=1)
        world = build_world_from_spec(spec)
        fig = plot_world(world)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# grammar_tree
# ---------------------------------------------------------------------------


def test_plot_task_tree_returns_figure():
    spec = generate_task(TaskFamily.AFFORDANCE, seed=0, split=SplitLabel.TRAIN)
    fig = plot_task_tree(spec)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_task_tree_all_families():
    for family in TaskFamily:
        spec = generate_task(family, seed=2)
        fig = plot_task_tree(spec)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------


def test_plot_intervention_trace_empty():
    fig = plot_intervention_trace([])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_intervention_trace_small():
    steps = _make_steps(20)
    fig = plot_intervention_trace(steps)
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# heatmaps
# ---------------------------------------------------------------------------


def test_plot_shortcut_exposure_deterministic():
    outcomes = _make_outcomes(10, split="train") + _make_outcomes(10, goal=False, split="test")
    fig1 = plot_shortcut_exposure(outcomes)
    plt.close(fig1)
    fig2 = plot_shortcut_exposure(outcomes)
    assert isinstance(fig2, Figure)
    plt.close(fig2)


def test_plot_recombination_heatmap_runs():
    train = _make_outcomes(5, split="train")
    test = _make_outcomes(3, goal=False, split="test")
    fig = plot_recombination_heatmap(train + test)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_failure_map_runs():
    steps = _make_steps(6)
    outcomes = _make_outcomes(3, goal=False)
    fig = plot_failure_map(steps, outcomes)
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# reports — _metric_table
# ---------------------------------------------------------------------------


def test_metric_table_no_aggregate_score(tmp_path: Path):
    from crucible.metrics import full_report
    from crucible.viz.reports import _metric_table

    trace = _minimal_trace(tmp_path)
    report = full_report(trace)
    table = _metric_table(report)
    assert "overall_score" not in table
    assert "task_success_rate" in table


# ---------------------------------------------------------------------------
# reports — generate_report
# ---------------------------------------------------------------------------


def test_generate_report_creates_file(tmp_path: Path):
    from crucible.viz.reports import generate_report

    trace = _minimal_trace(tmp_path)
    report_path = generate_report(trace, output_dir=tmp_path)
    assert report_path.exists()
    assert report_path.suffix == ".md"


def test_generate_report_has_metric_definitions(tmp_path: Path):
    from crucible.viz.reports import generate_report

    trace = _minimal_trace(tmp_path)
    report_path = generate_report(trace, output_dir=tmp_path)
    content = report_path.read_text()
    assert "task_success_rate" in content
    # MetricResult.definition text appears in the table
    assert "Fraction" in content or "fraction" in content or "episodes" in content.lower()


def test_generate_report_has_seed_metadata(tmp_path: Path):
    from crucible.viz.reports import generate_report

    trace = _minimal_trace(tmp_path)
    report_path = generate_report(
        trace, output_dir=tmp_path, config={"families": ["affordance"]}
    )
    content = report_path.read_text()
    assert "affordance" in content


def test_generate_report_no_aggregate_score(tmp_path: Path):
    from crucible.viz.reports import generate_report

    trace = _minimal_trace(tmp_path)
    report_path = generate_report(trace, output_dir=tmp_path)
    content = report_path.read_text()
    assert "overall_score" not in content
    assert "aggregate_score" not in content


def test_generate_report_handles_empty_trace(tmp_path: Path):
    from crucible.viz.reports import generate_report

    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    report_path = generate_report(empty, output_dir=tmp_path)
    assert report_path.exists()
