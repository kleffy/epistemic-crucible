"""Visualisation utilities for Epistemic Crucible.

This sub-package contains:
- world_graph: plot a WorldState as a 6x6 grid
- grammar_tree: plot a TaskSpec as a hierarchical tree
- traces: plot intervention trace matrices
- heatmaps: TSR heatmaps and failure-mode bar charts
- reports: generate full markdown evaluation reports

All functions require matplotlib. Install with:
    pip install epistemic-crucible[notebooks]
"""

from crucible.viz.factorial import render_factorial_world

__all__ = ["render_factorial_world"]
