"""Markdown report generator — metrics, plots, and metadata in one document."""

from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path

from crucible.utils.logging import get_logger

_log = get_logger(__name__)

_METRIC_ORDER = [
    "task_success_rate",
    "transfer_success",
    "shortcut_sensitivity",
    "intervention_validity",
    "intervention_efficiency",
    "counterfactual_accuracy",
    "concept_reuse_proxy",
    "failure_diversity",
    "curriculum_progression",
]

_PLOTS = [
    # (plot_name, module, function, args_builder)
    # args_builder receives (steps, outcomes) and returns positional args tuple
    ("intervention_trace", "crucible.viz.traces", "plot_intervention_trace",
     lambda steps, outcomes: (steps,)),
    ("shortcut_exposure", "crucible.viz.heatmaps", "plot_shortcut_exposure",
     lambda steps, outcomes: (outcomes,)),
    ("recombination_heatmap", "crucible.viz.heatmaps", "plot_recombination_heatmap",
     lambda steps, outcomes: (outcomes,)),
    ("failure_map", "crucible.viz.heatmaps", "plot_failure_map",
     lambda steps, outcomes: (steps, outcomes)),
]


def _metric_table(report: dict) -> str:
    """Format a dict[str, MetricResult] as a markdown table.

    Columns: Metric | Value | Count | Definition | Gaming Risk
    No aggregate or overall score row is added.
    """
    header = "| Metric | Value | Count | Definition | Gaming Risk |"
    sep = "| --- | --- | --- | --- | --- |"
    rows = [header, sep]
    for key in _METRIC_ORDER:
        if key not in report:
            continue
        mr = report[key]
        v = mr.value
        if v is None:
            v_str = "—"
        elif isinstance(v, float):
            v_str = f"{v:.4f}"
        elif isinstance(v, dict):
            raw = json.dumps(v, default=str)
            v_str = raw[:80] + "…" if len(raw) > 80 else raw
        else:
            v_str = str(v)
        risk = mr.gaming_risk
        if len(risk) > 80:
            risk = risk[:80] + "…"
        rows.append(f"| `{key}` | {v_str} | {mr.count} | {mr.definition} | {risk} |")
    return "\n".join(rows)


def generate_report(trace_path, *, output_dir=None, config=None) -> Path:
    """Load a trace, compute metrics, generate plots, write a markdown report.

    Handles missing optional plots gracefully — a failed plot is logged and
    skipped; the report is always written.

    Parameters
    ----------
    trace_path : path-like
        JSONL trace file produced by run_baselines.py or run_diagnostic.py.
    output_dir : path-like, optional
        Directory for the report and plot PNGs.  Defaults to trace_path.parent.
    config : dict, optional
        Arbitrary metadata embedded in the report's Configuration section.

    Returns
    -------
    pathlib.Path
        Path to the generated markdown file.
    """
    from crucible.metrics import full_report, load_trace

    trace_path = Path(trace_path)
    if output_dir is None:
        output_dir = trace_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"report_{timestamp}"
    report_path = output_dir / f"{stem}.md"

    # Load trace records for plotting
    try:
        steps, outcomes = load_trace(trace_path)
    except Exception as exc:
        _log.warning("Could not load trace %s: %s", trace_path, exc)
        steps, outcomes = [], []

    # Compute metric report
    try:
        report_dict = full_report(trace_path)
    except Exception as exc:
        _log.warning("Could not compute metrics: %s", exc)
        report_dict = {}

    # Generate plots
    plot_paths: dict[str, Path] = {}
    try:
        import matplotlib.pyplot as plt
        _has_mpl = True
    except ImportError:
        _has_mpl = False
        _log.warning("matplotlib not available — skipping all plots")

    if _has_mpl:
        for plot_name, module_name, fn_name, args_builder in _PLOTS:
            save_to = output_dir / f"{stem}_{plot_name}.png"
            try:
                mod = importlib.import_module(module_name)
                fn = getattr(mod, fn_name)
                args = args_builder(steps, outcomes)
                fig = fn(*args, save_to=save_to)
                plt.close(fig)
                plot_paths[plot_name] = save_to
            except Exception as exc:
                _log.warning("Plot %r skipped: %s", plot_name, exc)

    # Build markdown
    lines: list[str] = []
    lines.append("# Epistemic Crucible Evaluation Report\n")
    lines.append("**No aggregate score is reported.** "
                 "The metric vector below must be read as a whole.\n")

    lines.append(f"- **Trace**: `{trace_path.name}`")
    lines.append(f"- **Generated**: {datetime.now().isoformat(timespec='seconds')}\n")

    if config:
        lines.append("## Configuration\n")
        lines.append("```json")
        lines.append(json.dumps(config, indent=2, default=str))
        lines.append("```\n")

    lines.append("## Metric Results\n")
    if report_dict:
        lines.append(_metric_table(report_dict))
    else:
        lines.append("*No metric data available.*")
    lines.append("")

    if report_dict:
        lines.append("## Gaming Risk Notes\n")
        for key in _METRIC_ORDER:
            if key in report_dict:
                mr = report_dict[key]
                lines.append(f"**`{key}`**: {mr.gaming_risk}\n")

    if plot_paths:
        lines.append("## Visualizations\n")
        for name, path in plot_paths.items():
            try:
                rel = path.relative_to(output_dir)
            except ValueError:
                rel = path
            label = name.replace("_", " ").title()
            lines.append(f"### {label}\n")
            lines.append(f"![{label}]({rel})\n")

    report_path.write_text("\n".join(lines))
    _log.info("Report written to %s", report_path)
    return report_path
