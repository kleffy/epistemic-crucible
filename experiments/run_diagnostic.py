# ruff: noqa: E402
"""Diagnostic experiment runner — headless version of the diagnostic notebook.

Runs a small, deterministic experiment (affordance family by default) using four
baseline agents, computes all eight diagnostic metrics, and writes a JSON report.
No plotting or notebook dependencies required.

Usage:
    .venv/bin/python experiments/run_diagnostic.py
    .venv/bin/python experiments/run_diagnostic.py \\
        --families affordance --seeds 0 1 2 3 4 --agents random heuristic
    .venv/bin/python experiments/run_diagnostic.py --output-dir /tmp/diag

Output (in --output-dir, default results/):
    diagnostic_<timestamp>.jsonl
    diagnostic_<timestamp>_report.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

_HERE = pathlib.Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from crucible.metrics import (
    counterfactual_accuracy,
    curriculum_progression,
    failure_diversity,
    intervention_efficiency,
    intervention_validity,
    load_traces,
    shortcut_sensitivity,
    task_success_rate,
    transfer_success,
)
from crucible.utils.logging import get_logger
from experiments.run_baselines import run_all

_log = get_logger(__name__)

_DEFAULT_FAMILIES = ["affordance"]
_DEFAULT_SEEDS = list(range(10))
_DEFAULT_AGENTS = ["random", "heuristic", "memorization", "hybrid_rule_planner"]


def _format_value(value: Any, width: int = 55) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        items = list(value.items())[:2]
        snippet = ", ".join(f"{k}={v}" for k, v in items)
        suffix = "…" if len(value) > 2 else ""
        return f"{{{snippet}{suffix}}}"[:width]
    return str(value)[:width]


def _print_report(report: dict) -> None:
    col = [36, 55, 6]
    sep = "-" * sum(col)
    header = f"{'Metric':<{col[0]}} {'Value':<{col[1]}} {'N':<{col[2]}}"
    print(sep)
    print(header)
    print(sep)
    for name, result in report.items():
        row = f"{name:<{col[0]}} {_format_value(result.value):<{col[1]}} {result.count:<{col[2]}}"
        print(row)
    print(sep)


def main(argv: list[str] | None = None) -> dict[str, pathlib.Path]:
    """Run the diagnostic experiment and return {"trace": path, "report": path}."""
    args = _parse_args(argv)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "families": args.families,
        "seeds": args.seeds,
        "agents": args.agents,
        "grid_size": 6,
        "output_dir": str(out_dir),
    }

    _log.info(
        "diagnostic: families=%s  seeds=%s  agents=%s",
        config["families"],
        config["seeds"],
        config["agents"],
    )

    run_all(config, out_dir)
    all_traces = sorted(
        out_dir.glob("baselines_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not all_traces:
        raise RuntimeError("run_all produced no trace file in %s", out_dir)
    trace_path = all_traces[0]

    steps, outcomes = load_traces([trace_path])
    _log.info("loaded %d steps, %d outcomes", len(steps), len(outcomes))

    report = {
        "task_success_rate": task_success_rate(outcomes),
        "transfer_success": transfer_success(outcomes),
        "shortcut_sensitivity": shortcut_sensitivity(outcomes),
        "intervention_validity": intervention_validity(steps),
        "intervention_efficiency": intervention_efficiency(steps, outcomes),
        "counterfactual_accuracy": counterfactual_accuracy(steps, outcomes),
        "failure_diversity": failure_diversity(steps, outcomes),
        "curriculum_progression": curriculum_progression(outcomes),
    }

    _print_report(report)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"diagnostic_{timestamp}_report.json"
    report_path.write_text(json.dumps({name: r.to_dict() for name, r in report.items()}, indent=2))
    print(f"\nReport → {report_path}")

    return {"trace": trace_path, "report": report_path}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a diagnostic experiment and compute all eight metrics."
    )
    p.add_argument(
        "--families",
        nargs="+",
        default=_DEFAULT_FAMILIES,
        help="Task families to evaluate (default: affordance).",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=_DEFAULT_SEEDS,
        help="Seeds to evaluate (default: 0-9).",
    )
    p.add_argument(
        "--agents",
        nargs="+",
        default=_DEFAULT_AGENTS,
        help="Agent names to evaluate (default: random heuristic memorization hybrid_rule_planner).",  # noqa: E501
    )
    p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output files (default: results/).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
