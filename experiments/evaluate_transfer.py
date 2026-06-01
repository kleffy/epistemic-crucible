# ruff: noqa: E402
"""Evaluate baseline traces: compute all diagnostic metrics and save a JSON report.

Usage:
    .venv/bin/python experiments/evaluate_transfer.py
    .venv/bin/python experiments/evaluate_transfer.py --traces results/*.jsonl
    .venv/bin/python experiments/evaluate_transfer.py --output results/metric_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from typing import Any

_HERE = pathlib.Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from crucible.grammar import generate_task
from crucible.metrics import (
    MetricResult,
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

_log = get_logger(__name__)


def _get_spec(family: str, seed: int, split: str) -> Any:
    """Reconstruct a TaskSpec from (family, seed, split)."""
    from crucible.splits import SplitLabel

    return generate_task(family, seed=seed, split=SplitLabel(split))


def _format_value(value: Any, max_width: int = 60) -> str:
    """Format a metric value for tabular display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        # Show first few entries as a compact repr.
        items = list(value.items())[:3]
        snippet = ", ".join(f"{k}={v}" for k, v in items)
        suffix = "…" if len(value) > 3 else ""
        return f"{{{snippet}{suffix}}}"
    return str(value)[:max_width]


def _print_report(report: dict[str, MetricResult]) -> None:
    col_w = [36, 50, 8, 60]
    header = (
        f"{'Metric':<{col_w[0]}} {'Value':<{col_w[1]}} {'N':<{col_w[2]}} "
        f"{'Gaming risk (truncated)':<{col_w[3]}}"
    )
    sep = "-" * sum(col_w)
    print(sep)
    print(header)
    print(sep)
    for name, result in report.items():
        row = (
            f"{name:<{col_w[0]}} "
            f"{_format_value(result.value):<{col_w[1]}} "
            f"{result.count:<{col_w[2]}} "
            f"{result.gaming_risk[:col_w[3]]:<{col_w[3]}}"
        )
        print(row)
    print(sep)


def _report_to_serialisable(report: dict[str, MetricResult]) -> dict:
    return {name: result.to_dict() for name, result in report.items()}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Epistemic Crucible baseline traces.")
    p.add_argument(
        "--traces",
        nargs="+",
        default=None,
        help="JSONL trace files (glob patterns accepted). Defaults to results/*.jsonl.",
    )
    p.add_argument(
        "--output",
        default="results/metric_report.json",
        help="Path for JSON metric report output.",
    )
    p.add_argument(
        "--no-spec",
        action="store_true",
        help="Skip get_spec (disables intervention_efficiency and counterfactual_accuracy).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Resolve trace paths.
    raw_patterns = args.traces or ["results/*.jsonl"]
    trace_paths: list[pathlib.Path] = []
    for pattern in raw_patterns:
        matched = glob.glob(str(pattern))
        if matched:
            trace_paths.extend(pathlib.Path(m) for m in sorted(matched))
        else:
            p = pathlib.Path(pattern)
            if p.exists():
                trace_paths.append(p)

    if not trace_paths:
        print("No trace files found. Run experiments/run_baselines.py first.")
        sys.exit(1)

    _log.info("loading %d trace file(s)", len(trace_paths))
    steps, outcomes = load_traces(trace_paths)
    _log.info("loaded %d steps, %d outcomes", len(steps), len(outcomes))

    get_spec = None if args.no_spec else _get_spec

    # Compute full report.
    report = {
        "task_success_rate": task_success_rate(outcomes),
        "transfer_success": transfer_success(outcomes),
        "shortcut_sensitivity": shortcut_sensitivity(outcomes),
        "intervention_validity": intervention_validity(steps),
        "intervention_efficiency": intervention_efficiency(steps, outcomes, get_spec),
        "counterfactual_accuracy": counterfactual_accuracy(steps, outcomes, get_spec),
        "failure_diversity": failure_diversity(steps, outcomes),
        "curriculum_progression": curriculum_progression(outcomes),
    }

    _print_report(report)

    # Save JSON report.
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_report_to_serialisable(report), indent=2))
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
