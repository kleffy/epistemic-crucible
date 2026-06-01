"""Generate markdown evaluation reports from JSONL trace files.

Usage
-----
    .venv/bin/python experiments/generate_report.py --traces results/*.jsonl
    .venv/bin/python experiments/generate_report.py \\
        --traces results/baselines_*.jsonl --output results/ \\
        --config '{"families": ["affordance"], "seeds": [0, 1, 2]}'
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from crucible.utils.logging import get_logger
from crucible.viz.reports import generate_report

_log = get_logger(__name__)


def main(argv=None) -> dict[str, str]:
    parser = argparse.ArgumentParser(
        description="Generate a diagnostic markdown report from one or more JSONL trace files."
    )
    parser.add_argument(
        "--traces",
        nargs="+",
        default=["results/*.jsonl"],
        help="JSONL trace file(s) or glob patterns (default: results/*.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for reports and plots (default: results/)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON string embedded in the report's Configuration section",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output)

    # Parse optional config
    config = None
    if args.config:
        try:
            config = json.loads(args.config)
        except json.JSONDecodeError:
            _log.warning("Could not parse --config JSON; ignoring")

    # Expand glob patterns
    trace_files: list[Path] = []
    for pattern in args.traces:
        matched = glob.glob(pattern)
        if matched:
            trace_files.extend(Path(p) for p in sorted(matched))
        elif Path(pattern).exists():
            trace_files.append(Path(pattern))
        else:
            _log.warning("No files matched pattern: %s", pattern)

    if not trace_files:
        _log.error("No trace files found — nothing to do")
        return {}

    results: dict[str, str] = {}
    for trace_path in trace_files:
        _log.info("Processing %s", trace_path)
        report_path = generate_report(trace_path, output_dir=output_dir, config=config)
        print(f"Report: {report_path}")
        results[str(trace_path)] = str(report_path)

    return results


if __name__ == "__main__":
    main()
