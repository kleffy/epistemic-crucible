"""Plot v0.2 attribution coordinates with both analytic uniform references."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from crucible.metrics import analytic_chance_point  # noqa: E402
from crucible.utils.logging import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def plot_summary(summary_path: pathlib.Path, output: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    summary = json.loads(summary_path.read_text())
    reports = summary["reports"]
    figure, axis = plt.subplots(figsize=(6.2, 5.0))
    for label, report in reports.items():
        mechanism = report["mechanism_responsiveness"]["value"]
        cue = report["cue_susceptibility"]["value"]
        if mechanism is None or cue is None:
            continue
        axis.scatter(mechanism, cue, s=48)
        axis.annotate(label, (mechanism, cue), xytext=(4, 4), textcoords="offset points")
    references = (
        ("uniform over all 3 tools", analytic_chance_point(3), "x"),
        ("uniform over red/blue focal tools", analytic_chance_point(2), "+"),
    )
    for label, reference, marker in references:
        axis.scatter(
            reference["mechanism_responsiveness"],
            reference["cue_susceptibility"],
            marker=marker,
            s=90,
            color="black",
            label=label,
        )
    axis.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    axis.set_xlabel("Strict mechanism responsiveness")
    axis.set_ylabel("Cue susceptibility (committed pairs)")
    axis.legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)
    _LOG.info("wrote %s", output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    plot_summary(args.summary, args.output)


if __name__ == "__main__":
    main()
