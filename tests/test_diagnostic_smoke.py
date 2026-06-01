"""Smoke tests for experiments/run_diagnostic.py.

Tests call main() directly (not subprocess) for speed and to avoid process
overhead. Each test uses a tiny config: one family, 1-2 seeds, one agent.
"""

from __future__ import annotations

import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from experiments.run_diagnostic import main  # noqa: E402

_MINIMAL = ["--families", "affordance", "--seeds", "0", "1", "--agents", "random"]
_SINGLE = ["--families", "affordance", "--seeds", "0", "--agents", "random"]

_EXPECTED_METRICS = {
    "task_success_rate",
    "transfer_success",
    "shortcut_sensitivity",
    "intervention_validity",
    "intervention_efficiency",
    "counterfactual_accuracy",
    "failure_diversity",
    "curriculum_progression",
}


def test_run_diagnostic_produces_outputs(tmp_path: pathlib.Path) -> None:
    """main() creates both a JSONL trace and a JSON report."""
    result = main(_MINIMAL + ["--output-dir", str(tmp_path)])
    assert result["trace"].exists(), "trace JSONL not created"
    assert result["report"].exists(), "report JSON not created"
    lines = [ln for ln in result["trace"].read_text().splitlines() if ln.strip()]
    assert len(lines) > 0, "trace is empty"


def test_run_diagnostic_report_has_all_metrics(tmp_path: pathlib.Path) -> None:
    """JSON report contains exactly the eight expected metric keys."""
    result = main(_SINGLE + ["--output-dir", str(tmp_path)])
    report = json.loads(result["report"].read_text())
    assert set(report.keys()) == _EXPECTED_METRICS


def test_run_diagnostic_is_deterministic(tmp_path: pathlib.Path) -> None:
    """Identical seeds produce identical task_success_rate values."""
    args = _MINIMAL + ["--output-dir", str(tmp_path)]
    r1 = main(args)
    r2 = main(args)
    rep1 = json.loads(r1["report"].read_text())
    rep2 = json.loads(r2["report"].read_text())
    assert rep1["task_success_rate"]["value"] == rep2["task_success_rate"]["value"]
