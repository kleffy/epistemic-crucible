"""Tests for the dose-response and oracle-ladder analysers.

These exercise the directory-scanning, condition parsing, TSR aggregation, and
colour-reliance wiring of ``experiments/analyze_results.py`` against synthetic
traces whose object IDs come from the real generator (so the reliance metric
sees the true red-tool and gate IDs).
"""

from __future__ import annotations

import json
import pathlib

from crucible.grammar import TaskFamily, generate_task
from crucible.splits import SplitLabel
from experiments.analyze_results import dose_response, oracle_ladder

AGENT = "test-agent"


def _affordance_ids(seed: int, split: str) -> tuple[str | None, str]:
    """Return (a red tool id or None, the gate id) for one affordance instance."""
    spec = generate_task(TaskFamily.AFFORDANCE, seed=seed, split=SplitLabel(split))
    red = next(
        (
            o.obj_id
            for o in spec.object_specs
            if o.obj_type.value == "tool" and o.color.value == "red"
        ),
        None,
    )
    gate = next(o.obj_id for o in spec.object_specs if o.obj_type.value == "gate")
    return red, gate


def _red_seeds(n: int) -> list[int]:
    """First n test-split seeds whose affordance instance has a red tool."""
    out, seed = [], 0
    while len(out) < n:
        red, _ = _affordance_ids(seed, "test")
        if red is not None:
            out.append(seed)
        seed += 1
    return out


def _write_condition(sub: pathlib.Path, *, n: int, use_red: bool, success: bool) -> None:
    """Write n affordance test-split episodes, each applying a tool to the gate."""
    sub.mkdir(parents=True, exist_ok=True)
    lines = []
    seeds = _red_seeds(n) if use_red else list(range(n))
    for seed in seeds:
        red, gate = _affordance_ids(seed, "test")
        tool = red if use_red else f"{gate}__decoy"  # decoy id is never red
        lines.append(
            json.dumps(
                {
                    "kind": "step",
                    "agent": AGENT,
                    "family": "affordance",
                    "split": "test",
                    "seed": seed,
                    "episode": seed,
                    "step": 0,
                    "action": {"kind": "apply", "args": {"tool_id": tool, "target_id": gate}},
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "kind": "outcome",
                    "agent": AGENT,
                    "family": "affordance",
                    "split": "test",
                    "seed": seed,
                    "episode": seed,
                    "goal_achieved": success,
                }
            )
        )
    (sub / "trace.jsonl").write_text("\n".join(lines) + "\n")


def test_dose_response_parses_modes_and_reliance(tmp_path):
    # cue_k1: all red -> reliance 1.0; mechanistic_k1: no red -> reliance 0.0.
    _write_condition(tmp_path / "cue_k1", n=5, use_red=True, success=False)
    _write_condition(tmp_path / "mechanistic_k1", n=5, use_red=False, success=True)
    (tmp_path / "ignore_me").mkdir()  # non-matching dir is skipped

    out = dose_response(tmp_path)
    assert set(out["modes"]) == {"cue", "mechanistic"}
    cue = out["modes"]["cue"][0]
    mech = out["modes"]["mechanistic"][0]
    assert cue["k"] == 1 and mech["k"] == 1
    assert cue["reliance_test"]["reliance"] == 1.0
    assert mech["reliance_test"]["reliance"] == 0.0
    assert mech["tsr_test"]["mean"] == 1.0


def test_dose_response_baseline_shared_across_modes(tmp_path):
    _write_condition(tmp_path / "k0", n=4, use_red=True, success=False)
    _write_condition(tmp_path / "cue_k3", n=4, use_red=False, success=False)

    out = dose_response(tmp_path)
    ks = [r["k"] for r in out["modes"]["cue"]]
    assert ks == [0, 3]  # the k0 baseline is folded in as the K=0 point, sorted


def test_oracle_ladder_levels_and_gap(tmp_path):
    for level, success in (("intervention", False), ("rule", True)):
        sub = tmp_path / f"sonnet_{level}"
        sub.mkdir()
        lines = []
        for seed in range(5):
            for split in ("train", "test"):
                lines.append(
                    json.dumps(
                        {
                            "kind": "outcome",
                            "agent": "sonnet",
                            "family": "affordance",
                            "split": split,
                            "seed": seed,
                            "episode": seed,
                            "goal_achieved": success,
                        }
                    )
                )
        (sub / "t.jsonl").write_text("\n".join(lines) + "\n")

    out = oracle_ladder(tmp_path)
    assert set(out) == {"intervention", "rule"}
    assert out["rule"]["sonnet"]["affordance"]["test"]["mean"] == 1.0
    assert out["intervention"]["sonnet"]["affordance"]["test"]["mean"] == 0.0
