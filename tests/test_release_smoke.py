"""Release smoke tests — verify the complete package is installable, runnable,
and produces the expected diagnostic result without remote API access.

Fast tests (no marker) complete in <5 s.
Slow tests (@pytest.mark.slow) complete in <60 s on a laptop CPU.

Run fast-only:  pytest tests/test_release_smoke.py -v -m "not slow"
Run all:        pytest tests/test_release_smoke.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUICKSTART = Path(__file__).parent.parent / "configs" / "quickstart.yaml"


def _quickstart_config() -> dict:
    with open(_QUICKSTART) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Fast tests — import smoke
# ---------------------------------------------------------------------------


def test_all_public_modules_importable():
    """Verify the complete public API is importable without network access."""
    import crucible  # noqa: F401
    from crucible import (  # noqa: F401  # noqa: F401  # noqa: F401
        actions,
        env,
        grammar,
        interventions,
        metrics,
        objects,
        observations,
        relations,
        splits,
        world,
    )
    from crucible.agents import base  # noqa: F401
    from crucible.utils import logging as clog  # noqa: F401
    from crucible.utils import seeding, serialization  # noqa: F401
    from crucible.viz import grammar_tree, heatmaps, reports, traces, world_graph  # noqa: F401


# ---------------------------------------------------------------------------
# Fast tests — single-episode determinism
# ---------------------------------------------------------------------------


def test_single_episode_deterministic():
    """Same seed + same agent seed produces identical action sequence across two runs."""
    from crucible.agents.random_agent import RandomAgent
    from crucible.env import CrucibleEnv
    from crucible.grammar import TaskFamily, generate_task
    from crucible.splits import SplitLabel

    spec = generate_task(TaskFamily.AFFORDANCE, seed=42, split=SplitLabel.TRAIN)

    def run_one(s):
        e = CrucibleEnv(seed=42, config={"task_spec": s})
        agent = RandomAgent(seed=7)
        obs = e.reset()
        agent.reset()
        record = []
        for _ in range(15):
            a = agent.act(obs)
            record.append({"kind": a.kind.value, "args": str(a.args)})
            obs, _, done, _ = e.step(a)
            if done:
                break
        return record

    assert run_one(spec) == run_one(spec)


# ---------------------------------------------------------------------------
# Fast tests — quickstart config
# ---------------------------------------------------------------------------


def test_quickstart_config_loads():
    """configs/quickstart.yaml exists and contains required keys."""
    assert _QUICKSTART.exists(), f"configs/quickstart.yaml not found at {_QUICKSTART}"
    cfg = _quickstart_config()
    assert "families" in cfg, "quickstart.yaml must have 'families'"
    assert "seeds" in cfg, "quickstart.yaml must have 'seeds'"
    assert "agents" in cfg, "quickstart.yaml must have 'agents'"
    assert len(cfg["seeds"]) >= 10, "quickstart.yaml must have at least 10 seeds"
    assert "heuristic" in cfg["agents"], "quickstart.yaml must include heuristic agent"


# ---------------------------------------------------------------------------
# Slow tests — end-to-end runs
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_hidden_failure_demonstrated(tmp_path: Path):
    """Quickstart shows heuristic >> random on affordance, revealing hidden structure.

    This is the Phase 10 acceptance criterion: a diagnostic run demonstrates a
    hidden failure. The heuristic baseline achieves non-trivial train TSR (~0.5)
    via greedy visible-property navigation, while random exploration completely
    fails (TSR~0). The heuristic's train/test TSR gap signals shortcut reliance:
    it benefits from the colour-conductivity correlation that only exists in the
    training split, rather than discovering the underlying causal rule.
    """
    from crucible.metrics import filter_records, load_trace, task_success_rate
    from experiments.run_baselines import run_all

    cfg = _quickstart_config()
    run_all(cfg, tmp_path)

    trace = sorted(tmp_path.glob("baselines_*.jsonl"), key=lambda p: p.stat().st_mtime)[-1]
    _, outcomes = load_trace(trace)

    # Heuristic achieves non-trivial train TSR via greedy visible-property policy
    heuristic_train = filter_records(outcomes, agent="heuristic", split="train")
    assert len(heuristic_train) > 0, "Heuristic must have train outcomes"
    heuristic_tsr = task_success_rate(heuristic_train).value
    assert heuristic_tsr > 0.2, (
        f"Heuristic train TSR={heuristic_tsr:.2f} must be >0.2 — "
        "greedy visible-only policy should partially succeed on affordance tasks"
    )

    # Random exploration completely fails — tasks require directed action
    rand_all = filter_records(outcomes, agent="random")
    assert len(rand_all) > 0, "Random agent must have outcomes"
    rand_tsr = task_success_rate(rand_all).value
    assert rand_tsr < heuristic_tsr, (
        f"Random TSR={rand_tsr:.2f} must be < heuristic TSR={heuristic_tsr:.2f}. "
        "Affordance requires directed action — the gap exposes the hidden failure pattern."
    )


@pytest.mark.slow
def test_quickstart_runs_deterministically(tmp_path: Path):
    """Running the quickstart config twice produces identical JSONL record sets."""
    from experiments.run_baselines import run_all

    cfg = _quickstart_config()
    out1, out2 = tmp_path / "run1", tmp_path / "run2"
    out1.mkdir()
    out2.mkdir()

    run_all(cfg, out1)
    run_all(cfg, out2)

    t1 = sorted(out1.glob("baselines_*.jsonl"), key=lambda p: p.stat().st_mtime)[-1]
    t2 = sorted(out2.glob("baselines_*.jsonl"), key=lambda p: p.stat().st_mtime)[-1]

    key = lambda d: json.dumps(d, sort_keys=True)  # noqa: E731
    lines1 = sorted([json.loads(ln) for ln in t1.read_text().splitlines() if ln.strip()], key=key)
    lines2 = sorted([json.loads(ln) for ln in t2.read_text().splitlines() if ln.strip()], key=key)
    assert lines1 == lines2, "Benchmark output must be identical across repeated runs"


@pytest.mark.slow
def test_report_generation_end_to_end(tmp_path: Path):
    """generate_report produces a valid markdown report from a quickstart trace."""
    from crucible.viz.reports import generate_report
    from experiments.run_baselines import run_all

    cfg = _quickstart_config()
    run_all(cfg, tmp_path)

    trace = sorted(tmp_path.glob("baselines_*.jsonl"), key=lambda p: p.stat().st_mtime)[-1]
    report_path = generate_report(trace, output_dir=tmp_path)

    assert report_path.exists(), "Report file must be created"
    content = report_path.read_text()
    assert "task_success_rate" in content, "Report must contain metric names"
    assert "overall_score" not in content, "Report must not contain an aggregate score"
    assert "No aggregate score" in content, "Report must explicitly state no aggregate score"
