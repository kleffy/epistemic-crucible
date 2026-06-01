# ruff: noqa: E402
"""Experiment runner: executes all configured baselines and logs JSONL traces.

Usage:
    .venv/bin/python experiments/run_baselines.py
    .venv/bin/python experiments/run_baselines.py --config configs/baselines.yaml
    .venv/bin/python experiments/run_baselines.py --families affordance --seeds 0 1 2

Output:
    results/baselines_<timestamp>.jsonl   — one JSON record per step + outcome line
    results/baselines_<timestamp>_summary.json — per-agent-family aggregate metrics
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import yaml

# Make sure the package root is on sys.path when run directly.
_HERE = pathlib.Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from crucible.actions import Action, ActionKind
from crucible.agents.base import INTERVENTION_KINDS, Agent
from crucible.agents.heuristic_symbolic import HeuristicAgent
from crucible.agents.hybrid_rule_planner import HybridRulePlannerAgent
from crucible.agents.memorization import MemorizationAgent
from crucible.agents.random_agent import RandomAgent
from crucible.agents.tabular_rl import TabularRLAgent
from crucible.agents.world_model import WorldModelAgent
from crucible.env import CrucibleEnv
from crucible.grammar import TaskFamily, check_goal, generate_task
from crucible.splits import SplitLabel
from crucible.utils.logging import get_logger

_log = get_logger(__name__)

# Registry mapping name → factory callable.
_AGENT_FACTORIES: dict[str, Any] = {
    "random": lambda cfg: RandomAgent(seed=cfg.get("seed", 0), grid_size=cfg.get("grid_size", 6)),
    "heuristic": lambda cfg: HeuristicAgent(grid_size=cfg.get("grid_size", 6)),
    "memorization": lambda cfg: MemorizationAgent(),
    "tabular_rl": lambda cfg: TabularRLAgent(
        seed=cfg.get("seed", 0), grid_size=cfg.get("grid_size", 6)
    ),
    "world_model": lambda cfg: WorldModelAgent(
        seed=cfg.get("seed", 0), grid_size=cfg.get("grid_size", 6)
    ),
    "hybrid_rule_planner": lambda cfg: HybridRulePlannerAgent(
        seed=cfg.get("seed", 0), grid_size=cfg.get("grid_size", 6)
    ),
}


def _make_neural_agent(cfg: dict) -> Agent:
    """Eval-only factory: load a frozen neural_ppo checkpoint for the family.

    Imported lazily so the optional torch dependency is only required when the
    neural baseline is actually used — the default CPU suite stays torch-free.
    """
    from crucible.agents.neural_ppo import NeuralPPOAgent

    ckpt_dir = pathlib.Path(cfg.get("checkpoint_dir", "results/checkpoints"))
    ckpt = ckpt_dir / f"{cfg['family']}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"neural_ppo checkpoint not found: {ckpt}. "
            f"Train it first: python experiments/train_neural.py"
        )
    return NeuralPPOAgent.load(
        ckpt, grid_size=cfg.get("grid_size", 6), device=cfg.get("neural_device", "cpu")
    )


_AGENT_FACTORIES["neural_ppo"] = _make_neural_agent


def _make_agent(name: str, cfg: dict) -> Agent:
    factory = _AGENT_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown agent: {name!r}. Available: {list(_AGENT_FACTORIES)}")
    return factory(cfg)


def _oracle_actions(spec: Any) -> list[Action]:
    """Convert solution_certificate action_sequence dicts to Action objects."""
    actions = []
    for step in spec.solution_certificate.action_sequence:
        kind = ActionKind(step["kind"])
        args = dict(step.get("args", {}))
        if "direction" in args:
            from crucible.actions import Direction

            args["direction"] = Direction(args["direction"])
        actions.append(Action(kind=kind, args=args))
    return actions


def run_episode(
    agent: Agent,
    spec: Any,
    writer: Any,
    episode_idx: int,
) -> dict:
    """Run one episode. Returns outcome dict. Writes step records to writer."""
    env = CrucibleEnv(seed=spec.seed, config={"task_spec": spec})
    obs = env.reset()
    agent.reset()

    # Pre-populate public hash (RL/world-model agents) and goal text (LLM agent).
    from crucible.agents.prompting import describe_goal
    from crucible.counterfactuals import stable_state_hash

    goal_text = describe_goal(spec.goal)
    obs["_public_hash"] = stable_state_hash(env.world, public=True)
    obs["_goal_text"] = goal_text

    total_steps = 0
    illegal_count = 0
    intervention_count = 0
    effects_all: list[str] = []

    for step_i in range(spec.max_steps):
        action = agent.act(obs)
        obs_after, reward, done, info = env.step(action)

        # Inject public hash + goal text for next act().
        obs_after["_public_hash"] = info.get("public_state_hash_after", "")
        obs_after["_goal_text"] = goal_text

        agent.observe_result(obs_after, reward, done, info)

        legal = info.get("legal", False)
        effects = info.get("effects", [])

        if not legal:
            illegal_count += 1
        if action.kind in INTERVENTION_KINDS:
            intervention_count += 1
        effects_all.extend(effects)

        step_record = {
            "kind": "step",
            "episode": episode_idx,
            "seed": spec.seed,
            "family": spec.family.value,
            "split": spec.split.value,
            "agent": agent.name,
            "step": step_i,
            "action": {
                "kind": action.kind.value,
                "args": {k: str(v) for k, v in action.args.items()},
            },
            "effects": effects,
            "legal": legal,
            "energy": obs_after.get("agent", {}).get("energy", 100),
            "done": done,
        }
        writer.write(json.dumps(step_record) + "\n")

        obs = obs_after
        total_steps = step_i + 1

        if done:
            break

    # Check goal from final world state.
    goal_achieved = check_goal(spec.goal, env.world)
    energy_remaining = obs.get("agent", {}).get("energy", 0)
    illegal_rate = illegal_count / max(total_steps, 1)

    outcome = {
        "kind": "outcome",
        "episode": episode_idx,
        "seed": spec.seed,
        "family": spec.family.value,
        "split": spec.split.value,
        "agent": agent.name,
        "goal_achieved": goal_achieved,
        "steps": total_steps,
        "interventions": intervention_count,
        "energy_remaining": energy_remaining,
        "illegal_rate": round(illegal_rate, 4),
        "unique_effects": sorted(set(effects_all)),
    }
    writer.write(json.dumps(outcome) + "\n")
    return outcome


def run_all(config: dict, output_dir: pathlib.Path | None = None) -> list[dict]:
    """Run all agent × family × seed combinations. Returns list of outcome dicts."""
    out_dir = output_dir or pathlib.Path(config.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"baselines_{timestamp}.jsonl"
    summary_path = out_dir / f"baselines_{timestamp}_summary.json"

    families = [TaskFamily(f) for f in config.get("families", ["affordance"])]
    seeds: list[int] = [int(s) for s in config.get("seeds", [0])]
    agent_names: list[str] = config.get("agents", ["random"])
    grid_size: int = int(config.get("grid_size", 6))

    all_outcomes: list[dict] = []
    episode_idx = 0

    with jsonl_path.open("w") as fh:
        for family in families:
            for agent_name in agent_names:
                try:
                    agent = _make_agent(
                        agent_name,
                        {
                            "grid_size": grid_size,
                            "family": family.value,
                            "checkpoint_dir": config.get("checkpoint_dir", "results/checkpoints"),
                            "neural_device": config.get("neural_device", "cpu"),
                        },
                    )
                except FileNotFoundError as exc:
                    # A learned agent without a checkpoint for this family (e.g.
                    # neural_ppo on untrainable families) is skipped, not fatal.
                    _log.warning("skipping %s on %s: %s", agent_name, family.value, exc)
                    continue

                # Pre-populate memorization agent with oracle solutions.
                if isinstance(agent, MemorizationAgent):
                    for s in seeds:
                        for split in (SplitLabel.TRAIN, SplitLabel.TEST):
                            spec = generate_task(family, seed=s, split=split)
                            key = (family.value, s, split.value)
                            agent.store(key, _oracle_actions(spec))

                for seed in seeds:
                    spec = generate_task(family, seed=seed)

                    # Set memorization episode key before reset.
                    if isinstance(agent, MemorizationAgent):
                        agent.set_episode_key((family.value, seed, spec.split.value))

                    _log.info(
                        "episode %d  family=%s  agent=%s  seed=%d  split=%s",
                        episode_idx,
                        family.value,
                        agent_name,
                        seed,
                        spec.split.value,
                    )
                    outcome = run_episode(agent, spec, fh, episode_idx)
                    all_outcomes.append(outcome)
                    fh.flush()
                    episode_idx += 1

    # Write summary: aggregate by (family, agent).
    summary: dict[str, Any] = {}
    for o in all_outcomes:
        key = f"{o['family']}/{o['agent']}"
        bucket = summary.setdefault(key, {"episodes": 0, "goal_achieved": 0, "total_steps": 0})
        bucket["episodes"] += 1
        bucket["goal_achieved"] += int(o["goal_achieved"])
        bucket["total_steps"] += o["steps"]
    for key, v in summary.items():
        v["success_rate"] = round(v["goal_achieved"] / max(v["episodes"], 1), 4)
        v["avg_steps"] = round(v["total_steps"] / max(v["episodes"], 1), 2)

    summary_path.write_text(json.dumps(summary, indent=2))
    _log.info("wrote %d episodes → %s", episode_idx, jsonl_path)
    _log.info("summary → %s", summary_path)
    return all_outcomes


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Epistemic Crucible baseline agents.")
    p.add_argument("--config", default="configs/baselines.yaml", help="YAML config path")
    p.add_argument("--families", nargs="+", help="Override families list")
    p.add_argument("--seeds", nargs="+", type=int, help="Override seeds list")
    p.add_argument("--agents", nargs="+", help="Override agents list")
    p.add_argument("--output-dir", default=None, help="Override output directory")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = pathlib.Path(args.config)
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    if args.families:
        cfg["families"] = args.families
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.agents:
        cfg["agents"] = args.agents

    out_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    outcomes = run_all(cfg, output_dir=out_dir)

    success_count = sum(1 for o in outcomes if o["goal_achieved"])
    print(f"Completed {len(outcomes)} episodes. Goal achieved: {success_count}/{len(outcomes)}")


if __name__ == "__main__":
    main()
