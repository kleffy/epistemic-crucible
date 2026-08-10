"""Train matched cue-following and detector-following v0.2 BC policies."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict

import yaml

_HERE = pathlib.Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from crucible.actions import Action, ActionKind  # noqa: E402
from crucible.agents.neural_ppo import (  # noqa: E402
    NeuralPPOAgent,
    PPOConfig,
    StepFeatures,
    behavior_clone,
    featurize_step,
    match_action_index,
)
from crucible.factorial import (  # noqa: E402
    EpistemicAction,
    FactorialEpisode,
    MacroActionKind,
    QuartetCell,
    generate_affordance_quartet,
)
from crucible.factorial_metrics import compute_factorial_metrics  # noqa: E402


def collect_matched_demonstrations(
    regime: str,
    *,
    seeds: list[int],
    target_transitions: int,
    grid_size: int = 6,
) -> list[tuple[StepFeatures, int]]:
    """Collect equal-volume macro-action datasets on identical crossed worlds.

    ``cue`` commits directly to the red carrier; ``mechanism`` first obtains
    public detector evidence. The generated worlds and their order are exactly
    matched. The regimes intentionally also differ in evidence availability and
    action distribution; equal sample count is a capacity control, not a claim
    that demonstration policy is the sole causal difference.
    """
    if regime not in {"cue", "mechanism"}:
        raise ValueError(f"unknown demonstration regime {regime!r}")
    if target_transitions <= 0 or not seeds:
        raise ValueError("positive target_transitions and at least one seed are required")
    cells = [cell for seed in seeds for cell in generate_affordance_quartet(seed).cells.values()]
    demos: list[tuple[StepFeatures, int]] = []
    episode_index = 0
    while len(demos) < target_transitions:
        cell = cells[episode_index % len(cells)]
        episode_index += 1
        episode = FactorialEpisode(cell)
        episode.reset()
        while not episode.done and len(demos) < target_transitions:
            macro_actions, candidates = _macro_proxy_candidates(cell, episode)
            features, candidates = featurize_step(episode.obs, grid_size, candidates)
            if regime == "cue":
                target_macro = EpistemicAction(MacroActionKind.COMMIT, cell.cue_carrier_slot)
            else:
                identified = episode.identified_slot
                if identified is not None:
                    target_macro = EpistemicAction(MacroActionKind.COMMIT, identified)
                else:
                    query = next(
                        action for action in macro_actions if action.kind == MacroActionKind.QUERY
                    )
                    target_macro = query
            target_action = candidates[macro_actions.index(target_macro)]
            target = match_action_index(target_action, candidates)
            if target is None:
                raise ValueError(f"macro target not present in proxy set: {target_macro}")
            demos.append((features, target))
            episode.macro_step(target_macro)
    return demos


def evaluate_agent(agent: NeuralPPOAgent, seeds: list[int]) -> dict:
    outcomes = []
    for seed in seeds:
        for cell in generate_affordance_quartet(seed).cells.values():
            episode = FactorialEpisode(cell)
            episode.reset()
            while not episode.done:
                macro_actions, candidates = _macro_proxy_candidates(cell, episode)
                action = agent.act_from_candidates(episode.obs, candidates)
                index = match_action_index(action, candidates)
                assert index is not None
                episode.macro_step(macro_actions[index])
            outcomes.append(episode.outcome())
    return asdict(compute_factorial_metrics(outcomes))


def validate_disjoint_seed_sets(train_seeds: list[int], eval_seeds: list[int]) -> None:
    """Reject evaluation on worlds used to construct demonstrations."""
    overlap = sorted(set(train_seeds) & set(eval_seeds))
    if overlap:
        raise ValueError(
            "BC train_seeds and eval_seeds must be disjoint; "
            f"found {len(overlap)} overlapping seeds, beginning with {overlap[:10]}"
        )


def train_regime(regime: str, cfg: dict, output_dir: pathlib.Path, *, training_seed: int) -> dict:
    seed = training_seed
    train_seeds = [int(value) for value in cfg.get("train_seeds", range(100))]
    eval_seeds = [int(value) for value in cfg.get("eval_seeds", range(64))]
    bc_cfg = cfg.get("bc", {})
    agent_cfg = cfg.get("network", {})
    agent = NeuralPPOAgent(
        PPOConfig(
            embed_dim=int(agent_cfg.get("embed_dim", 32)),
            hidden=int(agent_cfg.get("hidden", 64)),
            device=str(cfg.get("device", "cpu")),
            seed=seed,
            metadata={
                "training_method": "behavior_cloning",
                "factorial_regime": regime,
                "protocol_version": "0.2",
            },
        )
    )
    demos = collect_matched_demonstrations(
        regime,
        seeds=train_seeds,
        target_transitions=int(bc_cfg.get("target_transitions", 4096)),
    )
    started = time.time()
    history = behavior_clone(
        agent,
        demos,
        epochs=int(bc_cfg.get("epochs", 300)),
        batch_size=int(bc_cfg.get("batch_size", 64)),
        lr=float(bc_cfg.get("lr", 1e-3)),
    )
    metrics = evaluate_agent(agent, eval_seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"factorial_{regime}_bc_seed{seed}.pt"
    agent.save(checkpoint)
    history_path = output_dir / f"factorial_{regime}_bc_seed{seed}_history.jsonl"
    history_path.write_text("".join(json.dumps(record) + "\n" for record in history))
    return {
        "regime": regime,
        "seed": seed,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "demonstration_transitions": len(demos),
        "optimization": {
            "epochs": int(bc_cfg.get("epochs", 300)),
            "batch_size": int(bc_cfg.get("batch_size", 64)),
            "lr": float(bc_cfg.get("lr", 1e-3)),
        },
        "metrics": metrics,
        "checkpoint": str(checkpoint),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def evaluate_existing(cfg: dict, output_dir: pathlib.Path) -> list[dict]:
    """Re-evaluate saved checkpoints on the configured held-out base seeds."""
    summary_path = output_dir / "factorial_bc_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing training summary: {summary_path}")
    previous = json.loads(summary_path.read_text())
    by_key = {(record["regime"], int(record["seed"])): record for record in previous}
    eval_seeds = [int(value) for value in cfg.get("eval_seeds", range(64))]
    training_seeds = [int(seed) for seed in cfg.get("training_seeds", range(5))]
    device = str(cfg.get("device", "cpu"))
    summaries = []
    for seed in training_seeds:
        for regime in ("cue", "mechanism"):
            record = dict(by_key[(regime, seed)])
            checkpoint = pathlib.Path(record["checkpoint"])
            started = time.time()
            agent = NeuralPPOAgent.load(checkpoint, device=device)
            record["metrics"] = evaluate_agent(agent, eval_seeds)
            record["eval_seeds"] = eval_seeds
            record["evaluation"] = {
                "base_seed_overlap": [],
                "device": str(agent.device),
                "elapsed_seconds": round(time.time() - started, 3),
                "held_out": True,
            }
            summaries.append(record)
    archive_path = output_dir / "factorial_bc_summary_in_sample_overlap.json"
    if not archive_path.exists():
        archive_path.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n")
    return summaries


def _macro_proxy_candidates(
    cell: QuartetCell,
    episode: FactorialEpisode,
) -> tuple[list[EpistemicAction], list[Action]]:
    """Encode QUERY/COMMIT choices for the existing parametric action network."""
    macros = episode.macro_actions()
    gate_id = next(spec.obj_id for spec in cell.task_spec.object_specs if spec.role == "gate")
    tool_ids = {
        int(spec.role.rsplit("_", 1)[-1]): spec.obj_id
        for spec in cell.task_spec.object_specs
        if spec.role.startswith("tool_slot_")
    }
    proxies = []
    for macro in macros:
        if macro.kind == MacroActionKind.QUERY:
            proxies.append(Action(ActionKind.INSPECT, {"obj_id": tool_ids[macro.slot]}))
        else:
            proxies.append(
                Action(
                    ActionKind.APPLY,
                    {"tool_id": tool_ids[macro.slot], "target_id": gate_id},
                )
            )
    return macros, proxies


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/factorial_bc_v02.yaml")
    parser.add_argument("--output-dir", default="results/factorial_v02/bc")
    parser.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="re-evaluate saved checkpoints without retraining",
    )
    args = parser.parse_args(argv)
    with pathlib.Path(args.config).open() as handle:
        cfg = yaml.safe_load(handle) or {}
    output_dir = pathlib.Path(args.output_dir)
    train_seeds = [int(value) for value in cfg.get("train_seeds", range(100))]
    eval_seeds = [int(value) for value in cfg.get("eval_seeds", range(64))]
    validate_disjoint_seed_sets(train_seeds, eval_seeds)
    training_seeds = [int(seed) for seed in cfg.get("training_seeds", range(5))]
    if args.evaluate_existing:
        summaries = evaluate_existing(cfg, output_dir)
    else:
        summaries = [
            train_regime(regime, cfg, output_dir, training_seed=seed)
            for seed in training_seeds
            for regime in ("cue", "mechanism")
        ]
    (output_dir / "factorial_bc_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
