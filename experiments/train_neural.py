"""Train the neural PPO baseline on TRAIN-split tasks (GPU-accelerated).

Trains one policy per task family on the TRAIN split using the opt-in reward
layer, checkpoints to ``results/checkpoints/<family>.pt``, and writes a learning
history JSONL. Evaluation on TRAIN vs TEST is done separately by the standard
runner (``run_baselines.py`` with the ``neural_ppo`` agent), so the learned
policy flows through the unchanged diagnostic metric vector.

Usage:
    python experiments/train_neural.py --config configs/neural.yaml
    python experiments/train_neural.py --families affordance --updates 200
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import yaml

_HERE = pathlib.Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from crucible.agents.heuristic_symbolic import HeuristicAgent  # noqa: E402
from crucible.agents.neural_ppo import (  # noqa: E402
    NeuralPPOAgent,
    PPOConfig,
    PPOTrainer,
    behavior_clone,
    collect_demonstrations,
)
from crucible.env import CrucibleEnv  # noqa: E402
from crucible.grammar import TaskFamily, generate_task  # noqa: E402
from crucible.rewards import RewardConfig  # noqa: E402
from crucible.splits import SplitLabel  # noqa: E402
from crucible.utils.logging import get_logger  # noqa: E402

_log = get_logger(__name__)


def _reward_config(cfg: dict) -> RewardConfig:
    r = cfg.get("reward", {})
    return RewardConfig(
        goal_reward=float(r.get("goal_reward", 1.0)),
        step_penalty=float(r.get("step_penalty", 0.002)),
        illegal_penalty=float(r.get("illegal_penalty", 0.0)),
        navigation_shaping=float(r.get("navigation_shaping", 0.05)),
        shaping_gamma=float(r.get("shaping_gamma", 0.99)),
    )


def _ppo_config(cfg: dict, seed: int) -> PPOConfig:
    p = cfg.get("ppo", {})
    return PPOConfig(
        embed_dim=int(p.get("embed_dim", 32)),
        hidden=int(p.get("hidden", 64)),
        lr=float(p.get("lr", 3e-4)),
        gamma=float(p.get("gamma", 0.99)),
        gae_lambda=float(p.get("gae_lambda", 0.95)),
        clip=float(p.get("clip", 0.2)),
        value_coef=float(p.get("value_coef", 0.5)),
        entropy_coef=float(p.get("entropy_coef", 0.01)),
        epochs=int(p.get("epochs", 4)),
        minibatch=int(p.get("minibatch", 256)),
        rollout_steps=int(p.get("rollout_steps", 2048)),
        device=str(cfg.get("device", "cuda")),
        seed=seed,
    )


def train_family(family: TaskFamily, cfg: dict, out_dir: pathlib.Path) -> dict:
    seeds = [int(s) for s in cfg.get("train_seeds", list(range(24)))]
    grid_size = int(cfg.get("grid_size", 6))
    updates = int(cfg.get("updates", 120))
    seed = int(cfg.get("seed", 0))
    mode = str(cfg.get("train_mode", "bc+ppo"))  # bc | ppo | bc+ppo
    bc = cfg.get("bc", {})

    specs = [generate_task(family, seed=s, split=SplitLabel.TRAIN) for s in seeds]
    reward_cfg = _reward_config(cfg)

    def eval_env(spec):
        return CrucibleEnv(seed=spec.seed, config={"task_spec": spec})

    def train_env(spec):
        return CrucibleEnv(
            seed=spec.seed,
            config={
                "task_spec": spec,
                "terminate_on_goal": True,
                "reward_config": reward_cfg,
            },
        )

    agent = NeuralPPOAgent(_ppo_config(cfg, seed), grid_size=grid_size)
    _log.info(
        "training family=%s mode=%s device=%s (%d seeds)",
        family.value,
        mode,
        agent.device,
        len(seeds),
    )
    t0 = time.time()
    bc_history: list[dict] = []
    ppo_history: list[dict] = []

    # Behavior-cloning warm-start from the heuristic solver (public features
    # only; the heuristic exploits the colour shortcut, which is exactly the
    # behaviour we want the learner to inherit and then be tested on).
    if "bc" in mode:
        demo_seeds = [int(s) for s in bc.get("demo_seeds", list(range(100)))]
        demo_specs = [generate_task(family, seed=s, split=SplitLabel.TRAIN) for s in demo_seeds]
        demos = collect_demonstrations(
            demo_specs, eval_env, HeuristicAgent(grid_size=grid_size), grid_size=grid_size
        )
        _log.info("family=%s collected %d demonstration steps", family.value, len(demos))
        bc_history = behavior_clone(
            agent,
            demos,
            epochs=int(bc.get("epochs", 300)),
            batch_size=int(bc.get("batch_size", 64)),
            lr=float(bc.get("lr", 1e-3)),
        )

    if "ppo" in mode:
        trainer = PPOTrainer(agent, train_env, reward_config=reward_cfg)
        ppo_history = trainer.train(specs, total_updates=updates)

    elapsed = time.time() - t0

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{family.value}.pt"
    agent.save(ckpt_path)

    hist_path = out_dir / f"train_{family.value}_history.jsonl"
    with hist_path.open("w") as fh:
        for rec in bc_history:
            fh.write(json.dumps({"phase": "bc", **rec}) + "\n")
        for rec in ppo_history:
            fh.write(json.dumps({"phase": "ppo", **rec}) + "\n")

    final_success = ppo_history[-1]["success_rate"] if ppo_history else None
    bc_acc = bc_history[-1]["bc_acc"] if bc_history else None
    _log.info(
        "family=%s done in %.1fs  bc_acc=%s  ppo_success=%s  -> %s",
        family.value,
        elapsed,
        f"{bc_acc:.3f}" if bc_acc is not None else "n/a",
        f"{final_success:.3f}" if final_success is not None else "n/a",
        ckpt_path,
    )
    return {
        "family": family.value,
        "mode": mode,
        "elapsed_sec": round(elapsed, 1),
        "bc_final_acc": bc_acc,
        "ppo_final_success_rate": final_success,
        "checkpoint": str(ckpt_path),
    }


def _log_device(cfg: dict) -> None:
    if cfg.get("device", "cuda") == "cuda":
        import torch

        if torch.cuda.is_available():
            _log.info(
                "CUDA device: %s  capability=%s",
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_capability(0),
            )
        else:
            _log.warning("device=cuda requested but CUDA unavailable; training on CPU")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train the neural PPO baseline.")
    p.add_argument("--config", default="configs/neural.yaml")
    p.add_argument("--families", nargs="+", help="Override families list")
    p.add_argument("--updates", type=int, help="Override number of PPO updates")
    p.add_argument("--device", help="Override device (cuda/cpu)")
    p.add_argument("--output-dir", default="results")
    args = p.parse_args(argv)

    with pathlib.Path(args.config).open() as f:
        cfg = yaml.safe_load(f) or {}
    if args.families:
        cfg["families"] = args.families
    if args.updates:
        cfg["updates"] = args.updates
    if args.device:
        cfg["device"] = args.device

    _log_device(cfg)
    out_dir = pathlib.Path(args.output_dir)
    families = [TaskFamily(f) for f in cfg.get("families", ["affordance"])]
    summary = [train_family(fam, cfg, out_dir) for fam in families]

    summary_path = out_dir / "train_neural_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _log.info("training summary -> %s", summary_path)


if __name__ == "__main__":
    main()
