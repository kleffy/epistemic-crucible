"""Tests for the neural PPO baseline. Run on CPU with a tiny config.

Marked ``gpu`` because they require the optional ``torch`` dependency; they do
not require an actual GPU (device is forced to cpu here).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from crucible.agents.heuristic_symbolic import HeuristicAgent  # noqa: E402
from crucible.agents.neural_ppo import (  # noqa: E402
    NeuralPPOAgent,
    PPOConfig,
    PPOTrainer,
    behavior_clone,
    collate,
    collect_demonstrations,
    featurize_step,
)
from crucible.env import CrucibleEnv  # noqa: E402
from crucible.grammar import TaskFamily, generate_task  # noqa: E402
from crucible.rewards import RewardConfig  # noqa: E402
from crucible.splits import SplitLabel  # noqa: E402

pytestmark = pytest.mark.gpu


def _obs(seed: int = 1):
    spec = generate_task(TaskFamily.AFFORDANCE, seed=seed, split=SplitLabel.TRAIN)
    env = CrucibleEnv(seed=spec.seed, config={"task_spec": spec})
    return env.reset(), spec


def _cpu_agent(seed: int = 0):
    return NeuralPPOAgent(PPOConfig(device="cpu", seed=seed))


def test_featurize_and_forward_shapes():
    obs, _ = _obs()
    feats, candidates = featurize_step(obs)
    assert feats.n_cand == len(candidates)
    agent = _cpu_agent()
    batch = collate([feats], agent.device)
    logits, value = agent.net(batch)
    assert logits.shape == (1, feats.n_cand)
    assert value.shape == (1,)
    assert torch.isfinite(logits).all(), "single-step logits must be finite (no padding)"
    assert math.isfinite(float(value.item()))


def test_act_is_deterministic_for_same_seed():
    obs, _ = _obs()
    a1, a2 = _cpu_agent(seed=7), _cpu_agent(seed=7)
    act1, act2 = a1.act(obs), a2.act(obs)
    assert act1.kind == act2.kind and act1.args == act2.args


def test_padding_masks_extra_candidates():
    """A batch with differing candidate counts masks padded logits to -inf."""
    obs_a, _ = _obs(seed=1)
    obs_b, _ = _obs(seed=2)
    fa, ca = featurize_step(obs_a)
    fb, cb = featurize_step(obs_b)
    agent = _cpu_agent()
    batch = collate([fa, fb], agent.device)
    logits, _ = agent.net(batch)
    c_max = max(fa.n_cand, fb.n_cand)
    assert logits.shape == (2, c_max)
    # The shorter candidate set must have -inf in its padded tail.
    shorter = 0 if fa.n_cand < fb.n_cand else 1
    n_short = min(fa.n_cand, fb.n_cand)
    if n_short < c_max:
        assert torch.isinf(logits[shorter, n_short:]).all()


def test_ppo_update_runs_and_changes_params():
    specs = [generate_task(TaskFamily.AFFORDANCE, seed=s, split=SplitLabel.TRAIN) for s in range(3)]

    def env_factory(spec):
        return CrucibleEnv(
            seed=spec.seed,
            config={
                "task_spec": spec,
                "terminate_on_goal": True,
                "reward_config": RewardConfig(step_penalty=0.001),
            },
        )

    agent = NeuralPPOAgent(PPOConfig(device="cpu", seed=0, rollout_steps=128, minibatch=64))
    before = [p.detach().clone() for p in agent.net.parameters()]
    trainer = PPOTrainer(agent, env_factory)
    history = trainer.train(specs, total_updates=1)

    assert len(history) == 1
    rec = history[0]
    assert all(math.isfinite(rec[k]) for k in ("loss_policy", "loss_value", "loss_entropy"))
    after = list(agent.net.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "PPO update must modify network parameters"


def test_save_load_roundtrip(tmp_path):
    obs, _ = _obs()
    agent = _cpu_agent(seed=3)
    path = tmp_path / "ckpt.pt"
    agent.save(path)
    loaded = NeuralPPOAgent.load(path, device="cpu")
    a1, a2 = agent.act(obs), loaded.act(obs)
    assert a1.kind == a2.kind and a1.args == a2.args


def test_behavior_cloning_fits_heuristic_demos():
    """Cloning the heuristic's demonstrations drives training accuracy upward."""
    specs = [
        generate_task(TaskFamily.AFFORDANCE, seed=s, split=SplitLabel.TRAIN) for s in range(20)
    ]

    def env_factory(spec):
        return CrucibleEnv(seed=spec.seed, config={"task_spec": spec})

    demos = collect_demonstrations(specs, env_factory, HeuristicAgent(grid_size=6), grid_size=6)
    assert len(demos) > 0, "heuristic should solve some train seeds and yield demos"

    agent = _cpu_agent(seed=0)
    history = behavior_clone(agent, demos, epochs=60, batch_size=32, lr=1e-3)
    assert history[-1]["bc_acc"] > history[0]["bc_acc"], "BC accuracy should improve"
    assert history[-1]["bc_loss"] < history[0]["bc_loss"], "BC loss should decrease"


def test_rollout_episode_boundary_is_terminal():
    """The last transition of an episode is terminal so GAE never bootstraps
    across the concatenated rollout (even on truncation)."""
    spec = generate_task(TaskFamily.AFFORDANCE, seed=0, split=SplitLabel.TRAIN)

    def env_factory(s):
        return CrucibleEnv(
            seed=s.seed,
            config={"task_spec": s, "terminate_on_goal": True, "reward_config": RewardConfig()},
        )

    agent = NeuralPPOAgent(PPOConfig(device="cpu", seed=0))
    trainer = PPOTrainer(agent, env_factory)
    transitions, _, _ = trainer._run_episode(spec)
    assert transitions, "episode should produce transitions"
    assert transitions[-1].done is True, "final transition must be a terminal boundary"
