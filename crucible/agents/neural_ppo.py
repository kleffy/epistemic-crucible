"""Neural PPO baseline with a parametric (variable-size) action space.

This is the first *learned* agent in the suite and the only one that benefits
from a GPU. It implements the standard ``Agent`` interface, reading only the
public observation dict (the hidden-property boundary is preserved).

Design notes
------------
The action set is variable per step (``enumerate_candidate_actions``), so the
policy *scores candidates* rather than using a fixed action head: objects are
embedded, pooled into a state vector, each candidate is embedded from its action
kind / direction / referenced objects, and a shared MLP scores
``[state, candidate]`` into a logit. A softmax over the candidate set is the
policy; a value head reads the pooled state.

Training (PPO + GAE) lives in :class:`PPOTrainer`; the env supplies reward via
the opt-in reward layer (``terminate_on_goal=True``). ``torch`` is an optional
``[gpu]`` dependency — importing this module requires it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crucible.actions import ActionKind, Direction
from crucible.agents.base import Agent, enumerate_candidate_actions
from crucible.objects import (
    ObjectColor,
    ObjectShape,
    ObjectSize,
    ObjectState,
    ObjectTexture,
    ObjectType,
)
from crucible.utils.logging import get_logger
from crucible.utils.seeding import seed_torch

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Categorical vocabularies (stable index maps derived from the enums)
# ---------------------------------------------------------------------------


def _index_map(values) -> dict[str, int]:
    return {v.value: i for i, v in enumerate(values)}


_TYPE = _index_map(ObjectType)
_COLOR = _index_map(ObjectColor)
_SHAPE = _index_map(ObjectShape)
_TEXTURE = _index_map(ObjectTexture)
_SIZE = _index_map(ObjectSize)
_STATE = _index_map(ObjectState)
_KIND = {k.value: i for i, k in enumerate(ActionKind)}
_DIR = {d.value: i for i, d in enumerate(Direction)}
_DIR_NONE = len(_DIR)  # extra slot for actions without a direction

_CAT_SIZES = [
    len(_TYPE),
    len(_COLOR),
    len(_SHAPE),
    len(_TEXTURE),
    len(_SIZE),
    len(_STATE),
]
_N_OBJ_FLOAT = 5  # pos_r, pos_c, held, at_agent, has_marker
_N_GLOBAL = 5  # agent_r, agent_c, energy, step_frac, inv_count


# ---------------------------------------------------------------------------
# Featurization (obs dict -> numpy arrays). Pure, no torch, deterministic.
# ---------------------------------------------------------------------------


@dataclass
class StepFeatures:
    obj_cat: np.ndarray  # [N, 6] int64
    obj_flt: np.ndarray  # [N, _N_OBJ_FLOAT] float32
    glob: np.ndarray  # [_N_GLOBAL] float32
    cand_kind: np.ndarray  # [C] int64
    cand_dir: np.ndarray  # [C] int64
    cand_ref0: np.ndarray  # [C] int64 (object index or N for null)
    cand_ref1: np.ndarray  # [C] int64
    n_obj: int
    n_cand: int


def featurize_step(obs: dict, grid_size: int = 6) -> tuple[StepFeatures, list]:
    """Featurize one observation. Returns (features, candidate Action list)."""
    obj_ids = sorted(obs["objects"].keys())
    idx_of = {oid: i for i, oid in enumerate(obj_ids)}
    n = len(obj_ids)
    null = n  # null reference slot index

    agent_pos = tuple(obs["agent"]["pos"])
    inventory = set(obs["agent"]["inventory"])

    obj_cat = np.zeros((n, 6), dtype=np.int64)
    obj_flt = np.zeros((n, _N_OBJ_FLOAT), dtype=np.float32)
    for i, oid in enumerate(obj_ids):
        o = obs["objects"][oid]
        obj_cat[i] = [
            _TYPE.get(o["type"], 0),
            _COLOR.get(o["color"], 0),
            _SHAPE.get(o["shape"], 0),
            _TEXTURE.get(o["texture"], 0),
            _SIZE.get(o["size"], 0),
            _STATE.get(o["state"], 0),
        ]
        held = oid in inventory
        pos = o["pos"]
        if pos is None:
            pr = pc = -1.0
        else:
            pr, pc = pos[0] / grid_size, pos[1] / grid_size
        at_agent = pos is not None and tuple(pos) == agent_pos
        obj_flt[i] = [pr, pc, float(held), float(at_agent), float(bool(o.get("marker")))]

    glob = np.array(
        [
            agent_pos[0] / grid_size,
            agent_pos[1] / grid_size,
            obs["agent"].get("energy", 100) / 100.0,
            obs.get("step", 0) / max(obs.get("max_steps", 1), 1),
            len(inventory) / max(n, 1),
        ],
        dtype=np.float32,
    )

    candidates = enumerate_candidate_actions(obs, grid_size)
    c = len(candidates)
    cand_kind = np.zeros(c, dtype=np.int64)
    cand_dir = np.full(c, _DIR_NONE, dtype=np.int64)
    cand_ref0 = np.full(c, null, dtype=np.int64)
    cand_ref1 = np.full(c, null, dtype=np.int64)
    for j, act in enumerate(candidates):
        cand_kind[j] = _KIND[act.kind.value]
        a = act.args
        if act.kind == ActionKind.MOVE:
            cand_dir[j] = _DIR[a["direction"].value]
        elif act.kind in (ActionKind.PICKUP, ActionKind.DROP, ActionKind.INSPECT):
            cand_ref0[j] = idx_of.get(a.get("obj_id"), null)
        elif act.kind == ActionKind.APPLY:
            cand_ref0[j] = idx_of.get(a.get("tool_id"), null)
            cand_ref1[j] = idx_of.get(a.get("target_id"), null)
        elif act.kind == ActionKind.COMBINE:
            cand_ref0[j] = idx_of.get(a.get("obj_id_a"), null)
            cand_ref1[j] = idx_of.get(a.get("obj_id_b"), null)

    feats = StepFeatures(
        obj_cat=obj_cat,
        obj_flt=obj_flt,
        glob=glob,
        cand_kind=cand_kind,
        cand_dir=cand_dir,
        cand_ref0=cand_ref0,
        cand_ref1=cand_ref1,
        n_obj=n,
        n_cand=c,
    )
    return feats, candidates


def collate(batch: list[StepFeatures], device: torch.device) -> dict:
    """Pad a list of StepFeatures into batched, masked tensors on ``device``."""
    b = len(batch)
    n_max = max(f.n_obj for f in batch)
    c_max = max(f.n_cand for f in batch)

    obj_cat = torch.zeros(b, n_max, 6, dtype=torch.long)
    obj_flt = torch.zeros(b, n_max, _N_OBJ_FLOAT)
    obj_mask = torch.zeros(b, n_max)
    glob = torch.zeros(b, _N_GLOBAL)
    cand_kind = torch.zeros(b, c_max, dtype=torch.long)
    cand_dir = torch.full((b, c_max), _DIR_NONE, dtype=torch.long)
    # Null reference slot is index n_max (one past the padded objects).
    cand_ref0 = torch.full((b, c_max), n_max, dtype=torch.long)
    cand_ref1 = torch.full((b, c_max), n_max, dtype=torch.long)
    cand_mask = torch.zeros(b, c_max)

    for i, f in enumerate(batch):
        n, c = f.n_obj, f.n_cand
        if n:
            obj_cat[i, :n] = torch.from_numpy(f.obj_cat)
            obj_flt[i, :n] = torch.from_numpy(f.obj_flt)
            obj_mask[i, :n] = 1.0
        glob[i] = torch.from_numpy(f.glob)
        cand_kind[i, :c] = torch.from_numpy(f.cand_kind)
        cand_dir[i, :c] = torch.from_numpy(f.cand_dir)
        # Remap per-step null (f.n_obj) to the batch null slot (n_max).
        r0 = np.where(f.cand_ref0 == f.n_obj, n_max, f.cand_ref0)
        r1 = np.where(f.cand_ref1 == f.n_obj, n_max, f.cand_ref1)
        cand_ref0[i, :c] = torch.from_numpy(r0)
        cand_ref1[i, :c] = torch.from_numpy(r1)
        cand_mask[i, :c] = 1.0

    return {
        "obj_cat": obj_cat.to(device),
        "obj_flt": obj_flt.to(device),
        "obj_mask": obj_mask.to(device),
        "glob": glob.to(device),
        "cand_kind": cand_kind.to(device),
        "cand_dir": cand_dir.to(device),
        "cand_ref0": cand_ref0.to(device),
        "cand_ref1": cand_ref1.to(device),
        "cand_mask": cand_mask.to(device),
        "n_max": n_max,
    }


# ---------------------------------------------------------------------------
# Policy / value network
# ---------------------------------------------------------------------------


class PolicyNetwork(nn.Module):
    def __init__(self, embed_dim: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(size, embed_dim) for size in _CAT_SIZES]
        )
        self.obj_proj = nn.Sequential(
            nn.Linear(embed_dim * len(_CAT_SIZES) + _N_OBJ_FLOAT, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.null_obj = nn.Parameter(torch.zeros(hidden))
        self.glob_proj = nn.Sequential(nn.Linear(_N_GLOBAL, hidden), nn.ReLU())
        self.state_proj = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())

        self.kind_embed = nn.Embedding(len(_KIND), embed_dim)
        self.dir_embed = nn.Embedding(_DIR_NONE + 1, embed_dim)
        self.cand_proj = nn.Sequential(
            nn.Linear(embed_dim * 2 + hidden * 2, hidden),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def _encode_objects(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        obj_cat, obj_flt, obj_mask = b["obj_cat"], b["obj_flt"], b["obj_mask"]
        bsz, n_max, _ = obj_cat.shape
        embeds = [emb(obj_cat[..., i]) for i, emb in enumerate(self.cat_embeds)]
        cat = torch.cat(embeds + [obj_flt], dim=-1)
        h_obj = self.obj_proj(cat)  # [B, N, H]
        # Masked mean pool for the state summary.
        mask = obj_mask.unsqueeze(-1)
        pooled = (h_obj * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # [B, H]
        # Append a null object row for empty reference slots.
        null = self.null_obj.expand(bsz, 1, -1)
        h_aug = torch.cat([h_obj, null], dim=1)  # [B, N+1, H]
        return h_aug, pooled

    def forward(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (candidate logits [B, C], value [B])."""
        h_aug, pooled = self._encode_objects(b)
        state = self.state_proj(torch.cat([pooled, self.glob_proj(b["glob"])], dim=-1))

        ref0 = _gather_refs(h_aug, b["cand_ref0"])  # [B, C, H]
        ref1 = _gather_refs(h_aug, b["cand_ref1"])
        kind = self.kind_embed(b["cand_kind"])  # [B, C, E]
        direction = self.dir_embed(b["cand_dir"])  # [B, C, E]
        cand = self.cand_proj(torch.cat([kind, direction, ref0, ref1], dim=-1))  # [B,C,H]

        c = cand.shape[1]
        state_exp = state.unsqueeze(1).expand(-1, c, -1)
        logits = self.scorer(torch.cat([state_exp, cand], dim=-1)).squeeze(-1)  # [B,C]
        logits = logits.masked_fill(b["cand_mask"] == 0, float("-inf"))
        value = self.value_head(state).squeeze(-1)  # [B]
        return logits, value


def _gather_refs(h_aug: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Gather object embeddings [B, N+1, H] by candidate ref indices [B, C]."""
    h_dim = h_aug.shape[-1]
    idx = ref.unsqueeze(-1).expand(-1, -1, h_dim)  # [B, C, H]
    return torch.gather(h_aug, 1, idx)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    embed_dim: int = 32
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    epochs: int = 4
    minibatch: int = 256
    rollout_steps: int = 2048
    max_grad_norm: float = 0.5
    device: str = "cuda"
    seed: int = 0
    metadata: dict = field(default_factory=dict)


class NeuralPPOAgent(Agent):
    """PPO policy over the parametric action set. CPU/GPU via ``config.device``."""

    name = "neural_ppo"

    def __init__(self, config: PPOConfig | None = None, grid_size: int = 6) -> None:
        self.config = config or PPOConfig()
        self._grid_size = grid_size
        seed_torch(self.config.seed)
        dev = self.config.device
        if dev == "cuda" and not torch.cuda.is_available():
            _log.warning("cuda requested but unavailable; falling back to cpu")
            dev = "cpu"
        self.device = torch.device(dev)
        self.net = PolicyNetwork(self.config.embed_dim, self.config.hidden).to(self.device)
        self._rng = np.random.default_rng(self.config.seed)
        self._training = False

    def reset(self) -> None:  # stateless across steps; nothing to reset
        pass

    @torch.no_grad()
    def act(self, obs: dict):
        """Deterministic (argmax) action for evaluation."""
        feats, candidates = featurize_step(obs, self._grid_size)
        batch = collate([feats], self.device)
        logits, _ = self.net(batch)
        idx = int(torch.argmax(logits[0]).item())
        return candidates[idx]

    @torch.no_grad()
    def act_train(self, obs: dict):
        """Sample an action; return (Action, features, idx, log_prob, value)."""
        feats, candidates = featurize_step(obs, self._grid_size)
        batch = collate([feats], self.device)
        logits, value = self.net(batch)
        dist = torch.distributions.Categorical(logits=logits[0])
        idx = dist.sample()
        return (
            candidates[int(idx.item())],
            feats,
            int(idx.item()),
            float(dist.log_prob(idx).item()),
            float(value[0].item()),
        )

    def save(self, path) -> None:
        torch.save(
            {"state_dict": self.net.state_dict(), "config": self.config.__dict__},
            path,
        )

    @classmethod
    def load(cls, path, grid_size: int = 6, device: str | None = None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = PPOConfig(**ckpt["config"])
        if device is not None:
            cfg.device = device
        agent = cls(cfg, grid_size=grid_size)
        agent.net.load_state_dict(ckpt["state_dict"])
        agent.net.eval()
        return agent


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------


@dataclass
class _Transition:
    feats: StepFeatures
    idx: int
    log_prob: float
    value: float
    reward: float
    done: bool


class PPOTrainer:
    """Collect on-policy rollouts over training task specs and run PPO updates.

    The env supplies reward through the opt-in reward layer; specs should be the
    TRAIN-split tasks. Evaluation on TRAIN vs TEST is done separately via the
    standard runner with a frozen checkpoint.
    """

    def __init__(self, agent: NeuralPPOAgent, env_factory, reward_config=None) -> None:
        self.agent = agent
        self.cfg = agent.config
        self.env_factory = env_factory  # (spec) -> CrucibleEnv (terminate_on_goal=True)
        self.reward_config = reward_config
        self.opt = torch.optim.Adam(agent.net.parameters(), lr=self.cfg.lr)

    def _run_episode(self, spec) -> tuple[list[_Transition], float, bool]:
        env = self.env_factory(spec)
        obs = env.reset()
        obs["_public_hash"] = ""
        transitions: list[_Transition] = []
        ep_return, solved = 0.0, False
        for _ in range(spec.max_steps):
            action, feats, idx, log_prob, value = self.agent.act_train(obs)
            obs, reward, done, info = env.step(action)
            obs["_public_hash"] = info.get("public_state_hash_after", "")
            transitions.append(_Transition(feats, idx, log_prob, value, reward, done))
            ep_return += reward
            solved = bool(info.get("solved", False))  # true goal achievement, not shaping
            if done:
                break
        # Rollouts concatenate whole episodes, so the final transition must be a
        # terminal boundary (even on truncation — e.g. illegal actions that never
        # advance world.step). Otherwise _compute_gae would bootstrap from the
        # next, unrelated episode's value, contaminating advantages/returns.
        if transitions:
            transitions[-1].done = True
        return transitions, ep_return, solved

    def collect_rollout(self, specs: list, start: int) -> tuple[list[_Transition], dict]:
        buf: list[_Transition] = []
        returns, successes, n_ep = [], 0, 0
        i = start
        while len(buf) < self.cfg.rollout_steps:
            spec = specs[i % len(specs)]
            i += 1
            transitions, ep_ret, solved = self._run_episode(spec)
            buf.extend(transitions)
            returns.append(ep_ret)
            successes += int(solved)
            n_ep += 1
        stats = {
            "episodes": n_ep,
            "mean_return": float(np.mean(returns)),
            "success_rate": successes / max(n_ep, 1),
            "next_start": i,
        }
        return buf, stats

    def _compute_gae(self, buf: list[_Transition]) -> tuple[np.ndarray, np.ndarray]:
        n = len(buf)
        adv = np.zeros(n, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(n)):
            nonterminal = 0.0 if buf[t].done else 1.0
            next_value = 0.0 if (t + 1 >= n or buf[t].done) else buf[t + 1].value
            delta = buf[t].reward + self.cfg.gamma * next_value * nonterminal - buf[t].value
            last_adv = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * last_adv
            adv[t] = last_adv
        returns = adv + np.array([t.value for t in buf], dtype=np.float32)
        return adv, returns

    def _update(self, buf: list[_Transition]) -> dict:
        adv, returns = self._compute_gae(buf)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        old_lp = np.array([t.log_prob for t in buf], dtype=np.float32)
        idxs = np.array([t.idx for t in buf], dtype=np.int64)
        n = len(buf)

        losses = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
        n_batches = 0
        for _ in range(self.cfg.epochs):
            order = self.agent._rng.permutation(n)
            for s in range(0, n, self.cfg.minibatch):
                mb = order[s: s + self.cfg.minibatch]
                batch = collate([buf[j].feats for j in mb], self.agent.device)
                logits, value = self.agent.net(batch)
                dist = torch.distributions.Categorical(logits=logits)
                chosen = torch.as_tensor(idxs[mb], device=self.agent.device)
                new_lp = dist.log_prob(chosen)
                ratio = torch.exp(new_lp - torch.as_tensor(old_lp[mb], device=self.agent.device))
                mb_adv = torch.as_tensor(adv[mb], device=self.agent.device)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                mb_ret = torch.as_tensor(returns[mb], device=self.agent.device)
                value_loss = F.mse_loss(value, mb_ret)
                entropy = dist.entropy().mean()
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                )
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                losses["policy"] += float(policy_loss.item())
                losses["value"] += float(value_loss.item())
                losses["entropy"] += float(entropy.item())
                n_batches += 1
        for k in losses:
            losses[k] /= max(n_batches, 1)
        return losses

    def train(self, specs: list, total_updates: int) -> list[dict]:
        """Run ``total_updates`` PPO updates; return per-update history dicts."""
        history: list[dict] = []
        self.agent.net.train()
        start = 0
        for update in range(total_updates):
            buf, stats = self.collect_rollout(specs, start)
            start = stats.pop("next_start")
            losses = self._update(buf)
            record = {"update": update, **stats, **{f"loss_{k}": v for k, v in losses.items()}}
            history.append(record)
            _log.info(
                "update %d  success=%.3f  return=%.3f  pi=%.4f  v=%.4f",
                update,
                stats["success_rate"],
                stats["mean_return"],
                losses["policy"],
                losses["value"],
            )
        return history


# ---------------------------------------------------------------------------
# Behavior cloning (warm-start from the heuristic solver)
# ---------------------------------------------------------------------------


def _match_action_index(action, candidates: list) -> int | None:
    """Index of ``action`` within the candidate list, or None if absent."""
    for i, cand in enumerate(candidates):
        if cand.kind == action.kind and cand.args == action.args:
            return i
    return None


def collect_demonstrations(
    specs: list, env_factory, demo_agent, grid_size: int = 6, *, only_successful: bool = True
) -> list[tuple[StepFeatures, int]]:
    """Roll out ``demo_agent`` on ``specs`` and record (features, chosen index).

    The demonstrator (e.g. the heuristic solver) supplies expert actions; we keep
    transitions from solved episodes so cloning learns competent behaviour. The
    demonstrator exploits only public features, so no hidden information leaks.
    """
    from crucible.grammar import check_goal

    demos: list[tuple[StepFeatures, int]] = []
    for spec in specs:
        env = env_factory(spec)
        obs = env.reset()
        obs["_public_hash"] = ""
        demo_agent.reset()
        traj: list[tuple[StepFeatures, int]] = []
        solved = False
        for _ in range(spec.max_steps):
            feats, candidates = featurize_step(obs, grid_size)
            action = demo_agent.act(obs)
            idx = _match_action_index(action, candidates)
            obs, reward, done, info = env.step(action)
            obs["_public_hash"] = info.get("public_state_hash_after", "")
            demo_agent.observe_result(obs, reward, done, info)
            if idx is not None:
                traj.append((feats, idx))
            if check_goal(spec.goal, env.world):
                solved = True
                break
            if done:
                break
        if solved or not only_successful:
            demos.extend(traj)
    return demos


def behavior_clone(
    agent: NeuralPPOAgent,
    demos: list[tuple[StepFeatures, int]],
    *,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> list[dict]:
    """Supervised cross-entropy pretraining of the policy on expert demos."""
    if not demos:
        _log.warning("behavior_clone: no demonstrations collected; skipping")
        return []
    opt = torch.optim.Adam(agent.net.parameters(), lr=lr)
    agent.net.train()
    history: list[dict] = []
    n = len(demos)
    for ep in range(epochs):
        order = agent._rng.permutation(n)
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        for s in range(0, n, batch_size):
            mb = order[s: s + batch_size]
            feats = [demos[j][0] for j in mb]
            targets = torch.as_tensor(
                [demos[j][1] for j in mb], dtype=torch.long, device=agent.device
            )
            batch = collate(feats, agent.device)
            logits, _ = agent.net(batch)
            loss = F.cross_entropy(logits, targets)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.net.parameters(), 0.5)
            opt.step()
            total_loss += float(loss.item())
            total_acc += float((logits.argmax(-1) == targets).float().mean().item())
            n_batches += 1
        rec = {
            "epoch": ep,
            "bc_loss": total_loss / max(n_batches, 1),
            "bc_acc": total_acc / max(n_batches, 1),
        }
        history.append(rec)
        _log.info("bc epoch %d  loss=%.4f  acc=%.3f", ep, rec["bc_loss"], rec["bc_acc"])
    return history
