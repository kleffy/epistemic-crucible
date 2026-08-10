"""Batched LLM evaluation harness.

Runs a panel of LLMs over family x seed task instances using *batched lockstep
rollouts*: all active episodes are advanced one step per batched generation, so
local open-weights models on a single GPU finish in reasonable time. Emits the
same JSONL trace schema as ``run_baselines.run_episode`` (agent name = model
label), so results flow through the unchanged diagnostic metric vector.

Usage:
    python experiments/run_llm_eval.py --config configs/llm.yaml
    python experiments/run_llm_eval.py --models Qwen/Qwen2.5-7B-Instruct --families affordance
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# Reduce CUDA fragmentation for long, growing LLM contexts (must precede torch).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml

_HERE = pathlib.Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from crucible.actions import is_legal  # noqa: E402
from crucible.agents.base import INTERVENTION_KINDS, enumerate_candidate_actions  # noqa: E402
from crucible.agents.llm_backends import ResponseCache, make_backend  # noqa: E402
from crucible.agents.prompting import (  # noqa: E402
    SYSTEM_PROMPT,
    anonymize_text,
    build_user_message,
    describe_goal,
    oracle_hint,
    parse_action,
)
from crucible.env import CrucibleEnv  # noqa: E402
from crucible.grammar import TaskFamily, check_goal, generate_task  # noqa: E402
from crucible.splits import SplitLabel  # noqa: E402
from crucible.utils.logging import get_logger  # noqa: E402

_log = get_logger(__name__)


@dataclass
class _Episode:
    idx: int
    spec: Any
    env: CrucibleEnv
    obs: dict
    goal_text: str
    prefix: list[dict] = field(default_factory=list)  # constant few-shot demos
    history: list[dict] = field(default_factory=list)  # interactive turns (windowed)
    label_map: dict = field(default_factory=dict)  # real id -> neutral label
    step_i: int = 0
    done: bool = False
    illegal: int = 0
    interventions: int = 0
    effects_all: list[str] = field(default_factory=list)
    candidates: list = field(default_factory=list)


def _label(model_id: str) -> str:
    """Short trace-friendly agent label, e.g. 'qwen2.5-7b-instruct'."""
    return model_id.split("/")[-1].lower()


def _cache_label(label: str, quantization: str | None) -> str:
    """Cache-file label namespaced by quantization.

    Quantization changes the model's outputs but is not part of ``gen_params``,
    so it does not enter ``ResponseCache.key``. Without namespacing, a 4-bit run
    and a full-precision run of the same model would share one cache file and
    silently reuse each other's completions, corrupting the comparison. Encoding
    the quantization in the file name keeps each configuration in its own cache.
    Full-precision runs keep the bare label, so existing caches stay valid.
    """
    return f"{label}-{quantization}" if quantization else label


def build_fewshot_prefix(
    family: TaskFamily, k: int, base_seed: int = 9000, mode: str = "cue"
) -> list[dict]:
    """In-context demonstrations for the dose-response shortcut probe.

    Modes (all stated by visible features only, no object IDs, so no split leaks):
    - ``cue``: TRAIN worlds where the conductive tool is always RED, so K demos
      teach the spurious 'red opens the gate' rule.
    - ``anticue``: TEST worlds where the conductive tool's colour varies across
      examples, so the demonstrations show colour is not predictive.
    - ``mechanistic``: demos describe the discovery procedure (test each tool, the
      one with an effect is the one that works) without naming a colour.
    """
    split = SplitLabel.TRAIN if mode in ("cue", "mechanistic") else SplitLabel.TEST
    lines: list[str] = []
    for i in range(k):
        spec = generate_task(family, seed=base_seed + i, split=split)
        correct = next((o for o in spec.object_specs if o.role == "correct_tool"), None)
        tools = [o for o in spec.object_specs if o.obj_type.value == "tool"]
        if correct is None or not tools:
            continue
        tool_desc = ", ".join(f"{t.color.value} {t.shape.value}" for t in tools)
        if mode == "mechanistic":
            lines.append(
                f"Example {i + 1}: tools were {tool_desc}. Applying each tool to the gate in "
                "turn, only one had an effect and opened the gate; the others did nothing. "
                "Testing the tools is how the working one was found."
            )
        else:  # cue or anticue
            lines.append(
                f"Example {i + 1}: tools were {tool_desc}; "
                f"applying the {correct.color.value} tool opened the gate."
            )
    if not lines:
        return []
    return [{"role": "user", "content": "Previously observed outcomes:\n" + "\n".join(lines)}]


def _step_record(ep: _Episode, action, info, label: str) -> dict:
    return {
        "kind": "step",
        "episode": ep.idx,
        "seed": ep.spec.seed,
        "family": ep.spec.family.value,
        "split": ep.spec.split.value,
        "agent": label,
        "step": ep.step_i,
        "action": {"kind": action.kind.value, "args": {k: str(v) for k, v in action.args.items()}},
        "effects": info.get("effects", []),
        "legal": info.get("legal", False),
        "energy": ep.obs.get("agent", {}).get("energy", 100),
        "done": info.get("solved", False) or ep.done,
    }


def _outcome_record(ep: _Episode, label: str) -> dict:
    goal_achieved = check_goal(ep.spec.goal, ep.env.world)
    return {
        "kind": "outcome",
        "episode": ep.idx,
        "seed": ep.spec.seed,
        "family": ep.spec.family.value,
        "split": ep.spec.split.value,
        "agent": label,
        "goal_achieved": goal_achieved,
        "steps": ep.step_i,
        "interventions": ep.interventions,
        "energy_remaining": ep.obs.get("agent", {}).get("energy", 0),
        "illegal_rate": round(ep.illegal / max(ep.step_i, 1), 4),
        "unique_effects": sorted(set(ep.effects_all)),
    }


def run_model(model_id: str, backend_name: str, cfg: dict, out_dir: pathlib.Path) -> dict:
    """Run one model over all family x seed instances; write a JSONL trace."""
    label = _label(model_id)
    families = [TaskFamily(f) for f in cfg.get("families", ["affordance"])]
    seeds = [int(s) for s in cfg.get("seeds", [0])]
    grid_size = int(cfg.get("grid_size", 6))
    max_steps_cap = cfg.get("max_steps")
    # Sliding context window (messages): bound prompt growth so batched rollouts
    # don't OOM. The observation is fully restated each step, so a short window
    # of recent (obs, action, result) turns is sufficient.
    max_history = int(cfg.get("max_history", 9))
    compact = bool(cfg.get("compact_layout", False))
    oracle_level = cfg.get("oracle")  # ablation ladder: intervention/property/rule
    cache_dir = pathlib.Path(cfg.get("cache_dir", "results/llm_cache"))
    cache = ResponseCache(cache_dir / f"{_cache_label(label, cfg.get('quantization'))}.jsonl")
    backend_kwargs = dict(
        cache=cache,
        batch_size=int(cfg.get("batch_size", 32)),
        max_new_tokens=int(cfg.get("max_new_tokens", 192)),
        concurrency=int(cfg.get("concurrency", 4)),
    )
    if cfg.get("quantization"):
        backend_kwargs["quantization"] = cfg["quantization"]
    backend = make_backend(backend_name, model_id, **backend_kwargs)

    # Optional few-shot in-context shortcut probe: prepend K solved TRAIN
    # demonstrations and evaluate on the forced split (default TEST).
    fewshot = cfg.get("fewshot") or {}
    fewshot_k = int(fewshot.get("k", 0))
    fewshot_mode = fewshot.get("mode", "cue")
    forced_split = SplitLabel(fewshot["split"]) if fewshot.get("split") else None
    # For a clean shortcut diagnostic, evaluate each seed under BOTH the train
    # and test variant (paired). Otherwise the per-seed auto-split is used.
    eval_splits = [SplitLabel(s) for s in cfg.get("eval_splits", [])] or [forced_split]

    # Build all episodes.
    episodes: list[_Episode] = []
    idx = 0
    for fam in families:
        prefix = build_fewshot_prefix(fam, fewshot_k, mode=fewshot_mode) if fewshot_k else []
        for seed in seeds:
            for split in eval_splits:
                spec = generate_task(fam, seed=seed, split=split)
                env = CrucibleEnv(
                    seed=spec.seed, config={"task_spec": spec, "compact_layout": compact}
                )
                obs = env.reset()
                episodes.append(
                    _Episode(
                        idx=idx,
                        spec=spec,
                        env=env,
                        obs=obs,
                        goal_text=describe_goal(spec.goal),
                        prefix=list(prefix),
                    )
                )
                idx += 1

    max_steps = max(e.spec.max_steps for e in episodes)
    if max_steps_cap:
        max_steps = min(max_steps, int(max_steps_cap))

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"llm_{label}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    with jsonl_path.open("w") as fh:
        for step in range(max_steps):
            active = [e for e in episodes if not e.done and e.step_i < e.spec.max_steps]
            if not active:
                break
            # Build prompts for all active episodes (lockstep).
            for e in active:
                e.obs["_goal_text"] = e.goal_text
                # Present only structurally-legal actions (the harness holds the
                # world). Legality is public; applying a wrong tool stays legal,
                # so the causal decision is preserved — only out-of-bounds moves,
                # pickup-not-here, apply-not-adjacent etc. are filtered out.
                legal = [
                    a
                    for a in enumerate_candidate_actions(e.obs, grid_size)
                    if is_legal(a, e.env.world)
                ]
                user_msg, e.candidates, e.label_map = build_user_message(
                    e.obs, e.goal_text, grid_size, candidates=legal, label_map=e.label_map
                )
                if oracle_level:
                    user_msg = oracle_hint(e.spec, e.label_map, oracle_level) + user_msg
                e.history.append({"role": "user", "content": user_msg})
            raws = backend.generate_batch(
                SYSTEM_PROMPT, [e.prefix + e.history[-max_history:] for e in active]
            )
            for e, raw in zip(active, raws):
                e.history.append({"role": "assistant", "content": raw})
                action = parse_action(raw, e.candidates)
                obs_after, reward, done, info = e.env.step(action)
                fh.write(json.dumps(_step_record(e, action, info, label)) + "\n")
                if not info.get("legal", False):
                    e.illegal += 1
                if action.kind in INTERVENTION_KINDS:
                    e.interventions += 1
                e.effects_all.extend(info.get("effects", []))
                effects = info.get("effects", [])
                if effects:
                    anon = anonymize_text(str(effects), e.label_map)
                    e.history.append({"role": "user", "content": f"Result of last action: {anon}"})
                e.obs = obs_after
                e.step_i += 1
                if check_goal(e.spec.goal, e.env.world) or done:
                    e.done = True
            _log.info("step %d: %d active episodes", step, len(active))
        for e in episodes:
            fh.write(json.dumps(_outcome_record(e, label)) + "\n")

    solved = sum(1 for e in episodes if check_goal(e.spec.goal, e.env.world))
    summary = {
        "model": model_id,
        "label": label,
        "episodes": len(episodes),
        "solved": solved,
        "tsr": round(solved / max(len(episodes), 1), 4),
        "elapsed_sec": round(time.time() - t0, 1),
        "trace": str(jsonl_path),
    }
    _log.info(
        "model=%s tsr=%.3f (%d/%d) in %.0fs",
        label,
        summary["tsr"],
        solved,
        len(episodes),
        summary["elapsed_sec"],
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Batched LLM evaluation.")
    p.add_argument("--config", default="configs/llm.yaml")
    p.add_argument("--models", nargs="+", help="Override model list")
    p.add_argument("--families", nargs="+", help="Override families")
    p.add_argument("--seeds", nargs="+", type=int, help="Override seeds")
    p.add_argument("--backend", help="Override backend (transformers/mock/anthropic/openai)")
    p.add_argument(
        "--compact", action="store_true", help="Decision-focused layout (objects near the agent)"
    )
    p.add_argument(
        "--both-splits",
        action="store_true",
        help="Evaluate each seed under both the train and test variant (paired)",
    )
    p.add_argument("--fewshot-k", type=int, help="Few-shot probe: K solved TRAIN demos")
    p.add_argument("--fewshot-split", default="test", help="Few-shot probe eval split")
    p.add_argument(
        "--fewshot-mode",
        default="cue",
        choices=["cue", "mechanistic", "anticue"],
        help="Demonstration style for the few-shot/dose-response probe",
    )
    p.add_argument("--quantization", choices=["4bit"], help="Local model quantization")
    p.add_argument("--batch-size", type=int, help="Local generation batch size")
    p.add_argument(
        "--oracle",
        choices=["intervention", "property", "rule"],
        help="Oracle-ladder ablation: inject a labelled ground-truth hint",
    )
    p.add_argument("--output-dir", default="results")
    args = p.parse_args(argv)

    with pathlib.Path(args.config).open() as f:
        cfg = yaml.safe_load(f) or {}
    if args.models:
        cfg["models"] = args.models
    if args.families:
        cfg["families"] = args.families
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.backend:
        cfg["backend"] = args.backend
    if args.compact:
        cfg["compact_layout"] = True
    if args.both_splits:
        cfg["eval_splits"] = ["train", "test"]
    if args.fewshot_k is not None:
        cfg["fewshot"] = {
            "k": args.fewshot_k,
            "split": args.fewshot_split,
            "mode": args.fewshot_mode,
        }
    if args.quantization:
        cfg["quantization"] = args.quantization
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.oracle:
        cfg["oracle"] = args.oracle

    backend_name = cfg.get("backend", "transformers")
    if backend_name == "transformers":
        import torch

        if torch.cuda.is_available():
            _log.info("CUDA device: %s", torch.cuda.get_device_name(0))

    out_dir = pathlib.Path(args.output_dir)
    summaries = [run_model(m, backend_name, cfg, out_dir) for m in cfg.get("models", [])]
    summary_path = out_dir / "llm_eval_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    _log.info("summary -> %s", summary_path)
    for s in summaries:
        print(
            f"{s['label']:28s} TSR={s['tsr']:.3f} "
            f"({s['solved']}/{s['episodes']}) {s['elapsed_sec']:.0f}s"
        )


if __name__ == "__main__":
    main()
