"""Run the v0.2 crossed-intervention controls or local vLLM study.

The model path targets an OpenAI-compatible *local* server. It does not make
paid provider calls. Start one pinned model server at a time, then select the
matching model label from ``configs/factorial_v02.yaml``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

_HERE = pathlib.Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from crucible.agents.llm_backends import ResponseCache, make_backend  # noqa: E402
from crucible.agents.prompting import build_label_map  # noqa: E402
from crucible.factorial import (  # noqa: E402
    PROTOCOL_NAME,
    TRACE_SCHEMA_VERSION,
    CommitOutcome,
    EpistemicAction,
    FactorialEpisode,
    MacroActionKind,
    QuartetCell,
    generate_affordance_challenge,
    generate_affordance_quartet,
    run_scripted_control,
)
from crucible.factorial_metrics import compute_factorial_metrics  # noqa: E402
from crucible.ledger import content_hash  # noqa: E402
from crucible.metrics import challenge_profile  # noqa: E402
from crucible.objects import ObjectType  # noqa: E402
from crucible.utils.serialization import to_dict  # noqa: E402

PROMPT_VERSION = "factorial-neutral-v1"
CANDIDATE_ORDER_VERSION = "macro-kind-slot-v1"
MACRO_COMPILER_VERSION = "legal-bfs-v1"
FACTORIAL_SYSTEM_PROMPT = """\
You control an epistemic policy over three tools. On each decision you receive
public tool features, any detector evidence, and a numbered list containing only
QUERY(tool slot) and COMMIT(tool slot) choices. QUERY reveals signal-positive or
signal-negative for that tool. A queried tool is removed from later QUERY choices.
COMMIT is terminal whether correct or incorrect. Reason briefly if useful, then end
with exactly one line: ACTION: <number>"""


@dataclass
class _StudyEpisode:
    index: int
    arm: str
    cell: QuartetCell
    protocol: FactorialEpisode
    prefix: list[dict[str, str]]
    label_map: dict[str, str]
    history: list[dict[str, str]] = field(default_factory=list)


_ACTION_INDEX_RE = re.compile(r"ACTION:\s*#?\s*(\d+)", re.IGNORECASE)


def build_macro_message(
    episode: FactorialEpisode,
    label_map: dict[str, str],
) -> tuple[str, list[EpistemicAction]]:
    """Render only public tool facts, accumulated evidence, and macro choices."""
    tools = []
    for slot in range(3):
        obj_id = next(
            spec.obj_id
            for spec in episode.cell.task_spec.object_specs
            if spec.role == f"tool_slot_{slot}"
        )
        visible = episode.obs["objects"][obj_id]
        marker = f", evidence={visible['marker']}" if visible.get("marker") else ""
        tools.append(
            f"  tool slot {slot} ({label_map[obj_id]}): {visible['color']}, "
            f"{visible['shape']}, {visible['texture']}, {visible['size']}{marker}"
        )
    actions = episode.macro_actions()
    choices = "\n".join(
        f"  {index}. {action.kind.value.upper()}(tool slot {action.slot})"
        for index, action in enumerate(actions)
    )
    message = (
        "GOAL: identify and commit the tool that opens the gate.\n\n"
        "Public tools:\n"
        + "\n".join(tools)
        + "\n\nAvailable epistemic actions:\n"
        + choices
        + "\n\nEnd with exactly `ACTION: <number>`. A COMMIT is terminal."
    )
    return message, actions


def parse_macro_action(
    response: str, candidates: list[EpistemicAction]
) -> tuple[EpistemicAction | None, str]:
    match = _ACTION_INDEX_RE.search(response)
    if match and 0 <= int(match.group(1)) < len(candidates):
        return candidates[int(match.group(1))], "parsed"
    return None, "invalid"


def build_factorial_fewshot_prefix(
    arm: str,
    *,
    count: int = 5,
    base_seed: int = 9_000,
) -> list[dict[str, str]]:
    """Build cue or detector-evidence demonstrations without raw object IDs."""
    if arm == "neutral":
        return []
    if arm not in {"cue", "mechanism"}:
        raise ValueError(f"unknown demonstration arm {arm!r}")
    lines: list[str] = []
    for index in range(count):
        quartet = generate_affordance_quartet(base_seed + index)
        if arm == "cue":
            slot = index % 2
            cell = quartet.cell(slot, slot)
        else:
            cell = quartet.cell(index % 2, (index // 2) % 2)
        tools = [obj for obj in cell.task_spec.object_specs if obj.obj_type == ObjectType.TOOL]
        descriptions = ", ".join(f"{obj.color.value} {obj.shape.value} tool" for obj in tools)
        correct = tools[cell.mechanism_carrier_slot]
        if arm == "cue":
            lines.append(
                f"Example {index + 1}: available tools were {descriptions}; "
                "the red tool was applied and opened the gate."
            )
        else:
            lines.append(
                f"Example {index + 1}: available tools were {descriptions}; a detector "
                f"marked the {correct.color.value} {correct.shape.value} tool "
                "signal-positive, and that tool opened the gate."
            )
    return [{"role": "user", "content": "Prior episodes:\n" + "\n".join(lines)}]


def run_controls(cfg: dict[str, Any], output_dir: pathlib.Path) -> dict[str, Any]:
    seeds = [int(seed) for seed in cfg.get("seeds", range(64))]
    controls = cfg.get(
        "controls",
        [
            "mechanism_oracle",
            "detector_policy",
            "cue_follower",
            "anti_cue",
            "fixed_slot",
            "random_committer",
            "abstain",
        ],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for control in controls:
        outcomes = [
            run_scripted_control(cell, control)
            for seed in seeds
            for cell in generate_affordance_quartet(seed).cells.values()
        ]
        report = compute_factorial_metrics(outcomes)
        reports[control] = asdict(report)
        path = output_dir / f"factorial_control_{control}.jsonl"
        with path.open("w") as handle:
            for outcome in outcomes:
                handle.write(json.dumps(_outcome_record(outcome, control), sort_keys=True) + "\n")
    summary = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "seeds": seeds,
        "reports": reports,
    }
    (output_dir / "factorial_control_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def run_local_model(
    cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    """Run one pinned model already served by a local OpenAI-compatible endpoint."""
    model_id = str(model_cfg["id"])
    revision = str(model_cfg["revision"])
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError(f"model {model_id} must use a full 40-character revision SHA")
    seeds = [int(seed) for seed in cfg.get("seeds", range(64))]
    arms = cfg.get("prompt_arms", _default_prompt_arms())
    variant = str(cfg.get("variant", "quartet_2x2"))
    history_mode = str(cfg.get("history_mode", "full"))
    if history_mode not in {"full", "compact_ledger"}:
        raise ValueError("history_mode must be 'full' or 'compact_ledger'")
    run_manifest = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "variant": variant,
        "history_mode": history_mode,
        "prompt_version": PROMPT_VERSION,
        "candidate_order_version": CANDIDATE_ORDER_VERSION,
        "macro_action_compiler_version": MACRO_COMPILER_VERSION,
        "repository_commit": _repository_commit(),
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_revision": model_cfg.get("tokenizer_revision", revision),
        "backend": "local_vllm",
        "server": cfg.get("base_url", "http://127.0.0.1:8000/v1"),
        "seeds": seeds,
        "arms": arms,
        "generation": model_cfg.get("generation", {}),
        "serving": model_cfg.get("serving", {}),
        "server_launch_manifest_hash": content_hash(model_cfg.get("serving", {})),
        "runtime_hardware": _runtime_hardware(),
    }
    manifest_hash = content_hash(run_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(output_dir / "cache" / f"{model_cfg['label']}-{manifest_hash[:12]}.jsonl")
    output_ceiling = int(model_cfg.get("output_ceiling", 1024))
    generation = {
        "max_new_tokens": output_ceiling,
        "max_completion_tokens": output_ceiling,
        "temperature": 0.0,
        "concurrency": int(cfg.get("concurrency", 8)),
        "run_manifest": run_manifest,
        **model_cfg.get("generation", {}),
    }
    backend = make_backend(
        "openai",
        model_id,
        cache=cache,
        base_url=cfg.get("base_url", "http://127.0.0.1:8000/v1"),
        model_revision=revision,
        **generation,
    )
    episodes = _build_episodes(seeds, arms, variant=variant)
    trace_path = output_dir / f"factorial_{model_cfg['label']}_{int(time.time())}.jsonl"
    with trace_path.open("w") as handle:
        while active := [episode for episode in episodes if not episode.protocol.done]:
            conversations: list[list[dict]] = []
            prompts: list[str] = []
            candidates_by_episode: list[list[EpistemicAction]] = []
            cache_contexts: list[dict[str, Any]] = []
            for episode in active:
                message, candidates = build_macro_message(episode.protocol, episode.label_map)
                episode.history.append({"role": "user", "content": message})
                episode.protocol.ledger.record_message("user", message)
                prompts.append(message)
                candidates_by_episode.append(candidates)
                # Full history is the primary protocol: no sliding truncation.
                if history_mode == "full":
                    conversations.append([*episode.prefix, *episode.history])
                else:
                    conversations.append(
                        [*episode.prefix, episode.protocol.ledger.compact_message()]
                    )
                cache_contexts.append(
                    {
                        "base_seed": episode.cell.task_spec.seed,
                        "condition_id": episode.cell.condition_id,
                        "prompt_arm": episode.arm,
                        "decision_index": len(episode.protocol.detector_evidence),
                    }
                )
            records = backend.generate_batch_records(
                FACTORIAL_SYSTEM_PROMPT,
                conversations,
                cache_contexts=cache_contexts,
            )
            for episode, prompt, candidates, record in zip(
                active, prompts, candidates_by_episode, records
            ):
                episode.history.append({"role": "assistant", "content": record.response})
                episode.protocol.ledger.record_message("assistant", record.response)
                action, parse_status = parse_macro_action(record.response, candidates)
                if action is None:
                    episode.protocol.force_commit(0)
                    action_record = {"kind": "forced_commit", "slot": 0}
                    effect_text = "Invalid action parse; terminal fallback COMMIT(tool slot 0)."
                    low_level_actions: list[dict] = []
                else:
                    macro_result = episode.protocol.macro_step(action)
                    action_record = {"kind": action.kind.value, "slot": action.slot}
                    low_level_actions = [
                        to_dict(item) for item in macro_result["low_level_actions"]
                    ]
                    effect_text = (
                        f"QUERY(tool slot {action.slot}) returned {macro_result['evidence']}."
                        if action.kind == MacroActionKind.QUERY
                        else f"COMMIT(tool slot {action.slot}) ended the episode."
                    )
                episode.history.append(
                    {"role": "user", "content": f"Result of last action: {effect_text}"}
                )
                episode.protocol.ledger.record_message(
                    "user", f"Result of last action: {effect_text}"
                )
                step_record = {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "protocol": PROTOCOL_NAME,
                    "run_manifest_hash": manifest_hash,
                    "base_seed": episode.cell.task_spec.seed,
                    "base_world_id": episode.cell.task_spec.metadata["base_world_id"],
                    "condition": {
                        "id": episode.cell.condition_id,
                        "mechanism_slot": episode.cell.mechanism_slot,
                        "cue_slot": episode.cell.cue_slot,
                        "mechanism_carrier_slot": episode.cell.mechanism_carrier_slot,
                        "cue_carrier_slot": episode.cell.cue_carrier_slot,
                    },
                    "prompt_arm": episode.arm,
                    "step": episode.protocol.env.world.step,
                    "prompt_hash": content_hash(prompt),
                    "candidate_set_hash": content_hash([asdict(item) for item in candidates]),
                    "label_map_hash": content_hash(episode.label_map),
                    "base_spec_hash": content_hash(to_dict(episode.cell.task_spec)),
                    "action": action_record,
                    "low_level_actions": low_level_actions,
                    "parse_status": parse_status,
                    "fallback_action": action_record if parse_status != "parsed" else None,
                    "raw_provider_status": record.finish_reason,
                    "generation": record.to_dict(),
                }
                handle.write(json.dumps(step_record, sort_keys=True, default=str) + "\n")
        outcomes = [episode.protocol.outcome() for episode in episodes]
        for episode, outcome in zip(episodes, outcomes):
            handle.write(
                json.dumps(
                    _outcome_record(outcome, str(model_cfg["label"]), episode.arm),
                    sort_keys=True,
                )
                + "\n"
            )
    arm_reports = {}
    for arm in arms:
        selected = [
            outcome for episode, outcome in zip(episodes, outcomes) if episode.arm == arm["name"]
        ]
        if variant == "challenge_3x2":
            arm_reports[arm["name"]] = asdict(
                challenge_profile(
                    _outcome_record(outcome, str(model_cfg["label"])) for outcome in selected
                )
            )
        else:
            arm_reports[arm["name"]] = asdict(compute_factorial_metrics(selected))
    summary = {
        "manifest": run_manifest,
        "manifest_hash": manifest_hash,
        "trace": str(trace_path),
        "reports": arm_reports,
    }
    (output_dir / f"factorial_{model_cfg['label']}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_HERE,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _runtime_hardware() -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        runtime.update(
            {
                "pytorch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        runtime.update({"pytorch": None, "cuda_runtime": None, "gpu": None})
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    runtime["driver"] = driver.stdout.strip() if driver.returncode == 0 else None
    return runtime


def _build_episodes(
    seeds: list[int],
    arms: list[dict[str, Any]],
    *,
    variant: str = "quartet_2x2",
) -> list[_StudyEpisode]:
    episodes: list[_StudyEpisode] = []
    if variant not in {"quartet_2x2", "challenge_3x2"}:
        raise ValueError(f"unknown factorial variant {variant!r}")
    generator = (
        generate_affordance_challenge if variant == "challenge_3x2" else generate_affordance_quartet
    )
    for seed in seeds:
        quartet = generator(seed)
        # Stable IDs make one base-world label map reusable across all cells.
        first = FactorialEpisode(quartet.cells[(0, 0)])
        first_obs = first.reset()
        base_label_map = build_label_map(first_obs)
        for arm in arms:
            default_arm_seeds = seeds[: int(arm.get("seed_count", len(seeds)))]
            arm_seeds = {int(value) for value in arm.get("seeds", default_arm_seeds)}
            if seed not in arm_seeds:
                continue
            prefix = build_factorial_fewshot_prefix(
                str(arm["mode"]),
                count=int(arm.get("count", 5)),
                base_seed=int(arm.get("base_seed", 9_000)),
            )
            for cell in quartet.cells.values():
                protocol = FactorialEpisode(cell)
                protocol.reset()
                episodes.append(
                    _StudyEpisode(
                        index=len(episodes),
                        arm=str(arm["name"]),
                        cell=cell,
                        protocol=protocol,
                        prefix=prefix,
                        label_map=dict(base_label_map),
                    )
                )
    return episodes


def _default_prompt_arms() -> list[dict[str, Any]]:
    return [
        {"name": "neutral", "mode": "neutral", "count": 0, "base_seed": 9_000},
        {"name": "cue_a", "mode": "cue", "count": 5, "base_seed": 9_000},
        {"name": "cue_b", "mode": "cue", "count": 5, "base_seed": 10_000},
        {
            "name": "mechanism_a",
            "mode": "mechanism",
            "count": 5,
            "base_seed": 11_000,
        },
        {
            "name": "mechanism_b",
            "mode": "mechanism",
            "count": 5,
            "base_seed": 12_000,
        },
    ]


def _outcome_record(
    outcome: CommitOutcome,
    agent: str,
    arm: str | None = None,
) -> dict[str, Any]:
    ledger = list(outcome.trace)
    return {
        "kind": "outcome",
        "schema_version": TRACE_SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "agent": agent,
        "prompt_arm": arm,
        **{key: value for key, value in asdict(outcome).items() if key != "trace"},
        "episode_ledger": ledger,
        "episode_ledger_hash": content_hash(ledger),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/factorial_v02.yaml")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--model", help="Model label from the config; local server must match")
    parser.add_argument("--output-dir", default="results/factorial_v02")
    args = parser.parse_args(argv)
    with pathlib.Path(args.config).open() as handle:
        cfg = yaml.safe_load(handle) or {}
    output_dir = pathlib.Path(args.output_dir)
    if args.controls:
        run_controls(cfg, output_dir)
        return
    if not args.model:
        parser.error("choose --controls or provide --model LABEL")
    models = {str(model["label"]): model for model in cfg.get("models", [])}
    if args.model not in models:
        parser.error(f"unknown model label {args.model!r}; choose from {sorted(models)}")
    run_local_model(cfg, models[args.model], output_dir)


if __name__ == "__main__":
    main()
