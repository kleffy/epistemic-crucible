"""Run the v0.2 crossed-intervention controls or local vLLM study.

The model path targets an OpenAI-compatible *local* server. It does not make
paid provider calls. Start one pinned model server at a time, then select the
matching model label from ``configs/factorial_v02.yaml``.

Live acquisition disables cache reads and validates uncached independent calls.
Artifact replay is cache-only and cannot emit a serving-validation pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import random
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
from crucible.factorial_metrics import (  # noqa: E402
    compute_factorial_metrics,
    paired_arm_contrast,
)
from crucible.ledger import content_hash  # noqa: E402
from crucible.metrics import challenge_profile  # noqa: E402
from crucible.objects import ObjectType  # noqa: E402
from crucible.utils.serialization import to_dict  # noqa: E402

PROMPT_VERSION = "factorial-neutral-v1"
CANDIDATE_ORDER_VERSION = "macro-kind-slot-v1"
MACRO_COMPILER_VERSION = "legal-bfs-v1"
LIVE_ACQUISITION = "live_acquisition"
ARTIFACT_REPLAY = "artifact_replay"
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
    count: int = 6,
    base_seed: int = 9_000,
) -> list[dict[str, str]]:
    """Build cue or detector-evidence demonstrations without raw object IDs."""
    if arm == "neutral":
        return []
    if arm not in {"cue", "mechanism"}:
        raise ValueError(f"unknown demonstration arm {arm!r}")
    examples = build_factorial_demo_examples(arm, count=count, base_seed=base_seed)
    return [
        {
            "role": "user",
            "content": "Prior episodes:\n" + "\n".join(example["text"] for example in examples),
        }
    ]


def build_factorial_demo_examples(
    arm: str,
    *,
    count: int = 6,
    base_seed: int = 9_000,
) -> list[dict[str, Any]]:
    """Return frozen, auditable six-shot demonstration examples."""
    if arm not in {"cue", "mechanism"}:
        raise ValueError(f"unknown demonstration arm {arm!r}")
    if count != 6:
        raise ValueError("v0.2 cue and mechanism demonstration arms require exactly six shots")
    order = list(range(6))
    random.Random(base_seed).shuffle(order)
    examples: list[dict[str, Any]] = []
    for display_index, source_index in enumerate(order, start=1):
        if arm == "cue":
            quartet = generate_affordance_quartet(base_seed + source_index)
            local_slot = source_index % 2
            cell = quartet.cell(local_slot, local_slot)
        else:
            challenge = generate_affordance_challenge(base_seed)
            conditions = [(mechanism, cue) for mechanism in range(3) for cue in range(2)]
            cell = challenge.cell(*conditions[source_index])
        tools = [obj for obj in cell.task_spec.object_specs if obj.obj_type == ObjectType.TOOL]
        descriptions = ", ".join(f"{obj.color.value} {obj.shape.value} tool" for obj in tools)
        correct = tools[cell.mechanism_carrier_slot]
        if arm == "cue":
            text = (
                f"Example {display_index}: available tools were {descriptions}; "
                "the red tool was applied and opened the gate."
            )
        else:
            text = (
                f"Example {display_index}: available tools were {descriptions}; a detector "
                f"marked the {correct.color.value} {correct.shape.value} tool "
                "signal-positive, and that tool opened the gate."
            )
        examples.append(
            {
                "text": text,
                "source_index": source_index,
                "condition_id": cell.condition_id,
                "positive_color": correct.color.value,
            }
        )
    return examples


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
            "focal_uniform",
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
    *,
    stage: str,
    stage_manifest: dict[str, Any],
    stage_manifest_hash: str,
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
    execution_mode = str(cfg.get("execution_mode", LIVE_ACQUISITION))
    if execution_mode not in {LIVE_ACQUISITION, ARTIFACT_REPLAY}:
        raise ValueError(f"execution_mode must be {LIVE_ACQUISITION!r} or {ARTIFACT_REPLAY!r}")
    cache_policy = {
        "read_enabled": execution_mode == ARTIFACT_REPLAY,
        "write_enabled": execution_mode == LIVE_ACQUISITION,
        "require_hits": execution_mode == ARTIFACT_REPLAY,
    }
    run_manifest = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "variant": variant,
        "history_mode": history_mode,
        "execution_mode": execution_mode,
        "cache_policy": cache_policy,
        "prompt_version": PROMPT_VERSION,
        "candidate_order_version": CANDIDATE_ORDER_VERSION,
        "macro_action_compiler_version": MACRO_COMPILER_VERSION,
        "repository_commit": _repository_commit(),
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_revision": model_cfg.get("tokenizer_revision", revision),
        "backend": "local_vllm",
        "server": cfg.get("base_url", "http://127.0.0.1:8000/v1"),
        "concurrency": int(cfg.get("concurrency", 8)),
        "seeds": seeds,
        "arms": arms,
        "generation": model_cfg.get("generation", {}),
        "serving": model_cfg.get("serving", {}),
        "server_launch_manifest_hash": content_hash(model_cfg.get("serving", {})),
        "runtime_hardware": _runtime_hardware(),
        "stage": stage,
        "stage_manifest_hash": stage_manifest_hash,
    }
    cache_identity_manifest = {
        **run_manifest,
        "execution_mode": LIVE_ACQUISITION,
        "cache_policy": {
            "read_enabled": False,
            "write_enabled": True,
            "require_hits": False,
        },
    }
    cache_namespace_hash = content_hash(cache_identity_manifest)
    run_manifest["cache_namespace_hash"] = cache_namespace_hash
    manifest_hash = content_hash(run_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(
        output_dir / "cache" / f"{model_cfg['label']}-{cache_namespace_hash[:12]}.jsonl",
        **cache_policy,
    )
    output_ceiling = int(model_cfg.get("output_ceiling", 1024))
    generation = {
        "max_new_tokens": output_ceiling,
        "max_completion_tokens": output_ceiling,
        "temperature": 0.0,
        "concurrency": int(cfg.get("concurrency", 8)),
        "run_manifest": run_manifest,
        **model_cfg.get("generation", {}),
    }
    cache_key_generation = {
        **generation,
        "run_manifest": cache_identity_manifest,
    }
    backend = make_backend(
        "openai",
        model_id,
        cache=cache,
        base_url=cfg.get("base_url", "http://127.0.0.1:8000/v1"),
        model_revision=revision,
        cache_key_params=cache_key_generation,
        **generation,
    )
    episodes = _build_episodes(seeds, arms, variant=variant)
    trace_path = output_dir / f"factorial_{model_cfg['label']}_{int(time.time())}.jsonl"
    step_records: list[dict[str, Any]] = []
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
            for episode, prompt, candidates, conversation, record in zip(
                active, prompts, candidates_by_episode, conversations, records
            ):
                decision_index = len(episode.protocol.detector_evidence)
                episode.history.append({"role": "assistant", "content": record.response})
                episode.protocol.ledger.record_message("assistant", record.response)
                action, parse_status = parse_macro_action(record.response, candidates)
                if action is None:
                    episode.protocol.terminate_without_commit("invalid_response")
                    action_record = {"kind": "invalid_response", "slot": None}
                    effect_text = "Invalid action parse; episode ended with A=⊥."
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
                    "kind": "step",
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "protocol": PROTOCOL_NAME,
                    "run_manifest_hash": manifest_hash,
                    "stage_manifest_hash": stage_manifest_hash,
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
                    "decision_index": decision_index,
                    "step": episode.protocol.env.world.step,
                    "prompt_hash": content_hash(
                        {"system": FACTORIAL_SYSTEM_PROMPT, "messages": conversation}
                    ),
                    "message_hash": content_hash(prompt),
                    "candidate_set_hash": content_hash([asdict(item) for item in candidates]),
                    "label_map_hash": content_hash(episode.label_map),
                    "base_spec_hash": content_hash(to_dict(episode.cell.task_spec)),
                    "action": action_record,
                    "low_level_actions": low_level_actions,
                    "parse_status": parse_status,
                    "fallback_action": None,
                    "raw_provider_status": record.finish_reason,
                    "generation": record.to_dict(),
                }
                step_records.append(step_record)
                handle.write(json.dumps(step_record, sort_keys=True, default=str) + "\n")
        outcomes = [episode.protocol.outcome() for episode in episodes]
        for episode, outcome in zip(episodes, outcomes):
            handle.write(
                json.dumps(
                    _outcome_record(
                        outcome,
                        str(model_cfg["label"]),
                        episode.arm,
                        stage_manifest_hash=stage_manifest_hash,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
    serving_validation = audit_identical_prompt_generations(
        step_records,
        cfg.get("serving_validation", {}),
        execution_mode=execution_mode,
    )
    arm_reports = {}
    paired_contrasts: dict[str, dict[str, Any]] = {}
    if serving_validation["scientific_results_eligible"] is True:
        for arm in arms:
            selected = [
                outcome
                for episode, outcome in zip(episodes, outcomes)
                if episode.arm == arm["name"]
            ]
            if variant == "challenge_3x2":
                arm_reports[arm["name"]] = asdict(
                    challenge_profile(
                        _outcome_record(outcome, str(model_cfg["label"])) for outcome in selected
                    )
                )
            else:
                arm_reports[arm["name"]] = asdict(compute_factorial_metrics(selected))
            parse_failures = sum(outcome.done_reason == "invalid_response" for outcome in selected)
            arm_reports[arm["name"]]["parse_failure_rate"] = {
                "value": parse_failures / len(selected) if selected else 0.0,
                "denominator": len(selected),
            }
        if variant == "quartet_2x2":
            paired_contrasts = _paired_prompt_arm_contrasts(episodes, outcomes)
    status = (
        "artifact-replay-no-live-validation"
        if execution_mode == ARTIFACT_REPLAY
        else "eligible-for-scientific-analysis"
        if serving_validation["passed"]
        else "invalid-serving-reproducibility"
    )
    summary = {
        "status": status,
        "manifest": run_manifest,
        "manifest_hash": manifest_hash,
        "stage_manifest": stage_manifest,
        "stage_manifest_hash": stage_manifest_hash,
        "trace": str(trace_path),
        "serving_validation": serving_validation,
        "reports": arm_reports,
        "paired_contrasts": paired_contrasts,
    }
    summary_path = output_dir / f"factorial_{model_cfg['label']}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if (
        execution_mode == LIVE_ACQUISITION
        and not serving_validation["passed"]
        and serving_validation["thresholds"]["fail_closed"]
    ):
        raise RuntimeError(
            "serving reproducibility gate failed; scientific reports were suppressed; "
            f"inspect {summary_path}"
        )
    return summary


def audit_identical_prompt_generations(
    step_records: list[dict[str, Any]],
    requirements: dict[str, Any] | None = None,
    *,
    execution_mode: str = LIVE_ACQUISITION,
) -> dict[str, Any]:
    """Audit independent calls for byte-identical public policy inputs.

    Crossed mechanism cells deliberately issue separate model calls even when
    their full visible conversations are identical. Any response or parsed-action
    disagreement within such a group is serving nondeterminism, not evidence that
    the policy responded to the hidden mechanism.
    """
    if execution_mode not in {LIVE_ACQUISITION, ARTIFACT_REPLAY}:
        raise ValueError(f"unknown execution mode {execution_mode!r}")
    cached_records = [
        record for record in step_records if record["generation"].get("cache_hit") is True
    ]
    uncached_records = [
        record for record in step_records if record["generation"].get("cache_hit") is not True
    ]
    if execution_mode == ARTIFACT_REPLAY:
        return {
            "performed": False,
            "passed": None,
            "scientific_results_eligible": False,
            "execution_mode": execution_mode,
            "reason": "cache-only artifact replay cannot validate the current serving process",
            "counts": {
                "step_records": len(step_records),
                "cache_hit_records": len(cached_records),
                "uncached_records": len(uncached_records),
                "server_contacted": bool(uncached_records),
            },
        }

    requirements = requirements or {}
    thresholds = {
        "minimum_replicated_prompt_groups": int(
            requirements.get("minimum_replicated_prompt_groups", 1)
        ),
        "minimum_initial_mechanism_axis_groups": int(
            requirements.get("minimum_initial_mechanism_axis_groups", 0)
        ),
        "maximum_response_conflict_groups": int(
            requirements.get("maximum_response_conflict_groups", 0)
        ),
        "maximum_action_conflict_groups": int(
            requirements.get("maximum_action_conflict_groups", 0)
        ),
        "maximum_candidate_set_conflict_groups": int(
            requirements.get("maximum_candidate_set_conflict_groups", 0)
        ),
        "require_all_uncached": bool(requirements.get("require_all_uncached", True)),
        "fail_closed": bool(requirements.get("fail_closed", True)),
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in uncached_records:
        grouped.setdefault(str(record["prompt_hash"]), []).append(record)
    replicated = {key: group for key, group in grouped.items() if len(group) > 1}

    initial_axis_groups: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for record in uncached_records:
        if int(record.get("decision_index", -1)) != 0:
            continue
        key = (
            int(record["base_seed"]),
            str(record["prompt_arm"]),
            int(record["condition"]["cue_slot"]),
        )
        initial_axis_groups.setdefault(key, []).append(record)
    complete_initial_axis_groups = {
        key: group
        for key, group in initial_axis_groups.items()
        if len({int(record["condition"]["mechanism_slot"]) for record in group}) >= 2
    }
    initial_axis_prompt_conflicts = [
        {
            "base_seed": key[0],
            "prompt_arm": key[1],
            "cue_slot": key[2],
            "prompt_hashes": sorted({str(record["prompt_hash"]) for record in group}),
        }
        for key, group in sorted(complete_initial_axis_groups.items())
        if len({str(record["prompt_hash"]) for record in group}) > 1
    ]

    def _conflicts(field) -> list[dict[str, Any]]:
        conflicts = []
        for prompt_hash, group in sorted(replicated.items()):
            values = {field(record) for record in group}
            if len(values) <= 1:
                continue
            conflicts.append(
                {
                    "prompt_hash": prompt_hash,
                    "call_count": len(group),
                    "base_seeds": sorted({int(record["base_seed"]) for record in group}),
                    "condition_ids": sorted({str(record["condition"]["id"]) for record in group}),
                    "prompt_arms": sorted({str(record["prompt_arm"]) for record in group}),
                }
            )
        return conflicts

    def _initial_axis_conflicts(field) -> list[dict[str, Any]]:
        conflicts = []
        for key, group in sorted(complete_initial_axis_groups.items()):
            if len({field(record) for record in group}) <= 1:
                continue
            conflicts.append(
                {
                    "base_seed": key[0],
                    "prompt_arm": key[1],
                    "cue_slot": key[2],
                    "condition_ids": sorted({str(record["condition"]["id"]) for record in group}),
                }
            )
        return conflicts

    response_conflicts = _conflicts(
        lambda record: content_hash(record["generation"].get("response", ""))
    )
    action_conflicts = _conflicts(lambda record: json.dumps(record["action"], sort_keys=True))
    candidate_conflicts = _conflicts(lambda record: str(record["candidate_set_hash"]))
    initial_axis_response_conflicts = _initial_axis_conflicts(
        lambda record: content_hash(record["generation"].get("response", ""))
    )
    initial_axis_action_conflicts = _initial_axis_conflicts(
        lambda record: json.dumps(record["action"], sort_keys=True)
    )
    initial_axis_candidate_conflicts = _initial_axis_conflicts(
        lambda record: str(record["candidate_set_hash"])
    )
    counts = {
        "step_records": len(step_records),
        "uncached_records": len(uncached_records),
        "cache_hit_records": len(cached_records),
        "server_contacted": bool(uncached_records),
        "unique_prompt_groups": len(grouped),
        "replicated_prompt_groups": len(replicated),
        "replicated_calls": sum(len(group) for group in replicated.values()),
        "initial_mechanism_axis_groups": len(complete_initial_axis_groups),
        "initial_mechanism_axis_prompt_conflict_groups": len(initial_axis_prompt_conflicts),
        "initial_mechanism_axis_response_conflict_groups": len(initial_axis_response_conflicts),
        "initial_mechanism_axis_action_conflict_groups": len(initial_axis_action_conflicts),
        "initial_mechanism_axis_candidate_set_conflict_groups": len(
            initial_axis_candidate_conflicts
        ),
        "response_conflict_groups": len(response_conflicts),
        "action_conflict_groups": len(action_conflicts),
        "candidate_set_conflict_groups": len(candidate_conflicts),
    }
    passed = (
        counts["server_contacted"]
        and (not thresholds["require_all_uncached"] or counts["cache_hit_records"] == 0)
        and counts["replicated_prompt_groups"] >= thresholds["minimum_replicated_prompt_groups"]
        and counts["initial_mechanism_axis_groups"]
        >= thresholds["minimum_initial_mechanism_axis_groups"]
        and counts["initial_mechanism_axis_prompt_conflict_groups"] == 0
        and counts["initial_mechanism_axis_response_conflict_groups"]
        <= thresholds["maximum_response_conflict_groups"]
        and counts["initial_mechanism_axis_action_conflict_groups"]
        <= thresholds["maximum_action_conflict_groups"]
        and counts["initial_mechanism_axis_candidate_set_conflict_groups"]
        <= thresholds["maximum_candidate_set_conflict_groups"]
        and counts["response_conflict_groups"] <= thresholds["maximum_response_conflict_groups"]
        and counts["action_conflict_groups"] <= thresholds["maximum_action_conflict_groups"]
        and counts["candidate_set_conflict_groups"]
        <= thresholds["maximum_candidate_set_conflict_groups"]
    )
    return {
        "performed": True,
        "passed": passed,
        "scientific_results_eligible": passed,
        "execution_mode": execution_mode,
        "unit": "independent calls sharing the full system-and-conversation hash",
        "counts": counts,
        "thresholds": thresholds,
        "conflicts": {
            "responses": response_conflicts,
            "actions": action_conflicts,
            "candidate_sets": candidate_conflicts,
            "initial_mechanism_axis_prompts": initial_axis_prompt_conflicts,
            "initial_mechanism_axis_responses": initial_axis_response_conflicts,
            "initial_mechanism_axis_actions": initial_axis_action_conflicts,
            "initial_mechanism_axis_candidate_sets": initial_axis_candidate_conflicts,
        },
    }


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_HERE,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _working_tree_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_HERE,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and not completed.stdout.strip()


def _model_lock(model_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": model_cfg["id"],
        "revision": model_cfg["revision"],
        "tokenizer_revision": model_cfg.get("tokenizer_revision", model_cfg["revision"]),
        "output_ceiling": int(model_cfg.get("output_ceiling", 1024)),
        "generation": model_cfg.get("generation", {}),
        "serving": model_cfg.get("serving", {}),
    }


def validate_stage_manifest(
    stage: str,
    manifest: dict[str, Any],
    cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    *,
    repository_commit: str | None = None,
    working_tree_clean: bool | None = None,
) -> None:
    """Fail closed when a staged run diverges from its declared manifest."""
    if manifest.get("stage") != stage:
        raise ValueError(f"manifest stage {manifest.get('stage')!r} does not match {stage!r}")
    variant = str(cfg.get("variant", "quartet_2x2"))
    expected_seeds = (
        manifest.get("ceiling_variant_seeds")
        if stage == "pilot" and variant == "challenge_3x2"
        else manifest.get("base_seeds")
    )
    if [int(seed) for seed in cfg.get("seeds", [])] != [int(seed) for seed in expected_seeds or []]:
        raise ValueError("configured base seeds do not match the stage manifest")
    if stage != "confirmatory":
        return
    if manifest.get("frozen") is not True:
        raise ValueError("confirmatory manifest must set frozen=true")
    if cfg.get("prompt_arms") != manifest.get("prompt_arms"):
        raise ValueError("configured prompt arms do not match the frozen manifest")
    if int(cfg.get("concurrency", 8)) != int(manifest.get("concurrency", -1)):
        raise ValueError("configured concurrency does not match the frozen manifest")
    if cfg.get("execution_mode", LIVE_ACQUISITION) != manifest.get("execution_mode"):
        raise ValueError("configured execution mode does not match the frozen manifest")
    if cfg.get("serving_validation") != manifest.get("serving_validation"):
        raise ValueError("configured serving validation does not match the frozen manifest")
    locked_models = manifest.get("models", {})
    label = str(model_cfg["label"])
    if locked_models.get(label) != _model_lock(model_cfg):
        raise ValueError(f"model lock for {label!r} does not match the frozen manifest")
    actual_commit = repository_commit if repository_commit is not None else _repository_commit()
    if manifest.get("protocol_commit") != actual_commit:
        raise ValueError("repository commit does not match frozen protocol_commit")
    clean = working_tree_clean if working_tree_clean is not None else _working_tree_clean()
    if not clean:
        raise ValueError("confirmatory runs require a clean working tree")


def load_stage_manifest(
    path: pathlib.Path,
    stage: str,
    cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    validate_stage_manifest(stage, manifest, cfg, model_cfg)
    return manifest, hashlib.sha256(raw).hexdigest()


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
    try:
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
    except FileNotFoundError:
        runtime["driver"] = None
    else:
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
                count=int(arm.get("count", 6)),
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
        {"name": "cue_a", "mode": "cue", "count": 6, "base_seed": 9_000},
        {"name": "cue_b", "mode": "cue", "count": 6, "base_seed": 10_000},
        {
            "name": "mechanism_a",
            "mode": "mechanism",
            "count": 6,
            "base_seed": 11_000,
        },
        {
            "name": "mechanism_b",
            "mode": "mechanism",
            "count": 6,
            "base_seed": 12_000,
        },
    ]


def _paired_prompt_arm_contrasts(
    episodes: list[_StudyEpisode],
    outcomes: list[CommitOutcome],
) -> dict[str, dict[str, Any]]:
    by_arm: dict[str, list[CommitOutcome]] = {}
    for episode, outcome in zip(episodes, outcomes, strict=True):
        by_arm.setdefault(episode.arm, []).append(outcome)
    if "neutral" not in by_arm:
        return {}
    metrics = (
        "cue_following",
        "mechanism_tracking",
        "detector_query_rate",
        "coverage",
        "choice_accuracy",
    )
    comparisons: list[tuple[str, str, str]] = []
    for treatment in sorted(name for name in by_arm if name != "neutral"):
        comparisons.append((f"{treatment}_minus_neutral", "neutral", treatment))
    for suffix in ("a", "b"):
        cue = f"cue_{suffix}"
        mechanism = f"mechanism_{suffix}"
        if cue in by_arm and mechanism in by_arm:
            comparisons.append((f"{cue}_minus_{mechanism}", mechanism, cue))
    return {
        label: {
            metric: asdict(paired_arm_contrast(by_arm[reference], by_arm[treatment], metric))
            for metric in metrics
        }
        for label, reference, treatment in comparisons
    }


def _outcome_record(
    outcome: CommitOutcome,
    agent: str,
    arm: str | None = None,
    *,
    stage_manifest_hash: str | None = None,
) -> dict[str, Any]:
    ledger = list(outcome.trace)
    return {
        "kind": "outcome",
        "schema_version": TRACE_SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "agent": agent,
        "prompt_arm": arm,
        "stage_manifest_hash": stage_manifest_hash,
        **{key: value for key, value in asdict(outcome).items() if key != "trace"},
        "episode_ledger": ledger,
        "episode_ledger_hash": content_hash(ledger),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/factorial_v02.yaml")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--model", help="Model label from the config; local server must match")
    parser.add_argument("--stage", choices=("pilot", "confirmatory"))
    parser.add_argument("--manifest", type=pathlib.Path)
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
    if not args.stage or args.manifest is None:
        parser.error("model runs require --stage and --manifest")
    models = {str(model["label"]): model for model in cfg.get("models", [])}
    if args.model not in models:
        parser.error(f"unknown model label {args.model!r}; choose from {sorted(models)}")
    stage_manifest, stage_manifest_hash = load_stage_manifest(
        args.manifest, args.stage, cfg, models[args.model]
    )
    run_local_model(
        cfg,
        models[args.model],
        output_dir,
        stage=args.stage,
        stage_manifest=stage_manifest,
        stage_manifest_hash=stage_manifest_hash,
    )


if __name__ == "__main__":
    main()
