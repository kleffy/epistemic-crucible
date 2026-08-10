"""Prompt construction for the LLM agent.

Turns a public observation + task goal into a natural-language prompt and a list
of legal candidate actions, and parses an action back out of a model response.
Uses only public observation fields and the goal predicate — no hidden
properties — so the observation boundary is preserved.
"""

from __future__ import annotations

import hashlib
import json
import re

from crucible.actions import Action, ActionKind, Direction
from crucible.agents.base import enumerate_candidate_actions
from crucible.grammar import GoalKind, GoalSpec

SYSTEM_PROMPT = """\
You control an agent in a grid world. Each turn you are shown the current state, \
a goal, and a NUMBERED list of the actions that are currently legal. Choose one \
action per turn to achieve the goal.

Action types and their arguments:
- move {"direction": "north"|"south"|"east"|"west"}
- pickup {"obj_id": ID}
- drop {"obj_id": ID}
- inspect {"obj_id": ID}
- apply {"tool_id": ID, "target_id": ID}
- combine {"obj_id_a": ID, "obj_id_b": ID}
- wait {}

Reason briefly if you wish, then end your reply with EXACTLY one line naming \
your choice by its number:
ACTION: <number>"""


def describe_goal(goal: GoalSpec) -> str:
    """One-line natural-language description of the task goal (public)."""
    if goal.kind == GoalKind.OPEN:
        return f"Open the gate '{goal.target_obj_id}' (make its state 'open')."
    if goal.kind == GoalKind.RETRIEVE:
        return f"Pick up and hold an object of type '{goal.target_type}'."
    if goal.kind == GoalKind.REACH:
        return f"Move the agent to position {tuple(goal.target_pos or ())}."
    if goal.kind == GoalKind.TRANSFORM:
        return f"Transform object '{goal.target_obj_id}' into its target state."
    if goal.kind == GoalKind.CLASSIFY:
        prop = goal.classify_property or "the relevant property"
        return f"Identify which object has {prop} and act on it."
    return "Achieve the task goal."


def build_label_map(obs: dict, existing: dict | None = None) -> dict:
    """Map real object IDs to neutral labels (object_1, object_2, ...).

    Object IDs encode the family, seed, and crucially the SPLIT (e.g.
    ``aff_1_train_tool0``) and the train-correct ordinal. Showing them to a
    text agent would let it read the split off the IDs and defeat the transfer
    diagnostic. Labels are assigned in hash order so they do not encode the
    train-correct ordering, and the map is extended (never reordered) across
    steps so labels stay stable within an episode.
    """
    label_map = dict(existing or {})
    new_ids = [i for i in obs.get("objects", {}) if i not in label_map]
    new_ids.sort(key=lambda x: hashlib.md5(x.encode()).hexdigest())
    start = len(label_map)
    for k, oid in enumerate(new_ids):
        label_map[oid] = f"object_{start + k + 1}"
    return label_map


def anonymize_text(text: str, label_map: dict) -> str:
    """Replace every real object ID with its neutral label (longest IDs first)."""
    for real_id in sorted(label_map, key=len, reverse=True):
        text = text.replace(real_id, label_map[real_id])
    return text


def oracle_hint(spec, label_map: dict, level: str | None) -> str:
    """A labelled oracle hint for the ablation ladder (empty for the genuine eval).

    Levels reveal progressively more ground truth to localize where failure
    occurs: ``intervention`` names the operative action, ``property`` names the
    object that opens the gate, and ``rule`` states the full local rule. The
    object is named by its anonymized label, so no real ID or split leaks.
    """
    if not level or level == "none":
        return ""
    if level == "intervention":
        return "Hint: applying a tool to the gate is the action that can open it.\n\n"
    correct = next((o for o in spec.object_specs if o.role == "correct_tool"), None)
    if correct is None:
        return ""
    label = label_map.get(correct.obj_id, "the correct tool")
    if level == "property":
        return f"Hint: {label} is the tool that opens the gate.\n\n"
    if level == "rule":
        return f"Hint: applying {label} to the gate opens it; the other tools have no effect.\n\n"
    return ""


def serialize_observation(obs: dict) -> str:
    """Compact, readable text rendering of a public observation."""
    agent = obs.get("agent", {})
    lines = [
        f"Step {obs.get('step', 0)}/{obs.get('max_steps', '?')}. "
        f"Agent at {tuple(agent.get('pos', ()))}, "
        f"energy {agent.get('energy', '?')}, "
        f"inventory {agent.get('inventory', [])}.",
        "Objects:",
    ]
    for oid, o in obs.get("objects", {}).items():
        pos = o.get("pos")
        where = "held" if pos is None else f"at {tuple(pos)}"
        marker = f" marker={o['marker']}" if o.get("marker") else ""
        lines.append(
            f"  - {oid}: {o['type']} ({o['color']}, {o['shape']}, {o['texture']}, "
            f"{o['size']}) state={o['state']} {where}{marker}"
        )
    relations = obs.get("relations", [])
    if relations:
        rels = ", ".join(f"{r['subject']} {r['kind']} {r['object']}" for r in relations)
        lines.append(f"Relations: {rels}")
    return "\n".join(lines)


def _action_to_payload(action: Action) -> dict:
    """JSON-serializable {kind, args} for an action (Direction → its value)."""
    args = {k: (v.value if isinstance(v, Direction) else v) for k, v in action.args.items()}
    return {"kind": action.kind.value, "args": args}


def describe_candidates(candidates: list[Action]) -> str:
    """Numbered list of legal candidate actions as JSON payloads."""
    return "\n".join(
        f"  {i}. {json.dumps(_action_to_payload(a))}" for i, a in enumerate(candidates)
    )


def build_user_message(
    obs: dict,
    goal_text: str,
    grid_size: int = 6,
    candidates: list[Action] | None = None,
    label_map: dict | None = None,
) -> tuple[str, list[Action], dict]:
    """Return (user_message, candidate_actions, label_map) for the observation.

    Object IDs are anonymized to neutral labels so the rendered prompt (goal,
    observation, candidate list) never leaks the split or the train-correct
    ordinal. ``label_map`` is built if not given and extended in place; pass it
    back on later steps (and use it to anonymize effect feedback) so labels stay
    stable within an episode. The returned ``candidates`` keep their real IDs —
    selection is by index, so the agent never types an ID.
    """
    if candidates is None:
        candidates = enumerate_candidate_actions(obs, grid_size)
    label_map = build_label_map(obs, label_map)
    msg = (
        f"GOAL: {goal_text}\n\n"
        f"{serialize_observation(obs)}\n\n"
        f"Legal actions (choose one by number):\n{describe_candidates(candidates)}\n\n"
        "Reason briefly, then output your choice as `ACTION: <number>` on the final line."
    )
    return anonymize_text(msg, label_map), candidates, label_map


_ACTION_IDX_RE = re.compile(r"ACTION:\s*#?\s*(\d+)", re.IGNORECASE)
_ANY_IDX_RE = re.compile(r"\b(\d+)\b")
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}")
_JSON_GREEDY_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(raw: str, candidates: list[Action] | None = None) -> Action:
    """Extract an Action from a model response. Falls back to WAIT on failure.

    Primary format is index selection (``ACTION: <number>``) into the numbered
    legal ``candidates`` list, which guarantees a legal action. Falls back to a
    JSON ``{"kind": ..., "args": ...}`` object if the model emits one, then to
    WAIT. The last integer in the reply is used if no explicit ACTION line.
    """
    # 1. Explicit index selection: ACTION: <number>.
    if candidates:
        m = _ACTION_IDX_RE.search(raw)
        if m and 0 <= int(m.group(1)) < len(candidates):
            return candidates[int(m.group(1))]

    # 2. JSON object with a kind (fallback for models that emit JSON). Try a
    # greedy outer match first (handles nested "args"), then innermost objects.
    flat = raw.replace("\n", " ")
    blobs = []
    greedy = _JSON_GREEDY_RE.search(flat)
    if greedy:
        blobs.append(greedy.group(0))
    blobs.extend(reversed(_JSON_OBJ_RE.findall(flat)))
    for blob in blobs:
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "kind" in payload:
            try:
                kind = ActionKind(str(payload["kind"]).lower())
                args = dict(payload.get("args", {}) or {})
                if "direction" in args:
                    args["direction"] = Direction(str(args["direction"]).lower())
                return Action(kind=kind, args=args)
            except (ValueError, KeyError, TypeError):
                break

    # 3. Last bare integer in the reply, as a lenient index.
    if candidates:
        nums = _ANY_IDX_RE.findall(raw)
        if nums and 0 <= int(nums[-1]) < len(candidates):
            return candidates[int(nums[-1])]

    return Action(kind=ActionKind.WAIT, args={})
