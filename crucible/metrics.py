"""Diagnostic metrics for Epistemic Crucible evaluation.

All metrics consume JSONL trace data produced by experiments/run_baselines.py.
Metrics are returned as a vector (dict[str, MetricResult]) — never collapsed
into a single aggregate score.

Every MetricResult carries an explicit gaming_risk note describing how an agent
could maximise the metric without genuine understanding.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from crucible.splits import is_shortcut_family
from crucible.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class AttributionProfile:
    """Unaggregated v0.2 behavioral-attribution coordinates."""

    mechanism_responsiveness: float | None
    cue_susceptibility: float | None
    cue_following: float | None
    coverage: float
    quartet_success: float | None
    mechanism_tracking: float | None
    cue_susceptibility_all: float | None
    coverage_by_cell: dict[str, float]
    n: dict[str, int]


@dataclass(frozen=True)
class ChallengeProfile:
    """Directional coordinates for the six-cell 3x2 ceiling pilot."""

    mechanism_accuracy: float | None
    cue_susceptibility: float | None
    cue_following: float | None
    coverage: float
    all_cells_success: float | None
    n: dict[str, int]


def analytic_chance_point(k: int) -> dict[str, float]:
    """Reference coordinates for independent uniform commitment over ``k`` slots."""
    if k < 2:
        raise ValueError("k must be at least 2")
    return {
        "mechanism_responsiveness": 1 / k**2,
        "cue_susceptibility": 1 - 1 / k,
        "cue_following": 1 / k,
        "coverage": 1.0,
    }


def attribution_profile(records: Any) -> AttributionProfile:
    """Compute exact factorial coordinates from dict-like outcome records.

    The base seed is the unit for the paired mechanism/cue axes. A seed is
    included on an axis only when both of that axis's pairs contain commitments.
    Directional cue-following is computed over individual committed cells.
    """
    records = list(records)
    grouped: dict[int, dict[tuple[int, int], dict]] = {}
    for record in records:
        seed = int(record["base_seed"])
        key = (int(record["mechanism_slot"]), int(record["cue_slot"]))
        if key in grouped.setdefault(seed, {}):
            raise ValueError(f"duplicate attribution record for seed={seed}, cell={key}")
        grouped[seed][key] = record

    mechanism_seed_values: list[float] = []
    cue_seed_values: list[float] = []
    cue_all_seed_values: list[float] = []
    quartet_values: list[float] = []
    mechanism_tracking_values: list[float] = []
    cue_following_values: list[float] = []
    committed = 0
    for cells in grouped.values():
        if set(cells) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            continue
        choices = {key: value.get("committed_slot") for key, value in cells.items()}
        for (mechanism, cue), choice in choices.items():
            if choice is not None:
                committed += 1
                cell = cells[(mechanism, cue)]
                mechanism_carrier = int(cell.get("mechanism_carrier_slot", mechanism))
                cue_carrier = int(cell.get("cue_carrier_slot", cue))
                mechanism_tracking_values.append(float(int(choice) == mechanism_carrier))
                cue_following_values.append(float(int(choice) == cue_carrier))
        mechanism_pairs = [
            (
                choices[(0, cue)],
                int(cells[(0, cue)].get("mechanism_carrier_slot", 0)),
                choices[(1, cue)],
                int(cells[(1, cue)].get("mechanism_carrier_slot", 1)),
            )
            for cue in (0, 1)
        ]
        if all(left is not None and right is not None for left, _, right, _ in mechanism_pairs):
            mechanism_seed_values.append(
                float(
                    np.mean(
                        [
                            left == left_carrier and right == right_carrier
                            for left, left_carrier, right, right_carrier in mechanism_pairs
                        ]
                    )
                )
            )
        cue_pairs = [(choices[(mechanism, 0)], choices[(mechanism, 1)]) for mechanism in (0, 1)]
        if all(left is not None and right is not None for left, right in cue_pairs):
            cue_seed_values.append(float(np.mean([left != right for left, right in cue_pairs])))
        cue_all_seed_values.append(float(np.mean([left != right for left, right in cue_pairs])))
        quartet_values.append(float(all(bool(value.get("solved")) for value in cells.values())))

    def mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    total = len(records)
    coverage_by_cell = {}
    for mechanism, cue in ((0, 0), (0, 1), (1, 0), (1, 1)):
        cell_records = [
            record
            for record in records
            if int(record["mechanism_slot"]) == mechanism and int(record["cue_slot"]) == cue
        ]
        coverage_by_cell[f"m{mechanism}_c{cue}"] = (
            sum(record.get("committed_slot") is not None for record in cell_records)
            / len(cell_records)
            if cell_records
            else 0.0
        )
    return AttributionProfile(
        mechanism_responsiveness=mean_or_none(mechanism_seed_values),
        cue_susceptibility=mean_or_none(cue_seed_values),
        cue_following=mean_or_none(cue_following_values),
        coverage=committed / total if total else 0.0,
        quartet_success=mean_or_none(quartet_values),
        mechanism_tracking=mean_or_none(mechanism_tracking_values),
        cue_susceptibility_all=mean_or_none(cue_all_seed_values),
        coverage_by_cell=coverage_by_cell,
        n={
            "mechanism_responsiveness": len(mechanism_seed_values),
            "cue_susceptibility": len(cue_seed_values),
            "cue_following": len(cue_following_values),
            "coverage": total,
            "quartet_success": len(quartet_values),
            "mechanism_tracking": len(mechanism_tracking_values),
            "cue_susceptibility_all": len(cue_all_seed_values),
        },
    )


def challenge_profile(records: Any) -> ChallengeProfile:
    """Compute the pre-registered directional 3x2 ceiling-pilot profile."""
    records = list(records)
    grouped: dict[int, dict[tuple[int, int], dict]] = {}
    for record in records:
        seed = int(record["base_seed"])
        key = (int(record["mechanism_slot"]), int(record["cue_slot"]))
        if key in grouped.setdefault(seed, {}):
            raise ValueError(f"duplicate challenge record for seed={seed}, cell={key}")
        grouped[seed][key] = record

    expected = {(mechanism, cue) for mechanism in (0, 1, 2) for cue in (0, 1)}
    mechanism_values: list[float] = []
    cue_values: list[float] = []
    cue_following_values: list[float] = []
    all_success: list[float] = []
    committed = 0
    for cells in grouped.values():
        if set(cells) != expected:
            continue
        choices = {key: cell.get("committed_slot") for key, cell in cells.items()}
        for (mechanism, cue), choice in choices.items():
            if choice is not None:
                committed += 1
                mechanism_carrier = int(
                    cells[(mechanism, cue)].get("mechanism_carrier_slot", mechanism)
                )
                cue_carrier = int(cells[(mechanism, cue)].get("cue_carrier_slot", cue))
                mechanism_values.append(float(int(choice) == mechanism_carrier))
                cue_following_values.append(float(int(choice) == cue_carrier))
        for mechanism in (0, 1, 2):
            left, right = choices[(mechanism, 0)], choices[(mechanism, 1)]
            if left is not None and right is not None:
                cue_values.append(float(left != right))
        all_success.append(float(all(bool(cell.get("solved")) for cell in cells.values())))

    def mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    total = len(records)
    return ChallengeProfile(
        mechanism_accuracy=mean_or_none(mechanism_values),
        cue_susceptibility=mean_or_none(cue_values),
        cue_following=mean_or_none(cue_following_values),
        coverage=committed / total if total else 0.0,
        all_cells_success=mean_or_none(all_success),
        n={
            "mechanism_accuracy": len(mechanism_values),
            "cue_susceptibility": len(cue_values),
            "cue_following": len(cue_following_values),
            "coverage": total,
            "all_cells_success": len(all_success),
        },
    )


def bootstrap_ci(
    values: list[float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) with a percentile bootstrap CI over ``values``."""
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return (float(arr.mean()), float(lo), float(hi))


def tsr_with_ci(
    outcomes: list[dict],
    *,
    family: str | None = None,
    agent: str | None = None,
    split: str | None = None,
) -> dict:
    """Mean task success rate with a bootstrap 95% CI over episodes."""
    filtered = filter_records(outcomes, family=family, agent=agent, split=split)
    vals = [1.0 if r.get("goal_achieved") else 0.0 for r in filtered]
    mean, lo, hi = bootstrap_ci(vals)
    return {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(vals)}


# Action kinds treated as interventions (must match Phase 5 runner).
_INTERVENTION_KINDS = {"apply", "combine", "inspect"}

# Maximum number of steps for standard task families (used to detect timeout).
_DEFAULT_MAX_STEPS = 40


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """Container for one metric computation result."""

    name: str
    value: float | dict | None
    count: int
    definition: str
    gaming_risk: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "count": self.count,
            "definition": self.definition,
            "gaming_risk": self.gaming_risk,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------


def load_trace(path: str | pathlib.Path) -> tuple[list[dict], list[dict]]:
    """Parse a JSONL trace file. Returns (step_records, outcome_records)."""
    steps: list[dict] = []
    outcomes: list[dict] = []
    with pathlib.Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("kind") == "step":
                steps.append(record)
            elif record.get("kind") == "outcome":
                outcomes.append(record)
    _log.debug("load_trace: %d steps, %d outcomes from %s", len(steps), len(outcomes), path)
    return steps, outcomes


def load_traces(paths: list[str | pathlib.Path]) -> tuple[list[dict], list[dict]]:
    """Merge multiple JSONL trace files into combined step and outcome lists."""
    all_steps: list[dict] = []
    all_outcomes: list[dict] = []
    for p in paths:
        s, o = load_trace(p)
        all_steps.extend(s)
        all_outcomes.extend(o)
    return all_steps, all_outcomes


def filter_records(
    records: list[dict],
    *,
    family: str | None = None,
    agent: str | None = None,
    split: str | None = None,
) -> list[dict]:
    """Filter trace records by any combination of family/agent/split."""
    out = records
    if family is not None:
        out = [r for r in out if r.get("family") == family]
    if agent is not None:
        out = [r for r in out if r.get("agent") == agent]
    if split is not None:
        out = [r for r in out if r.get("split") == split]
    return out


# ---------------------------------------------------------------------------
# Metric 1: Task Success Rate
# ---------------------------------------------------------------------------


def task_success_rate(
    outcomes: list[dict],
    *,
    family: str | None = None,
    agent: str | None = None,
    split: str | None = None,
) -> MetricResult:
    """Fraction of episodes where goal_achieved=True."""
    filtered = filter_records(outcomes, family=family, agent=agent, split=split)
    count = len(filtered)
    if count == 0:
        return MetricResult(
            name="task_success_rate",
            value=0.0,
            count=0,
            definition="Fraction of episodes with goal_achieved=True.",
            gaming_risk=(
                "Agents can maximise TSR on OPEN-goal families by memorising "
                "seed→action maps without causal understanding."
            ),
        )
    value = sum(1 for r in filtered if r.get("goal_achieved")) / count
    return MetricResult(
        name="task_success_rate",
        value=round(value, 4),
        count=count,
        definition="Fraction of episodes with goal_achieved=True.",
        gaming_risk=(
            "Agents can maximise TSR on OPEN-goal families by memorising "
            "seed→action maps without causal understanding."
        ),
    )


# ---------------------------------------------------------------------------
# Metric 2: Transfer Success
# ---------------------------------------------------------------------------


def transfer_success(
    outcomes: list[dict],
    *,
    family: str | None = None,
    agent: str | None = None,
) -> MetricResult:
    """Per-family/agent train TSR vs test TSR and their delta (test - train).

    Negative delta indicates probable memorisation or shortcut exploitation.
    """
    families = sorted({r["family"] for r in outcomes if "family" in r})
    agents = sorted({r["agent"] for r in outcomes if "agent" in r})
    if family is not None:
        families = [f for f in families if f == family]
    if agent is not None:
        agents = [a for a in agents if a == agent]

    results: dict[str, dict] = {}
    total_count = 0

    for fam in families:
        for ag in agents:
            key = f"{fam}/{ag}"
            train = filter_records(outcomes, family=fam, agent=ag, split="train")
            test = filter_records(outcomes, family=fam, agent=ag, split="test")
            if not train and not test:
                continue
            train_tsr = (
                round(sum(1 for r in train if r.get("goal_achieved")) / len(train), 4)
                if train
                else None
            )
            test_tsr = (
                round(sum(1 for r in test if r.get("goal_achieved")) / len(test), 4)
                if test
                else None
            )
            delta = (
                round(test_tsr - train_tsr, 4)
                if (test_tsr is not None and train_tsr is not None)
                else None
            )
            results[key] = {"train_tsr": train_tsr, "test_tsr": test_tsr, "delta": delta}
            total_count += len(train) + len(test)

    return MetricResult(
        name="transfer_success",
        value=results,
        count=total_count,
        definition="Per-(family,agent) train TSR vs test TSR; delta=test-train.",
        gaming_risk=(
            "Positive delta from lucky seeds is not reliable; "
            "report standard deviation across seeds alongside the mean."
        ),
    )


# ---------------------------------------------------------------------------
# Metric 3: Shortcut Sensitivity
# ---------------------------------------------------------------------------


def shortcut_sensitivity(
    outcomes: list[dict],
    *,
    agent: str | None = None,
) -> MetricResult:
    """train_TSR - test_TSR for shortcut-exposed families (affordance, tool_substitution).

    High sensitivity indicates the agent exploits a surface feature
    (colour, shape) rather than a hidden causal property.
    """
    agents = sorted({r["agent"] for r in outcomes if "agent" in r})
    if agent is not None:
        agents = [a for a in agents if a == agent]

    results: dict[str, float | None] = {}
    total_count = 0

    shortcut_fams = sorted(
        {r["family"] for r in outcomes if "family" in r and is_shortcut_family(r["family"])}
    )
    for fam in shortcut_fams:
        for ag in agents:
            key = f"{fam}/{ag}"
            train = filter_records(outcomes, family=fam, agent=ag, split="train")
            test = filter_records(outcomes, family=fam, agent=ag, split="test")
            if not train and not test:
                continue
            train_tsr = (
                sum(1 for r in train if r.get("goal_achieved")) / len(train) if train else None
            )
            test_tsr = sum(1 for r in test if r.get("goal_achieved")) / len(test) if test else None
            if train_tsr is not None and test_tsr is not None:
                results[key] = round(train_tsr - test_tsr, 4)
            else:
                results[key] = None
            total_count += len(train) + len(test)

    return MetricResult(
        name="shortcut_sensitivity",
        value=results,
        count=total_count,
        definition=(
            "train_TSR - test_TSR for affordance and tool_substitution families. "
            "High value = surface-feature exploitation."
        ),
        gaming_risk=(
            "Memorisation agents show SS≈0 trivially because they use different "
            "lookup keys per split; SS must be read alongside TSR."
        ),
    )


# ---------------------------------------------------------------------------
# Metric 4: Intervention Validity
# ---------------------------------------------------------------------------


def intervention_validity(
    steps: list[dict],
    *,
    family: str | None = None,
    agent: str | None = None,
) -> MetricResult:
    """Fraction of APPLY/COMBINE/INSPECT steps that produced non-empty effects."""
    filtered = filter_records(steps, family=family, agent=agent)
    interventions = [s for s in filtered if s.get("action", {}).get("kind") in _INTERVENTION_KINDS]
    count = len(interventions)
    if count == 0:
        return MetricResult(
            name="intervention_validity",
            value=0.0,
            count=0,
            definition=("Fraction of APPLY/COMBINE/INSPECT steps with non-empty effect strings."),
            gaming_risk=(
                "INSPECT on a DETECTOR object always produces a marker effect; "
                "excluding INSPECT yields a stricter causal-intervention validity metric."
            ),
        )
    effective = sum(1 for s in interventions if s.get("effects"))
    return MetricResult(
        name="intervention_validity",
        value=round(effective / count, 4),
        count=count,
        definition="Fraction of APPLY/COMBINE/INSPECT steps with non-empty effect strings.",
        gaming_risk=(
            "INSPECT on a DETECTOR object always produces a marker effect; "
            "excluding INSPECT yields a stricter causal-intervention validity metric."
        ),
    )


# ---------------------------------------------------------------------------
# Metric 5: Intervention Efficiency
# ---------------------------------------------------------------------------


def intervention_efficiency(
    steps: list[dict],
    outcomes: list[dict],
    get_spec: Callable[[str, int, str], Any] | None = None,
    *,
    family: str | None = None,
    agent: str | None = None,
) -> MetricResult:
    """For successful episodes: oracle_steps / agent_steps (capped at 1.0).

    Requires get_spec callable: (family_str, seed_int, split_str) → TaskSpec.
    Returns value=None with a metadata note when get_spec is not provided.
    """
    _definition = (
        "For successful episodes: oracle_action_count / agent_steps, capped at 1.0. "
        "Higher = more efficient relative to the oracle."
    )
    _gaming = (
        "Short lucky episodes score higher than systematic solvers; "
        "efficiency only applies to successful episodes."
    )

    if get_spec is None:
        return MetricResult(
            name="intervention_efficiency",
            value=None,
            count=0,
            definition=_definition,
            gaming_risk=_gaming,
            metadata={"note": "get_spec not provided; cannot compute oracle lengths."},
        )

    successful = filter_records(
        [o for o in outcomes if o.get("goal_achieved")], family=family, agent=agent
    )
    count = len(successful)
    if count == 0:
        return MetricResult(
            name="intervention_efficiency",
            value=0.0,
            count=0,
            definition=_definition,
            gaming_risk=_gaming,
        )

    ratios: list[float] = []
    for ep in successful:
        try:
            spec = get_spec(ep["family"], ep["seed"], ep["split"])
            oracle_len = len(spec.solution_certificate.action_sequence)
            agent_steps = max(ep.get("steps", 1), 1)
            ratios.append(min(oracle_len / agent_steps, 1.0))
        except Exception as exc:
            _log.debug("intervention_efficiency: skipping episode: %s", exc)

    value = round(sum(ratios) / len(ratios), 4) if ratios else 0.0
    return MetricResult(
        name="intervention_efficiency",
        value=value,
        count=count,
        definition=_definition,
        gaming_risk=_gaming,
    )


# ---------------------------------------------------------------------------
# Metric 6: Counterfactual Accuracy
# ---------------------------------------------------------------------------


def counterfactual_accuracy(
    steps: list[dict],
    outcomes: list[dict],
    get_spec: Callable[[str, int, str], Any] | None = None,
    *,
    mode: str = "behavioral",
) -> MetricResult:
    """Counterfactual accuracy in behavioral or prediction mode.

    behavioral mode: For COUNTERFACTUAL family, checks whether the agent applied
    the source to the object that actually has the relevant property
    (classify_correct_obj_id from TaskSpec). Requires get_spec.

    prediction mode: Counts PREDICT action steps and checks predicted_effect vs
    subsequent actual effect. Returns count=0 if no PREDICT actions found.
    """
    _definition = (
        "Counterfactual accuracy: behavioral = correct intervention on classified object; "
        "prediction = predicted effect matches actual effect."
    )
    _gaming = (
        "Agents can achieve high behavioral CA on COUNTERFACTUAL by always applying to "
        "block0 (which is correct on TRAIN); this fails on TEST where the property is swapped."
    )

    if mode == "prediction":
        predict_steps = [s for s in steps if s.get("action", {}).get("kind") == "predict"]
        count = len(predict_steps)
        if count == 0:
            return MetricResult(
                name="counterfactual_accuracy",
                value=None,
                count=0,
                definition=_definition,
                gaming_risk=_gaming,
                metadata={
                    "mode": "prediction",
                    "note": "No PREDICT actions found in trace; metric not computable.",
                },
            )
        correct = sum(
            1
            for s in predict_steps
            if s.get("action", {}).get("args", {}).get("predicted_effect")
            == s.get("effects", [None])[0]
        )
        return MetricResult(
            name="counterfactual_accuracy",
            value=round(correct / count, 4),
            count=count,
            definition=_definition,
            gaming_risk=_gaming,
            metadata={"mode": "prediction"},
        )

    # behavioral mode
    if get_spec is None:
        return MetricResult(
            name="counterfactual_accuracy",
            value=None,
            count=0,
            definition=_definition,
            gaming_risk=_gaming,
            metadata={
                "mode": "behavioral",
                "note": "get_spec not provided; cannot determine correct object.",
            },
        )

    cf_outcomes = filter_records(outcomes, family="counterfactual")
    count = len(cf_outcomes)
    if count == 0:
        return MetricResult(
            name="counterfactual_accuracy",
            value=0.0,
            count=0,
            definition=_definition,
            gaming_risk=_gaming,
            metadata={"mode": "behavioral"},
        )

    correct_count = 0
    for ep in cf_outcomes:
        try:
            spec = get_spec(ep["family"], ep["seed"], ep["split"])
            correct_obj = spec.goal.classify_correct_obj_id
            if correct_obj is None:
                continue
            ep_steps = filter_records(
                [
                    s
                    for s in steps
                    if s.get("episode") == ep.get("episode")
                    and s.get("seed") == ep.get("seed")
                    and s.get("split") == ep.get("split")
                ],
            )
            # Check if agent applied source to the correct object.
            for s in ep_steps:
                act = s.get("action", {})
                target_match = act.get("args", {}).get("target_id") == correct_obj
                if act.get("kind") == "apply" and target_match:
                    correct_count += 1
                    break
        except Exception as exc:
            _log.debug("counterfactual_accuracy: skipping episode: %s", exc)

    return MetricResult(
        name="counterfactual_accuracy",
        value=round(correct_count / count, 4) if count else 0.0,
        count=count,
        definition=_definition,
        gaming_risk=_gaming,
        metadata={"mode": "behavioral"},
    )


# ---------------------------------------------------------------------------
# Metric 7: Failure Diversity
# ---------------------------------------------------------------------------


def failure_diversity(steps: list[dict], outcomes: list[dict]) -> MetricResult:
    """Classify each failed episode into failure modes and count distinct modes.

    Modes: timeout, no_interaction, high_illegal, no_effects, energy_depleted.
    """
    failed = [o for o in outcomes if not o.get("goal_achieved")]
    if not failed:
        empty_counts: dict[str, int] = {
            m: 0
            for m in ("timeout", "no_interaction", "high_illegal", "no_effects", "energy_depleted")
        }
        empty_counts["distinct_modes"] = 0
        return MetricResult(
            name="failure_diversity",
            value=empty_counts,
            count=0,
            definition="Count of distinct failure modes across all failed episodes.",
            gaming_risk=(
                "Failure diversity does not penalise agents that fail identically "
                "in predictable ways; high diversity is not inherently desirable."
            ),
        )

    mode_counts: dict[str, int] = {
        "timeout": 0,
        "no_interaction": 0,
        "high_illegal": 0,
        "no_effects": 0,
        "energy_depleted": 0,
    }
    episodes_with_mode: set[str] = set()

    for ep in failed:
        ep_key = f"{ep.get('episode')}_{ep.get('seed')}_{ep.get('split')}_{ep.get('agent')}"
        # timeout: last step is done=True and max steps reached
        if ep.get("steps", 0) >= _DEFAULT_MAX_STEPS:
            mode_counts["timeout"] += 1
            episodes_with_mode.add(f"{ep_key}:timeout")

        if ep.get("interventions", 0) == 0:
            mode_counts["no_interaction"] += 1
            episodes_with_mode.add(f"{ep_key}:no_interaction")

        if ep.get("illegal_rate", 0.0) > 0.5:
            mode_counts["high_illegal"] += 1
            episodes_with_mode.add(f"{ep_key}:high_illegal")

        if not ep.get("unique_effects"):
            mode_counts["no_effects"] += 1
            episodes_with_mode.add(f"{ep_key}:no_effects")

        if ep.get("energy_remaining", 100) <= 0:
            mode_counts["energy_depleted"] += 1
            episodes_with_mode.add(f"{ep_key}:energy_depleted")

    active_modes = [m for m, c in mode_counts.items() if c > 0]
    mode_counts["distinct_modes"] = len(active_modes)

    return MetricResult(
        name="failure_diversity",
        value=mode_counts,
        count=len(failed),
        definition="Count of distinct failure modes across all failed episodes.",
        gaming_risk=(
            "Failure diversity does not penalise agents that fail identically "
            "in predictable ways; high diversity is not inherently desirable."
        ),
        metadata={"active_modes": active_modes},
    )


# ---------------------------------------------------------------------------
# Metric 8: Curriculum Progression
# ---------------------------------------------------------------------------


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    """Compute slope of least-squares line through (xs, ys)."""
    n = len(xs)
    if n < 2:
        return 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


def curriculum_progression(
    outcomes: list[dict],
    *,
    agent: str | None = None,
    family: str | None = None,
    window: int = 5,
) -> MetricResult:
    """TSR trend across seeds (proxy for learning improvement over time).

    Seeds are used as a temporal ordering proxy within each agent+family run.
    Computes rolling TSR in windows of `window` seeds and the overall slope.
    """
    agents = sorted({r["agent"] for r in outcomes if "agent" in r})
    families = sorted({r["family"] for r in outcomes if "family" in r})
    if agent is not None:
        agents = [a for a in agents if a == agent]
    if family is not None:
        families = [f for f in families if f == family]

    results: dict[str, dict] = {}
    total_count = 0

    for fam in families:
        for ag in agents:
            eps = sorted(
                filter_records(outcomes, family=fam, agent=ag),
                key=lambda r: r.get("seed", 0),
            )
            if not eps:
                continue
            total_count += len(eps)
            key = f"{fam}/{ag}"

            # Compute windowed TSR.
            windows: list[float] = []
            for start in range(0, len(eps), window):
                chunk = eps[start : start + window]
                if chunk:
                    w_tsr = sum(1 for r in chunk if r.get("goal_achieved")) / len(chunk)
                    windows.append(round(w_tsr, 4))

            xs = list(range(len(windows)))
            slope = round(_linear_slope(xs, windows), 6) if len(windows) >= 2 else 0.0
            results[key] = {"windows": windows, "slope": slope}

    return MetricResult(
        name="curriculum_progression",
        value=results,
        count=total_count,
        definition=(
            "Rolling TSR over seed-ordered windows within each agent+family run. "
            "Positive slope indicates improvement; negative indicates forgetting."
        ),
        gaming_risk=(
            "Slope is statistically meaningless for random agents or small seed counts; "
            "require at least 3 windows before interpreting trend direction."
        ),
    )


# ---------------------------------------------------------------------------
# Metric 9: Shortcut Exposure Score  (perturbation-based)
# ---------------------------------------------------------------------------


def shortcut_exposure_score(
    clean_outcomes: list[dict],
    perturbed_outcomes: list[dict],
    *,
    family: str | None = None,
    agent: str | None = None,
) -> MetricResult:
    """TSR drop when a surface-feature perturbation is applied.

    High value → agent relies on the perturbed surface feature (colour, shape,
    marker, position) rather than hidden causal structure.
    Low value  → agent is perturbation-resilient (uses hidden properties).

    Unlike shortcut_sensitivity, this metric is computed from paired
    (clean, perturbed) runs rather than from train/test splits.
    """
    clean_f = filter_records(clean_outcomes, family=family, agent=agent)
    pert_f = filter_records(perturbed_outcomes, family=family, agent=agent)
    clean_tsr = task_success_rate(clean_f).value if clean_f else 0.0
    pert_tsr = task_success_rate(pert_f).value if pert_f else 0.0
    count = len(clean_f) + len(pert_f)
    return MetricResult(
        name="shortcut_exposure_score",
        value=round(max(0.0, clean_tsr - pert_tsr), 4),
        count=count,
        definition=(
            "TSR drop between clean and perturbed episodes. "
            "Higher = greater reliance on the perturbed surface shortcut."
        ),
        gaming_risk=(
            "SES=0 is achieved trivially by always failing (TSR=0 on both). "
            "Interpret alongside absolute clean TSR."
        ),
    )


# ---------------------------------------------------------------------------
# Metric 10: Concept Reuse Proxy
# ---------------------------------------------------------------------------


# INSPECT/APPLY are the evidence-gathering interventions; effects that are empty
# or merely "no_effect"/"illegal_action" carry no causal information.
_EVIDENCE_KINDS = {"inspect", "apply"}
_UNINFORMATIVE_EFFECTS = {"no_effect", "illegal_action"}


def _episode_key(record: dict) -> tuple:
    # family is required: episode indices are per-(family, agent), so omitting it
    # would match same-index episodes across different families.
    return (
        record.get("family"),
        record.get("episode"),
        record.get("seed"),
        record.get("split"),
        record.get("agent"),
    )


def concept_reuse_proxy(
    steps: list[dict],
    outcomes: list[dict],
    *,
    family: str | None = None,
    agent: str | None = None,
) -> MetricResult:
    """Reuse of latent concepts across appearances (playbook §6).

    ConceptReuse = success_on_new_surface_forms_given_prior_evidence
                 - success_without_prior_evidence

    "New surface forms" = the TEST split (same hidden rule, changed visible
    features). "Prior evidence" = the agent performed at least one informative
    INSPECT/APPLY intervention (one producing a real effect) during the episode.
    A positive value means gathering causal evidence transfers to novel
    appearances — i.e. the agent reuses concepts rather than surface routines.
    """
    _definition = (
        "Test-split TSR with prior evidence minus test-split TSR without prior "
        "evidence. Evidence = an informative INSPECT/APPLY intervention in-episode."
    )
    _gaming = (
        "Inflated if hidden-rule structure leaks through object types/names, or by "
        "an agent that always intervenes trivially; read with intervention_validity."
    )

    test_outcomes = filter_records(outcomes, family=family, agent=agent, split="test")
    if not test_outcomes:
        return MetricResult(
            name="concept_reuse_proxy",
            value=None,
            count=0,
            definition=_definition,
            gaming_risk=_gaming,
            metadata={"note": "no test-split episodes; metric not computable."},
        )

    evidence_keys: set[tuple] = set()
    for s in steps:
        if s.get("split") != "test":
            continue
        if s.get("action", {}).get("kind") not in _EVIDENCE_KINDS:
            continue
        effects = s.get("effects") or []
        if any(e not in _UNINFORMATIVE_EFFECTS for e in effects):
            evidence_keys.add(_episode_key(s))

    with_ev = [o for o in test_outcomes if _episode_key(o) in evidence_keys]
    without_ev = [o for o in test_outcomes if _episode_key(o) not in evidence_keys]

    def _tsr(group: list[dict]) -> float | None:
        if not group:
            return None
        return sum(1 for r in group if r.get("goal_achieved")) / len(group)

    w, wo = _tsr(with_ev), _tsr(without_ev)
    reuse = round(w - wo, 4) if (w is not None and wo is not None) else None

    return MetricResult(
        name="concept_reuse_proxy",
        value={
            "with_evidence_tsr": round(w, 4) if w is not None else None,
            "without_evidence_tsr": round(wo, 4) if wo is not None else None,
            "concept_reuse": reuse,
            "n_with_evidence": len(with_ev),
            "n_without_evidence": len(without_ev),
        },
        count=len(test_outcomes),
        definition=_definition,
        gaming_risk=_gaming,
    )


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def full_report(
    trace_path: str | pathlib.Path,
    get_spec: Callable[[str, int, str], Any] | None = None,
) -> dict[str, MetricResult]:
    """Compute the full metric vector from a JSONL trace file.

    Args:
        trace_path: Path to the JSONL file produced by run_baselines.py.
        get_spec: Optional callable (family_str, seed_int, split_str) → TaskSpec.
                  Required for intervention_efficiency and counterfactual_accuracy.
    """
    steps, outcomes = load_trace(trace_path)
    return {
        "task_success_rate": task_success_rate(outcomes),
        "transfer_success": transfer_success(outcomes),
        "shortcut_sensitivity": shortcut_sensitivity(outcomes),
        "intervention_validity": intervention_validity(steps),
        "intervention_efficiency": intervention_efficiency(steps, outcomes, get_spec),
        "counterfactual_accuracy": counterfactual_accuracy(steps, outcomes, get_spec),
        "concept_reuse_proxy": concept_reuse_proxy(steps, outcomes),
        "failure_diversity": failure_diversity(steps, outcomes),
        "curriculum_progression": curriculum_progression(outcomes),
    }
