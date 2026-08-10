"""Execute the non-scientific v0.2 integrity gates and emit an audit report."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time
from dataclasses import asdict

_HERE = pathlib.Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from crucible.agents.base import enumerate_candidate_actions  # noqa: E402
from crucible.agents.prompting import build_label_map  # noqa: E402
from crucible.factorial import (  # noqa: E402
    compile_factorial_certificate,
    execute_compiled_actions,
    generate_affordance_quartet,
    run_scripted_control,
    validate_affordance_quartet,
)
from crucible.factorial_metrics import compute_factorial_metrics  # noqa: E402
from crucible.grammar import build_world_from_spec  # noqa: E402
from crucible.metrics import analytic_chance_point  # noqa: E402
from crucible.observations import observe  # noqa: E402
from crucible.utils.logging import get_logger  # noqa: E402
from crucible.utils.serialization import to_dict  # noqa: E402

_LOG = get_logger(__name__)


def run_integrity_gate(
    *,
    invariant_seeds: int = 10_000,
    oracle_seeds: int = 1_000,
    control_seeds: int = 64,
) -> dict:
    started = time.time()
    violations: list[str] = []
    mechanism_public_pairs = 0
    mechanism_public_matches = 0
    label_pairs = 0
    label_matches = 0
    cue_masked_pairs = 0
    cue_masked_matches = 0
    candidate_pairs = 0
    candidate_matches = 0
    neutral_counts = [0, 0, 0]
    mechanism_carrier_counts = [0, 0, 0]
    cue_carrier_counts = [0, 0, 0]

    for seed in range(invariant_seeds):
        quartet = generate_affordance_quartet(seed)
        errors = validate_affordance_quartet(quartet)
        violations.extend(f"seed {seed}: {error}" for error in errors)
        observations = {
            (cell.mechanism_slot, cell.cue_slot): observe(
                build_world_from_spec(cell.task_spec, compact=True)
            )
            for cell in quartet.cells.values()
        }
        labels = {key: build_label_map(obs) for key, obs in observations.items()}
        candidates = {
            key: [to_dict(action) for action in enumerate_candidate_actions(obs)]
            for key, obs in observations.items()
        }
        neutral_counts[int(quartet.cell(0, 0).task_spec.metadata["neutral_slot"])] += 1
        for cell in quartet.cells.values():
            mechanism_carrier_counts[cell.mechanism_carrier_slot] += 1
            cue_carrier_counts[cell.cue_carrier_slot] += 1
        for cue in (0, 1):
            mechanism_public_pairs += 1
            if observations[(0, cue)] == observations[(1, cue)]:
                mechanism_public_matches += 1
            label_pairs += 1
            if labels[(0, cue)] == labels[(1, cue)]:
                label_matches += 1
            candidate_pairs += 1
            if candidates[(0, cue)] == candidates[(1, cue)]:
                candidate_matches += 1
        for mechanism in (0, 1):
            cue_masked_pairs += 1
            left = _mask_public_colours(observations[(mechanism, 0)])
            right = _mask_public_colours(observations[(mechanism, 1)])
            if left == right:
                cue_masked_matches += 1

    if max(neutral_counts) - min(neutral_counts) > 1:
        violations.append(f"neutral global-slot imbalance: {neutral_counts}")
    if max(mechanism_carrier_counts) - min(mechanism_carrier_counts) > 4:
        violations.append(f"mechanism global-slot imbalance: {mechanism_carrier_counts}")
    if max(cue_carrier_counts) - min(cue_carrier_counts) > 4:
        violations.append(f"cue global-slot imbalance: {cue_carrier_counts}")

    oracle_checks = 0
    for seed in range(oracle_seeds):
        for cell in generate_affordance_quartet(seed).cells.values():
            for use_detector in (False, True):
                actions = compile_factorial_certificate(cell, use_detector=use_detector)
                outcome = execute_compiled_actions(cell, actions)
                oracle_checks += 1
                if not outcome.solved or outcome.committed_slot != cell.mechanism_carrier_slot:
                    violations.append(
                        f"seed {seed} {cell.condition_id}: oracle failure detector={use_detector}"
                    )
                if outcome.steps > cell.task_spec.constraints.max_steps:
                    violations.append(
                        f"seed {seed} {cell.condition_id}: oracle exceeded step budget"
                    )
                if outcome.interventions > cell.task_spec.constraints.max_interventions:
                    violations.append(
                        f"seed {seed} {cell.condition_id}: oracle exceeded intervention budget"
                    )

    fixed_slot_cue_tracking = sum(
        cell.cue_carrier_slot == 0
        for seed in range(control_seeds)
        for cell in generate_affordance_quartet(seed).cells.values()
    ) / (4 * control_seeds)
    expected = {
        "mechanism_oracle": (1.0, 0.0, 0.5, 1.0, 1.0),
        "detector_policy": (1.0, 0.0, 0.5, 1.0, 1.0),
        "cue_follower": (0.0, 1.0, 1.0, 1.0, 0.0),
        "anti_cue": (0.0, 1.0, 0.0, 1.0, 0.0),
        "fixed_slot": (0.0, 0.0, fixed_slot_cue_tracking, 1.0, 0.0),
    }
    control_reports = {}
    for control, target in expected.items():
        outcomes = [
            run_scripted_control(cell, control)
            for seed in range(control_seeds)
            for cell in generate_affordance_quartet(seed).cells.values()
        ]
        report = compute_factorial_metrics(outcomes)
        control_reports[control] = asdict(report)
        observed = (
            report.mechanism_responsiveness.value,
            report.cue_susceptibility.value,
            report.cue_following.value,
            report.coverage.value,
            report.quartet_success.value,
        )
        if observed != target:
            violations.append(f"control {control}: expected {target}, observed {observed}")

    abstain = compute_factorial_metrics(
        run_scripted_control(cell, "abstain")
        for seed in range(control_seeds)
        for cell in generate_affordance_quartet(seed).cells.values()
    )
    if (
        abstain.coverage.value != 0.0
        or abstain.mechanism_responsiveness.value is not None
        or abstain.cue_susceptibility.value is not None
    ):
        violations.append("abstention missingness semantics failed")
    control_reports["abstain"] = asdict(abstain)
    random_outcomes = [
        run_scripted_control(cell, "random_committer")
        for seed in range(control_seeds)
        for cell in generate_affordance_quartet(seed).cells.values()
    ]
    control_reports["random_committer"] = asdict(compute_factorial_metrics(random_outcomes))
    focal_outcomes = [
        run_scripted_control(cell, "focal_uniform")
        for seed in range(control_seeds)
        for cell in generate_affordance_quartet(seed).cells.values()
    ]
    control_reports["focal_uniform"] = asdict(compute_factorial_metrics(focal_outcomes))

    report = {
        "schema_version": "0.2",
        "gate": "factorial_integrity",
        "passed": not violations,
        "counts": {
            "invariant_seeds": invariant_seeds,
            "oracle_seeds": oracle_seeds,
            "oracle_checks": oracle_checks,
            "control_seeds": control_seeds,
        },
        "leakage": {
            "mechanism_public_match_rate": mechanism_public_matches
            / max(mechanism_public_pairs, 1),
            "label_match_rate": label_matches / max(label_pairs, 1),
            "candidate_order_match_rate": candidate_matches / max(candidate_pairs, 1),
            "cue_axis_colour_masked_match_rate": cue_masked_matches / max(cue_masked_pairs, 1),
            "interpretation": (
                "Matched public observations are identical within each cue value, so a "
                "public-state classifier has exactly chance information about mechanism."
            ),
        },
        "slot_balance": {
            "neutral": neutral_counts,
            "mechanism_carrier": mechanism_carrier_counts,
            "cue_carrier": cue_carrier_counts,
        },
        "controls": control_reports,
        "analytic_uniform_reference_points": {
            "all_three_tools": analytic_chance_point(3),
            "red_blue_focal_tools": analytic_chance_point(2),
        },
        "violations": violations,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return report


def _mask_public_colours(observation: dict) -> dict:
    masked = copy.deepcopy(observation)
    for obj in masked.get("objects", {}).values():
        obj["color"] = "<manipulated-colour>"
    return masked


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invariant-seeds", type=int, default=10_000)
    parser.add_argument("--oracle-seeds", type=int, default=1_000)
    parser.add_argument("--control-seeds", type=int, default=64)
    parser.add_argument("--output", default="results/factorial_v02/integrity_report.json")
    args = parser.parse_args(argv)
    report = run_integrity_gate(
        invariant_seeds=args.invariant_seeds,
        oracle_seeds=args.oracle_seeds,
        control_seeds=args.control_seeds,
    )
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _LOG.info("integrity gate passed=%s output=%s", report["passed"], output)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
