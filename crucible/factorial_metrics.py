"""Paired behavioral-attribution metrics for v0.2 affordance quartets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from crucible.factorial import CommitOutcome


@dataclass(frozen=True)
class MetricEstimate:
    value: float | None
    ci_low: float | None
    ci_high: float | None
    denominator: int


@dataclass(frozen=True)
class FactorialMetricReport:
    mechanism_tracking: MetricEstimate
    mechanism_responsiveness: MetricEstimate
    cue_susceptibility: MetricEstimate
    cue_susceptibility_all: MetricEstimate
    cue_following: MetricEstimate
    coverage: MetricEstimate
    natural_coverage: MetricEstimate
    choice_accuracy: MetricEstimate
    detector_query_rate: MetricEstimate
    identification_coverage: MetricEstimate
    evidence_consistent_commitment: MetricEstimate
    first_query_cue_bias: MetricEstimate
    unique_queries: MetricEstimate
    coverage_by_cell: dict[str, float]
    quartet_success: MetricEstimate
    cell_success: MetricEstimate
    aligned_success: MetricEstimate
    crossed_success: MetricEstimate
    base_seeds: int
    complete_quartets: int


@dataclass(frozen=True)
class ChallengeMetricReport:
    """Seed-clustered report for the promoted six-cell 3x2 assay."""

    mechanism_accuracy: MetricEstimate
    cue_susceptibility: MetricEstimate
    cue_following: MetricEstimate
    coverage: MetricEstimate
    detector_query_rate: MetricEstimate
    all_cells_success: MetricEstimate
    cell_success: MetricEstimate
    coverage_by_cell: dict[str, float]
    base_seeds: int
    complete_challenges: int


def compute_factorial_metrics(
    outcomes: Iterable[CommitOutcome],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> FactorialMetricReport:
    """Compute metrics and paired base-seed bootstrap confidence intervals."""
    outcomes = list(outcomes)
    grouped = _group_complete(outcomes)
    seed_values = [_seed_values(cells) for _, cells in sorted(grouped.items())]
    rng = np.random.default_rng(bootstrap_seed)
    return FactorialMetricReport(
        mechanism_tracking=_estimate(
            seed_values,
            lambda value: value["mechanism_tracking_values"],
            bootstrap_samples,
            rng,
        ),
        mechanism_responsiveness=_estimate(
            seed_values,
            lambda value: value["mechanism_values"],
            bootstrap_samples,
            rng,
        ),
        cue_susceptibility=_estimate(
            seed_values,
            lambda value: value["cue_values"],
            bootstrap_samples,
            rng,
        ),
        cue_susceptibility_all=_estimate(
            seed_values,
            lambda value: value["cue_all_values"],
            bootstrap_samples,
            rng,
        ),
        cue_following=_estimate(
            seed_values,
            lambda value: value["cue_following_values"],
            bootstrap_samples,
            rng,
        ),
        coverage=_estimate(
            seed_values,
            lambda value: [value["coverage"]],
            bootstrap_samples,
            rng,
        ),
        natural_coverage=_estimate(
            seed_values,
            lambda value: [value["natural_coverage"]],
            bootstrap_samples,
            rng,
        ),
        choice_accuracy=_estimate(
            seed_values,
            lambda value: value["choice_accuracy_values"],
            bootstrap_samples,
            rng,
        ),
        detector_query_rate=_estimate(
            seed_values,
            lambda value: value["detector_query_values"],
            bootstrap_samples,
            rng,
        ),
        identification_coverage=_estimate(
            seed_values,
            lambda value: value["identification_values"],
            bootstrap_samples,
            rng,
        ),
        evidence_consistent_commitment=_estimate(
            seed_values,
            lambda value: value["evidence_consistency_values"],
            bootstrap_samples,
            rng,
        ),
        first_query_cue_bias=_estimate(
            seed_values,
            lambda value: value["first_query_cue_values"],
            bootstrap_samples,
            rng,
        ),
        unique_queries=_estimate(
            seed_values,
            lambda value: value["unique_query_values"],
            bootstrap_samples,
            rng,
        ),
        coverage_by_cell={
            f"m{mechanism}_c{cue}": _cell_coverage(grouped, mechanism, cue)
            for mechanism in (0, 1)
            for cue in (0, 1)
        },
        quartet_success=_estimate(
            seed_values,
            lambda value: [value["quartet_success"]],
            bootstrap_samples,
            rng,
        ),
        cell_success=_estimate(
            seed_values,
            lambda value: [value["cell_success"]],
            bootstrap_samples,
            rng,
        ),
        aligned_success=_estimate(
            seed_values,
            lambda value: [value["aligned_success"]],
            bootstrap_samples,
            rng,
        ),
        crossed_success=_estimate(
            seed_values,
            lambda value: [value["crossed_success"]],
            bootstrap_samples,
            rng,
        ),
        base_seeds=len({outcome.base_seed for outcome in outcomes}),
        complete_quartets=len(grouped),
    )


def paired_arm_contrast(
    reference: Iterable[CommitOutcome],
    treatment: Iterable[CommitOutcome],
    metric: str,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> MetricEstimate:
    """Estimate a treatment-minus-reference effect with paired seed resampling."""
    selectors: dict[str, Callable[[dict], list[float]]] = {
        "mechanism_tracking": lambda value: value["mechanism_tracking_values"],
        "mechanism_responsiveness": lambda value: value["mechanism_values"],
        "cue_following": lambda value: value["cue_following_values"],
        "cue_susceptibility": lambda value: value["cue_values"],
        "detector_query_rate": lambda value: value["detector_query_values"],
        "coverage": lambda value: [value["coverage"]],
        "choice_accuracy": lambda value: value["choice_accuracy_values"],
    }
    if metric not in selectors:
        raise ValueError(f"unknown paired contrast metric {metric!r}")
    reference_grouped = _group_complete(reference)
    treatment_grouped = _group_complete(treatment)
    if set(reference_grouped) != set(treatment_grouped):
        raise ValueError("paired arm contrasts require identical complete base-seed sets")
    selector = selectors[metric]
    seed_differences: list[float] = []
    for seed in sorted(reference_grouped):
        reference_values = selector(_seed_values(reference_grouped[seed]))
        treatment_values = selector(_seed_values(treatment_grouped[seed]))
        if reference_values and treatment_values:
            seed_differences.append(float(np.mean(treatment_values) - np.mean(reference_values)))
    if not seed_differences:
        return MetricEstimate(None, None, None, 0)
    observed = float(np.mean(seed_differences))
    if bootstrap_samples <= 0:
        return MetricEstimate(observed, None, None, len(seed_differences))
    rng = np.random.default_rng(bootstrap_seed)
    values = np.asarray(seed_differences)
    indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
    boot = values[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return MetricEstimate(observed, float(low), float(high), len(seed_differences))


def compute_challenge_metrics(
    outcomes: Iterable[CommitOutcome],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> ChallengeMetricReport:
    """Compute 3x2 metrics with base seed as the bootstrap unit."""
    outcomes = list(outcomes)
    grouped = _group_complete_challenge(outcomes)
    seed_values = [_challenge_seed_values(cells) for _, cells in sorted(grouped.items())]
    rng = np.random.default_rng(bootstrap_seed)
    return ChallengeMetricReport(
        mechanism_accuracy=_estimate(
            seed_values,
            lambda value: value["mechanism_accuracy_values"],
            bootstrap_samples,
            rng,
        ),
        cue_susceptibility=_estimate(
            seed_values,
            lambda value: value["cue_susceptibility_values"],
            bootstrap_samples,
            rng,
        ),
        cue_following=_estimate(
            seed_values,
            lambda value: value["cue_following_values"],
            bootstrap_samples,
            rng,
        ),
        coverage=_estimate(
            seed_values,
            lambda value: [value["coverage"]],
            bootstrap_samples,
            rng,
        ),
        detector_query_rate=_estimate(
            seed_values,
            lambda value: value["detector_query_values"],
            bootstrap_samples,
            rng,
        ),
        all_cells_success=_estimate(
            seed_values,
            lambda value: [value["all_cells_success"]],
            bootstrap_samples,
            rng,
        ),
        cell_success=_estimate(
            seed_values,
            lambda value: [value["cell_success"]],
            bootstrap_samples,
            rng,
        ),
        coverage_by_cell={
            f"m{mechanism}_c{cue}": _cell_coverage(grouped, mechanism, cue)
            for mechanism in (0, 1, 2)
            for cue in (0, 1)
        },
        base_seeds=len({outcome.base_seed for outcome in outcomes}),
        complete_challenges=len(grouped),
    )


def paired_challenge_arm_contrast(
    reference: Iterable[CommitOutcome],
    treatment: Iterable[CommitOutcome],
    metric: str,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> MetricEstimate:
    """Estimate a treatment-minus-reference 3x2 effect by paired seed resampling."""
    selectors: dict[str, Callable[[dict], list[float]]] = {
        "mechanism_accuracy": lambda value: value["mechanism_accuracy_values"],
        "cue_susceptibility": lambda value: value["cue_susceptibility_values"],
        "cue_following": lambda value: value["cue_following_values"],
        "detector_query_rate": lambda value: value["detector_query_values"],
        "coverage": lambda value: [value["coverage"]],
        "all_cells_success": lambda value: [value["all_cells_success"]],
        "cell_success": lambda value: [value["cell_success"]],
    }
    if metric not in selectors:
        raise ValueError(f"unknown paired challenge metric {metric!r}")
    reference_grouped = _group_complete_challenge(reference)
    treatment_grouped = _group_complete_challenge(treatment)
    if set(reference_grouped) != set(treatment_grouped):
        raise ValueError("paired challenge contrasts require identical complete base-seed sets")
    selector = selectors[metric]
    seed_differences: list[float] = []
    for seed in sorted(reference_grouped):
        reference_values = selector(_challenge_seed_values(reference_grouped[seed]))
        treatment_values = selector(_challenge_seed_values(treatment_grouped[seed]))
        if reference_values and treatment_values:
            seed_differences.append(float(np.mean(treatment_values) - np.mean(reference_values)))
    if not seed_differences:
        return MetricEstimate(None, None, None, 0)
    observed = float(np.mean(seed_differences))
    if bootstrap_samples <= 0:
        return MetricEstimate(observed, None, None, len(seed_differences))
    rng = np.random.default_rng(bootstrap_seed)
    values = np.asarray(seed_differences)
    indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
    boot = values[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return MetricEstimate(observed, float(low), float(high), len(seed_differences))


def _group_complete(
    outcomes: Iterable[CommitOutcome],
) -> dict[int, dict[tuple[int, int], CommitOutcome]]:
    grouped: dict[int, dict[tuple[int, int], CommitOutcome]] = {}
    for outcome in outcomes:
        key = (outcome.mechanism_slot, outcome.cue_slot)
        cells = grouped.setdefault(outcome.base_seed, {})
        if key in cells:
            raise ValueError(f"duplicate outcome for seed={outcome.base_seed}, cell={key}")
        cells[key] = outcome
    expected = {(0, 0), (0, 1), (1, 0), (1, 1)}
    complete: dict[int, dict[tuple[int, int], CommitOutcome]] = {}
    for seed, cells in grouped.items():
        if set(cells) == expected:
            complete[seed] = cells
    return complete


def _group_complete_challenge(
    outcomes: Iterable[CommitOutcome],
) -> dict[int, dict[tuple[int, int], CommitOutcome]]:
    grouped: dict[int, dict[tuple[int, int], CommitOutcome]] = {}
    for outcome in outcomes:
        key = (outcome.mechanism_slot, outcome.cue_slot)
        cells = grouped.setdefault(outcome.base_seed, {})
        if key in cells:
            raise ValueError(f"duplicate outcome for seed={outcome.base_seed}, cell={key}")
        cells[key] = outcome
    expected = {(mechanism, cue) for mechanism in (0, 1, 2) for cue in (0, 1)}
    return {seed: cells for seed, cells in grouped.items() if set(cells) == expected}


def _seed_values(cells: dict[tuple[int, int], CommitOutcome]) -> dict:
    mechanism_values: list[float] = []
    cue_values: list[float] = []
    for cue in (0, 1):
        left_cell = cells[(0, cue)]
        right_cell = cells[(1, cue)]
        left = left_cell.committed_slot
        right = right_cell.committed_slot
        if left is not None and right is not None:
            mechanism_values.append(
                float(
                    left == _mechanism_carrier(left_cell)
                    and right == _mechanism_carrier(right_cell)
                )
            )
    for mechanism in (0, 1):
        left = cells[(mechanism, 0)].committed_slot
        right = cells[(mechanism, 1)].committed_slot
        if left is not None and right is not None:
            cue_values.append(float(left != right))
    values = list(cells.values())
    cue_all_values = [
        float(cells[(mechanism, 0)].committed_slot != cells[(mechanism, 1)].committed_slot)
        for mechanism in (0, 1)
    ]
    return {
        "mechanism_values": mechanism_values,
        "cue_values": cue_values,
        "cue_all_values": cue_all_values,
        "mechanism_tracking_values": [
            float(value.committed_slot == _mechanism_carrier(value))
            for value in values
            if value.committed_slot is not None
        ],
        "cue_following_values": [
            float(value.committed_slot == _cue_carrier(value))
            for value in values
            if value.committed_slot is not None
        ],
        "coverage": sum(value.committed_slot is not None for value in values) / 4.0,
        "natural_coverage": sum(value.commit_mode == "natural" for value in values) / 4.0,
        "choice_accuracy_values": [
            float(
                value.committed_slot is not None
                and value.committed_slot == _mechanism_carrier(value)
            )
            for value in values
        ],
        "detector_query_values": [float(value.detector_queries > 0) for value in values],
        "identification_values": [
            float(value.mechanism_identified_before_commit) for value in values
        ],
        "evidence_consistency_values": [
            float(value.evidence_consistent_commit)
            for value in values
            if value.evidence_consistent_commit is not None
        ],
        "first_query_cue_values": [
            float(value.first_query_slot == _cue_carrier(value))
            for value in values
            if value.first_query_slot is not None
        ],
        "unique_query_values": [float(value.unique_queries) for value in values],
        "quartet_success": float(all(value.solved for value in values)),
        "cell_success": sum(value.solved for value in values) / 4.0,
        "aligned_success": (float(cells[(0, 0)].solved) + float(cells[(1, 1)].solved)) / 2.0,
        "crossed_success": (float(cells[(0, 1)].solved) + float(cells[(1, 0)].solved)) / 2.0,
    }


def _challenge_seed_values(cells: dict[tuple[int, int], CommitOutcome]) -> dict:
    values = list(cells.values())
    cue_susceptibility_values: list[float] = []
    for mechanism in (0, 1, 2):
        left = cells[(mechanism, 0)].committed_slot
        right = cells[(mechanism, 1)].committed_slot
        if left is not None and right is not None:
            cue_susceptibility_values.append(float(left != right))
    return {
        "mechanism_accuracy_values": [
            float(value.committed_slot == _mechanism_carrier(value))
            for value in values
            if value.committed_slot is not None
        ],
        "cue_susceptibility_values": cue_susceptibility_values,
        "cue_following_values": [
            float(value.committed_slot == _cue_carrier(value))
            for value in values
            if value.committed_slot is not None
        ],
        "coverage": sum(value.committed_slot is not None for value in values) / 6.0,
        "detector_query_values": [float(value.detector_queries > 0) for value in values],
        "all_cells_success": float(all(value.solved for value in values)),
        "cell_success": sum(value.solved for value in values) / 6.0,
    }


def _mechanism_carrier(outcome: CommitOutcome) -> int:
    return (
        outcome.mechanism_carrier_slot
        if outcome.mechanism_carrier_slot is not None
        else outcome.mechanism_slot
    )


def _cue_carrier(outcome: CommitOutcome) -> int:
    return outcome.cue_carrier_slot if outcome.cue_carrier_slot is not None else outcome.cue_slot


def _cell_coverage(
    grouped: dict[int, dict[tuple[int, int], CommitOutcome]],
    mechanism: int,
    cue: int,
) -> float:
    values = [cells[(mechanism, cue)] for cells in grouped.values()]
    return (
        sum(value.committed_slot is not None for value in values) / len(values) if values else 0.0
    )


def _estimate(
    seed_values: list[dict],
    selector: Callable[[dict], list[float]],
    samples: int,
    rng: np.random.Generator,
) -> MetricEstimate:
    observed = [item for seed_value in seed_values for item in selector(seed_value)]
    if not observed:
        return MetricEstimate(None, None, None, 0)
    value = float(np.mean(observed))
    if samples <= 0 or not seed_values:
        return MetricEstimate(value, None, None, len(observed))
    boot: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(seed_values), size=len(seed_values))
        selected = [item for index in indices for item in selector(seed_values[int(index)])]
        if selected:
            boot.append(float(np.mean(selected)))
    if not boot:
        return MetricEstimate(value, None, None, len(observed))
    low, high = np.quantile(np.asarray(boot), [0.025, 0.975])
    return MetricEstimate(value, float(low), float(high), len(observed))
