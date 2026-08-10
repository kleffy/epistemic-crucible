from __future__ import annotations

from enum import Enum
from typing import Any

from crucible.utils.seeding import make_rng

_SPLIT_RNG_OFFSET = 1_000_000

# Families where the train→test split deliberately decorrelates a visible surface
# feature from the hidden causal property that determines task success.
_SHORTCUT_FAMILIES: frozenset[str] = frozenset({"affordance", "tool_substitution"})


def is_shortcut_family(family: str | Any) -> bool:
    """Return True for families where train/test decorrelates a surface shortcut.

    Accepts either a string name or a TaskFamily enum value.
    """
    val = family.value if hasattr(family, "value") else str(family)
    return val in _SHORTCUT_FAMILIES


class SplitLabel(str, Enum):
    TRAIN = "train"
    TEST = "test"


class SplitPolicy(str, Enum):
    FEATURE_HOLDOUT = "feature_holdout"
    COMPOSITION_HOLDOUT = "composition_holdout"
    RULE_HOLDOUT = "rule_holdout"
    COUNTERFACTUAL_HOLDOUT = "counterfactual_holdout"


def assign_split(seed: int, train_ratio: float = 0.8) -> SplitLabel:
    """Deterministically assign a seed to TRAIN or TEST.

    Uses seed + _SPLIT_RNG_OFFSET to avoid correlation with the world-gen RNG
    that uses the bare seed.
    """
    rng = make_rng(seed + _SPLIT_RNG_OFFSET)
    return SplitLabel.TRAIN if float(rng.random()) < train_ratio else SplitLabel.TEST


def split_seeds(seeds: list[int], train_ratio: float = 0.8) -> dict[SplitLabel, list[int]]:
    """Partition a list of seeds into TRAIN and TEST groups."""
    result: dict[SplitLabel, list[int]] = {SplitLabel.TRAIN: [], SplitLabel.TEST: []}
    for s in seeds:
        result[assign_split(s, train_ratio)].append(s)
    return result
