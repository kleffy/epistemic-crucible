"""Tests for train/test split assignment logic."""

from crucible.grammar import TaskFamily, generate_task
from crucible.splits import SplitLabel, assign_split, is_shortcut_family, split_seeds


def test_assign_split_deterministic():
    for seed in (0, 42, 999):
        assert assign_split(seed) == assign_split(seed)


def test_assign_split_roughly_80_20():
    seeds = list(range(1000))
    train_count = sum(1 for s in seeds if assign_split(s) == SplitLabel.TRAIN)
    ratio = train_count / len(seeds)
    assert 0.75 <= ratio <= 0.85, f"Train ratio {ratio:.2f} outside expected range"


def test_split_seeds_partition():
    seeds = list(range(200))
    groups = split_seeds(seeds)
    combined = sorted(groups[SplitLabel.TRAIN] + groups[SplitLabel.TEST])
    assert combined == sorted(seeds)
    assert len(set(combined)) == len(combined)  # no duplicates


def test_split_seeds_no_overlap():
    seeds = list(range(100))
    groups = split_seeds(seeds)
    assert set(groups[SplitLabel.TRAIN]).isdisjoint(set(groups[SplitLabel.TEST]))


def test_task_spec_has_split_label():
    for family in TaskFamily:
        spec = generate_task(family, seed=42)
        assert spec.split in (SplitLabel.TRAIN, SplitLabel.TEST)


def test_explicit_split_override():
    spec_train = generate_task(TaskFamily.AFFORDANCE, seed=0, split=SplitLabel.TRAIN)
    spec_test = generate_task(TaskFamily.AFFORDANCE, seed=0, split=SplitLabel.TEST)
    assert spec_train.split == SplitLabel.TRAIN
    assert spec_test.split == SplitLabel.TEST


def test_train_test_differ_for_affordance():
    spec_train = generate_task(TaskFamily.AFFORDANCE, seed=42, split=SplitLabel.TRAIN)
    spec_test = generate_task(TaskFamily.AFFORDANCE, seed=42, split=SplitLabel.TEST)
    # Training: RED tool is always the correct one (index 0)
    train_correct = next(s for s in spec_train.object_specs if s.role == "correct_tool")
    assert train_correct.color.value == "red"
    # Test: task_id differs, so correct tool may be different
    assert spec_train.task_id != spec_test.task_id


# ---------------------------------------------------------------------------
# is_shortcut_family
# ---------------------------------------------------------------------------


def test_is_shortcut_family_affordance():
    assert is_shortcut_family("affordance") is True
    assert is_shortcut_family(TaskFamily.AFFORDANCE) is True


def test_is_shortcut_family_tool_substitution():
    assert is_shortcut_family("tool_substitution") is True
    assert is_shortcut_family(TaskFamily.TOOL_SUBSTITUTION) is True


def test_is_shortcut_family_non_shortcut():
    for fam in ("causal_gate", "counterfactual", "contradiction"):
        assert is_shortcut_family(fam) is False

    for fam in (TaskFamily.CAUSAL_GATE, TaskFamily.COUNTERFACTUAL, TaskFamily.CONTRADICTION):
        assert is_shortcut_family(fam) is False
