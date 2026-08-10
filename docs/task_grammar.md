# Task Grammar

## Overview

A `TaskSpec` is a complete blueprint for one evaluation episode. It encodes:

- **ObjectSpec list**, the exact objects in the world, including all hidden causal properties
- **GoalSpec**, what the agent must achieve
- **ConstraintSpec**, budget for interventions, steps, and energy
- **SolutionCertificate**, the oracle action sequence and which rules it requires
- **SplitLabel**, whether the world belongs to the training or test distribution
- **pressure_labels**, which diagnostic pressures this task is designed to exercise

## Task Families

| ID | Family | Key objects | Hidden property | Goal | Train/Test split |
|----|--------|-------------|-----------------|------|-----------------|
| A | `affordance` | 3 TOOLs + GATE | `conductivity` | OPEN gate | Train: RED=conductive; Test: colour decorrelated |
| B | `causal_gate` | KEY + SOURCE + GATE | `magnetism` on KEY | OPEN gate | Seeds vary layout; structure stable |
| C | `counterfactual` | 2 BLOCKs + SOURCE | `solubility` | CLASSIFY which block | Test: block property assignment swapped |
| D | `tool_substitution` | 2 TOOLs + GATE | `conductivity` on correct tool | OPEN gate | Train: ROD shape; Test: FLAT shape (novel) |
| E | `contradiction` | 2 KEYs + GATE | `affinity` on gate | OPEN gate | Train: no affinity; Test: affinity required |

## Goal kinds

| GoalKind | Predicate | Machine-checkable |
|----------|-----------|-------------------|
| `OPEN` | `obj.visible.state == OPEN` | Yes |
| `RETRIEVE` | target type in agent inventory | Yes |
| `REACH` | `agent.pos == target_pos` | Yes |
| `CLASSIFY` | correct object identified via trace | No |
| `TRANSFORM` | `obj.visible.state.value == target_state` | Yes |

## Object ID convention

Object IDs embed the task ID: `{task_id}_<role_suffix>`.

Task IDs encode the family prefix, seed, and split value, e.g. `aff_42_train_gate`.

## Generating tasks

```python
from crucible.grammar import TaskFamily, generate_task, validate_task, build_world_from_spec

spec = generate_task(TaskFamily.AFFORDANCE, seed=42)
assert validate_task(spec) == []
world = build_world_from_spec(spec)
assert len(world.objects) == len(spec.object_specs)
```

## Registry

```python
from crucible.registry import TaskRegistry

specs = TaskRegistry.generate_batch(TaskFamily.AFFORDANCE, seeds=range(100))
families = TaskRegistry.list_families()
```

## Validity rules

`validate_task(spec)` returns an empty list if and only if all of the following hold:

1. `constraints.max_interventions > 0` and `constraints.max_steps > 0` and `constraints.energy_budget > 0`
2. `pressure_labels` is non-empty
3. `solution_certificate.action_sequence` is non-empty
4. `split` is `TRAIN` or `TEST`
5. Any `target_obj_id` or `classify_correct_obj_id` in the goal exists in `object_specs`
6. Any object ID referenced in solution action args (`obj_id`, `tool_id`, `target_id`) exists in `object_specs`

## Train/test split

Splits are assigned deterministically per seed using a separate RNG stream (`seed + 1_000_000`) to avoid correlation with the world-generation RNG:

```python
from crucible.splits import assign_split, split_seeds, SplitLabel

label = assign_split(seed=42)  # TRAIN or TEST
groups = split_seeds(list(range(100)), train_ratio=0.8)
```
