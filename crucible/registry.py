from __future__ import annotations

from typing import Iterable

from crucible.grammar import TaskFamily, TaskSpec, generate_task
from crucible.splits import SplitLabel


class TaskRegistry:
    """Thin coordination layer over generate_task for batch and family-listing operations."""

    @classmethod
    def generate(
        cls, family: str | TaskFamily, seed: int, split: SplitLabel | None = None
    ) -> TaskSpec:
        return generate_task(family, seed, split)

    @classmethod
    def generate_batch(
        cls,
        family: str | TaskFamily,
        seeds: Iterable[int],
        split: SplitLabel | None = None,
    ) -> list[TaskSpec]:
        return [generate_task(family, s, split) for s in seeds]

    @classmethod
    def list_families(cls) -> list[TaskFamily]:
        return list(TaskFamily)
