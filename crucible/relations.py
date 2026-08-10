from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RelationKind(str, Enum):
    ADJACENT = "adjacent"
    INSIDE = "inside"
    HELD = "held"
    CONNECTED_TO = "connected_to"
    BLOCKS = "blocks"
    UNLOCKS = "unlocks"
    POWERED_BY = "powered_by"
    MODIFIED_BY = "modified_by"


@dataclass
class Relation:
    kind: RelationKind
    subject: str  # obj_id or "agent"
    object_: str  # obj_id, "agent", or "path"
    metadata: dict = field(default_factory=dict)
