from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ObjectType(str, Enum):
    BLOCK = "block"
    KEY = "key"
    GATE = "gate"
    SOURCE = "source"
    TOOL = "tool"
    DETECTOR = "detector"
    CATALYST = "catalyst"
    HAZARD = "hazard"
    TOKEN = "token"


class ObjectColor(str, Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    GREY = "grey"


class ObjectShape(str, Enum):
    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    ROD = "rod"
    FLAT = "flat"


class ObjectTexture(str, Enum):
    SMOOTH = "smooth"
    ROUGH = "rough"
    STRIPED = "striped"
    DOTTED = "dotted"


class ObjectSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ObjectState(str, Enum):
    DEFAULT = "default"
    ACTIVE = "active"
    OPEN = "open"
    CLOSED = "closed"
    DESTROYED = "destroyed"


AFFINITY_VALUES: tuple[str | None, ...] = ("metal", "organic", "ceramic", None)


@dataclass
class VisibleObjectState:
    obj_type: ObjectType
    color: ObjectColor
    shape: ObjectShape
    texture: ObjectTexture
    size: ObjectSize
    marker: str | None
    pos: tuple[int, int] | None  # None when held in inventory
    state: ObjectState


@dataclass
class HiddenObjectProps:
    conductivity: bool
    solubility: bool
    magnetism: bool
    charge: bool
    fragility: bool
    affinity: str | None


@dataclass
class CrucibleObject:
    obj_id: str
    visible: VisibleObjectState
    hidden: HiddenObjectProps = field(repr=False)
