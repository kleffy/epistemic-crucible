from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

_PUBLIC_OMIT_FIELDS = {"hidden", "rules", "pending_effects"}


def to_dict(obj: Any, *, public: bool = True) -> Any:
    """Recursively serialise dataclasses/enums to JSON-safe primitives.

    When public=True the 'hidden' field of CrucibleObject is omitted,
    enforcing the visible/hidden boundary for agent-facing data.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in dataclasses.fields(obj):
            if public and f.name in _PUBLIC_OMIT_FIELDS:
                continue
            result[f.name] = to_dict(getattr(obj, f.name), public=public)
        return result
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_dict(v, public=public) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        serialised = [to_dict(v, public=public) for v in obj]
        return type(obj)(serialised) if isinstance(obj, tuple) else serialised
    return obj


def from_dict(cls: type, d: Any) -> Any:
    """Reconstruct a dataclass instance from a plain dict.

    Handles nested dataclasses, enums, and tuple positions.
    """
    if not dataclasses.is_dataclass(cls) or isinstance(d, cls):
        return d
    if d is None:
        return None

    kwargs: dict[str, Any] = {}
    field_map = {f.name: f for f in dataclasses.fields(cls)}

    for fname, f in field_map.items():
        if fname not in d:
            continue
        val = d[fname]
        ftype = f.type
        kwargs[fname] = _coerce(ftype, val)

    return cls(**kwargs)


def _coerce(type_hint: Any, value: Any) -> Any:
    """Best-effort coercion used by from_dict for common type patterns."""
    if value is None:
        return None
    # Resolve string annotations
    if isinstance(type_hint, str):
        return value
    # Enums
    if isinstance(type_hint, type) and issubclass(type_hint, Enum):
        return type_hint(value)
    # Dataclasses
    if dataclasses.is_dataclass(type_hint) and isinstance(value, dict):
        return from_dict(type_hint, value)
    # tuple position: list [r, c] -> tuple
    if type_hint is tuple or (hasattr(type_hint, "__origin__") and type_hint.__origin__ is tuple):
        if isinstance(value, (list, tuple)):
            return tuple(value)
    return value
