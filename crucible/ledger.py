"""Lossless v0.2 episode ledger and compact prompt rendering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class LedgerEvent:
    sequence: int
    kind: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "kind": self.kind, "payload": self.payload}


class EpisodeLedger:
    """Append-only event log used as the source of truth for prompts and traces."""

    schema_version = "0.2"

    def __init__(self, events: Iterable[LedgerEvent] | None = None) -> None:
        self._events = list(events or [])

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        return tuple(self._events)

    def append(self, kind: str, **payload: Any) -> LedgerEvent:
        event = LedgerEvent(len(self._events), kind, payload)
        self._events.append(event)
        return event

    def record_message(self, role: str, content: str) -> LedgerEvent:
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role {role!r}")
        return self.append("message", role=role, content=content)

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": event.payload["role"], "content": event.payload["content"]}
            for event in self._events
            if event.kind == "message"
        ]

    def compact_message(self) -> dict[str, str]:
        """Return one lossless JSON message for the compact-ledger ablation."""
        return {
            "role": "user",
            "content": "EPISODE_LEDGER_JSONL\n" + self.to_jsonl(),
        }

    def semantic_projection(self) -> list[dict[str, Any]]:
        """Project action/effect evidence identically across prompt renderings."""
        return [
            event.to_dict()
            for event in self._events
            if event.kind
            in {
                "action",
                "effect",
                "observation",
                "commit",
                "forced_commit",
                "macro_action",
            }
        ]

    def to_jsonl(self) -> str:
        return "\n".join(canonical_json(event.to_dict()) for event in self._events)

    @classmethod
    def from_jsonl(cls, text: str) -> EpisodeLedger:
        events: list[LedgerEvent] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            events.append(
                LedgerEvent(
                    sequence=int(record["sequence"]),
                    kind=str(record["kind"]),
                    payload=dict(record["payload"]),
                )
            )
        if [event.sequence for event in events] != list(range(len(events))):
            raise ValueError("ledger sequence must be contiguous and zero-based")
        return cls(events)

    @property
    def hash(self) -> str:
        return content_hash([event.to_dict() for event in self._events])
