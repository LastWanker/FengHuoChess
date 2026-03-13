"""Domain events emitted by pure logic steps."""

from dataclasses import dataclass, field
from typing import Any

Color = tuple[int, int, int]


@dataclass(frozen=True)
class LogicEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def broadcast(text: str, color: Color) -> "LogicEvent":
        return LogicEvent(kind="BROADCAST", payload={"text": text, "color": color})

