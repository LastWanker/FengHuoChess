"""Trace recorder for training sample export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.fenghuo_chess.application.logic_api import Action
from src.fenghuo_chess.domain.events import LogicEvent
from src.fenghuo_chess.domain.models import GameState
from src.fenghuo_chess.domain.serialization import state_to_dict


@dataclass
class TraceRecorder:
    rows: list[dict] = field(default_factory=list)

    def record_step(
        self,
        *,
        state: GameState,
        legal_actions: list[Action],
        action: Action,
        player: int,
        events: list[LogicEvent],
    ) -> None:
        self.rows.append(
            {
                "state": state_to_dict(state, core_only=True),
                "legal_actions": [[a.row, a.col] for a in legal_actions],
                "action": [action.row, action.col],
                "player": int(player),
                "events": [event.kind for event in events],
                "outcome": None,
            }
        )

    def finalize_outcome(self, winner: int) -> None:
        for row in self.rows:
            if winner == 0:
                row["outcome"] = 0
            elif int(row["player"]) == int(winner):
                row["outcome"] = 1
            else:
                row["outcome"] = -1

    def write_jsonl(self, output_path: str | Path, append: bool = True) -> int:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")
        return len(self.rows)

