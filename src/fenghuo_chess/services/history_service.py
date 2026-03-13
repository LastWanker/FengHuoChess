"""Snapshot-based history service for undo."""

import copy
from dataclasses import dataclass

from src.fenghuo_chess.domain.models import GameState


@dataclass
class HistoryService:
    snapshots: list[GameState]
    cursor: int

    def __init__(self) -> None:
        self.snapshots = []
        self.cursor = -1

    def reset(self, initial_state: GameState) -> None:
        self.snapshots = [copy.deepcopy(initial_state)]
        self.cursor = 0

    def push(self, state: GameState) -> None:
        if self.cursor < len(self.snapshots) - 1:
            self.snapshots = self.snapshots[: self.cursor + 1]
        self.snapshots.append(copy.deepcopy(state))
        self.cursor = len(self.snapshots) - 1

    def undo(self) -> GameState | None:
        if self.cursor <= 0:
            return None
        self.cursor -= 1
        return copy.deepcopy(self.snapshots[self.cursor])


