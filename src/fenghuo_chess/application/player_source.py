"""Player input sources for human and AI-controlled turns."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.fenghuo_chess.application.logic_api import Action
from src.fenghuo_chess.domain.models import GameState


class PlayerSource(Protocol):
    def select_action(self, state: GameState, legal_actions: Sequence[Action]) -> Action | None:
        ...


class HumanPlayerSource:
    def __init__(self) -> None:
        self._pending_action: Action | None = None

    def submit_action(self, action: Action) -> None:
        self._pending_action = action

    def clear_pending(self) -> None:
        self._pending_action = None

    def select_action(self, state: GameState, legal_actions: Sequence[Action]) -> Action | None:
        action = self._pending_action
        self._pending_action = None
        if action is None:
            return None
        if action in legal_actions:
            return action
        return None

