"""Random legal-action AI baseline."""

from __future__ import annotations

import random
from collections.abc import Sequence

from src.fenghuo_chess.application.logic_api import Action
from src.fenghuo_chess.domain.models import GameState


class RandomAISource:
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def select_action(self, state: GameState, legal_actions: Sequence[Action]) -> Action | None:
        if not legal_actions:
            return None
        return self._rng.choice(list(legal_actions))

