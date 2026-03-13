"""Mode switching rules."""

from src.fenghuo_chess.domain.models import GameState


def set_mode(state: GameState, mode: str) -> bool:
    if mode not in {"fast", "slow"}:
        return False
    if state.mode_locked:
        return False
    state.game_mode = mode
    return True


