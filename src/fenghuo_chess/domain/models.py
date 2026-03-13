"""Domain data models."""

from dataclasses import dataclass, field

import numpy as np

from src.fenghuo_chess.constants.gameplay import BOARD_SIZE, STAGE_TITLES


@dataclass
class GameState:
    board: np.ndarray
    stage: int = 1
    current_player: int = 1
    game_over: bool = False
    winner: int = 0
    message: str = STAGE_TITLES[1]

    exposure_message: str = ""
    stage_positions: list[tuple[int, int]] = field(default_factory=list)
    exposure_occurred: bool = False
    exposure_positions: list[tuple[int, int]] = field(default_factory=list)
    exposure_markers: list[tuple[int, int]] = field(default_factory=list)
    checked_positions: set[tuple[int, int]] = field(default_factory=set)
    last_exposure_player: int = 0
    consecutive_exposure: bool = False

    show_intro: bool = True
    intro_p1_source: str = "human"
    intro_p2_source: str = "baseline"
    end_overlay_closed: bool = False
    game_mode: str = "slow"
    mode_locked: bool = False

    restart_button_state: str = "normal"
    restart_button_press_time: float = 0.0

    def clone(self) -> "GameState":
        """Return a lightweight deep copy for simulation usage."""
        return GameState(
            board=self.board.copy(),
            stage=int(self.stage),
            current_player=int(self.current_player),
            game_over=bool(self.game_over),
            winner=int(self.winner),
            message=str(self.message),
            exposure_message=str(self.exposure_message),
            stage_positions=list(self.stage_positions),
            exposure_occurred=bool(self.exposure_occurred),
            exposure_positions=list(self.exposure_positions),
            exposure_markers=list(self.exposure_markers),
            checked_positions=set(self.checked_positions),
            last_exposure_player=int(self.last_exposure_player),
            consecutive_exposure=bool(self.consecutive_exposure),
            show_intro=bool(self.show_intro),
            intro_p1_source=str(self.intro_p1_source),
            intro_p2_source=str(self.intro_p2_source),
            end_overlay_closed=bool(self.end_overlay_closed),
            game_mode=str(self.game_mode),
            mode_locked=bool(self.mode_locked),
            restart_button_state=str(self.restart_button_state),
            restart_button_press_time=float(self.restart_button_press_time),
        )


def create_initial_state() -> GameState:
    from src.fenghuo_chess.domain.rules import get_stage_positions

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    state = GameState(board=board)
    state.stage_positions = get_stage_positions(state.stage)
    return state


