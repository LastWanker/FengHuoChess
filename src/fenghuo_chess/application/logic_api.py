"""Pure move application API shared by UI, AI and headless runners."""

from __future__ import annotations
from dataclasses import dataclass

from src.fenghuo_chess.constants.colors import RED, WHITE
from src.fenghuo_chess.constants.gameplay import STAGE_BROADCASTS, WIN_BROADCASTS
from src.fenghuo_chess.domain.events import LogicEvent
from src.fenghuo_chess.domain.models import GameState
from src.fenghuo_chess.domain.rules import advance_stage_if_needed, check_win, count_stones, is_valid_move
from src.fenghuo_chess.domain.scoring import determine_winner
from src.fenghuo_chess.services.exposure_service import ExposureService


@dataclass(frozen=True)
class Action:
    row: int
    col: int


@dataclass
class StepResult:
    ok: bool
    state: GameState
    events: list[LogicEvent]
    reason: str | None = None


def legal_actions(state: GameState) -> list[Action]:
    actions: list[Action] = []
    for row, col in state.stage_positions:
        if state.board[row, col] == 0:
            actions.append(Action(row=row, col=col))
    return actions


def apply_action(state: GameState, action: Action, exposure_service: ExposureService | None = None) -> StepResult:
    if state.game_over:
        return StepResult(ok=False, state=state, events=[], reason="game_over")
    if not is_valid_move(state, action.row, action.col):
        return StepResult(ok=False, state=state, events=[], reason="invalid_move")

    service = exposure_service or ExposureService()
    events: list[LogicEvent] = []

    state.board[action.row, action.col] = state.current_player
    events.append(
        LogicEvent(
            kind="MOVE_APPLIED",
            payload={"row": action.row, "col": action.col, "player": state.current_player},
        )
    )

    if check_win(state.board, action.row, action.col, state.stage, state.game_mode):
        state.game_over = True
        state.winner = state.current_player
        state.end_overlay_closed = False
        _emit_stage_win_events(state, events)
        events.append(LogicEvent(kind="GAME_OVER", payload={"winner": state.winner, "reason": "line_win"}))
        return StepResult(ok=True, state=state, events=events)

    state.exposure_occurred = False
    state.exposure_markers = []
    state.exposure_positions = []
    state.exposure_message = ""

    def _on_exposure_broadcast(text: str, color: tuple[int, int, int]) -> None:
        events.append(LogicEvent.broadcast(text, color))

    while True:
        new_exposure = service.check_all_exposures(state, _on_exposure_broadcast)
        if not new_exposure or state.game_over:
            break
        state.exposure_occurred = True

    if state.exposure_message:
        events.append(LogicEvent(kind="EXPOSURE_MESSAGE", payload={"text": state.exposure_message}))

    # Exposure penalties can create immediate wins; stop the turn flow right here.
    exposure_winner = _winner_from_positions(state, state.exposure_positions)
    if exposure_winner != 0:
        state.game_over = True
        state.winner = exposure_winner
        state.end_overlay_closed = False

    if state.game_over:
        events.append(LogicEvent(kind="GAME_OVER", payload={"winner": state.winner, "reason": "exposure_win"}))
        return StepResult(ok=True, state=state, events=events)

    stage_event = advance_stage_if_needed(state)
    if stage_event == "stage":
        events.append(LogicEvent(kind="STAGE_CHANGED", payload={"stage": state.stage}))
        events.append(LogicEvent.broadcast(STAGE_BROADCASTS[state.stage], WHITE))
    elif stage_event == "final_scoring":
        events.append(LogicEvent.broadcast("茫茫焦土", WHITE))
        state.winner = determine_winner(state.board)
        state.end_overlay_closed = False
        events.append(LogicEvent(kind="GAME_OVER", payload={"winner": state.winner, "reason": "final_scoring"}))

    if (
        not state.show_intro
        and not state.game_over
        and not state.exposure_occurred
        and count_stones(state.board) == 1
    ):
        events.append(LogicEvent.broadcast("一阶段-小型冲突", WHITE))

    if not state.game_over:
        state.current_player = 3 - state.current_player
        events.append(LogicEvent(kind="TURN_SWITCHED", payload={"current_player": state.current_player}))

    return StepResult(ok=True, state=state, events=events)


def simulate_action(state: GameState, action: Action, exposure_service: ExposureService | None = None) -> StepResult:
    sim_state = state.clone()
    return apply_action(sim_state, action, exposure_service=exposure_service)


def _emit_stage_win_events(state: GameState, events: list[LogicEvent]) -> None:
    if state.stage not in WIN_BROADCASTS:
        return

    text = WIN_BROADCASTS[state.stage]
    if state.stage == 2:
        events.append(LogicEvent.broadcast(text, (255, 150, 150)))
    elif state.stage == 4:
        events.append(LogicEvent.broadcast(text, (255, 215, 0)))
    elif state.stage == 3:
        events.append(LogicEvent.broadcast(text, RED))
    else:
        events.append(LogicEvent.broadcast(text, WHITE))


def _winner_from_positions(state: GameState, positions: list[tuple[int, int]]) -> int:
    for row, col in positions:
        if not (0 <= row < state.board.shape[0] and 0 <= col < state.board.shape[1]):
            continue
        if int(state.board[row, col]) == 0:
            continue
        if check_win(state.board, row, col, state.stage, state.game_mode):
            return int(state.board[row, col])
    return 0
