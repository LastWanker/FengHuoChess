from src.fenghuo_chess.domain.models import create_initial_state
from src.fenghuo_chess.domain.rules import get_stage_positions
from src.fenghuo_chess.services.exposure_service import ExposureService


def _stage3_state(mode: str):
    state = create_initial_state()
    state.show_intro = False
    state.stage = 3
    state.game_mode = mode
    state.stage_positions = get_stage_positions(3)
    return state


def _noop_broadcast(text: str, color: tuple[int, int, int]) -> None:
    _ = text, color


def test_stage3_slow_three_in_line_does_not_trigger_exposure():
    state = _stage3_state("slow")
    state.board[7, 9] = 1
    state.board[7, 10] = 1
    state.board[7, 11] = 1

    service = ExposureService()
    triggered = service.check_all_exposures(state, _noop_broadcast)

    assert not triggered
    assert state.exposure_positions == []
    assert not state.game_over


def test_stage3_slow_four_in_line_triggers_exposure():
    state = _stage3_state("slow")
    state.board[7, 8] = 1
    state.board[7, 9] = 1
    state.board[7, 10] = 1
    state.board[7, 11] = 1

    service = ExposureService()
    triggered = service.check_all_exposures(state, _noop_broadcast)

    assert triggered
    assert (7, 12) in state.exposure_positions
    assert int(state.board[7, 12]) == 2
    assert not state.game_over


def test_stage3_fast_three_in_line_still_triggers_exposure():
    state = _stage3_state("fast")
    state.board[7, 9] = 1
    state.board[7, 10] = 1
    state.board[7, 11] = 1

    service = ExposureService()
    triggered = service.check_all_exposures(state, _noop_broadcast)

    assert triggered
    assert (7, 12) in state.exposure_positions
    assert int(state.board[7, 12]) == 2
