from src.fenghuo_chess.application.logic_api import Action, apply_action
from src.fenghuo_chess.domain.models import create_initial_state


class _FakeExposureService:
    def check_all_exposures(self, state, broadcast):
        # Simulate one exposure penalty that creates a line for player 2.
        state.board[7, 7] = 2
        state.board[7, 8] = 2
        state.board[7, 9] = 2
        state.exposure_positions = [(7, 9)]
        return False


def test_apply_action_checks_win_after_exposure_penalty():
    state = create_initial_state()
    state.show_intro = False
    state.stage = 1
    state.stage_positions = [(7, 6), (7, 7), (7, 8), (7, 9)]

    result = apply_action(state, Action(7, 6), exposure_service=_FakeExposureService())

    assert result.ok
    assert state.game_over
    assert state.winner == 2
    assert any(event.kind == "GAME_OVER" for event in result.events)

