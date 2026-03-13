from src.fenghuo_chess.application.logic_api import Action, legal_actions, simulate_action
from src.fenghuo_chess.domain.models import create_initial_state


def test_simulate_action_does_not_mutate_original_state():
    state = create_initial_state()
    state.show_intro = False

    action = Action(7, 7)
    before = state.board.copy()
    simulated = simulate_action(state, action)

    assert simulated.ok
    assert (state.board == before).all()
    assert simulated.state.board[7, 7] == 1


def test_legal_actions_initial_stage_is_center_3x3():
    state = create_initial_state()
    actions = legal_actions(state)

    assert len(actions) == 9
    assert Action(7, 7) in actions
