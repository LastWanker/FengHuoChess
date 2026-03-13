import numpy as np

from src.fenghuo_chess.application.game_controller import GameController


def test_pve_human_then_ai_can_progress():
    c = GameController(match_mode="pve", with_ui=False)
    c.state.show_intro = False

    assert c.make_move(7, 7)
    if not c.state.game_over:
        assert c.step_turn()

    assert int(np.count_nonzero(c.state.board)) >= 2


def test_eve_can_advance_without_human_input():
    c = GameController(match_mode="eve", with_ui=False)
    c.state.show_intro = False

    for _ in range(6):
        if c.state.game_over:
            break
        assert c.step_turn()

    assert int(np.count_nonzero(c.state.board)) >= 1
    assert not c.make_move(7, 7)


def test_pve_human_white_starts_with_ai_black_move():
    c = GameController(match_mode="pve", with_ui=False, human_player=2)
    c.state.show_intro = False

    assert c.step_turn()
    assert int(np.count_nonzero(c.state.board)) >= 1
    if not c.state.game_over:
        assert c.state.current_player == 2


def test_controller_default_intro_slots_are_human_vs_baseline():
    c = GameController(with_ui=True)
    assert c.state.intro_p1_source == "human"
    assert c.state.intro_p2_source == "baseline"
