from src.fenghuo_chess.application.game_controller import GameController
from src.fenghuo_chess.ui.input_mapper import compute_layout
from src.fenghuo_chess.ui.pygame_main import _handle_click


def test_end_overlay_close_does_not_resume_finished_game():
    c = GameController(match_mode="eve", with_ui=True)
    c.state.show_intro = False
    c.state.game_over = True
    c.state.winner = 1

    layout = compute_layout(1280, 820)
    result = _handle_click(c, layout, layout.end_close_rect.center)

    assert result == "end_close"
    assert c.state.game_over
    assert c.state.end_overlay_closed


def test_undo_is_available_after_game_over():
    c = GameController(match_mode="pvp", with_ui=True)
    c.state.show_intro = False
    assert c.make_move(7, 7)
    c.state.game_over = True

    layout = compute_layout(1280, 820)
    result = _handle_click(c, layout, layout.undo_rect.center)

    assert result == "undo"
    assert c.state.board[7, 7] == 0


def test_restart_is_available_after_game_over():
    c = GameController(match_mode="pvp", with_ui=True)
    c.state.show_intro = False
    c.state.game_over = True

    layout = compute_layout(1280, 820)
    first = _handle_click(c, layout, layout.restart_rect.center)
    second = _handle_click(c, layout, layout.restart_rect.center)

    assert first == "restart"
    assert second == "restart"
    assert not c.state.game_over
