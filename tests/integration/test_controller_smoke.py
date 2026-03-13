from src.fenghuo_chess.application.game_controller import GameController


def test_controller_can_make_and_undo_move():
    c = GameController()
    c.state.show_intro = False
    assert c.make_move(7, 7)
    assert c.undo()


