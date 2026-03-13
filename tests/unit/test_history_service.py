from src.fenghuo_chess.domain.models import create_initial_state
from src.fenghuo_chess.services.history_service import HistoryService


def test_undo_moves_cursor_back_one_snapshot():
    state = create_initial_state()
    history = HistoryService()
    history.reset(state)

    state.board[7, 7] = 1
    history.push(state)
    state.board[7, 8] = 2
    history.push(state)

    restored = history.undo()
    assert restored is not None
    assert restored.board[7, 7] == 1
    assert restored.board[7, 8] == 0


