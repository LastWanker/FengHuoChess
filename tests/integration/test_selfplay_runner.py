from src.fenghuo_chess.ai.selfplay_runner import run_selfplay


def test_run_selfplay_single_game_returns_summary():
    summary = run_selfplay(games=1)

    assert summary.games == 1
    assert summary.steps > 0
    assert summary.p1_wins + summary.p2_wins + summary.draws == 1
