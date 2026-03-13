"""Application entrypoint."""

import argparse

from src.fenghuo_chess.ai.selfplay_runner import run_selfplay
from src.fenghuo_chess.application.match_runner import MatchMode


def main() -> None:
    parser = argparse.ArgumentParser(description="FengHuoChess launcher")
    parser.add_argument("--match-mode", choices=["pvp", "pve", "eve"], default="pvp")
    parser.add_argument("--human-player", choices=[1, 2], type=int, default=1, help="For PVE: 1=black, 2=white")
    parser.add_argument("--headless", action="store_true", help="Run without pygame window")
    parser.add_argument("--games", type=int, default=1, help="Number of games for headless run")
    parser.add_argument("--output", type=str, default="", help="Optional JSONL output path for headless run")
    args = parser.parse_args()

    match_mode: MatchMode = args.match_mode  # type: ignore[assignment]
    if args.headless:
        if match_mode != "eve":
            parser.error("--headless currently supports --match-mode eve only")
        summary = run_selfplay(games=max(1, args.games), output_path=(args.output or None))
        print(
            f"games={summary.games} p1_wins={summary.p1_wins} "
            f"p2_wins={summary.p2_wins} draws={summary.draws} steps={summary.steps}"
        )
        return

    from src.fenghuo_chess.ui.pygame_main import run

    run(match_mode=match_mode, human_player=args.human_player)


if __name__ == "__main__":
    main()
