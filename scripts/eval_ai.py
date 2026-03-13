"""Evaluate baseline AI sources in headless EVE matches."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fenghuo_chess.ai.baselines.heuristic_ai import HeuristicAISource
from src.fenghuo_chess.ai.baselines.random_ai import RandomAISource
from src.fenghuo_chess.ai.selfplay_runner import SelfPlaySummary, run_selfplay
from src.fenghuo_chess.application.match_runner import build_match_runner


def _make_source(name: str, rng_seed: int) -> object:
    key = name.strip().lower()
    if key == "random":
        return RandomAISource(rng=random.Random(rng_seed))
    if key == "heuristic":
        return HeuristicAISource(rng=random.Random(rng_seed))
    raise ValueError(f"Unsupported source: {name}. Supported: random, heuristic")


def _print_round(label: str, summary: SelfPlaySummary) -> None:
    print(
        f"{label}: games={summary.games} "
        f"p1_wins={summary.p1_wins} p2_wins={summary.p2_wins} "
        f"draws={summary.draws} steps={summary.steps}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline AIs in headless mode")
    parser.add_argument("--candidate", default="heuristic", choices=["random", "heuristic"])
    parser.add_argument("--opponent", default="random", choices=["random", "heuristic"])
    parser.add_argument("--games", type=int, default=100, help="Games per color assignment")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for source RNGs")
    parser.add_argument("--verbose", action="store_true", help="Print per-round summaries")
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be > 0")

    # Run pair once for optional verbose line output.
    black_1 = _make_source(args.candidate, args.seed + 11)
    white_1 = _make_source(args.opponent, args.seed + 23)
    runner_1 = build_match_runner("eve", with_ui=False, black_source=black_1, white_source=white_1)
    s1 = run_selfplay(games=args.games, runner=runner_1)

    black_2 = _make_source(args.opponent, args.seed + 37)
    white_2 = _make_source(args.candidate, args.seed + 53)
    runner_2 = build_match_runner("eve", with_ui=False, black_source=black_2, white_source=white_2)
    s2 = run_selfplay(games=args.games, runner=runner_2)

    if args.verbose:
        _print_round(f"{args.candidate}(black) vs {args.opponent}(white)", s1)
        _print_round(f"{args.opponent}(black) vs {args.candidate}(white)", s2)

    candidate_wins = s1.p1_wins + s2.p2_wins
    opponent_wins = s1.p2_wins + s2.p1_wins
    draws = s1.draws + s2.draws
    total_games = args.games * 2
    total_steps = s1.steps + s2.steps
    win_rate = candidate_wins / total_games
    avg_steps = total_steps / total_games

    print("--- Eval Summary ---")
    print(f"candidate={args.candidate} opponent={args.opponent}")
    print(f"games={total_games} candidate_wins={candidate_wins} opponent_wins={opponent_wins} draws={draws}")
    print(f"candidate_win_rate={win_rate:.3f} avg_steps={avg_steps:.2f}")


if __name__ == "__main__":
    main()
