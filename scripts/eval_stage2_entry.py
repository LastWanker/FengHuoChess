"""Evaluate whether self-play reliably enters stage 2."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fenghuo_chess.ai.baselines.heuristic_ai import HeuristicAISource, MasterAISource
from src.fenghuo_chess.ai.baselines.random_ai import RandomAISource
from src.fenghuo_chess.application.logic_api import apply_action, legal_actions
from src.fenghuo_chess.application.match_runner import build_match_runner
from src.fenghuo_chess.domain.models import create_initial_state
from src.fenghuo_chess.services.exposure_service import ExposureService


SourceName = str


@dataclass
class Stage2EvalSummary:
    games: int
    stage2_entries: int
    stage1_finishes: int
    p1_wins: int
    p2_wins: int
    draws: int
    total_steps: int

    @property
    def stage2_entry_rate(self) -> float:
        return self.stage2_entries / self.games if self.games else 0.0

    @property
    def avg_steps(self) -> float:
        return self.total_steps / self.games if self.games else 0.0


def _make_source(name: SourceName, seed: int) -> object:
    key = name.strip().lower()
    rng = random.Random(seed)
    if key == "weak":
        return RandomAISource(rng=rng)
    if key == "baseline":
        return HeuristicAISource(rng=rng)
    if key == "master":
        return MasterAISource(rng=rng)
    raise ValueError(f"Unsupported source: {name}. Supported: weak, baseline, master")


def _run_one_game(*, p1: SourceName, p2: SourceName, seed: int) -> tuple[bool, int, int, int]:
    black = _make_source(p1, seed + 11)
    white = _make_source(p2, seed + 29)
    runner = build_match_runner("eve", with_ui=False, black_source=black, white_source=white)

    state = create_initial_state()
    state.show_intro = False
    exposure = ExposureService()
    reached_stage2 = state.stage >= 2
    steps = 0

    while not state.game_over:
        legal = legal_actions(state)
        if not legal:
            break

        action = runner.select_action(state, legal)
        if action is None:
            break

        step = apply_action(state, action, exposure_service=exposure)
        if not step.ok:
            break

        steps += 1
        if state.stage >= 2:
            reached_stage2 = True

    return reached_stage2, state.stage, state.winner, steps


def evaluate_stage2_entry(*, games: int, p1: SourceName, p2: SourceName, seed: int) -> Stage2EvalSummary:
    stage2_entries = 0
    stage1_finishes = 0
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_steps = 0

    for game_idx in range(games):
        reached_stage2, final_stage, winner, steps = _run_one_game(
            p1=p1,
            p2=p2,
            seed=seed + game_idx * 97,
        )
        if reached_stage2:
            stage2_entries += 1
        if final_stage == 1:
            stage1_finishes += 1

        if winner == 1:
            p1_wins += 1
        elif winner == 2:
            p2_wins += 1
        else:
            draws += 1
        total_steps += steps

    return Stage2EvalSummary(
        games=games,
        stage2_entries=stage2_entries,
        stage1_finishes=stage1_finishes,
        p1_wins=p1_wins,
        p2_wins=p2_wins,
        draws=draws,
        total_steps=total_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate how often headless AI self-play reaches stage 2"
    )
    parser.add_argument("--games", type=int, default=30, help="Number of games")
    parser.add_argument("--p1", default="baseline", choices=["weak", "baseline", "master"])
    parser.add_argument("--p2", default="baseline", choices=["weak", "baseline", "master"])
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be > 0")

    summary = evaluate_stage2_entry(
        games=args.games,
        p1=args.p1,
        p2=args.p2,
        seed=args.seed,
    )

    print("--- Stage2 Entry Eval ---")
    print(f"p1={args.p1} p2={args.p2} games={summary.games} seed={args.seed}")
    print(
        "stage2_entries="
        f"{summary.stage2_entries} stage1_finishes={summary.stage1_finishes} "
        f"stage2_entry_rate={summary.stage2_entry_rate:.3f}"
    )
    print(
        f"p1_wins={summary.p1_wins} p2_wins={summary.p2_wins} draws={summary.draws} "
        f"avg_steps={summary.avg_steps:.2f}"
    )


if __name__ == "__main__":
    main()
